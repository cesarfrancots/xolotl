"""Xolotl bench runner: runs Xolotl headless on a task suite and records pass rate, tokens and timing.

Standard library only. The pure helpers come first; orchestration (run_attempt, selfcheck, main) lives below them.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Header + separator of results/HISTORY.md. Ends with a newline so a row can be appended straight after it.
HISTORY_HEADER = (
    "| Date | Run | Policy | Pass | Tokens (prompt / completion) | Median wall | Note |\n"
    "|---|---|---|---|---|---|---|\n"
)


def slug(path: str) -> str:
    """Same rule as Rust `store::project_slug`: anything but ASCII alphanumerics, '-', '_' and '.' becomes '-'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", path)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _str_list(v: object) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def load_task(task_dir: Path) -> dict:
    """Read and validate task.json, filling defaults. Raises ValueError naming the missing or bad field."""
    def bad(msg: str) -> ValueError:
        return ValueError(f"task {task_dir.name}: {msg}")

    try:
        raw = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise bad(f"cannot read task.json: {e}") from e
    if not isinstance(raw, dict):
        raise bad("task.json must be a JSON object")
    t = {"tags": [], "requires": [], "timeout_s": 600, "verify_timeout_s": 120, "budget": None, **raw}
    for k in ("id", "title", "prompt"):
        if not isinstance(t.get(k), str) or not t[k].strip():
            raise bad(f"field '{k}' is required and must be a non-empty string")
    if t["id"] != task_dir.name:
        raise bad(f"field 'id' ({t['id']!r}) must equal the folder name")
    if not _str_list(t.get("verify")) or not t["verify"]:  # a blank command would pass vacuously
        raise bad("field 'verify' is required and must be a non-empty list of command strings")
    for k in ("tags", "requires"):
        if not _str_list(t[k]):
            raise bad(f"field '{k}' must be a list of non-empty strings")
    for k in ("timeout_s", "verify_timeout_s"):
        if not _is_int(t[k]) or t[k] <= 0:
            raise bad(f"field '{k}' must be a positive integer")
    if t["budget"] is not None and (not _is_int(t["budget"]) or t["budget"] <= 0):
        raise bad("field 'budget' must be a positive integer or null")
    return t


def _events(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        try:
            ev = json.loads(line)
        except (ValueError, RecursionError):  # truncated, binary, or absurdly nested garbage
            continue
        if isinstance(ev, dict):
            yield ev


def _count(usage: object, key: str) -> int:
    v = usage.get(key) if isinstance(usage, dict) else None
    return v if _is_int(v) else 0


def metrics_from_events(lines: Iterable[str]) -> dict:
    """Metrics for one attempt from Xolotl's JSON-lines event stream. Lines that aren't JSON objects are ignored."""
    m = {"model_calls": 0, "tool_calls": 0, "tokens": {"prompt": 0, "completion": 0, "cached": 0},
         "compactions": {"stage1": 0, "stage2": 0}, "completion_check": False, "errors": 0,
         "policy_hash": "", "interrupted": False}
    for ev in _events(lines):
        kind = ev.get("type")
        if kind == "assistant":
            m["model_calls"] += 1
            for k in m["tokens"]:
                m["tokens"][k] += _count(ev.get("usage"), k)
        elif kind == "tool_result":
            m["tool_calls"] += 1
        elif kind == "compaction" and _is_int(ev.get("stage")) and ev["stage"] in (1, 2):
            m["compactions"][f"stage{ev['stage']}"] += 1
        elif kind == "nudge" and ev.get("reason") == "completion_check":
            m["completion_check"] = True
        elif kind == "error":
            m["errors"] += 1
        elif kind == "interrupt":
            m["interrupted"] = True
        elif kind == "session_start" and isinstance(ev.get("policy_hash"), str):
            m["policy_hash"] = ev["policy_hash"]
    return m


def session_dir(home: Path, cwd: str) -> Path | None:
    """Newest session folder for `cwd` under <home>/.xolotl/projects/<slug>/sessions, or None."""
    root = home / ".xolotl" / "projects" / slug(cwd) / "sessions"
    if not root.is_dir():
        return None
    # ponytail: newest = max folder name (ids are UTC YYYYMMDD-HHMMSS-xxxx, as in Rust); two sessions in one second tie on the random suffix.
    return max((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, default=None)


def summarize(attempts: list[dict]) -> dict:
    """Run summary. Skipped attempts count in `total` but not in `pass_rate` or `per_task` denominators."""
    ran = [a for a in attempts if a.get("status") != "skipped"]
    passed = sum(1 for a in ran if a.get("pass"))
    per: dict[str, list[int]] = {}  # task -> [passes, attempts that ran]
    for a in attempts:
        counts = per.setdefault(a["task"], [0, 0])
        if a.get("status") != "skipped":
            counts[0] += bool(a.get("pass"))
            counts[1] += 1
    walls = [a["wall_s"] for a in ran if isinstance(a.get("wall_s"), (int, float))]
    return {
        "pass": passed,
        "total": len(attempts),
        "skipped": len(attempts) - len(ran),
        "pass_rate": round(passed / len(ran), 3) if ran else 0.0,
        "per_task": {t: f"{p}/{n}" if n else "skipped" for t, (p, n) in per.items()},
        "tokens": {k: sum(_count(a.get("tokens"), k) for a in attempts) for k in ("prompt", "completion", "cached")},
        "median_wall_s": round(statistics.median(walls), 3) if walls else 0.0,
        "total_wall_s": round(sum(walls), 3),
    }


def history_row(result: dict) -> str:
    """One HISTORY.md table row (no trailing newline) for a results/<run_id>.json dict."""
    s = result["summary"]
    ran = s["total"] - s["skipped"]
    cells = [
        datetime.fromisoformat(result["started"]).strftime("%Y-%m-%d %H:%M"),
        result["run_id"],
        (result.get("policy_hash") or "")[:8],
        f"{s['pass']}/{ran} ({s['pass_rate'] * 100:.1f}%)",
        f"{s['tokens']['prompt']} / {s['tokens']['completion']}",
        f"{s['median_wall_s']:.1f}s",
        (result.get("note") or "").replace("|", "\\|").replace("\n", " "),  # a note must not break the table
    ]
    return "| " + " | ".join(cells) + " |"


# ---- orchestration ---------------------------------------------------------------------------------------------


def find_bash() -> list[str]:
    """Argv prefix for running a verify command through a shell."""
    if os.name != "nt":
        return ["bash"]
    # Git for Windows' bash, found from `git` (never System32\bash.exe, the WSL stub). git.exe lives in <Git>\cmd or
    # <Git>\bin, or in <Git>\mingw64\bin when PATH comes from Git Bash: <Git> is parents[1] or parents[2].
    git = shutil.which("git")
    for root in Path(git).parents[1:3] if git else []:
        if (root / "bin" / "bash.exe").is_file():
            return [str(root / "bin" / "bash.exe")]
    return ["cmd", "/c"]


def _rm(path: Path) -> None:
    """Delete a file or a tree. Git's read-only object files would make a plain rmtree fail on Windows."""
    def writable(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, **{"onexc" if sys.version_info >= (3, 12) else "onerror": writable})
    elif path.is_symlink() or path.exists():
        path.unlink()


def _pump(stream, buf: bytearray) -> None:
    for chunk in iter(lambda: stream.read1(65536), b""):
        buf += chunk


def _join(threads: list[threading.Thread], timeout: float) -> None:
    end = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15)
        else:
            os.killpg(proc.pid, signal.SIGKILL)  # it leads its own session, so pgid == pid even once reaped
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_tree(argv: list[str] | str, cwd: str | Path | None, timeout: float) -> tuple[int | None, bytes, bytes, bool]:
    """Run `argv` with stdout and stderr drained on threads (a full pipe can't block the child) and kill its whole
    process tree after `timeout` seconds. Returns (exit code, stdout, stderr, timed_out)."""
    # ponytail: no Windows Job Object, so a grandchild that outlives a child which exited on its own is not killed
    # (readers just give up on it after 3 s); upgrade path: assign the child to a Job Object with KILL_ON_JOB_CLOSE.
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **flags)
    out, err = bytearray(), bytearray()
    readers = [threading.Thread(target=_pump, args=(s, b), daemon=True) for s, b in ((proc.stdout, out), (proc.stderr, err))]
    for t in readers:
        t.start()
    try:
        deadline = time.monotonic() + timeout
        while proc.poll() is None and time.monotonic() < deadline:  # poll, not wait(): stays interruptible by Ctrl+C
            time.sleep(0.05)
        timed_out = proc.poll() is None
        if not timed_out:
            _join(readers, 3)
        if timed_out or any(t.is_alive() for t in readers):  # on timeout, or when something still holds the pipes
            _kill_tree(proc)
        _join(readers, 3)  # short: an orphan holding a pipe must not hang the runner
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    except BaseException:  # Ctrl+C included: never leave the child tree running
        _kill_tree(proc)
        raise
    finally:
        for stream, t in zip((proc.stdout, proc.stderr), readers):
            if not t.is_alive():  # a reader still blocked on an orphan's pipe must not have it closed under it
                stream.close()
    return proc.returncode, bytes(out), bytes(err), timed_out


def _verify(task: dict, task_dir: Path, work: Path) -> tuple[bool, str]:
    """Put the real hidden checks in work/verify (replacing anything already there), run every verify command in
    `work`, stop at the first failure. Returns (all passed, last 2 KB of output)."""
    _rm(work / "verify")
    shutil.copytree(task_dir / "verify", work / "verify")
    shell = find_bash()
    py = f'"{sys.executable}"' if shell[0] == "cmd" else shlex.quote(sys.executable)
    out, ok = b"", True
    for cmd in task["verify"]:
        cmd = cmd.replace("{python}", py)
        argv = f'cmd /c "{cmd}"' if shell[0] == "cmd" else [*shell, "-c", cmd]
        rc, o, e, timed_out = _run_tree(argv, work, task["verify_timeout_s"])
        out += o + e
        if timed_out:
            out += f"\n[verify timed out after {task['verify_timeout_s']}s]\n".encode()
        ok = rc == 0 and not timed_out
        if not ok:
            break
    return ok, out[-2048:].decode("utf-8", errors="replace")


def _init_repo(path: str) -> None:
    """git init plus one commit, so the agent has a repository to work in."""
    commit = ["-c", "user.name=bench", "-c", "user.email=bench@local", "-c", "commit.gpgsign=false",
              "commit", "-q", "--allow-empty", "-m", "init"]
    for args in (["init", "-q"], ["add", "-A"], commit):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, stdin=subprocess.DEVNULL)


def run_attempt(task: dict, task_dir: Path, xolotl_cmd: list[str], home: Path,
                budget: int | None, keep: bool, attempt: int = 1) -> dict:
    """One attempt: fresh git-initialised copy of repo/, run Xolotl on it, replace verify/ with the real one, verify,
    collect metrics. The temp dir is deleted only if the attempt passed and `keep` is off."""
    missing = [r for r in task["requires"] if shutil.which(r) is None]
    base = {"task": task["id"], "attempt": attempt}
    if missing:
        return {**base, "status": "skipped", "pass": False, "exit_code": None, "timed_out": False, "wall_s": 0.0,
                "verify_output": f"missing toolchain: {', '.join(missing)}", "kept_dir": None, "session_dir": None,
                "model": "", **metrics_from_events([])}

    tmp = tempfile.mkdtemp(prefix="xolotl-bench-")  # the same string goes to --cwd and session_dir (slug parity)
    try:
        shutil.copytree(task_dir / "repo", tmp, dirs_exist_ok=True)
        _init_repo(tmp)
        argv = [*xolotl_cmd, "-p", task["prompt"], "--mode", "yolo", "--cwd", tmp]
        budget = task["budget"] if task["budget"] is not None else budget
        if budget is not None:
            argv += ["--budget", str(budget)]
        t0 = time.monotonic()
        exit_code, stdout, _stderr, timed_out = _run_tree(argv, None, task["timeout_s"])
        wall_s = round(time.monotonic() - t0, 3)

        ok, verify_output = _verify(task, task_dir, Path(tmp))
        lines = stdout.decode("utf-8", errors="replace").split("\n")  # not splitlines(): U+2028 may sit inside a JSON string
        model = next((e["model"] for e in _events(lines) if e.get("type") == "session_start" and isinstance(e.get("model"), str)), "")
        sd = session_dir(home, tmp)
        kept = tmp
        if ok and not keep:
            try:
                _rm(Path(tmp))
                kept = None
            except OSError:  # locked by a straggler process: the dir is still there, so report it
                pass
    except BaseException:  # includes Ctrl+C: the attempt is dropped, so don't leave its dir behind
        _rm(Path(tmp))
        raise
    return {**base, "status": "pass" if ok else "fail", "pass": ok, "exit_code": exit_code, "timed_out": timed_out,
            "wall_s": wall_s, "verify_output": verify_output, "kept_dir": kept, "session_dir": str(sd) if sd else None,
            "model": model, **metrics_from_events(lines)}


def _last_lines(text: str, n: int = 3, limit: int = 300) -> str:
    return " | ".join([ln.strip() for ln in text.splitlines() if ln.strip()][-n:])[-limit:]


def selfcheck(task: dict, task_dir: Path) -> tuple[bool, str]:
    """No model calls: the reference solution must pass the hidden checks, and the untouched repo must fail them."""
    missing = [r for r in task["requires"] if shutil.which(r) is None]
    if missing:
        return True, f"skipped, missing toolchain: {', '.join(missing)}"
    for d in ("repo", "verify", "solution"):
        if not (task_dir / d).is_dir():
            return False, f"missing {d}/ directory"
    with tempfile.TemporaryDirectory(prefix="xolotl-selfcheck-") as tmp:
        solved, unsolved = Path(tmp) / "solved", Path(tmp) / "unsolved"
        shutil.copytree(task_dir / "repo", solved)
        shutil.copytree(task_dir / "solution", solved, dirs_exist_ok=True)
        ok, output = _verify(task, task_dir, solved)
        if not ok:
            return False, f"solution fails verify: {_last_lines(output)}"
        shutil.copytree(task_dir / "repo", unsolved)
        if _verify(task, task_dir, unsolved)[0]:
            return False, "unsolved repo already passes"
    return True, "solution passes, unsolved repo fails"


def _positive(s: str) -> int:
    if not s.isdigit() or int(s) <= 0:
        raise argparse.ArgumentTypeError(f"{s!r} is not a positive integer")
    return int(s)


def _parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(prog="run.py", description="Run Xolotl on the bench task suite.")
    p.add_argument("--tasks", help="comma-separated task ids (default: all)")
    p.add_argument("--repeat", type=_positive, default=1, help="attempts per task (default 1)")
    p.add_argument("--budget", type=_positive, help="--budget passed to Xolotl; a task's own budget wins")
    p.add_argument("--xolotl", help="Xolotl executable, or a .py stand-in (default: build this source tree)")
    p.add_argument("--note", default="", help="free text recorded in the results and HISTORY.md")
    p.add_argument("--keep", action="store_true", help="keep every attempt's temp dir, not only failed ones")
    p.add_argument("--selfcheck", action="store_true", help="check each task is solvable and not already solved (no model calls)")
    p.add_argument("--tasks-dir", type=Path, default=here / "tasks", help="default: bench/tasks")
    p.add_argument("--results-dir", type=Path, default=here / "results", help=argparse.SUPPRESS)
    return p


def _git(*args: str) -> str:
    """Stdout of a git command in the repo root, or "" if it fails."""
    try:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.decode("utf-8", errors="replace").strip() if r.returncode == 0 else ""


def _print_summary(result: dict, path: Path) -> None:
    s = result["summary"]
    width = max((len(t) for t in s["per_task"]), default=4)
    print(f"\n{'task':<{width}}  result")
    for t, r in s["per_task"].items():
        print(f"{t:<{width}}  {r}")
    t = s["tokens"]
    print(f"\n{s['pass']}/{s['total'] - s['skipped']} passed ({s['pass_rate'] * 100:.1f}%), {s['skipped']} skipped; "
          f"tokens {t['prompt']} prompt / {t['completion']} completion ({t['cached']} cached); "
          f"median {s['median_wall_s']:.1f}s, total {s['total_wall_s']:.1f}s")
    print(f"results: {path}")


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if not args.tasks_dir.is_dir():
        print(f"error: tasks dir not found: {args.tasks_dir}", file=sys.stderr)
        return 2
    names = [n for n in args.tasks.split(",") if n] if args.tasks else sorted(p.name for p in args.tasks_dir.iterdir() if p.is_dir())
    try:
        tasks = [(load_task(args.tasks_dir / n), args.tasks_dir / n) for n in names]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not tasks:
        print(f"error: no tasks in {args.tasks_dir}", file=sys.stderr)
        return 2

    if args.selfcheck:
        failed = 0
        for task, task_dir in tasks:
            ok, reason = selfcheck(task, task_dir)
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {task['id']}: {reason}")
        return 1 if failed else 0

    bin_dir = None
    try:
        if args.xolotl:
            cmd = [sys.executable, args.xolotl] if args.xolotl.endswith(".py") else [args.xolotl]
        else:
            print("building: cargo build --release", flush=True)
            try:
                built = subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT).returncode == 0
            except OSError as e:
                print(f"error: cannot run cargo: {e}", file=sys.stderr)
                return 1
            if not built:
                print("error: cargo build failed", file=sys.stderr)
                return 1
            bin_dir = tempfile.mkdtemp(prefix="xolotl-bin-")  # a copy, so a later rebuild can't hit a locked .exe
            cmd = [shutil.copy2(REPO_ROOT / "target" / "release" / ("xolotl.exe" if os.name == "nt" else "xolotl"), bin_dir)]

        try:
            v = subprocess.run([*cmd, "--version"], capture_output=True, stdin=subprocess.DEVNULL, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"error: cannot run xolotl: {e}", file=sys.stderr)
            return 1
        if v.returncode != 0:
            print(f"error: `xolotl --version` exited with {v.returncode}", file=sys.stderr)
            return 1

        started = datetime.now().astimezone()
        sha = _git("rev-parse", "HEAD") or "unknown"
        run_id = f"{started:%Y%m%d-%H%M}-{sha[:7]}"
        total = len(tasks) * args.repeat
        attempts: list[dict] = []
        partial = False
        try:
            for task, task_dir in tasks:
                for i in range(1, args.repeat + 1):
                    a = run_attempt(task, task_dir, cmd, Path.home(), args.budget, args.keep, i)
                    attempts.append(a)
                    note = f" ({a['verify_output']})" if a["status"] == "skipped" else " (timed out)" if a["timed_out"] else ""
                    print(f"[{len(attempts)}/{total}] {task['id']} #{i}: {a['status']:<7} {a['wall_s']:7.1f}s  "
                          f"{a['model_calls']} model calls{note}", flush=True)
        except KeyboardInterrupt:
            partial = True
            print("\ninterrupted: writing partial results", flush=True)

        result = {
            "run_id": run_id, "started": started.isoformat(timespec="seconds"), "git_sha": sha,
            "dirty": bool(_git("status", "--porcelain")),
            "xolotl_version": v.stdout.decode("utf-8", errors="replace").strip(),
            "model": next((a["model"] for a in attempts if a["model"]), ""),
            "policy_hash": next((a["policy_hash"] for a in attempts if a["policy_hash"]), ""),
            "repeat": args.repeat, "budget": args.budget, "note": args.note, "partial": partial,
            "attempts": attempts, "summary": summarize(attempts),
        }
        args.results_dir.mkdir(parents=True, exist_ok=True)
        path = args.results_dir / f"{run_id}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        if not partial:
            history = args.results_dir / "HISTORY.md"
            header = HISTORY_HEADER if not history.exists() or history.stat().st_size == 0 else ""
            with history.open("a", encoding="utf-8", newline="\n") as f:
                f.write(header + history_row(result) + "\n")
        _print_summary(result, path)
        return 130 if partial else 0
    finally:
        if bin_dir:
            shutil.rmtree(bin_dir, ignore_errors=True)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")  # verify output may hold characters the console codepage can't print
    sys.exit(main(sys.argv[1:]))
