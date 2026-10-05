"""Terminal-Bench 2.0 wrapper for Xolotl: build the Linux binary, run `harbor run`, parse the job, scan it for the API key,
then write results/<run_id>.json and a HISTORY.md row. Standard library only (Harbor itself is a separate uv tool).

    py bench/tb2/run.py --smoke --note "tb2 smoke"       # the 2 smoke tasks
    py bench/tb2/run.py --note "tb2 v1 baseline"         # tasks.txt (15 tasks)
    py bench/tb2/run.py --task fix-git --dry-run         # print the Harbor command (key redacted) and let Harbor validate it
"""
import argparse
import hashlib
import itertools
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))  # `python bench/tb2/run.py` and Harbor's PYTHONPATH both import xolotl_harbor as a top-level package
from xolotl_harbor import core  # noqa: E402

DATASET = "terminal-bench@2.0"
AGENT = "xolotl_harbor.agent:XolotlAgent"
MODEL = "zai/glm-5.3-flash"
HISTORY_HEADER = (
    "| Date | Run | Policy | Tasks | Pass | Tokens (in / out) | Median agent | Note |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def read_task_names(path: Path) -> list[str]:
    """Task names, one per line; blank lines and `#` comments are skipped."""
    lines = (ln.strip() for ln in path.read_text(encoding="utf-8").splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def tasks_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:12]


def _num(v: object) -> int | float | None:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _agent_seconds(timing: object) -> float | None:
    try:
        t = timing if isinstance(timing, dict) else {}
        return round((datetime.fromisoformat(t["finished_at"]) - datetime.fromisoformat(t["started_at"])).total_seconds(), 3)
    except (KeyError, TypeError, ValueError):
        return None


def parse_trial(trial_dir: Path) -> dict | None:
    """One Harbor trial folder (<job>/<task>__<id>/) as a flat dict, or None if it holds no result.json.

    The verifier decides: reward >= 1 is a pass whatever Xolotl's exit code or an agent timeout. Without a reward the
    verifier never ran (setup, environment or verifier exception), which says nothing about the agent: infra_error. A
    reward below 1 is infra_error only if xolotl.jsonl shows the agent never really ran (core.metrics_from_jsonl)."""
    try:
        r = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(r, dict) or not isinstance(r.get("task_name"), str):
        return None
    rewards = (r.get("verifier_result") or {}).get("rewards") or {}
    reward = _num(rewards.get("reward", next(iter(rewards.values()), None)) if isinstance(rewards, dict) else None)
    exc = (r.get("exception_info") or {}).get("exception_type")
    try:
        text = (trial_dir / "agent" / "xolotl.jsonl").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    m = core.metrics_from_jsonl(text)
    try:
        exit_code = int((trial_dir / "agent" / "exit_code").read_text().strip())
    except (OSError, ValueError):
        exit_code = None
    if reward is not None and reward >= 1.0:
        status, infra = "pass", ""
    elif reward is None:
        status, infra = "infra_error", exc or "no verifier result"
    elif m["infra_reason"]:
        status, infra = "infra_error", m["infra_reason"]
    else:
        status, infra = "fail", ""
    ar = r.get("agent_result") or {}
    tokens = {k: _num(ar.get(f)) if _num(ar.get(f)) is not None else m["tokens"][k]
              for k, f in (("prompt", "n_input_tokens"), ("completion", "n_output_tokens"), ("cached", "n_cache_tokens"))}
    return {
        "task": r["task_name"], "trial": trial_dir.name, "status": status, "reward": reward, "exception": exc,
        "infra_reason": infra, "exit_code": exit_code, "tokens": tokens, "policy_hash": m["policy_hash"],
        "metadata": ar.get("metadata") or {}, "agent_time_s": _agent_seconds(r.get("agent_execution")),
        "xolotl_version": (r.get("agent_info") or {}).get("version") or "",
    }


def parse_job(job_dir: Path) -> list[dict]:
    """One dict per trial of a Harbor job folder, ordered by task name."""
    trials = (parse_trial(d) for d in sorted(job_dir.iterdir()) if d.is_dir())
    return sorted((t for t in trials if t), key=lambda t: (t["task"], t["trial"]))


def summarize(trials: list[dict]) -> dict:
    """Pass rate excludes infra_error trials (they say nothing about the agent); tokens are summed over every trial."""
    ran = [t for t in trials if t["status"] != "infra_error"]
    passed = sum(t["status"] == "pass" for t in ran)
    times = [t["agent_time_s"] for t in ran if t["agent_time_s"] is not None]
    return {
        "pass": passed, "total": len(trials), "infra": len(trials) - len(ran),
        "pass_rate": round(passed / len(ran), 3) if ran else 0.0,
        "tokens": {k: sum(t["tokens"][k] for t in trials) for k in ("prompt", "completion", "cached")},
        "median_agent_time_s": round(statistics.median(times), 3) if times else 0.0,
    }


def history_row(result: dict) -> str:
    """One HISTORY.md table row (no trailing newline) for a results/<run_id>.json dict."""
    s = result["summary"]
    ran = s["total"] - s["infra"]
    cells = [
        datetime.fromisoformat(result["started"]).strftime("%Y-%m-%d %H:%M"),
        result["run_id"] + (" (dirty)" if result.get("dirty") else ""),
        (result.get("policy_hash") or "")[:8],
        result.get("tasks_hash") or "",
        f"{s['pass']}/{ran} ({s['pass_rate'] * 100:.1f}%)" + (f" \u26a0{s['infra']} infra" if s["infra"] else ""),
        f"{s['tokens']['prompt']} / {s['tokens']['completion']}",
        f"{s['median_agent_time_s']:.1f}s",
        (result.get("note") or "").replace("|", "\\|").replace("\n", " "),
    ]
    return "| " + " | ".join(cells) + " |"


def secret_scan(root: Path, key: str) -> list[Path]:
    """Files under `root` containing the first 8 characters of `key`. Never prints or returns the key itself."""
    needle = key[:8].encode("utf-8")
    hits = []
    for p in sorted(root.rglob("*")):
        try:
            if p.is_file() and needle in p.read_bytes():
                hits.append(p)
        except OSError:
            continue
    return hits


def harbor_command(harbor: str, tasks: list[str], binary: Path, jobs_dir: Path, job_name: str, n: int) -> list[str]:
    cmd = [harbor, "run", "-d", DATASET, "-a", AGENT, "-m", MODEL, "--ak", f"binary={binary}", "-n", str(n),
           "-o", str(jobs_dir), "--job-name", job_name, "-q"]
    for t in tasks:
        cmd += ["-i", t]
    return cmd


def harbor_env(key: str | None) -> dict[str, str]:
    """The Harbor child's environment: the key travels only here (never argv, files or logs). PYTHONPATH makes
    xolotl_harbor importable inside Harbor's own interpreter; PYTHONUTF8 keeps its rich output from crashing on cp1252."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(HERE), env.get("PYTHONPATH")) if p)
    env["PYTHONUTF8"] = "1"
    if key:
        env[core.KEY_ENV] = key
    return env


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.decode("utf-8", errors="replace").strip() if r.returncode == 0 else ""


def _print_summary(result: dict, path: Path) -> None:
    s = result["summary"]
    width = max((len(t["task"]) for t in result["trials"]), default=4)
    print(f"\n{'task':<{width}}  status       reward  agent time  tokens in/out")
    for t in result["trials"]:
        agent_s = "-" if t["agent_time_s"] is None else f"{t['agent_time_s']:.0f}s"
        print(f"{t['task']:<{width}}  {t['status']:<11}  {t['reward']!s:<6}  {agent_s:<10}  "
              f"{t['tokens']['prompt']}/{t['tokens']['completion']}"
              + (f"  ({t['infra_reason']})" if t["infra_reason"] else ""))
    ran = s["total"] - s["infra"]
    print(f"\n{s['pass']}/{ran} passed ({s['pass_rate'] * 100:.1f}%), {s['infra']} infra; "
          f"tokens {s['tokens']['prompt']} in / {s['tokens']['completion']} out ({s['tokens']['cached']} cached); "
          f"median agent time {s['median_agent_time_s']:.1f}s")
    if s["infra"]:
        print(f"WARNING: {s['infra']} trial(s) are infra_error and excluded from the pass rate; re-run them before trusting this row")
    print(f"results: {path}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="run smoke.txt (2 tasks)")
    g.add_argument("--tasks-file", type=Path, help="one task name per line (default: tasks.txt)")
    g.add_argument("--task", action="append", metavar="NAME", help="a task name; repeat for several")
    p.add_argument("-n", type=int, default=1, help="Harbor concurrency (default 1: least Docker load)")
    p.add_argument("--note", default="", help="free text for the HISTORY row")
    p.add_argument("--no-build", action="store_true", help="use the already built Linux binary")
    p.add_argument("--dry-run", action="store_true", help="print the Harbor command (key redacted) and let Harbor validate it; no run")
    p.add_argument("--jobs-dir", type=Path, default=HERE / "jobs", help="Harbor job output (git-ignored)")
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    return p


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.task:
        tasks = args.task
    else:
        tasks = read_task_names(args.tasks_file or HERE / ("smoke.txt" if args.smoke else "tasks.txt"))
    if not tasks:
        print("error: no tasks selected", file=sys.stderr)
        return 1
    harbor = shutil.which("harbor")
    if not harbor:
        print("error: `harbor` is not on PATH; install it with `uv tool install harbor`", file=sys.stderr)
        return 1
    import build_linux  # sibling module (HERE is on sys.path)
    binary = build_linux.binary_path(REPO_ROOT)
    key = core.read_api_key()
    started = datetime.now().astimezone()
    sha = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--", ".", ":(exclude)bench/tb2/results"))
    base = f"{started:%Y%m%d-%H%M}-{sha[:7]}"
    run_id = next(r for r in (base if n == 1 else f"{base}-{n}" for n in itertools.count(1))
                  if not (args.jobs_dir / r).exists() and not (args.results_dir / f"{r}.json").exists())
    cmd = harbor_command(harbor, tasks, binary, args.jobs_dir, run_id, args.n)

    if args.dry_run:
        print("harbor command:", subprocess.list2cmdline(cmd), flush=True)
        print(f"env: {core.KEY_ENV}={'<redacted>' if key else '<NOT FOUND>'}  PYTHONPATH={HERE}  PYTHONUTF8=1", flush=True)
        print(f"tasks ({len(tasks)}), tasks_hash {tasks_hash(tasks)}, run_id {run_id}", flush=True)
        return subprocess.run([*cmd, "--dry-run"], env=harbor_env(key), stdin=subprocess.DEVNULL).returncode
    if not key:
        print(f"error: {core.KEY_ENV} not found in the environment or the Windows User scope", file=sys.stderr)
        return 1
    if not args.no_build and build_linux.main([]) != 0:
        return 1
    if not binary.is_file():
        print(f"error: {binary} does not exist; run without --no-build", file=sys.stderr)
        return 1

    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}: {len(tasks)} task(s), -n {args.n}", flush=True)
    code = subprocess.run(cmd, env=harbor_env(key), stdin=subprocess.DEVNULL).returncode  # streams Harbor's output
    job_dir = args.jobs_dir / run_id
    if not job_dir.is_dir():
        print(f"error: harbor exited {code} and produced no job folder {job_dir}", file=sys.stderr)
        return 1

    hits = secret_scan(job_dir, key)
    if hits:  # paths only; results are not written
        print("error: the API key's prefix was found in the job folder; refusing to write results:", file=sys.stderr)
        for p in hits:
            print(f"  {p}", file=sys.stderr)
        return 3

    trials = parse_job(job_dir)
    partial = code != 0 or len(trials) < len(tasks)
    result = {
        "run_id": run_id, "started": started.isoformat(timespec="seconds"), "git_sha": sha, "dirty": dirty,
        "harbor_version": subprocess.run([harbor, "--version"], capture_output=True, text=True).stdout.strip(),
        "xolotl_version": next((t["xolotl_version"] for t in trials if t["xolotl_version"]), ""),
        "dataset": DATASET, "model": MODEL, "concurrency": args.n,
        "policy_hash": next((t["policy_hash"] for t in trials if t["policy_hash"]), ""),
        "tasks": tasks, "tasks_hash": tasks_hash(tasks), "note": args.note, "partial": partial,
        "harbor_exit_code": code, "trials": trials, "summary": summarize(trials),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / f"{run_id}.json"
    with path.open("x", encoding="utf-8", newline="\n") as f:  # exclusive: never overwrite an earlier run (run_id is unique)
        f.write(json.dumps(result, indent=2) + "\n")
    if not partial:
        history = args.results_dir / "HISTORY.md"
        header = HISTORY_HEADER if not history.exists() or history.stat().st_size == 0 else ""
        with history.open("a", encoding="utf-8", newline="\n") as f:
            f.write(header + history_row(result) + "\n")
    else:
        print(f"WARNING: partial run (harbor exit {code}, {len(trials)}/{len(tasks)} trials); no HISTORY row", file=sys.stderr)
    _print_summary(result, path)
    print(f"jobs: {job_dir}")
    return 1 if partial else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    sys.exit(main(sys.argv[1:]))
