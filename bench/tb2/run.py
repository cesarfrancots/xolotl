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
import re
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


# Harbor exceptions raised by the verifier phase. With a real Xolotl session behind them they are the agent's failure
# (a verifier that times out or leaves no/garbled reward is the task's way of saying "not solved"), not infrastructure.
# Anything else without a reward (setup, install, Docker, AddTestsDirError, DownloadVerifierDirError...) is infra.
VERIFIER_EXCEPTIONS = {"VerifierTimeoutError", "RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError"}


def parse_trial(trial_dir: Path) -> dict | None:
    """One Harbor trial folder (<job>/<task>__<id>/) as a flat dict, or None if it holds no result.json.

    The verifier decides: reward >= 1 is a pass whatever Xolotl's exit code or an agent timeout. A reward below 1 is
    a fail, unless xolotl.jsonl shows the agent never really ran (core.metrics_from_jsonl): then infra_error. Without
    a reward, a verifier-phase exception (VERIFIER_EXCEPTIONS) is a fail when a real session ran; any other exception
    (setup, install, environment) says nothing about the agent: infra_error.

    Tokens and metadata come from xolotl.jsonl itself; Harbor's agent_result is only the fallback when that file is
    missing."""
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
        text = (trial_dir / "agent" / "xolotl.jsonl").read_bytes().decode("utf-8", errors="replace")
        have_jsonl = True
    except OSError:
        text, have_jsonl = "", False
    m = core.metrics_from_jsonl(text)
    try:
        exit_code = int((trial_dir / "agent" / "exit_code").read_text().strip())
    except (OSError, ValueError):
        exit_code = None
    if reward is not None and reward >= 1.0:
        status, infra = "pass", ""
    elif reward is None and exc not in VERIFIER_EXCEPTIONS:
        status, infra = "infra_error", exc or "no verifier result"
    elif m["infra_reason"]:
        status, infra = "infra_error", m["infra_reason"]
    else:
        status, infra = "fail", ""
    rec = _load_json(trial_dir / "agent" / core.TIMEOUT_FILE)  # what XolotlAgent.run() resolved; {} if it never got that far
    source = rec.get("source")
    ar = r.get("agent_result") or {}
    if have_jsonl:
        tokens = m["tokens"]
        metadata = {"model_calls": m["model_calls"], "tool_calls": m["tool_calls"], "compactions": m["compactions"],
                    "policy_hash": m["policy_hash"], "exit_code": exit_code, "infra_reason": m["infra_reason"]}
    else:
        tokens = {k: _num(ar.get(f)) or 0
                  for k, f in (("prompt", "n_input_tokens"), ("completion", "n_output_tokens"), ("cached", "n_cache_tokens"))}
        metadata = ar.get("metadata") or {}
    return {
        "task": r["task_name"], "trial": trial_dir.name, "task_checksum": r.get("task_checksum"), "status": status,
        "reward": reward, "exception": exc, "infra_reason": infra, "exit_code": exit_code, "tokens": tokens,
        "policy_hash": m["policy_hash"], "metadata": metadata, "agent_time_s": _agent_seconds(r.get("agent_execution")),
        "xolotl_version": (r.get("agent_info") or {}).get("version") or "",
        "agent_timeout_sec": _num(rec.get("agent_timeout_sec")), "timeout_source": source if isinstance(source, str) else None,
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


def secret_scan(root: Path, key: str) -> tuple[list[Path], list[Path]]:
    """(hits, unreadable): files under `root` containing the first 8 characters of `key`, and files or folders that
    could not be read. An unreadable path is not clean, so the caller must treat a non-empty second list as a failed
    scan. Never prints or returns the key itself."""
    needle = key[:8].encode("utf-8")
    hits: list[Path] = []
    unreadable: list[Path] = []
    for dirpath, _, names in os.walk(root, onerror=lambda e: unreadable.append(Path(e.filename))):
        for name in sorted(names):
            p = Path(dirpath) / name
            try:
                if needle in p.read_bytes():
                    hits.append(p)
            except OSError:
                unreadable.append(p)
    return sorted(hits), sorted(unreadable)


def _scan_failed(job_dir: Path, key: str) -> int:
    """0 if the job folder is clean; else 3 after printing paths only (the caller then writes no results)."""
    hits, unreadable = secret_scan(job_dir, key)
    for p in unreadable:
        print(f"WARNING: cannot read {p}", file=sys.stderr)
    if unreadable:
        print("error: the secrets scan is incomplete (unreadable paths above), so it counts as failed; results are not "
              f"written. The job folder {job_dir} is intact: fix the access problem and run "
              f"`py bench/tb2/run.py --reparse {job_dir}`.", file=sys.stderr)
    if hits:
        print("error: the API key's prefix was found in the job folder; refusing to write results:", file=sys.stderr)
        for p in hits:
            print(f"  {p}", file=sys.stderr)
    return 3 if hits or unreadable else 0


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
    print(f"\n{'task':<{width}}  status       reward  agent time  limit  tokens in/out")
    for t in result["trials"]:
        agent_s = "-" if t["agent_time_s"] is None else f"{t['agent_time_s']:.0f}s"
        limit = "-" if t["agent_timeout_sec"] is None else f"{t['agent_timeout_sec']:.0f}s"
        print(f"{t['task']:<{width}}  {t['status']:<11}  {t['reward']!s:<6}  {agent_s:<10}  {limit:<5}  "
              f"{t['tokens']['prompt']}/{t['tokens']['completion']}"
              + (f"  ({t['infra_reason']})" if t["infra_reason"] else ""))
    ran = s["total"] - s["infra"]
    print(f"\n{s['pass']}/{ran} passed ({s['pass_rate'] * 100:.1f}%), {s['infra']} infra; "
          f"tokens {s['tokens']['prompt']} in / {s['tokens']['completion']} out ({s['tokens']['cached']} cached); "
          f"median agent time {s['median_agent_time_s']:.1f}s")
    if s["infra"]:
        print(f"WARNING: {s['infra']} trial(s) are infra_error and excluded from the pass rate; re-run them before trusting this row")
    fallback = sum(t["timeout_source"] == "default" for t in result["trials"])
    if fallback:
        print(f"WARNING: {fallback} trial(s) used the {core.DEFAULT_TIMEOUT_S} s default timeout because the task's task.toml was "
              "not found; their budget may be shorter than the task's real limit")
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
    p.add_argument("--reparse", type=Path, metavar="JOB_DIR",
                   help="no build, no Harbor: scan an existing job folder for the key, parse it with the current code, "
                        "write results/<job name>.json and the HISTORY row (use --note)")
    p.add_argument("--jobs-dir", type=Path, default=HERE / "jobs", help="Harbor job output (git-ignored)")
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    return p


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_result(trials: list[dict], tasks: list[str], **fields) -> dict:
    """The results/<run_id>.json dict: `fields` (provenance) plus everything derived from the parsed trials."""
    return {
        **fields,
        "xolotl_version": next((t["xolotl_version"] for t in trials if t["xolotl_version"]), ""),
        "policy_hash": next((t["policy_hash"] for t in trials if t["policy_hash"]), ""),
        "tasks": tasks, "tasks_hash": tasks_hash(tasks), "trials": trials, "summary": summarize(trials),
    }


def record(result: dict, results_dir: Path) -> Path:
    """Write results/<run_id>.json (exclusive create; `-2`, `-3` appended to run_id if that file exists, so a result is
    never overwritten), append the HISTORY row unless the run is partial, and print the summary."""
    results_dir.mkdir(parents=True, exist_ok=True)
    base = result["run_id"]
    for n in itertools.count(1):
        result["run_id"] = base if n == 1 else f"{base}-{n}"
        path = results_dir / f"{result['run_id']}.json"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(result, indent=2) + "\n")
            break
        except FileExistsError:
            continue
    if not result["partial"]:
        history = results_dir / "HISTORY.md"
        header = HISTORY_HEADER if not history.exists() or history.stat().st_size == 0 else ""
        with history.open("a", encoding="utf-8", newline="\n") as f:
            f.write(header + history_row(result) + "\n")
    else:
        print(f"WARNING: partial run ({len(result['trials'])}/{len(result['tasks'])} trials); no HISTORY row", file=sys.stderr)
    _print_summary(result, path)
    return path


def _load_json(path: Path) -> dict:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def reparse(args: argparse.Namespace) -> int:
    """Re-derive results from an existing Harbor job folder with the current parser: no build, no Harbor, no Docker.
    Provenance Harbor's folder doesn't hold stays unknown: git_sha is only the sha7 in the job name, harbor_version and
    dirty are null, and binary_sha256 is that of the binary on disk now (the run's own only if it wasn't rebuilt)."""
    job_dir = args.reparse
    if not job_dir.is_dir():
        print(f"error: {job_dir} is not a directory", file=sys.stderr)
        return 1
    key = core.read_api_key()
    if not key:
        print(f"error: {core.KEY_ENV} not found (needed for the secrets scan)", file=sys.stderr)
        return 1
    if _scan_failed(job_dir, key):
        return 3
    trials = parse_job(job_dir)
    if not trials:
        print(f"error: no trial results under {job_dir}", file=sys.stderr)
        return 1
    cfg, job = _load_json(job_dir / "config.json"), _load_json(job_dir / "result.json")
    dataset = ((cfg.get("datasets") or [{}])[0] or {})
    names = dataset.get("task_names")
    tasks = names if isinstance(names, list) and names and all(isinstance(t, str) for t in names) \
        else sorted({t["task"] for t in trials})
    try:  # Harbor writes the job's start as a naive local time
        started = datetime.fromisoformat(job["started_at"]).astimezone()
    except (KeyError, TypeError, ValueError):
        started = datetime.now().astimezone()
    import build_linux
    binary = build_linux.binary_path(REPO_ROOT)
    m = re.search(r"-([0-9a-f]{7})(?:-\d+)?$", job_dir.name)
    result = make_result(
        trials, tasks, run_id=job_dir.name, job_name=job_dir.name, reparsed=True,
        parser_git_sha=_git("rev-parse", "HEAD") or "unknown", started=started.isoformat(timespec="seconds"),
        git_sha=m.group(1) if m else "unknown", dirty=None, harbor_version=None,
        binary_sha256=file_sha256(binary) if binary.is_file() else None,
        dataset=DATASET, model=((cfg.get("agents") or [{}])[0] or {}).get("model_name") or MODEL,
        concurrency=cfg.get("n_concurrent_trials"), note=args.note, partial=len(trials) < len(tasks),
        harbor_exit_code=None)
    record(result, args.results_dir)
    return 1 if result["partial"] else 0


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.reparse:
        return reparse(args)
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
    binary_sha256 = file_sha256(binary)  # what this run uploads into every container

    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}: {len(tasks)} task(s), -n {args.n}", flush=True)
    code = subprocess.run(cmd, env=harbor_env(key), stdin=subprocess.DEVNULL).returncode  # streams Harbor's output
    job_dir = args.jobs_dir / run_id
    if not job_dir.is_dir():
        print(f"error: harbor exited {code} and produced no job folder {job_dir}", file=sys.stderr)
        return 1

    if _scan_failed(job_dir, key):  # results are not written; the job folder is left intact
        return 3

    trials = parse_job(job_dir)
    partial = code != 0 or len(trials) < len(tasks)
    result = make_result(
        trials, tasks, run_id=run_id, job_name=run_id, reparsed=False, started=started.isoformat(timespec="seconds"),
        git_sha=sha, dirty=dirty, binary_sha256=binary_sha256,
        harbor_version=subprocess.run([harbor, "--version"], capture_output=True, text=True).stdout.strip(),
        dataset=DATASET, model=MODEL, concurrency=args.n, note=args.note, partial=partial, harbor_exit_code=code)
    record(result, args.results_dir)
    print(f"jobs: {job_dir}")
    return 1 if partial else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    sys.exit(main(sys.argv[1:]))
