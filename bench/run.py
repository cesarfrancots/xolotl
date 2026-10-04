"""Xolotl bench runner: runs Xolotl headless on a task suite and records pass rate, tokens and timing.

Standard library only. This part holds the pure helpers; orchestration lives below them.
"""
import json
import re
import statistics
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

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
        elif kind == "compaction" and ev.get("stage") in (1, 2):
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
