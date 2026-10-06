"""Pure helpers for the Xolotl Harbor agent. Standard library only; no Harbor import, so they run under plain `py`.

`_is_int`, `_events`, `_count`, `metrics_from_events` and `_infra_reason` are copied verbatim from bench/run.py (Harbor's
interpreter can't import the `bench` package without path hacks); test_tb2 asserts both produce identical output.
"""
import base64
import json
import os
import shlex
import subprocess
from collections.abc import Iterable, Iterator

DEFAULT_MODEL = "glm-5.3-flash"
DEFAULT_TIMEOUT_S = 900  # TB2's usual agent timeout; Harbor doesn't tell the agent the task's own value
BINARY = "/installed-agent/xolotl"
LOGS = "/logs/agent"
KEY_ENV = "ZAI_API_KEY"


def model_arg(harbor_model: str | None) -> str:
    """Harbor's `-m provider/model` as Xolotl's `--model`: the provider prefix is dropped."""
    return (harbor_model or DEFAULT_MODEL).split("/", 1)[-1]


def timeout_budget(agent_timeout_sec: float | None) -> int:
    """Seconds Xolotl may run: the task's agent timeout minus 60 s, at least 60, so it stops before Harbor kills the
    container and the logs still get copied."""
    return max(60, int(agent_timeout_sec if agent_timeout_sec is not None else DEFAULT_TIMEOUT_S) - 60)


def run_command(instruction: str, model: str, timeout_s: int) -> str:
    """The shell command run in the task container. The instruction travels base64-encoded, so no quote, `$()`, newline
    or unicode in it can break or inject into the command. Always exits 0: Xolotl's own exit code goes to
    /logs/agent/exit_code, so it never becomes a Harbor exception and the verifier decides pass or fail. `timeout -k 5`
    sends SIGKILL 5 s after SIGTERM, so Xolotl is guaranteed to be gone before the logs are copied."""
    b64 = base64.b64encode(instruction.encode("utf-8")).decode("ascii")
    return (
        f'cd "$PWD"\n'  # the task's own WORKDIR, not always /app
        f"timeout -k 5 {int(timeout_s)} {BINARY} -p \"$(printf %s '{b64}' | base64 -d)\" --mode yolo --cwd \"$PWD\" "
        f"--model {shlex.quote(model)} > {LOGS}/xolotl.jsonl 2> {LOGS}/xolotl.stderr < /dev/null\n"
        f"echo $? > {LOGS}/exit_code\n"
        f"mkdir -p {LOGS}/xolotl-home && cp -r /root/.xolotl/. {LOGS}/xolotl-home/ 2>/dev/null\n"
        "true"
    )


def read_api_key() -> str | None:
    """The API key from the process env, else the Windows User scope (as Xolotl's main.rs does). Never prints it."""
    key = os.environ.get(KEY_ENV)
    if key:
        return key
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["reg", "query", r"HKCU\Environment", "/v", KEY_ENV], capture_output=True,
                             stdin=subprocess.DEVNULL, timeout=30).stdout.decode("utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():  # `    NAME    REG_SZ    data`
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].upper() == KEY_ENV and parts[1].startswith("REG_"):
            return parts[2].strip() or None
    return None


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


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


def _infra_reason(events: list[dict]) -> str:
    """Why an attempt says nothing about the agent, or "": no session ever started (e.g. the API key variable is
    unset), or the model API failed for good (quota, auth, network, HTTP). Not infra: an `llm` error marked retry,
    which is a context overflow the agent compacted away and retried, and a repeated overflow, which is the agent's
    own context management failing (messages as in llm.rs `LlmError`'s Display)."""
    if not any(e.get("type") == "session_start" for e in events):
        return "no session_start event"
    for e in events:
        if e.get("type") == "error" and e.get("kind") == "llm" and e.get("retry") is not True:
            msg = str(e.get("message", ""))
            if not msg.startswith("context window exceeded"):
                return f"llm error: {msg[:200]}"
    return ""


def metrics_from_jsonl(text: str) -> dict:
    """bench.run's metrics for the text of xolotl.jsonl, plus `infra_reason` (None when the run says something about
    the agent)."""
    lines = text.split("\n")  # not splitlines(): U+2028, U+2029 and U+0085 may sit unescaped inside a JSON string
    m = metrics_from_events(lines)
    m["infra_reason"] = _infra_reason(list(_events(lines))) or None
    return m
