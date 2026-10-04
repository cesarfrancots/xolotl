"""Stand-in for the xolotl binary, used by bench/test_run.py. The -p prompt picks the behaviour:

ok      write answer.txt = 42 and emit a short event stream
plant   plant a verify/test_planted.py that always passes, nothing else
hang    spawn a long-sleeping child, record its pid in child.pid, then sleep too
garbage print 1 MB of non-JSON bytes, then one assistant event
"""
import json
import subprocess
import sys
import time
from pathlib import Path


def emit(type_: str, **fields) -> None:
    sys.stdout.write(json.dumps({"seq": 0, "t_ms": 0, "type": type_, **fields}) + "\n")
    sys.stdout.flush()


def assistant() -> None:
    emit("assistant", text="", reasoning="", tool_calls=[],
         usage={"prompt": 10, "completion": 5, "cached": 2}, finish_reason="stop", latency_ms=1)


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print("xolotl-fake 0.0.0")
        return 0
    cwd = Path(argv[argv.index("--cwd") + 1])
    prompt = argv[argv.index("-p") + 1]
    if prompt == "ok":
        (cwd / "answer.txt").write_text("42", encoding="utf-8")
        emit("session_start", cwd=str(cwd), model="fake-model", mode="yolo", policy_hash="fakehash", system="")
        assistant()
        emit("session_end", status="ok")
    elif prompt == "plant":
        (cwd / "verify").mkdir(exist_ok=True)
        (cwd / "verify" / "test_planted.py").write_text("assert True\n", encoding="utf-8")
    elif prompt == "hang":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])  # inherits our pipes
        (cwd / "child.pid").write_text(str(child.pid), encoding="utf-8")
        time.sleep(60)
    elif prompt == "garbage":
        sys.stdout.flush()
        sys.stdout.buffer.write(b"\xff\xfe garbage \x00" * 70_000 + b"\n")  # ~1 MB, one huge non-UTF-8 line
        sys.stdout.buffer.flush()
        assistant()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
