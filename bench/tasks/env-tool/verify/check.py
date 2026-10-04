"""Hidden check for env-tool.

Runs report.py (with this Python) if it exists, else report.sh (with bash), and compares its stdout line for line
with "<sha256 hex>  <name>" for every data/*.txt in plain lexicographic name order. The expected lines are computed
from the files themselves. The script is then run a second time in a scratch directory holding different data, so
a script that just prints fixed lines can't pass.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_bash() -> str:
    """Git for Windows' bash (found from git, never the System32 WSL stub); plain `bash` elsewhere."""
    if os.name == "nt":
        git = shutil.which("git")
        for root in Path(git).parents[1:3] if git else []:
            if (root / "bin" / "bash.exe").is_file():
                return str(root / "bin" / "bash.exe")
    return shutil.which("bash") or "bash"


def expected(data: Path) -> list[str]:
    files = [p for p in data.glob("*.txt") if p.is_file()]
    return [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(files, key=lambda p: p.name)]


def run_report(workdir: Path) -> list[str]:
    """Run the script found in workdir with workdir as cwd; returns its output lines (trailing blank lines dropped)."""
    if (workdir / "report.py").is_file():
        argv = [sys.executable, "report.py"]
    elif (workdir / "report.sh").is_file():
        argv = [find_bash(), "report.sh"]
    else:
        raise SystemExit("FAIL: neither report.py nor report.sh exists")
    p = subprocess.run(argv, cwd=workdir, capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
    if p.returncode != 0:
        raise SystemExit(f"FAIL: {' '.join(argv)} exited {p.returncode}: "
                         f"{p.stderr.decode('utf-8', 'replace')[-500:]}")
    lines = p.stdout.decode("utf-8", errors="replace").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def compare(label: str, got: list[str], want: list[str]) -> None:
    if got != want:
        print(f"FAIL ({label}): output differs from the expected lines")
        print("expected:", *want, sep="\n  ")
        print("got:", *got, sep="\n  ")
        raise SystemExit(1)


def main() -> int:
    want = expected(ROOT / "data")
    if len(want) < 2:
        print("FAIL: data/ no longer holds the .txt files it started with")
        return 1
    compare("data/", run_report(ROOT), want)

    with tempfile.TemporaryDirectory(prefix="env-tool-check-") as tmp:
        work = Path(tmp)
        for script in ("report.py", "report.sh"):
            if (ROOT / script).is_file():
                shutil.copyfile(ROOT / script, work / script)
        (work / "data").mkdir()
        for name, content in {"n7.txt": b"seven\nseven\n", "e.txt": b"", "n12.txt": b"twelve\n", "k.txt": b"kilo\n",
                              "skip.md": b"not a txt file\n"}.items():
            (work / "data" / name).write_bytes(content)
        (work / "top.txt").write_bytes(b"outside data/\n")  # a .txt that is not in data/ must not be listed
        compare("different data", run_report(work), expected(work / "data"))
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
