"""Hidden check for long-context.

FINDINGS.md must name exactly the functions that write to the global CONFIG: the set of `mod_NN:function` tokens in
it has to equal verify/expected.txt (nothing missing, nothing extra).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # Windows PowerShell's `>` writes UTF-16
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def main() -> int:
    findings = ROOT / "FINDINGS.md"
    if not findings.is_file():
        print("FAIL: FINDINGS.md does not exist")
        return 1
    want = {ln.strip() for ln in (HERE / "expected.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
    got = set(re.findall(r"mod_\d\d:\w+", read_text(findings)))
    missing, extra = sorted(want - got), sorted(got - want)
    if missing or extra:
        print(f"FAIL: {len(got & want)}/{len(want)} writers found")
        if missing:
            print("missing:", *missing, sep="\n  ")
        if extra:
            print("not writers:", *extra, sep="\n  ")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
