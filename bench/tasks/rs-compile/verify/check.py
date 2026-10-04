"""Hidden check for rs-compile.

1. Every file listed in hashes.json (tests/calc.rs) must be byte-for-byte what was shipped.
2. `cargo test --quiet` must exit 0.
3. The calc integration target must have actually run all of its tests (so a Cargo.toml edit that switches the
   target off can't make step 2 pass vacuously).
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path: Path) -> str:
    # CRLF -> LF first: a Windows checkout with autocrlf must hash the same as the LF file the hash was taken from.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cargo_test(*args: str) -> subprocess.CompletedProcess:
    p = subprocess.run(["cargo", "test", "--quiet", *args], cwd=ROOT, capture_output=True, encoding="utf-8",
                       errors="replace", timeout=100)
    print(p.stdout[-1500:] + p.stderr[-1500:])
    return p


def main() -> int:
    for rel, want in json.loads((HERE / "hashes.json").read_text(encoding="utf-8")).items():
        f = ROOT / rel
        if not f.is_file():
            print(f"FAIL: {rel} is missing")
            return 1
        if sha256(f) != want:
            print(f"FAIL: {rel} was modified; fix the code under src/ instead")
            return 1

    if cargo_test().returncode != 0:
        print("FAIL: cargo test failed")
        return 1

    n_tests = len(re.findall(r"^#\[test\]", (ROOT / "tests" / "calc.rs").read_text(encoding="utf-8"), re.M))
    p = cargo_test("--test", "calc")
    if p.returncode != 0 or not re.search(rf"test result: ok\. {n_tests} passed; 0 failed; 0 ignored", p.stdout):
        print(f"FAIL: the calc integration tests did not all run and pass (expected {n_tests})")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
