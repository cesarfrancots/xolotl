"""Hidden check for rs-method.

1. src/ must contain a #[test] function that calls peek (the unit test the prompt asks for).
2. With verify/peek_hidden.rs copied to tests/peek_hidden.rs, `cargo test --quiet` must exit 0.
3. The hidden target must have actually run all of its tests (so a Cargo.toml edit that switches the target off
   can't make step 2 pass vacuously).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIDDEN = ROOT / "tests" / "peek_hidden.rs"

# Comments and string literals, matched left to right so a `//` inside a string (or a quote inside a comment) is
# handled by whichever starts first.
NOISE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.S)


def peek_tests(src: str) -> list[str]:
    """Names of the #[test] fns in `src` whose body calls peek. Brace matching on comment- and string-free text.
    ponytail: raw strings and '{' char literals would confuse it; fine for hand-written unit tests."""
    src = NOISE.sub(lambda m: '""' if m.group(0).startswith('"') else "", src)
    found = []
    for t in re.finditer(r"#\[test\]", src):
        fn = re.compile(r"\bfn\s+(\w+)[^{]*\{").search(src, t.end())
        if not fn:
            continue
        depth, i = 1, fn.end()
        while i < len(src) and depth:
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        if re.search(r"\bpeek\s*\(", src[fn.end():i]):
            found.append(fn.group(1))
    return found


def cargo_test(*args: str) -> subprocess.CompletedProcess:
    p = subprocess.run(["cargo", "test", "--quiet", *args], cwd=ROOT, capture_output=True, encoding="utf-8",
                       errors="replace", timeout=100)
    print(p.stdout[-1500:] + p.stderr[-1500:])
    return p


def main() -> int:
    if not any(peek_tests(f.read_text(encoding="utf-8", errors="replace")) for f in sorted((ROOT / "src").rglob("*.rs"))):
        print("FAIL: no #[test] function in src/ calls peek")
        return 1

    hidden_src = (HERE / "peek_hidden.rs").read_text(encoding="utf-8")
    n_tests = len(re.findall(r"^#\[test\]", hidden_src, re.M))
    HIDDEN.parent.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "peek_hidden.rs", HIDDEN)
    try:
        if cargo_test().returncode != 0:
            print("FAIL: cargo test failed")
            return 1
        p = cargo_test("--test", "peek_hidden")
        if p.returncode != 0 or not re.search(rf"test result: ok\. {n_tests} passed; 0 failed; 0 ignored", p.stdout):
            print(f"FAIL: the hidden peek tests did not all run and pass (expected {n_tests})")
            return 1
    finally:
        HIDDEN.unlink(missing_ok=True)  # leave the agent's tree as it was
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
