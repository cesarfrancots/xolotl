"""Builds a static x86_64 musl `xolotl` inside Docker (rust:alpine), for upload into TB2 task containers.

Standard library only. Run from anywhere: `py bench/tb2/build_linux.py`. The last stdout line is the binary path.
"""
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE = "rust:1.95-alpine"
TARGET = "x86_64-unknown-linux-musl"
REGISTRY_VOLUME = "xolotl-cargo-registry"  # named volume: cargo's registry survives between builds


def binary_path(repo_root: Path) -> Path:
    return repo_root / "target" / TARGET / "release" / "xol"


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], stdin=subprocess.DEVNULL)


def is_static(data: bytes) -> bool:
    """True for a 64-bit little-endian ELF with no PT_INTERP segment: it asks for no dynamic loader, so it runs on any
    Linux. Rust's musl target links static-pie, which musl's `ldd` reports as the loader itself, so `ldd` can't tell."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return False
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    return all(struct.unpack_from("<I", data, phoff + i * phentsize)[0] != 3 for i in range(phnum))  # 3 = PT_INTERP


def main(argv: list[str]) -> int:
    try:
        ok = subprocess.run(["docker", "info"], capture_output=True, stdin=subprocess.DEVNULL, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    if not ok:
        print("Docker Desktop is not running", file=sys.stderr)
        return 1
    repo = str(REPO_ROOT)
    build = ("apk add --no-cache musl-dev >/dev/null && "
             f"cargo build --release --locked --target {TARGET}")  # --locked: the build must use Cargo.lock as committed
    r = _docker("run", "--rm", "-v", f"{repo}:/src", "-v", f"{REGISTRY_VOLUME}:/usr/local/cargo/registry",
                "-w", "/src", IMAGE, "sh", "-c", build)
    if r.returncode != 0:
        print(f"error: docker build failed ({r.returncode})", file=sys.stderr)
        return 1
    binary = binary_path(REPO_ROOT)
    if not binary.is_file():
        print(f"error: {binary} was not produced", file=sys.stderr)
        return 1
    rel = binary.relative_to(REPO_ROOT).as_posix()
    if not is_static(binary.read_bytes()):
        print(f"error: {rel} is not static (it has a PT_INTERP dynamic loader)", file=sys.stderr)
        return 1
    print(str(binary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
