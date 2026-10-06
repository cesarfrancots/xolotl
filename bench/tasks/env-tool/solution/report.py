import hashlib
from pathlib import Path

for f in sorted(Path("data").glob("*.txt"), key=lambda p: p.name):
    print(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
