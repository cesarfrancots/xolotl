from pathlib import Path

assert Path("answer.txt").read_text(encoding="utf-8").strip() == "42", "answer.txt must contain 42"
