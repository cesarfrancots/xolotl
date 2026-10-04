import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Common Log Format: host ident user [time] "METHOD path PROTO" status bytes
CLF = re.compile(r'^\S+ \S+ \S+ \[[^\]]+\] "(\S+) (\S+) [^"]*" (\d{3}) (\S+)\s*$')


def expected_summary(lines):
    """The summary, computed independently from the raw log lines."""
    total, by_status, by_path = 0, {}, {}
    for line in lines:
        if not line.strip():
            continue
        m = CLF.match(line)
        assert m, f"unparseable log line: {line!r}"
        _method, path, status, _bytes = m.groups()
        total += 1
        by_status[status] = by_status.get(status, 0) + 1
        by_path[path] = by_path.get(path, 0) + 1
    top = sorted(by_path.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return {"total": total, "by_status": by_status, "top_paths": [[p, c] for p, c in top]}


def run_summarize(workdir):
    """Run summarize.py fresh in workdir (any old summary.json removed first) and return what it wrote."""
    out = os.path.join(workdir, "summary.json")
    if os.path.exists(out):
        os.remove(out)
    p = subprocess.run([sys.executable, "summarize.py"], cwd=workdir, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, f"summarize.py exited {p.returncode}: {p.stderr[-500:]}"
    assert os.path.exists(out), "summarize.py did not write summary.json"
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def check(test, got, want):
    test.assertEqual(got["total"], want["total"])
    test.assertIsInstance(got["total"], int)
    test.assertEqual(got["by_status"], want["by_status"])
    test.assertTrue(all(isinstance(k, str) and isinstance(v, int) for k, v in got["by_status"].items()))
    test.assertEqual(got["top_paths"], want["top_paths"])


class LogparseHidden(unittest.TestCase):
    def read_log(self):
        with open(os.path.join(ROOT, "access.log"), encoding="utf-8") as f:
            return f.read().splitlines()

    def test_fixture_shape(self):
        want = expected_summary(self.read_log())
        self.assertEqual(want["total"], 60)
        self.assertEqual(sorted(want["by_status"]), ["200", "301", "404", "500"])
        tied = {c for _, c in want["top_paths"][1:]}
        self.assertEqual(len(tied), 1, "the log must tie the paths at ranks 2 and 3")

    def test_summary_of_access_log(self):
        check(self, run_summarize(ROOT), expected_summary(self.read_log()))

    def test_summary_of_a_different_log(self):
        # A different log in a scratch dir: the script must really read access.log, not repeat fixed numbers.
        lines = self.read_log()
        variant = [ln for i, ln in enumerate(reversed(lines)) if i % 3 != 0] + lines[:7]
        with tempfile.TemporaryDirectory() as tmp:
            # Copy the whole solution (summarize.py and any helper modules), but not the checks, git data or old outputs.
            skip = {"verify", ".git", "access.log", "summary.json"}
            shutil.copytree(ROOT, tmp, dirs_exist_ok=True, ignore=lambda src, names: skip & set(names) if src == ROOT else ())
            with open(os.path.join(tmp, "access.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(variant) + "\n")
            check(self, run_summarize(tmp), expected_summary(variant))


if __name__ == "__main__":
    unittest.main()
