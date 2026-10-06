import csv
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run_top(*args):
    return subprocess.run([sys.executable, "top.py", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)


def expected():
    """(name, score) pairs of scores.csv, highest score first (the scores are all distinct)."""
    with open(os.path.join(ROOT, "scores.csv"), newline="", encoding="utf-8") as f:
        rows = [(r["name"], int(r["score"])) for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: r[1], reverse=True)


def printed(proc):
    return [(n, int(s)) for n, s in (line.split() for line in proc.stdout.splitlines() if line.strip())]


class TopHidden(unittest.TestCase):
    def test_fixture_has_eight_distinct_scores(self):
        rows = expected()
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({s for _, s in rows}), 8)

    def test_no_flag_prints_all_rows(self):
        p = run_top("scores.csv")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(printed(p), expected())

    def test_limit_three(self):
        p = run_top("--limit", "3", "scores.csv")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(printed(p), expected()[:3])

    def test_limit_after_path_and_equals_form(self):
        for args in (("scores.csv", "--limit", "2"), ("--limit=2", "scores.csv")):
            p = run_top(*args)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(printed(p), expected()[:2])

    def test_limit_one(self):
        p = run_top("--limit", "1", "scores.csv")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(printed(p), expected()[:1])

    def test_limit_larger_than_rows_prints_all(self):
        p = run_top("--limit", "100", "scores.csv")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(printed(p), expected())

    def test_invalid_limits_exit_nonzero(self):
        for bad in ("0", "x", "-2", "2.5", ""):
            with self.subTest(limit=bad):
                p = run_top("--limit", bad, "scores.csv")
                self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
