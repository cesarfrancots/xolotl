import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fast import dedupe

LIMIT_S = 2.0

# Run in a child process so a quadratic implementation is killed after 15 s instead of running for minutes.
# 200,000 ints, 100,000 distinct (50% duplicates), shuffled with a fixed seed. Prints the time dedupe() took.
CHILD = """
import random, time
from fast import dedupe
items = list(range(100_000)) * 2
random.Random(1234).shuffle(items)
t0 = time.perf_counter()
out = dedupe(items)
dt = time.perf_counter() - t0
assert out == list(dict.fromkeys(items)), "wrong result on the large input"
print(dt)
"""


class DedupeHidden(unittest.TestCase):
    def test_keeps_first_occurrence_in_order(self):
        self.assertEqual(dedupe([3, 1, 3, 2, 1, 4]), [3, 1, 2, 4])
        self.assertEqual(dedupe(["b", "a", "b", "c", "a"]), ["b", "a", "c"])

    def test_mixed_ints_and_strings(self):
        self.assertEqual(dedupe([1, "1", 1, "a", "a", 2, "1"]), [1, "1", "a", 2])

    def test_trivial_inputs(self):
        self.assertEqual(dedupe([]), [])
        self.assertEqual(dedupe([7]), [7])
        self.assertEqual(dedupe([5, 5, 5]), [5])

    def test_returns_a_new_list_and_leaves_input_alone(self):
        items = [2, 1, 2]
        out = dedupe(items)
        self.assertIsInstance(out, list)
        self.assertEqual(items, [2, 1, 2])
        self.assertIsNot(out, items)

    def test_accepts_any_iterable(self):
        self.assertEqual(dedupe(iter([4, 4, 5])), [4, 5])
        self.assertEqual(dedupe(x % 3 for x in range(10)), [0, 1, 2])

    def test_large_input_is_fast_and_correct(self):
        try:
            p = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            self.fail(f"dedupe() on 200,000 items did not finish within 15 s (limit {LIMIT_S} s)")
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        elapsed = float(p.stdout.strip().splitlines()[-1])
        self.assertLess(elapsed, LIMIT_S, f"dedupe() took {elapsed:.2f} s for 200,000 items")


if __name__ == "__main__":
    unittest.main()
