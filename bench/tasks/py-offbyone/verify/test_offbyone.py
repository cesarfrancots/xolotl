import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import moving_average


def brute(xs, k):
    return [sum(xs[i:i + k]) / k for i in range(len(xs) - k + 1)]


class MovingAverageHidden(unittest.TestCase):
    def test_window_two(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])

    def test_window_equals_length(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 4), [2.5])

    def test_window_one_is_identity_as_floats(self):
        got = moving_average([1, 2, 3, 4], 1)
        self.assertEqual(got, [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(all(isinstance(v, float) for v in got))

    def test_window_longer_than_input(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 5), [])
        self.assertEqual(moving_average([], 1), [])

    def test_non_positive_window_raises(self):
        for k in (0, -1, -5):
            with self.assertRaises(ValueError):
                moving_average([1, 2, 3], k)
        with self.assertRaises(ValueError):
            moving_average([], 0)

    def test_every_window_size_matches_brute_force(self):
        xs = [4, 8, 15, 16, 23, 42, -7, 0.5]
        for k in range(1, len(xs) + 3):
            got, want = moving_average(xs, k), brute(xs, k)  # brute is [] when k > len(xs)
            self.assertEqual(len(got), len(want), f"k={k}")
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w)

    def test_input_is_not_modified(self):
        xs = [1, 2, 3, 4]
        moving_average(xs, 2)
        self.assertEqual(xs, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
