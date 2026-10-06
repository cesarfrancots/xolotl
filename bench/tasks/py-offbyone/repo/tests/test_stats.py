import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import moving_average


class MovingAverageTest(unittest.TestCase):
    def test_window_of_two(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])


if __name__ == "__main__":
    unittest.main()
