import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orders import order_total


class OrderTotalTest(unittest.TestCase):
    def test_discounted_order(self):
        self.assertEqual(order_total([(10.0, 2), (5.0, 1)], 0.2), 20.0)

    def test_no_discount(self):
        self.assertEqual(order_total([(10.0, 2), (5.0, 1)], 0.0), 25.0)


if __name__ == "__main__":
    unittest.main()
