import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orders import order_total
from pricing import apply_discount


class ApplyDiscountHidden(unittest.TestCase):
    def test_fractions(self):
        for price, pct, want in [(100, 0, 100), (100, 0.25, 75), (100, 1.0, 0), (80, 0.5, 40), (19.99, 0, 19.99)]:
            with self.subTest(price=price, pct=pct):
                self.assertAlmostEqual(apply_discount(price, pct), want)


class OrderTotalHidden(unittest.TestCase):
    ITEMS = [(10.0, 2), (5.0, 1)]  # subtotal 25.0

    def test_discount_cases(self):
        for pct, want in [(0.0, 25.0), (0.2, 20.0), (0.5, 12.5), (1.0, 0.0)]:
            with self.subTest(pct=pct):
                self.assertEqual(order_total(self.ITEMS, pct), want)

    def test_default_is_no_discount(self):
        self.assertEqual(order_total(self.ITEMS), 25.0)

    def test_empty_order(self):
        self.assertEqual(order_total([], 0.3), 0.0)

    def test_rounds_to_cents(self):
        self.assertEqual(order_total([(9.99, 3)], 0.1), 26.97)  # 29.97 * 0.9 = 26.973


if __name__ == "__main__":
    unittest.main()
