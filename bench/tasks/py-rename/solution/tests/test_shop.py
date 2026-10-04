import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shop import orders, reports
from shop.users import get_user


class ShopTest(unittest.TestCase):
    def test_get_user(self):
        self.assertEqual(get_user(1)["name"], "Ana")
        with self.assertRaises(KeyError):
            get_user(99)

    def test_order_label(self):
        self.assertEqual(orders.order_label(2, 40), "Ben: 40.00")

    def test_user_report(self):
        self.assertEqual(reports.user_report([1, 99]), "1: Ana (gold)\n99: unknown")


if __name__ == "__main__":
    unittest.main()
