import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import shop
from shop import orders, reports, users

OLD = re.compile(r"\bget_usr\b")


class RenameHidden(unittest.TestCase):
    def test_no_old_name_left_in_any_py_file(self):
        left = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in ("verify", "__pycache__") and not d.startswith(".")]
            for name in filenames:
                if name.endswith(".py"):
                    path = os.path.join(dirpath, name)
                    with open(path, encoding="utf-8", errors="replace") as f:
                        for n, line in enumerate(f, 1):
                            if OLD.search(line):
                                left.append(f"{os.path.relpath(path, ROOT)}:{n}: {line.strip()}")
        self.assertEqual(left, [], "get_usr is still referenced")

    def test_new_name_exists(self):
        self.assertTrue(callable(users.get_user))
        self.assertIs(shop.get_user, users.get_user)
        self.assertFalse(hasattr(users, "get_usr"))
        self.assertFalse(hasattr(shop, "get_usr"))

    def test_get_user_behavior(self):
        self.assertEqual(users.get_user(1), {"name": "Ana", "tier": "gold"})
        self.assertEqual(users.get_user(2), {"name": "Ben", "tier": "basic"})
        with self.assertRaises(KeyError):
            users.get_user(99)

    def test_orders_unchanged(self):
        self.assertEqual(orders.order_total(1, 100), 90.0)
        self.assertEqual(orders.order_total(2, 100), 100.0)
        self.assertEqual(orders.order_total(3, 19.99), 17.99)
        self.assertEqual(orders.order_label(1, 100), "Ana: 90.00")
        self.assertEqual(orders.order_label(2, 40), "Ben: 40.00")
        with self.assertRaises(KeyError):
            orders.order_total(99, 10)

    def test_reports_unchanged(self):
        self.assertEqual(reports.user_report([]), "")
        self.assertEqual(reports.user_report([2]), "2: Ben (basic)")
        self.assertEqual(reports.user_report([3, 99, 1]), "3: Cy (gold)\n99: unknown\n1: Ana (gold)")


if __name__ == "__main__":
    unittest.main()
