import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_utils import slugify

CASES = [
    ("Hello, World!", "hello-world"),
    ("  Crème Brûlée  ", "creme-brulee"),
    ("a--b__c", "a-b-c"),
    ("", ""),
    ("123 Go", "123-go"),
    ("Ünïcödé", "unicode"),
    ("---", ""),
    ("   ", ""),
    ("already-a-slug", "already-a-slug"),
    ("-lead and trail-", "lead-and-trail"),
    ("Tab\tand\nnewline", "tab-and-newline"),
    ("ÀÉÎÕÜ àéîõü", "aeiou-aeiou"),
    ("Ça va? Très bien!", "ca-va-tres-bien"),
    ("x", "x"),
    ("日本 go", "go"),  # step 3: anything outside a-z and 0-9 is a separator
]


class SlugifyHidden(unittest.TestCase):
    def test_cases(self):
        for text, want in CASES:
            with self.subTest(text=text):
                self.assertEqual(slugify(text), want)

    def test_returns_str(self):
        self.assertIsInstance(slugify("Hello"), str)


if __name__ == "__main__":
    unittest.main()
