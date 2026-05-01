import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_lr.data.windowing import generate_synthetic_examples


class WindowingTests(unittest.TestCase):
    def test_synthetic_examples_cover_requested_count(self):
        rows = generate_synthetic_examples("train", 10, 7)
        self.assertEqual(len(rows), 10)
        self.assertTrue(any("hpoly" in row["example_id"] for row in rows))


if __name__ == "__main__":
    unittest.main()

