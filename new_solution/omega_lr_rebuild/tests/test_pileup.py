import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_lr.data.pileup import compute_support_features


class PileupTests(unittest.TestCase):
    def test_feature_extraction(self):
        features = compute_support_features("ACGT", ["ACGT", "A-GT"], ["+", "-"], [0, 1, 0, 0], [[0, 0, 0, 0], [0, 1, 0, 0]])
        self.assertEqual(features["support_base_counts"][0], [2, 0, 0, 0])
        self.assertEqual(features["support_del_count"][1], 1)
        self.assertEqual(features["support_ins_count"][1], 1)


if __name__ == "__main__":
    unittest.main()

