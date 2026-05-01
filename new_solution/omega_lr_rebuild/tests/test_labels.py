import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_lr.constants import EDIT_TO_ID
from omega_lr.data.labels import generate_labels
from omega_lr.data.pileup import compute_support_features
from omega_lr.data.windowing import _support_for_ins


class LabelTests(unittest.TestCase):
    def test_simple_substitution(self):
        labels = generate_labels("ACGT", "AGGT", 3)
        self.assertEqual(labels["edit_labels"][1], EDIT_TO_ID["SUB_G"])

    def test_simple_deletion(self):
        labels = generate_labels("ACGT", "AGT", 3)
        self.assertEqual(labels["edit_labels"][1], EDIT_TO_ID["DEL"])
        self.assertEqual(labels["delete_length_labels"][1], 1)

    def test_homopolymer_deletion_normalization(self):
        labels = generate_labels("AAAAT", "AAAT", 3)
        self.assertIn(1, labels["delete_candidate_labels"])

    def test_boundary_insertion_anchor_matches_support(self):
        target = "ACGT"
        support, events, base_counts = _support_for_ins(target, 0, "A")
        insertion_counts = [sum(read[pos] for read in events) for pos in range(len(target))]
        features = compute_support_features(
            target,
            support,
            ["+", "-", "+"],
            insertion_counts=insertion_counts,
            insertion_base_counts=base_counts,
        )
        labels = generate_labels(target, "AACGT", 3)
        self.assertEqual(labels["edit_labels"][0], EDIT_TO_ID["INS_A"])
        self.assertGreater(features["support_ins_count"][0], 0)
        self.assertEqual(features["support_ins_base_counts"][0][0], 2)


if __name__ == "__main__":
    unittest.main()
