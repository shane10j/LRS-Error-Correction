import unittest

from omega_lr.baseline.support_rule import predict
from omega_lr.data.support_rules import derive_support_rule_labels


class SupportRuleTests(unittest.TestCase):
    def test_support_rule_labels_capture_sub_ins_del_evidence(self):
        features = {
            "support_base_counts": [[0, 2, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            "support_ins_base_counts": [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
            "support_ins_count": [0, 2, 0],
            "support_del_count": [0, 0, 2],
        }
        labels = derive_support_rule_labels("AGT", features)
        self.assertEqual(labels["support_majority_base"], [1, 0, 3])
        self.assertEqual(labels["support_inserted_base"], [1, 2, 3])
        self.assertEqual(labels["support_suggests_sub"], [1, 1, 0])
        self.assertEqual(labels["support_suggests_ins"], [0, 1, 0])
        self.assertEqual(labels["support_suggests_del"], [0, 0, 1])
        self.assertEqual(labels["support_rule_type"], [1, 1, 2])

    def test_support_rule_baseline_uses_visible_support_events(self):
        example = {
            "target_seq": "ACGT",
            "features": {
                "support_base_counts": [[0, 2, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]],
                "support_ins_count": [0, 2, 0, 0],
                "support_ins_base_counts": [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "support_del_count": [0, 0, 2, 0],
                "support_depth": [3, 3, 3, 3],
                "support_agreement": [2 / 3, 2 / 3, 2 / 3, 1.0],
            },
        }
        result = predict(example, agreement_threshold=0.60, insertion_threshold=0.50, deletion_threshold=0.50)
        self.assertEqual(result["predicted_labels"], ["SUB_C", "INS_G", "DEL", "COPY"])
        self.assertEqual(result["prediction"], "CCGT")


if __name__ == "__main__":
    unittest.main()
