import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_lr.model.decode import decode_example


class DecodeTests(unittest.TestCase):
    def test_threshold_behavior(self):
        target_seq = "AC"
        example = {
            "target_seq": target_seq,
            "features": {
                "support_agreement": [0.9, 0.9],
                "support_entropy": [0.1, 0.1],
                "support_del_count": [0, 0],
                "support_ins_count": [0, 0],
                "support_depth": [2, 2],
                "gap_length_hist": [[0], [0]],
                "support_base_counts": [[0, 2, 0, 0], [0, 0, 1, 1]],
            }
        }
        outputs = {
            "type_logits": torch.tensor([[0.1, 5.0, 0.1, 0.1], [5.0, 0.1, 0.1, 0.1]]),
            "sub_base_logits": torch.tensor([[4.0, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "ins_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.tensor([[5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),
            "delete_candidate_logits": torch.tensor([0.0, 0.0]),
            "delete_length_logits": torch.tensor([[5.0, 0.1], [5.0, 0.1]]),
            "trust": torch.tensor([0.9, 0.9]),
        }
        decoded = decode_example(target_seq, example, outputs, {"sub_threshold": 0.5, "del_threshold": 0.5, "ins_threshold": 0.5, "trust_threshold": 0.5, "max_deletion_length": 1})
        self.assertEqual(decoded["prediction"], "AC")
        self.assertEqual(decoded["predicted_labels"][1], "COPY")

    def test_debug_decode_bypasses_candidate_and_trust_veto(self):
        target_seq = "AC"
        example = {
            "target_seq": target_seq,
            "features": {
                "support_agreement": [0.2, 0.2],
                "support_entropy": [2.0, 2.0],
                "support_del_count": [0, 0],
                "support_ins_count": [0, 0],
                "support_depth": [2, 2],
                "gap_length_hist": [[0], [0]],
                "support_base_counts": [[0, 0, 1, 0], [0, 0, 1, 0]],
            },
        }
        outputs = {
            "type_logits": torch.tensor([[0.1, 4.0, 0.1, 0.1], [5.0, 0.1, 0.1, 0.1]]),
            "sub_base_logits": torch.tensor([[0.1, 0.1, 4.0, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "ins_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.tensor([[0.1, 0.1, 0.1, 4.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),
            "delete_candidate_logits": torch.tensor([-5.0, 0.0]),
            "delete_length_logits": torch.tensor([[5.0, 0.1], [5.0, 0.1]]),
            "trust": torch.tensor([0.05, 0.9]),
        }
        decoded = decode_example(
            target_seq,
            example,
            outputs,
            {
                "sub_threshold": 0.5,
                "del_threshold": 0.5,
                "ins_threshold": 0.5,
                "trust_threshold": 0.9,
                "max_deletion_length": 1,
                "mode": "debug",
                "use_trust_threshold": False,
                "use_delete_candidate_veto": False,
                "restrict_supported_candidates": False,
                "consistency_check": False,
                "full_trace": True,
            },
        )
        self.assertEqual(decoded["predicted_labels"][0], "SUB_G")
        self.assertTrue(decoded["trace"])

    def test_hybrid_rule_copy_veto_blocks_neural_sub_a_false_positive(self):
        target_seq = "CC"
        example = {
            "target_seq": target_seq,
            "features": {
                "support_agreement": [1.0, 1.0],
                "support_entropy": [0.0, 0.0],
                "support_del_count": [0, 0],
                "support_ins_count": [0, 0],
                "support_depth": [4, 4],
                "gap_length_hist": [[0], [0]],
                "support_base_counts": [[0, 4, 0, 0], [0, 4, 0, 0]],
            },
        }
        outputs = {
            "type_logits": torch.tensor([[0.1, 8.0, 0.1, 0.1], [5.0, 0.1, 0.1, 0.1]]),
            "sub_base_logits": torch.tensor([[8.0, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "ins_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.zeros(2, 10),
            "delete_candidate_logits": torch.tensor([0.0, 0.0]),
            "delete_length_logits": torch.tensor([[5.0, 0.1], [5.0, 0.1]]),
            "trust": torch.tensor([1.0, 1.0]),
        }
        decoded = decode_example(
            target_seq,
            example,
            outputs,
            {
                "sub_threshold": 0.0,
                "del_threshold": 0.0,
                "ins_threshold": 0.0,
                "trust_threshold": 0.0,
                "max_deletion_length": 1,
                "mode": "debug",
                "full_trace": True,
                "hybrid_rule_decode": True,
                "hybrid_negative_veto": True,
                "hybrid_sub_a_copy_safety": True,
            },
        )
        self.assertEqual(decoded["predicted_labels"][0], "COPY")
        self.assertIn("hybrid_rule_copy_veto", decoded["trace"][0]["veto_reasons"])
        self.assertIn("hybrid_sub_a_safety_veto", decoded["trace"][0]["veto_reasons"])

    def test_hybrid_positive_rule_forcing_still_applies(self):
        target_seq = "AC"
        example = {
            "target_seq": target_seq,
            "features": {
                "support_agreement": [1.0, 1.0],
                "support_entropy": [0.0, 0.0],
                "support_del_count": [0, 0],
                "support_ins_count": [0, 0],
                "support_depth": [4, 4],
                "gap_length_hist": [[0], [0]],
                "support_base_counts": [[0, 4, 0, 0], [0, 4, 0, 0]],
            },
        }
        outputs = {
            "type_logits": torch.tensor([[3.0, 2.0, 0.1, 0.1], [5.0, 0.1, 0.1, 0.1]]),
            "sub_base_logits": torch.tensor([[0.1, 8.0, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "ins_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.zeros(2, 10),
            "delete_candidate_logits": torch.tensor([0.0, 0.0]),
            "delete_length_logits": torch.tensor([[5.0, 0.1], [5.0, 0.1]]),
            "trust": torch.tensor([1.0, 1.0]),
        }
        decoded = decode_example(
            target_seq,
            example,
            outputs,
            {
                "sub_threshold": 0.99,
                "del_threshold": 0.99,
                "ins_threshold": 0.99,
                "trust_threshold": 0.0,
                "max_deletion_length": 1,
                "mode": "debug",
                "full_trace": True,
                "hybrid_rule_decode": True,
                "hybrid_sub_payload_threshold": 0.80,
                "hybrid_sub_min_type_prob": 0.05,
                "hybrid_sub_min_copy_margin": -0.75,
                "hybrid_negative_veto": True,
            },
        )
        self.assertEqual(decoded["predicted_labels"][0], "SUB_C")
        self.assertTrue(decoded["trace"][0]["forced_by_rule"])

    def test_hybrid_neural_rescue_allows_borderline_rule_positive_sub(self):
        target_seq = "AC"
        example = {
            "example_id": "neighbor_borderline_sub",
            "target_seq": target_seq,
            "features": {
                "support_agreement": [2 / 3, 1.0],
                "support_entropy": [0.9183, 0.0],
                "support_del_count": [0, 0],
                "support_ins_count": [0, 0],
                "support_depth": [3, 3],
                "gap_length_hist": [[0], [0]],
                "support_base_counts": [[1, 2, 0, 0], [0, 3, 0, 0]],
            },
        }
        outputs = {
            "type_logits": torch.tensor([[0.0, 7.0, 0.0, 0.0], [5.0, 0.1, 0.1, 0.1]]),
            "sub_base_logits": torch.tensor([[0.1, 8.0, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "ins_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.zeros(2, 10),
            "delete_candidate_logits": torch.tensor([0.0, 0.0]),
            "delete_length_logits": torch.tensor([[5.0, 0.1], [5.0, 0.1]]),
            "trust": torch.tensor([1.0, 1.0]),
        }
        decoded = decode_example(
            target_seq,
            example,
            outputs,
            {
                "sub_threshold": 0.0,
                "del_threshold": 0.0,
                "ins_threshold": 0.0,
                "trust_threshold": 0.0,
                "max_deletion_length": 1,
                "mode": "debug",
                "full_trace": True,
                "hybrid_rule_decode": True,
                "hybrid_sub_payload_threshold": 0.85,
                "hybrid_neighbor_min_support_fraction": 0.90,
                "hybrid_neighbor_max_entropy": 0.50,
                "hybrid_neural_rescue_enabled": True,
                "hybrid_neural_rescue_min_type_prob": 0.95,
                "hybrid_neural_rescue_min_payload_prob": 0.95,
                "hybrid_neural_rescue_min_support_fraction": 0.60,
                "hybrid_neural_rescue_min_agreement": 0.60,
                "hybrid_require_rule_agreement_for_neural_edits": True,
            },
        )
        self.assertEqual(decoded["predicted_labels"][0], "SUB_C")
        self.assertTrue(decoded["trace"][0]["rescued_by_neural"])
        self.assertFalse(decoded["trace"][0]["forced_by_rule"])

    def test_hybrid_neural_rescue_does_not_require_strong_type_confidence(self):
        target_seq = "AC"
        example = {
            "example_id": "neighbor_borderline_sub",
            "target_seq": target_seq,
            "features": {
                "support_agreement": [2 / 3, 1.0],
                "support_entropy": [0.9183, 0.0],
                "support_del_count": [0, 0],
                "support_ins_count": [0, 0],
                "support_depth": [3, 3],
                "gap_length_hist": [[0], [0]],
                "support_base_counts": [[1, 2, 0, 0], [0, 3, 0, 0]],
            },
        }
        outputs = {
            "type_logits": torch.tensor([[0.0, 2.0, 0.0, 0.0], [5.0, 0.1, 0.1, 0.1]]),
            "sub_base_logits": torch.tensor([[0.1, 8.0, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "ins_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.zeros(2, 10),
            "delete_candidate_logits": torch.tensor([0.0, 0.0]),
            "delete_length_logits": torch.tensor([[5.0, 0.1], [5.0, 0.1]]),
            "trust": torch.tensor([1.0, 1.0]),
        }
        decoded = decode_example(
            target_seq,
            example,
            outputs,
            {
                "sub_threshold": 0.0,
                "del_threshold": 0.0,
                "ins_threshold": 0.0,
                "trust_threshold": 0.0,
                "max_deletion_length": 1,
                "mode": "debug",
                "full_trace": True,
                "hybrid_rule_decode": True,
                "hybrid_sub_payload_threshold": 0.85,
                "hybrid_neighbor_min_support_fraction": 0.90,
                "hybrid_neighbor_max_entropy": 0.50,
                "hybrid_neural_rescue_enabled": True,
                "hybrid_neural_rescue_min_type_prob": 0.0,
                "hybrid_neural_rescue_min_payload_prob": 0.95,
                "hybrid_neural_rescue_min_support_fraction": 0.60,
                "hybrid_neural_rescue_min_agreement": 0.60,
                "hybrid_require_rule_agreement_for_neural_edits": True,
            },
        )
        self.assertEqual(decoded["predicted_labels"][0], "SUB_C")
        self.assertTrue(decoded["trace"][0]["rescued_by_neural"])

    def test_noisy_two_of_three_deletion_stays_copy_when_rule_negative(self):
        target_seq = "A"
        example = {
            "example_id": "noisy_del_like_copy",
            "target_seq": target_seq,
            "features": {
                "support_agreement": [2 / 3],
                "support_entropy": [0.9183],
                "support_del_count": [2],
                "support_ins_count": [0],
                "support_depth": [3],
                "gap_length_hist": [[0]],
                "support_base_counts": [[1, 0, 0, 0]],
                "homopolymer_run_length": [4],
                "tandem_repeat_flag": [1],
            },
        }
        outputs = {
            "type_logits": torch.tensor([[0.1, 0.1, 8.0, 0.1]]),
            "sub_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1]]),
            "ins_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.zeros(1, 10),
            "delete_candidate_logits": torch.tensor([8.0]),
            "delete_length_logits": torch.tensor([[0.1, 8.0]]),
            "trust": torch.tensor([1.0]),
        }
        decoded = decode_example(
            target_seq,
            example,
            outputs,
            {
                "sub_threshold": 0.0,
                "del_threshold": 0.0,
                "ins_threshold": 0.0,
                "trust_threshold": 0.0,
                "max_deletion_length": 1,
                "mode": "debug",
                "full_trace": True,
                "hybrid_rule_decode": True,
                "hybrid_negative_veto": True,
                "hybrid_require_rule_agreement_for_neural_edits": True,
                "support_rule_deletion_threshold": 0.75,
                "hybrid_negative_veto_min_type_prob": 0.95,
                "hybrid_negative_veto_min_support_fraction": 0.95,
            },
        )
        self.assertEqual(decoded["predicted_labels"][0], "COPY")
        self.assertEqual(decoded["trace"][0]["support_rule_label"], "COPY")
        self.assertIn("hybrid_rule_copy_veto", decoded["trace"][0]["veto_reasons"])

    def test_hybrid_ins_support_payload_rescue_uses_support_base(self):
        target_seq = "CC"
        example = {
            "example_id": "true_ins_a_boundary",
            "target_seq": target_seq,
            "features": {
                "support_agreement": [1.0, 1.0],
                "support_entropy": [0.0, 0.0],
                "support_del_count": [0, 0],
                "support_ins_count": [2, 0],
                "support_ins_base_counts": [[2, 0, 0, 0], [0, 0, 0, 0]],
                "support_depth": [3, 3],
                "gap_length_hist": [[0], [0]],
                "support_base_counts": [[0, 3, 0, 0], [0, 3, 0, 0]],
                "homopolymer_run_length": [1, 1],
                "tandem_repeat_flag": [0, 0],
            },
        }
        outputs = {
            "type_logits": torch.tensor([[0.1, 0.1, 0.1, 4.0], [5.0, 0.1, 0.1, 0.1]]),
            "sub_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            # Neural payload is wrong (C), but support insertion payload is A.
            "ins_base_logits": torch.tensor([[0.1, 8.0, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.zeros(2, 10),
            "delete_candidate_logits": torch.tensor([0.0, 0.0]),
            "delete_length_logits": torch.tensor([[5.0, 0.1], [5.0, 0.1]]),
            "trust": torch.tensor([1.0, 1.0]),
        }
        decoded = decode_example(
            target_seq,
            example,
            outputs,
            {
                "sub_threshold": 0.0,
                "del_threshold": 0.0,
                "ins_threshold": 0.0,
                "trust_threshold": 0.0,
                "max_deletion_length": 1,
                "mode": "debug",
                "full_trace": True,
                "hybrid_rule_decode": True,
                "support_rule_insertion_threshold": 0.50,
                "hybrid_ins_support_payload_rescue": True,
                "hybrid_ins_support_payload_min_fraction": 2 / 3,
                "hybrid_ins_support_payload_min_count": 2,
                "hybrid_ins_support_payload_allow_neighbor": False,
                "hybrid_require_rule_agreement_for_neural_edits": True,
                "hybrid_rule_agree_min_support_fraction": 0.90,
            },
        )
        self.assertEqual(decoded["predicted_labels"][0], "INS_A")
        self.assertEqual(decoded["prediction"], "CAC")
        self.assertTrue(decoded["trace"][0]["rescued_by_support_payload"])

    def test_hybrid_ins_support_payload_rescue_blocks_neighbor_conflict(self):
        target_seq = "CC"
        example = {
            "example_id": "neighbor_true_ins_a",
            "target_seq": target_seq,
            "edit_labels": [0, 0],
            "features": {
                "support_agreement": [1.0, 1.0],
                "support_entropy": [0.0, 0.0],
                "support_del_count": [0, 0],
                "support_ins_count": [2, 0],
                "support_ins_base_counts": [[2, 0, 0, 0], [0, 0, 0, 0]],
                "support_depth": [3, 3],
                "gap_length_hist": [[0], [0]],
                "support_base_counts": [[0, 3, 0, 0], [0, 3, 0, 0]],
                "homopolymer_run_length": [1, 1],
                "tandem_repeat_flag": [0, 0],
            },
        }
        outputs = {
            "type_logits": torch.tensor([[0.1, 0.1, 0.1, 4.0], [5.0, 0.1, 0.1, 0.1]]),
            "sub_base_logits": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "ins_base_logits": torch.tensor([[8.0, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
            "edit_logits": torch.zeros(2, 10),
            "delete_candidate_logits": torch.tensor([0.0, 0.0]),
            "delete_length_logits": torch.tensor([[5.0, 0.1], [5.0, 0.1]]),
            "trust": torch.tensor([1.0, 1.0]),
        }
        decoded = decode_example(
            target_seq,
            example,
            outputs,
            {
                "sub_threshold": 0.0,
                "del_threshold": 0.0,
                "ins_threshold": 0.0,
                "trust_threshold": 0.0,
                "max_deletion_length": 1,
                "mode": "debug",
                "full_trace": True,
                "hybrid_rule_decode": True,
                "support_rule_insertion_threshold": 0.50,
                "hybrid_ins_support_payload_rescue": True,
                "hybrid_ins_support_payload_min_fraction": 2 / 3,
                "hybrid_ins_support_payload_min_count": 2,
                "hybrid_ins_support_payload_allow_neighbor": False,
                "hybrid_require_rule_agreement_for_neural_edits": True,
            },
        )
        self.assertEqual(decoded["predicted_labels"][0], "COPY")
        self.assertIn("hybrid_ins_support_payload_neighbor_conflict", decoded["trace"][0]["veto_reasons"])


if __name__ == "__main__":
    unittest.main()
