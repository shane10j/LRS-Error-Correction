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


if __name__ == "__main__":
    unittest.main()
