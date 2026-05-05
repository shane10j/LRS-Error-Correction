import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_lr.train.losses import build_class_weights, build_type_weights, compute_losses
from omega_lr.eval.core import safe_divide


class LossTests(unittest.TestCase):
    def test_metric_denominator_safety(self):
        self.assertEqual(safe_divide(1, 0), 0.0)

    def test_loss_computation(self):
        config = {
            "train": {
                "class_weights": {"COPY": 0.3, "SUB": 1.5, "DEL": 2.0, "INS": 1.2},
            }
        }
        self.assertEqual(len(build_class_weights(config)), 10)
        self.assertEqual(len(build_type_weights(config)), 4)
        batch = {
            "attention_mask": torch.tensor([[1.0, 1.0]]),
            "edit_labels": torch.tensor([[0, 5]]),
            "delete_candidate_labels": torch.tensor([[0.0, 1.0]]),
            "delete_length_labels": torch.tensor([[0, 1]]),
        }
        outputs = {
            "type_logits": torch.randn(1, 2, 4),
            "sub_base_logits": torch.randn(1, 2, 4),
            "ins_base_logits": torch.randn(1, 2, 4),
            "edit_logits": torch.randn(1, 2, 10),
            "delete_candidate_logits": torch.randn(1, 2),
            "delete_length_logits": torch.randn(1, 2, 4),
            "trust": torch.rand(1, 2),
        }
        losses = compute_losses(batch, outputs, config, torch.device("cpu"))
        self.assertIn("total", losses)
        self.assertIn("support_sub_base_loss", losses)

    def test_support_sub_base_loss_teaches_majority_payload(self):
        config = {
            "model": {"full": {"use_support": True}},
            "_active_run_name": "full",
            "train": {
                "class_weights": {"COPY": 0.3, "SUB": 1.5, "DEL": 2.0, "INS": 1.2},
                "support_sub_base_loss_weight": 1.0,
                "support_sub_base_weights": {"T": 4.0},
            },
        }
        batch = {
            "attention_mask": torch.tensor([[1.0]]),
            "edit_labels": torch.tensor([[0]]),
            "delete_candidate_labels": torch.tensor([[0.0]]),
            "delete_length_labels": torch.tensor([[0]]),
            "support_majority_base_labels": torch.tensor([[3]]),
            "support_suggests_sub_labels": torch.tensor([[1.0]]),
            "support_suggests_ins_labels": torch.tensor([[0.0]]),
            "support_suggests_del_labels": torch.tensor([[0.0]]),
            "support_rule_type_labels": torch.tensor([[1]]),
            "target_tokens": torch.tensor([[1]]),
        }
        outputs = {
            "type_logits": torch.randn(1, 1, 4),
            "sub_base_logits": torch.tensor([[[4.0, 3.0, 2.0, -1.0]]]),
            "ins_base_logits": torch.randn(1, 1, 4),
            "edit_logits": torch.randn(1, 1, 10),
            "delete_candidate_logits": torch.randn(1, 1),
            "delete_length_logits": torch.randn(1, 1, 4),
            "trust": torch.rand(1, 1),
        }
        losses = compute_losses(batch, outputs, config, torch.device("cpu"))
        self.assertGreater(float(losses["support_sub_base_loss"]), 1.0)


if __name__ == "__main__":
    unittest.main()
