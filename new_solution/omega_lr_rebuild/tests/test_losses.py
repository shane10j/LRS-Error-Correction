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


if __name__ == "__main__":
    unittest.main()
