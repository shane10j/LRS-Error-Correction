import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python"


class IntegrationTests(unittest.TestCase):
    def test_debug_tiny_pipeline(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            dataset_dir = output_root / "dataset"
            config_path = Path(tmp_dir) / "debug.yaml"
            config = {
                "name": "debug_tiny",
                "seed": 5,
                "dataset": {
                    "kind": "synthetic",
                    "sample_id": "DEBUG",
                    "output_dir": str(dataset_dir),
                    "splits": {"train": 8, "val": 4, "test": 4},
                    "max_window_length": 32,
                    "overlap": 8,
                    "max_support_reads": 4,
                    "max_deletion_length": 3,
                    "synthetic_seed": 5,
                },
                "model": {
                    "target_only": {"use_support": False},
                    "full": {"use_support": True},
                    "d_model": 16,
                    "conv_kernel_size": 3,
                    "support_hidden_dim": 16,
                },
                "train": {
                    "output_dir": str(output_root),
                    "epochs": 1,
                    "batch_size": 2,
                    "lr": 0.001,
                    "patience": 1,
                    "class_weights": {"COPY": 0.3, "SUB": 1.5, "DEL": 2.0, "INS": 1.2},
                },
                "decode": {
                    "sub_threshold": 0.55,
                    "del_threshold": 0.60,
                    "ins_threshold": 0.60,
                    "trust_threshold": 0.50,
                    "max_deletion_length": 3,
                },
                "baseline": {
                    "consensus_agreement_threshold": 0.75,
                    "deletion_threshold": 0.70,
                },
            }
            import yaml

            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            subprocess.run([PYTHON, "scripts/preprocess_dataset.py", "--config", str(config_path)], cwd=PROJECT_ROOT, check=True, env=env)
            subprocess.run([PYTHON, "scripts/train_model.py", "--config", str(config_path), "--run-name", "target_only"], cwd=PROJECT_ROOT, check=True, env=env)
            summary_path = output_root / "target_only" / "test_summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("usable_score", summary)


if __name__ == "__main__":
    unittest.main()
