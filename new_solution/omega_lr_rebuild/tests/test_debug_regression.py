import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from omega_lr.constants import ID_TO_EDIT
from omega_lr.utils import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python"


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}


class DebugRegressionTests(unittest.TestCase):
    def test_full_debug_tiny_exact_edit_learning_regression(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "outputs"
            dataset_dir = output_root / "dataset"
            config_path = root / "debug_full.yaml"
            config = {
                "name": "debug_tiny_full_regression",
                "seed": 7,
                "dataset": {
                    "kind": "synthetic",
                    "sample_id": "DEBUG",
                    "output_dir": str(dataset_dir),
                    "splits": {"train": 4, "val": 4, "test": 4},
                    "max_window_length": 32,
                    "overlap": 8,
                    "max_support_reads": 4,
                    "max_deletion_length": 3,
                    "synthetic_seed": 7,
                    "shared_examples_across_splits": True,
                    "synthetic_case_names": ["sub", "ins", "del", "copy"],
                },
                "model": {
                    "target_only": {"use_support": False},
                    "full": {"use_support": True},
                    "d_model": 32,
                    "conv_kernel_size": 3,
                    "support_hidden_dim": 16,
                },
                "model_debug": {
                    "use_trust_gate": False,
                    "use_delete_length_head": False,
                },
                "train": {
                    "output_dir": str(output_root),
                    "epochs": 200,
                    "batch_size": 1,
                    "lr": 0.003,
                    "patience": 200,
                    "class_weights": {"COPY": 1.0, "SUB": 3.5, "DEL": 1.0, "INS": 2.5},
                    "false_positive_penalty_weight": 0.02,
                    "type_loss_weight": 4.0,
                    "hard_type_loss_weight": 10.0,
                    "sub_payload_loss_weight": 1.0,
                    "ins_payload_loss_weight": 5.0,
                    "type_margin": 0.50,
                    "type_margin_weight": 1.00,
                    "sub_copy_margin": 1.00,
                    "sub_copy_margin_weight": 1.50,
                    "sub_del_margin": 1.00,
                    "sub_del_margin_weight": 1.50,
                    "ins_del_margin": 0.75,
                    "ins_del_margin_weight": 1.00,
                    "non_del_margin": 0.75,
                    "non_del_margin_weight": 1.00,
                    "del_fallback_penalty_weight": 1.00,
                    "hard_copy_penalty_weight": 1.00,
                    "delete_candidate_aux_weight": 0.05,
                    "encourage_edits_schedule": {
                        "warmup_epochs": 20,
                        "early_false_positive_penalty_weight": 0.02,
                        "late_false_positive_penalty_weight": 0.35,
                        "early_copy_to_sub_penalty_weight": 0.00,
                        "late_copy_to_sub_penalty_weight": 0.60,
                        "early_copy_to_del_penalty_weight": 0.00,
                        "late_copy_to_del_penalty_weight": 0.90,
                        "early_copy_to_ins_penalty_weight": 0.00,
                        "late_copy_to_ins_penalty_weight": 0.60,
                        "early_positive_hard_edit_reward_weight": 1.20,
                        "late_positive_hard_edit_reward_weight": 0.70,
                        "early_gate_open_bias": 0.0,
                        "late_gate_open_bias": 0.0,
                        "early_trust_regularization_weight": 0.0,
                        "late_trust_regularization_weight": 0.0,
                        "early_curriculum_fraction": 1.0,
                        "late_curriculum_fraction": 1.0,
                        "soft_decode_thresholds": {
                            "sub_threshold": 0.05,
                            "del_threshold": 0.05,
                            "ins_threshold": 0.05,
                            "trust_threshold": 0.0,
                        },
                    },
                },
                "decode": {
                    "sub_threshold": 0.05,
                    "del_threshold": 0.05,
                    "ins_threshold": 0.05,
                    "trust_threshold": 0.0,
                    "max_deletion_length": 3,
                    "mode": "debug",
                    "use_trust_threshold": False,
                    "use_delete_candidate_veto": False,
                    "restrict_supported_candidates": False,
                    "consistency_check": False,
                    "full_trace": True,
                },
                "baseline": {
                    "consensus_agreement_threshold": 0.75,
                    "deletion_threshold": 0.70,
                },
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            env = _env()
            subprocess.run([PYTHON, "scripts/preprocess_dataset.py", "--config", str(config_path)], cwd=PROJECT_ROOT, check=True, env=env)
            subprocess.run([PYTHON, "scripts/train_model.py", "--config", str(config_path), "--run-name", "full"], cwd=PROJECT_ROOT, check=True, env=env)
            run_dir = output_root / "full"
            summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["sequence"]["identity"], 1.0)
            self.assertEqual(summary["sequence"]["edit_distance"], 0.0)
            self.assertEqual(summary["safety"]["overcorrection_rate"], 0.0)
            self.assertEqual(summary["safety"]["hard_edit_false_positive_rate"], 0.0)

            subprocess.run(
                [
                    PYTHON,
                    "scripts/debug_probe.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(run_dir / "best.ckpt"),
                    "--mode",
                    "classwise",
                    "--run-output-dir",
                    str(run_dir),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                env=env,
            )
            classwise = json.loads((run_dir / "classwise_probe.json").read_text(encoding="utf-8"))
            self.assertEqual(classwise["class_stats"]["SUB_C"]["argmax_match_rate"], 1.0)
            self.assertEqual(classwise["class_stats"]["INS_T"]["argmax_match_rate"], 1.0)
            self.assertEqual(classwise["class_stats"]["DEL"]["argmax_match_rate"], 1.0)

    def test_harder_synthetic_dataset_is_nonshared_and_payload_complete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = yaml.safe_load((PROJECT_ROOT / "configs" / "debug_synthetic_generalization.yaml").read_text(encoding="utf-8"))
            config["dataset"]["output_dir"] = str(Path(tmp_dir) / "dataset")
            config["train"]["output_dir"] = str(Path(tmp_dir) / "outputs")
            config_path = Path(tmp_dir) / "harder.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            subprocess.run([PYTHON, "scripts/preprocess_dataset.py", "--config", str(config_path)], cwd=PROJECT_ROOT, check=True, env=_env())

            train_rows = load_jsonl(Path(config["dataset"]["output_dir"]) / "train.jsonl")
            test_rows = load_jsonl(Path(config["dataset"]["output_dir"]) / "test.jsonl")
            train_pairs = {(row["target_seq"], row["truth_seq"]) for row in train_rows}
            test_pairs = {(row["target_seq"], row["truth_seq"]) for row in test_rows}
            self.assertTrue(train_pairs.isdisjoint(test_pairs))

            observed = {
                ID_TO_EDIT[int(label)]
                for row in train_rows + test_rows
                for label in row["edit_labels"]
            }
            for label in ["SUB_A", "SUB_C", "SUB_G", "SUB_T", "INS_A", "INS_C", "INS_G", "INS_T", "DEL"]:
                self.assertIn(label, observed)

            for row in test_rows:
                for pos, label_id in enumerate(row["edit_labels"]):
                    label = ID_TO_EDIT[int(label_id)]
                    if not label.startswith("INS_"):
                        continue
                    base_idx = "ACGT".index(label[-1])
                    self.assertGreater(
                        row["features"]["support_ins_count"][pos],
                        0,
                        msg=f"missing insertion count at {row['example_id']} pos={pos} label={label}",
                    )
                    self.assertGreater(
                        row["features"]["support_ins_base_counts"][pos][base_idx],
                        0,
                        msg=f"missing insertion base support at {row['example_id']} pos={pos} label={label}",
                    )

    def test_large_synthetic_dataset_has_boundary_insertions_with_support(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = yaml.safe_load((PROJECT_ROOT / "configs" / "debug_synthetic_generalization_large.yaml").read_text(encoding="utf-8"))
            config["dataset"]["output_dir"] = str(Path(tmp_dir) / "dataset")
            config["train"]["output_dir"] = str(Path(tmp_dir) / "outputs")
            config["dataset"]["splits"] = {"train": 80, "val": 20, "test": 40}
            config_path = Path(tmp_dir) / "large.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            subprocess.run([PYTHON, "scripts/preprocess_dataset.py", "--config", str(config_path)], cwd=PROJECT_ROOT, check=True, env=_env())

            rows = load_jsonl(Path(config["dataset"]["output_dir"]) / "test.jsonl")
            boundary_insertions = []
            for row in rows:
                for pos, label_id in enumerate(row["edit_labels"]):
                    label = ID_TO_EDIT[int(label_id)]
                    if label.startswith("INS_") and pos == 0:
                        boundary_insertions.append((row, pos, label))
            self.assertTrue(boundary_insertions)
            for row, pos, label in boundary_insertions:
                base_idx = "ACGT".index(label[-1])
                self.assertGreater(row["features"]["support_ins_count"][pos], 0)
                self.assertGreater(row["features"]["support_ins_base_counts"][pos][base_idx], 0)
