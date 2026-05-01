import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python"


class OverfitTests(unittest.TestCase):
    def run_overfit_case(self, case_names: list[str], run_name: str = "target_only") -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            output_root = tmp_root / "outputs"
            dataset_dir = output_root / "dataset"
            config_path = tmp_root / "overfit.yaml"
            probe_dir = output_root / "target_only"
            config = {
                "name": "debug_tiny_overfit_ci",
                "seed": 7,
                "dataset": {
                    "kind": "synthetic",
                    "sample_id": "DEBUG",
                    "output_dir": str(dataset_dir),
                    "splits": {"train": len(case_names), "val": len(case_names), "test": len(case_names)},
                    "max_window_length": 32,
                    "overlap": 8,
                    "max_support_reads": 4,
                    "max_deletion_length": 3,
                    "synthetic_seed": 7,
                    "shared_examples_across_splits": True,
                    "synthetic_case_names": case_names,
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
            subprocess.run([PYTHON, "scripts/preprocess_dataset.py", "--config", str(config_path)], cwd=PROJECT_ROOT, check=True, env=env)
            subprocess.run([PYTHON, "scripts/train_model.py", "--config", str(config_path), "--run-name", run_name], cwd=PROJECT_ROOT, check=True, env=env)
            subprocess.run(
                    [
                        PYTHON,
                        "scripts/debug_probe.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(output_root / run_name / "best.ckpt"),
                    "--mode",
                    "argmax",
                        "--run-output-dir",
                        str(output_root / run_name),
                        "--example-filters",
                        "sub_,ins_,del_,copy_",
                    ],
                cwd=PROJECT_ROOT,
                check=True,
                env=env,
            )
            reports = json.loads(((output_root / run_name) / "argmax_probe.json").read_text(encoding="utf-8"))
            best_selection = json.loads(((output_root / run_name) / "best_selection_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(best_selection["selection_mode"], "exact_correction_quality")
            self.assertTrue(best_selection["correction_quality"]["exact_zero_false_edits"])
            self.assertTrue(reports)
            for report in reports:
                self.assertEqual(
                    report["prediction"],
                    report["truth_seq"],
                    msg=f"Decoded sequence mismatch for {report['example_id']}",
                )
                gold_labels = report["gold_labels"]
                hard_positions = [idx for idx, label in enumerate(gold_labels) if label != "COPY"]
                for position in report["positions"]:
                    self.assertEqual(
                        position["argmax_label"],
                        position["gold_label"],
                        msg=f"Failed overfit for {case_names}: {report['example_id']} pos {position['pos']}",
                    )
                    gold_type = position["gold_label"].split("_", 1)[0]
                    self.assertGreater(
                        position["type_probs"][gold_type],
                        0.9,
                        msg=f"Gold type probability too low for {report['example_id']} pos {position['pos']}",
                    )
                for hard_pos in hard_positions:
                    predicted_label = report["argmax_labels"][hard_pos]
                    self.assertEqual(
                        predicted_label,
                        gold_labels[hard_pos],
                        msg=f"Decoded label mismatch for {report['example_id']} pos {hard_pos}",
                    )
                    for neighbor in [hard_pos - 1, hard_pos + 1]:
                        if 0 <= neighbor < len(gold_labels) and gold_labels[neighbor] == "COPY":
                            self.assertEqual(
                                report["argmax_labels"][neighbor],
                                "COPY",
                                msg=f"False neighboring edit for {report['example_id']} pos {neighbor}",
                            )
                    if gold_labels[hard_pos] == "DEL":
                        trace = report["positions"][0]["decoder_trace"]
                        self.assertEqual(
                            trace["delete_length"],
                            report["delete_length_labels"][hard_pos],
                            msg=f"Wrong DEL length for {report['example_id']} pos {hard_pos}",
                        )
                for pos, gold_label in enumerate(gold_labels):
                    if gold_label == "COPY":
                        self.assertEqual(
                            report["argmax_labels"][pos],
                            "COPY",
                            msg=f"False edit on COPY position for {report['example_id']} pos {pos}",
                        )

    def test_single_sub_overfit(self):
        self.run_overfit_case(["sub"])

    def test_single_ins_overfit(self):
        self.run_overfit_case(["ins"])

    def test_single_del_overfit(self):
        self.run_overfit_case(["del"])

    def test_mixed_four_example_overfit(self):
        self.run_overfit_case(["sub", "ins", "del", "copy"], run_name="target_only")


if __name__ == "__main__":
    unittest.main()
