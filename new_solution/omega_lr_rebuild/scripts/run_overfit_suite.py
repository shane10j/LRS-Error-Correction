#!/usr/bin/env python
"""Run mandatory staged overfit proofs before benchmark interpretation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from omega_lr.utils import ensure_dir, save_json


PYTHON = "/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python"


PROOFS = [
    {"name": "single_sub", "cases": ["sub"]},
    {"name": "single_ins", "cases": ["ins"]},
    {"name": "single_del", "cases": ["del"]},
    {"name": "mixed_four", "cases": ["sub", "ins", "del", "copy"]},
]


def strict_overfit_config(output_root: Path, cases: list[str], seed: int) -> dict:
    return {
        "name": f"debug_tiny_overfit_{'_'.join(cases)}",
        "seed": seed,
        "dataset": {
            "kind": "synthetic",
            "sample_id": "DEBUG",
            "output_dir": str(output_root / "dataset"),
            "splits": {"train": len(cases), "val": len(cases), "test": len(cases)},
            "max_window_length": 32,
            "overlap": 8,
            "max_support_reads": 4,
            "max_deletion_length": 3,
            "synthetic_seed": seed,
            "shared_examples_across_splits": True,
            "synthetic_case_names": cases,
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


def run_command(args: list[str], cwd: Path, env: dict) -> None:
    print(" ".join(args))
    subprocess.run(args, cwd=cwd, check=True, env=env)


def validate_report(report: dict) -> list[str]:
    failures = []
    if report["prediction"] != report["truth_seq"]:
        failures.append(f"{report['example_id']}: decoded sequence does not match truth")
    gold_labels = report["gold_labels"]
    for position in report["positions"]:
        pos = position["pos"]
        if position["argmax_label"] != position["gold_label"]:
            failures.append(f"{report['example_id']} pos {pos}: wrong edit label")
        gold_type = position["gold_label"].split("_", 1)[0]
        if position["type_probs"][gold_type] <= 0.90:
            failures.append(f"{report['example_id']} pos {pos}: weak {gold_type} type probability")
        if position["gold_label"] == "DEL":
            expected_length = report["delete_length_labels"][pos]
            observed_length = position["decoder_trace"]["delete_length"]
            if observed_length != expected_length:
                failures.append(f"{report['example_id']} pos {pos}: wrong deletion length")
    for pos, gold in enumerate(gold_labels):
        if gold == "COPY" and report["argmax_labels"][pos] != "COPY":
            failures.append(f"{report['example_id']} pos {pos}: false edit on COPY position")
    for hard_pos in report["hard_positions"]:
        for neighbor in [hard_pos - 1, hard_pos + 1]:
            if 0 <= neighbor < len(gold_labels) and gold_labels[neighbor] == "COPY":
                if report["argmax_labels"][neighbor] != "COPY":
                    failures.append(f"{report['example_id']} pos {neighbor}: false neighboring edit")
    return failures


def run_proof(project_root: Path, output_root: Path, proof: dict, seed: int) -> dict:
    proof_root = ensure_dir(output_root / proof["name"])
    config_path = proof_root / "overfit.yaml"
    run_name = "target_only"
    config = strict_overfit_config(proof_root, proof["cases"], seed)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}

    run_command([PYTHON, "scripts/preprocess_dataset.py", "--config", str(config_path)], project_root, env)
    run_command([PYTHON, "scripts/train_model.py", "--config", str(config_path), "--run-name", run_name], project_root, env)
    for checkpoint_name in ["best.ckpt", "last.ckpt"]:
        run_command(
            [
                PYTHON,
                "scripts/debug_probe.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(proof_root / run_name / checkpoint_name),
                "--mode",
                "argmax",
                "--run-output-dir",
                str(proof_root / run_name / checkpoint_name.replace(".ckpt", "")),
                "--example-filters",
                "sub_,ins_,del_,copy_",
            ],
            project_root,
            env,
        )

    reports = json.loads((proof_root / run_name / "best" / "argmax_probe.json").read_text(encoding="utf-8"))
    failures = []
    for report in reports:
        failures.extend(validate_report(report))
    history = json.loads((proof_root / run_name / "history.json").read_text(encoding="utf-8"))
    best_selection = json.loads((proof_root / run_name / "best_selection_summary.json").read_text(encoding="utf-8"))
    if not best_selection["correction_quality"]["exact_zero_false_edits"]:
        failures.append("best.ckpt is not an exact zero-false-edit checkpoint")
    result = {
        "name": proof["name"],
        "cases": proof["cases"],
        "passed": not failures,
        "failures": failures,
        "reports": reports,
        "best_selection": best_selection,
        "final_train_metrics": history[-1]["train"] if history else {},
    }
    save_json(result, proof_root / "overfit_proof.json")
    if failures:
        raise AssertionError("\n".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/mandatory_overfit")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_root = ensure_dir(project_root / args.output_dir)
    results = [run_proof(project_root, output_root, proof, args.seed) for proof in PROOFS]
    suite = {"passed": all(result["passed"] for result in results), "proofs": results}
    save_json(suite, output_root / "overfit_suite.json")
    print(json.dumps({"passed": suite["passed"], "proofs": [result["name"] for result in results]}, indent=2))


if __name__ == "__main__":
    main()
