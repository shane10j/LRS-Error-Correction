#!/usr/bin/env python
"""Create a temporary debug config without requiring notebook-side package imports."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from omega_lr.utils import print_config, read_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--preset", default="debug_tiny")
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--enable-low-threshold-debug", action="store_true")
    parser.add_argument("--enable-overfit-debug", action="store_true")
    parser.add_argument("--overfit-run-name", default="target_only")
    parser.add_argument("--overfit-num-examples", type=int, default=4)
    parser.add_argument("--overfit-cases", default="sub,del,ins,copy")
    args = parser.parse_args()

    config = deepcopy(read_config(args.base_config))
    config["debug_run_name"] = args.run_name

    if args.preset == "debug_tiny" and args.enable_low_threshold_debug:
        config["name"] = f"{config['name']}_full_low_threshold_probe"
        config["train"]["output_dir"] = "outputs/debug_tiny_low_threshold_probe"
        config["decode"]["sub_threshold"] = 0.20
        config["decode"]["del_threshold"] = 0.20
        config["decode"]["ins_threshold"] = 0.20
        config["decode"]["trust_threshold"] = 0.05
        config["decode"]["mode"] = "debug"
        config["decode"]["use_trust_threshold"] = False
        config["decode"]["use_delete_candidate_veto"] = False
        config["decode"]["restrict_supported_candidates"] = False
        config["decode"]["consistency_check"] = False
        config["decode"]["full_trace"] = True

    if args.preset == "debug_tiny" and args.enable_overfit_debug:
        config["name"] = f"{config['name']}_{args.overfit_run_name}_overfit_probe"
        config["dataset"]["kind"] = "synthetic"
        config["dataset"]["output_dir"] = "outputs/debug_tiny_overfit_probe/dataset"
        config["dataset"]["splits"] = {
            "train": args.overfit_num_examples,
            "val": args.overfit_num_examples,
            "test": args.overfit_num_examples,
        }
        config["dataset"]["shared_examples_across_splits"] = True
        config["dataset"]["synthetic_case_names"] = [value for value in args.overfit_cases.split(",") if value]
        config["train"]["output_dir"] = "outputs/debug_tiny_overfit_probe"
        config["train"]["epochs"] = 200
        config["train"]["patience"] = 200
        config["train"]["batch_size"] = 1
        config["train"]["lr"] = 0.003
        config["train"]["class_weights"] = {"COPY": 1.0, "SUB": 3.5, "DEL": 1.0, "INS": 2.5}
        config["train"]["false_positive_penalty_weight"] = 0.02
        config["train"]["type_loss_weight"] = 4.0
        config["train"]["hard_type_loss_weight"] = 10.0
        config["train"]["sub_payload_loss_weight"] = 1.0
        config["train"]["ins_payload_loss_weight"] = 5.0
        config["train"]["type_margin"] = 0.50
        config["train"]["type_margin_weight"] = 1.00
        config["train"]["sub_copy_margin"] = 1.00
        config["train"]["sub_copy_margin_weight"] = 1.50
        config["train"]["sub_del_margin"] = 1.00
        config["train"]["sub_del_margin_weight"] = 1.50
        config["train"]["ins_del_margin"] = 0.75
        config["train"]["ins_del_margin_weight"] = 1.00
        config["train"]["non_del_margin"] = 0.75
        config["train"]["non_del_margin_weight"] = 1.00
        config["train"]["del_fallback_penalty_weight"] = 1.00
        config["train"]["hard_copy_penalty_weight"] = 1.00
        config["train"]["delete_candidate_aux_weight"] = 0.05
        config["train"]["encourage_edits_schedule"] = {
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
                "sub_threshold": config["decode"]["sub_threshold"],
                "del_threshold": config["decode"]["del_threshold"],
                "ins_threshold": config["decode"]["ins_threshold"],
                "trust_threshold": config["decode"]["trust_threshold"],
            },
        }
        config["decode"]["mode"] = "debug"
        config["decode"]["use_trust_threshold"] = False
        config["decode"]["use_delete_candidate_veto"] = False
        config["decode"]["restrict_supported_candidates"] = False
        config["decode"]["consistency_check"] = False
        config["decode"]["full_trace"] = True
        config["model_debug"] = {
            "use_trust_gate": False,
            "use_delete_length_head": False,
        }
        config["debug_run_name"] = args.overfit_run_name

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print_config(config)
    print(f"wrote_debug_config={output_path}")


if __name__ == "__main__":
    main()
