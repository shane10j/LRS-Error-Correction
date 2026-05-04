#!/usr/bin/env python
"""Run thresholded/argmax/classwise debug probes and save outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from omega_lr.eval.inspect import (
    argmax_decode_reports,
    calibration_allowed_missed_edits,
    classwise_edit_statistics,
    false_sub_diagnostics,
    false_hard_edit_diagnostics,
    hard_edit_learnability_summary,
    hybrid_gap_report,
    hybrid_miss_diagnostics,
    insertion_payload_diagnostics,
    inspect_debug_examples,
    missed_hard_edit_evidence,
    print_argmax_reports,
    print_calibration_allowed_missed_edits,
    print_false_sub_reports,
    print_false_hard_edit_reports,
    print_hybrid_gap_report,
    print_hybrid_miss_reports,
    print_insertion_payload_reports,
    print_inspection_report,
    print_missed_evidence_reports,
    print_support_rule_audit,
    print_support_rule_calibration_report,
    print_vetoed_true_edit_reports,
    support_rule_calibration_report,
    support_rule_positive_audit,
    vetoed_true_edit_diagnostics,
)
from omega_lr.utils import ensure_dir, print_config, read_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "thresholded",
            "argmax",
            "classwise",
            "missed_evidence",
            "hybrid_gap",
            "hybrid_miss",
            "false_sub",
            "false_hard",
            "ins_payload",
            "support_rule_audit",
            "rule_calibration",
            "calibration_gap",
            "vetoed_true",
        ],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--example-filters", default="sub_,del_,ins_")
    args = parser.parse_args()

    config = read_config(args.config)
    print_config(config)
    output_dir = ensure_dir(Path(args.run_output_dir))
    checkpoint = Path(args.checkpoint)
    example_filters = tuple(value for value in args.example_filters.split(",") if value)

    if args.mode == "thresholded":
        reports = inspect_debug_examples(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
            example_filters=example_filters,
        )
        print_inspection_report(reports)
        save_json(reports, output_dir / "thresholded_probe.json")
        return

    if args.mode == "argmax":
        reports = argmax_decode_reports(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
            example_filters=example_filters,
        )
        print_argmax_reports(reports)
        save_json(reports, output_dir / "argmax_probe.json")
        return

    if args.mode == "missed_evidence":
        reports = missed_hard_edit_evidence(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_missed_evidence_reports(reports)
        save_json(reports, output_dir / "missed_evidence_probe.json")
        return

    if args.mode == "hybrid_gap":
        report = hybrid_gap_report(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_hybrid_gap_report(report)
        save_json(report, output_dir / "hybrid_gap_probe.json")
        return

    if args.mode == "hybrid_miss":
        reports = hybrid_miss_diagnostics(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_hybrid_miss_reports(reports)
        save_json(reports, output_dir / "hybrid_miss_probe.json")
        return

    if args.mode == "false_sub":
        reports = false_sub_diagnostics(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_false_sub_reports(reports)
        save_json(reports, output_dir / "false_sub_probe.json")
        return

    if args.mode == "false_hard":
        reports = false_hard_edit_diagnostics(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_false_hard_edit_reports(reports)
        save_json(reports, output_dir / "false_hard_probe.json")
        return

    if args.mode == "vetoed_true":
        reports = vetoed_true_edit_diagnostics(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_vetoed_true_edit_reports(reports)
        save_json(reports, output_dir / "vetoed_true_probe.json")
        return

    if args.mode == "ins_payload":
        reports = insertion_payload_diagnostics(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_insertion_payload_reports(reports)
        save_json(reports, output_dir / "ins_payload_probe.json")
        return

    if args.mode == "support_rule_audit":
        report = support_rule_positive_audit(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_support_rule_audit(report)
        save_json(report, output_dir / "support_rule_audit_probe.json")
        return

    if args.mode == "rule_calibration":
        report = support_rule_calibration_report(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_support_rule_calibration_report(report)
        save_json(report, output_dir / "rule_calibration_probe.json")
        return

    if args.mode == "calibration_gap":
        reports = calibration_allowed_missed_edits(
            config=config,
            checkpoint_path=checkpoint,
            split=args.split,
        )
        print_calibration_allowed_missed_edits(reports)
        save_json(reports, output_dir / "calibration_gap_probe.json")
        return

    class_stats = classwise_edit_statistics(
        config=config,
        checkpoint_path=checkpoint,
        split=args.split,
    )
    summary = {
        "class_stats": class_stats,
        "hard_edit_learnability": hard_edit_learnability_summary(class_stats),
    }
    save_json(summary, output_dir / "classwise_probe.json")
    print(summary)


if __name__ == "__main__":
    main()
