#!/usr/bin/env python
"""Run no_edit and consensus baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from omega_lr.baseline import consensus, no_edit, support_rule
from omega_lr.eval.summaries import benchmark_record, build_summary
from omega_lr.logging_utils import get_logger
from omega_lr.utils import dump_yaml, ensure_dir, load_json, load_jsonl, print_config, read_config, save_json, save_jsonl


LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    print_config(config)
    dataset_dir = Path(config["dataset"]["output_dir"])
    output_root = ensure_dir(Path(config["train"]["output_dir"]) / "baselines")
    manifest = load_json(dataset_dir / "manifest.json")
    rows = load_jsonl(dataset_dir / "test.jsonl")
    for name, predictor in {
        "no_edit": lambda example: no_edit.predict(example),
        "support_rule": lambda example: support_rule.predict(
            example,
            agreement_threshold=config["baseline"].get("support_rule_agreement_threshold", 0.60),
            insertion_threshold=config["baseline"].get("support_rule_insertion_threshold", 0.50),
            deletion_threshold=config["baseline"].get("support_rule_deletion_threshold", 0.50),
            use_confidence=config["baseline"].get("support_rule_use_confidence", False),
            confidence_min_fraction=config["baseline"].get("support_rule_confidence_min_fraction", 0.75),
            confidence_min_margin=config["baseline"].get("support_rule_confidence_min_margin", 1.0),
            confidence_max_entropy=config["baseline"].get("support_rule_confidence_max_entropy", 0.95),
            deletion_confidence_min_fraction=config["baseline"].get("support_rule_deletion_confidence_min_fraction", 0.90),
        ),
        "consensus": lambda example: consensus.predict(
            example,
            agreement_threshold=config["baseline"]["consensus_agreement_threshold"],
            deletion_threshold=config["baseline"]["deletion_threshold"],
        ),
    }.items():
        run_dir = ensure_dir(output_root / name)
        dump_yaml(config, run_dir / "config_snapshot.yaml")
        save_json(manifest, run_dir / "manifest.json")
        predictions = []
        for example in rows:
            result = predictor(example)
            predictions.append(
                {
                    "example_id": example["example_id"],
                    "target_seq": example["target_seq"],
                    "truth_seq": example["truth_seq"],
                    "prediction": result["prediction"],
                    "predicted_labels": result["predicted_labels"],
                    "gold_edit_labels": example["edit_labels"],
                    "features": example["features"],
                    "masks": example["masks"],
                    "trust": result["trust"],
                }
            )
        summary = build_summary(predictions)
        save_jsonl(predictions, run_dir / "test_predictions.jsonl")
        save_json(summary, run_dir / "test_summary.json")
        save_json(benchmark_record(name, summary), run_dir / "benchmark_summary.json")
        LOGGER.info("saved %s baseline summary to %s", name, run_dir)


if __name__ == "__main__":
    main()
