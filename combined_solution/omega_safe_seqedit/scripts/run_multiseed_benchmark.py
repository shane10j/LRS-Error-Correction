#!/usr/bin/env python
"""Run a Mac-sized multiseed benchmark and aggregate safety diagnostics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.baselines import conservative_consensus, no_edit, support_rule
from omega_safe_seqedit.config import load_config, print_resolved_config
from omega_safe_seqedit.io_utils import ensure_dir, read_jsonl, write_json
from omega_safe_seqedit.metrics import summarize_predictions
from omega_safe_seqedit.preprocess import write_dataset
from omega_safe_seqedit.trainer import evaluate_checkpoint, train


def _hash_file(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _score_baseline(records: list[dict], fn) -> dict:
    decoded = [{**record, **fn(record)} for record in records]
    return summarize_predictions(decoded)


def _seed_config(config: dict, seed: int, base_out: Path) -> dict:
    cfg = copy.deepcopy(config)
    cfg["seed"] = seed
    cfg["paths"]["output_dir"] = str(base_out / f"seed_{seed}")
    cfg["paths"]["dataset_dir"] = str(base_out / f"seed_{seed}" / "dataset")
    cfg["paths"]["baseline_dir"] = str(base_out / f"seed_{seed}" / "baselines")
    cfg["paths"]["runs_dir"] = str(base_out / f"seed_{seed}" / "runs")
    return cfg


def _aggregate(seed_summaries: list[dict]) -> dict:
    numeric: dict[str, list[float]] = {}
    for row in seed_summaries:
        for run_name, summary in row["summaries"].items():
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    numeric.setdefault(f"{run_name}.{key}", []).append(float(value))
    aggregate = {}
    for key, values in numeric.items():
        aggregate[key] = {
            "mean": sum(values) / max(len(values), 1),
            "min": min(values),
            "max": max(values),
            "values": values,
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--skip-train", action="store_true", help="Only aggregate/evaluate existing checkpoints.")
    args = parser.parse_args()
    config = load_config(args.config)
    print_resolved_config(config)
    seeds = args.seeds or config.get("multiseed", {}).get("seeds", [47, 48])
    runs = config.get("multiseed", {}).get("runs", ["target_only", "full"])
    modes_by_run = config.get("multiseed", {}).get("modes", {"target_only": ["neural"], "full": ["neural", "hybrid"]})
    base_out = ensure_dir(config["paths"]["output_dir"])
    seed_summaries = []
    for seed in seeds:
        seed_cfg = _seed_config(config, seed, base_out)
        print(f"\n=== seed {seed} ===")
        if not args.skip_train:
            write_dataset(seed_cfg)
            for run_name in runs:
                train(seed_cfg, run_name)
        test_path = Path(seed_cfg["paths"]["dataset_dir"]) / "test.jsonl"
        records = read_jsonl(test_path)
        summaries = {
            "no_edit": _score_baseline(records, no_edit),
            "consensus": _score_baseline(records, conservative_consensus),
            "support_rule": _score_baseline(records, support_rule),
        }
        for run_name in runs:
            for mode in modes_by_run.get(run_name, ["neural"]):
                key = f"{run_name}_{mode}"
                summaries[key] = evaluate_checkpoint(seed_cfg, run_name, split="test", mode=mode)
        label_counts = {"COPY": 0, "SUB": 0, "DEL": 0, "INS": 0}
        for record in records:
            label_counts["COPY"] += sum(1 for x in record["labels"]["main_type"] if x == 0)
            label_counts["SUB"] += sum(1 for x in record["labels"]["main_type"] if x == 1)
            label_counts["DEL"] += sum(1 for x in record["labels"]["main_type"] if x == 2)
            label_counts["INS"] += sum(1 for x in record["labels"]["insert_before"] if x > 0)
        seed_row = {
            "seed": seed,
            "dataset_hash": _hash_file(test_path),
            "label_counts": label_counts,
            "summaries": summaries,
        }
        seed_summaries.append(seed_row)
        write_json(Path(seed_cfg["paths"]["output_dir"]) / "seed_summary.json", seed_row)
    payload = {
        "config": config,
        "seed_summaries": seed_summaries,
        "aggregate": _aggregate(seed_summaries),
    }
    write_json(base_out / "multiseed_summary.json", payload)
    print(json.dumps({"wrote": str(base_out / "multiseed_summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
