#!/usr/bin/env python
"""Run the larger synthetic benchmark across multiple random seeds."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from omega_lr.utils import ensure_dir, read_config, save_json


def run(project_root: Path, *args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    command = [sys.executable, *args]
    print(" ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)


def seed_config(base: dict, seed: int, output_root: Path, train_examples: int | None, val_examples: int | None, test_examples: int | None) -> dict:
    config = deepcopy(base)
    config["name"] = f"{base['name']}_seed_{seed}"
    config["dataset"]["synthetic_seed"] = seed
    config["dataset"]["shared_examples_across_splits"] = False
    config["dataset"]["output_dir"] = str(output_root / f"seed_{seed}" / "dataset")
    config["train"]["output_dir"] = str(output_root / f"seed_{seed}")
    if train_examples is not None:
        config["dataset"]["splits"]["train"] = train_examples
    if val_examples is not None:
        config["dataset"]["splits"]["val"] = val_examples
    if test_examples is not None:
        config["dataset"]["splits"]["test"] = test_examples
    return config


def load_record(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def file_md5(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_run_metrics(record: dict) -> dict:
    compact = {}
    for run_name, run_record in record.get("runs", {}).items():
        if run_record is None:
            compact[run_name] = None
            continue
        compact[run_name] = {
            key: run_record.get(key)
            for key in [
                "identity",
                "edit_distance",
                "normalized_edit_distance",
                "overcorrection_rate",
                "hard_edit_false_positive_rate",
                "usable_score",
            ]
        }
        compact[run_name]["confusion_matrix"] = run_record.get("test_summary", {}).get("confusion_matrix", {})
    return compact


def build_audit(index: dict, records: list[dict], output_root: Path) -> dict:
    seed_records = []
    for seed, row in index.items():
        dataset_dir = Path(row["output_dir"]) / "dataset"
        summary_path = Path(row["summary"])
        record = load_record(summary_path) if summary_path.exists() else None
        seed_records.append(
            {
                "seed": int(seed),
                "config": row["config"],
                "dataset_hashes": {
                    split: file_md5(dataset_dir / f"{split}.jsonl")
                    for split in ["train", "val", "test"]
                },
                "summary": str(summary_path),
                "summary_md5": file_md5(summary_path),
                "metrics": compact_run_metrics(record or {}),
            }
        )
    uniqueness = {}
    for split in ["train", "val", "test"]:
        hashes = [row["dataset_hashes"][split] for row in seed_records]
        uniqueness[f"{split}_dataset_hashes_unique"] = len(set(hashes)) == len(hashes)
        uniqueness[f"{split}_unique_hash_count"] = len(set(hashes))
    metric_digests = [
        hashlib.md5(str(row["metrics"]).encode("utf-8")).hexdigest()
        for row in seed_records
    ]
    summary_hashes = [row["summary_md5"] for row in seed_records]
    uniqueness["summary_files_unique"] = len(set(summary_hashes)) == len(summary_hashes)
    uniqueness["summary_file_unique_count"] = len(set(summary_hashes))
    uniqueness["metric_records_unique"] = len(set(metric_digests)) == len(metric_digests)
    uniqueness["metric_record_unique_count"] = len(set(metric_digests))
    audit = {
        "output_root": str(output_root),
        "num_seed_records": len(seed_records),
        "uniqueness": uniqueness,
        "seed_records": seed_records,
    }
    if records and not uniqueness["metric_records_unique"]:
        audit["note"] = (
            "Datasets may still be seed-unique even when metrics are identical; "
            "inspect dataset hashes and false-positive dumps before assuming reuse."
        )
    return audit


def summarize(records: list[dict]) -> dict:
    metric_names = ["identity", "usable_score", "hard_edit_false_positive_rate", "overcorrection_rate"]
    runs = sorted({run for record in records for run in record["runs"] if record["runs"][run] is not None})
    summary = {"num_seeds": len(records), "runs": {}}
    for run_name in runs:
        summary["runs"][run_name] = {}
        for metric in metric_names:
            values = [record["runs"][run_name][metric] for record in records if record["runs"].get(run_name) is not None]
            if not values:
                continue
            summary["runs"][run_name][metric] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/debug_synthetic_generalization_large.yaml")
    parser.add_argument("--output-root", default="outputs/debug_synthetic_generalization_multiseed")
    parser.add_argument("--seeds", default="47,48,49,50,51")
    parser.add_argument("--runs", default="target_only,full")
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--val-examples", type=int)
    parser.add_argument("--test-examples", type=int)
    parser.add_argument("--skip-training", action="store_true", help="Only write per-seed configs for inspection.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base = read_config(args.base_config)
    output_root = ensure_dir(Path(args.output_root)).resolve()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    selected_runs = [run_name for run_name in args.runs.split(",") if run_name]

    index = {}
    records = []
    for seed in seeds:
        config = seed_config(base, seed, output_root, args.train_examples, args.val_examples, args.test_examples)
        config_path = output_root / f"seed_{seed}" / "config.yaml"
        ensure_dir(config_path.parent)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        index[str(seed)] = {
            "config": str(config_path),
            "output_dir": config["train"]["output_dir"],
            "summary": str(Path(config["train"]["output_dir"]) / "benchmark_summary.json"),
        }
        if args.skip_training:
            continue
        run(project_root, "scripts/preprocess_dataset.py", "--config", str(config_path))
        run(project_root, "scripts/run_baselines.py", "--config", str(config_path))
        for run_name in selected_runs:
            run(project_root, "scripts/train_model.py", "--config", str(config_path), "--run-name", run_name)
        run(project_root, "scripts/export_summary.py", "--config", str(config_path), "--warn-on-regression-failure")
        if "full" in selected_runs:
            full_dir = Path(config["train"]["output_dir"]) / "full"
            checkpoint = full_dir / "best.ckpt"
            if checkpoint.exists():
                run(
                    project_root,
                    "scripts/debug_probe.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--mode",
                    "false_hard",
                    "--run-output-dir",
                    str(full_dir),
                )
        records.append(load_record(Path(config["train"]["output_dir"]) / "benchmark_summary.json"))

    save_json(index, output_root / "multiseed_index.json")
    save_json(build_audit(index, records, output_root), output_root / "multiseed_audit.json")
    if records:
        save_json(summarize(records), output_root / "multiseed_summary.json")


if __name__ == "__main__":
    main()
