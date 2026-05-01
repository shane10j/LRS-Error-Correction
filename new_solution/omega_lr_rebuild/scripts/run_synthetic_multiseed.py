#!/usr/bin/env python
"""Run the larger synthetic benchmark across multiple random seeds."""

from __future__ import annotations

import argparse
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
    output_root = ensure_dir(Path(args.output_root))
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
        records.append(load_record(Path(config["train"]["output_dir"]) / "benchmark_summary.json"))

    save_json(index, output_root / "multiseed_index.json")
    if records:
        save_json(summarize(records), output_root / "multiseed_summary.json")


if __name__ == "__main__":
    main()
