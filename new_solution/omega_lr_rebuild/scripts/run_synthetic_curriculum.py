#!/usr/bin/env python
"""Run staged synthetic generalization curricula from the harder synthetic base."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from omega_lr.utils import ensure_dir, print_config, read_config, save_json


STAGES = {
    "copy_sub": ["sub_a", "sub_c", "sub_g", "sub_t", "copy_harder_0", "copy_harder_1", "copy_harder_2", "copy_harder_3"],
    "copy_ins": ["ins_a", "ins_c", "ins_g", "ins_t", "copy_harder_0", "copy_harder_1", "copy_harder_2", "copy_harder_3"],
    "copy_del": ["del_ctx_0", "del_ctx_1", "del_ctx_2", "hpoly_del_harder", "copy_harder_0", "copy_harder_1", "copy_harder_2", "copy_harder_3"],
    "mixed_hard": ["sub_a", "sub_c", "sub_g", "sub_t", "ins_a", "ins_c", "ins_g", "ins_t", "del_ctx_0", "del_ctx_1", "del_ctx_2", "hpoly_del_harder"],
    "full_harder": [],
}


def run(project_root: Path, *args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    command = [sys.executable, *args]
    print(" ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)


def stage_config(base: dict, stage: str, output_root: Path) -> dict:
    config = deepcopy(base)
    config["name"] = f"{base['name']}_{stage}"
    config["dataset"]["shared_examples_across_splits"] = False
    config["dataset"]["synthetic_suite"] = "harder"
    config["dataset"]["output_dir"] = str(output_root / stage / "dataset")
    config["train"]["output_dir"] = str(output_root / stage)
    if STAGES[stage]:
        config["dataset"]["synthetic_case_names"] = STAGES[stage]
    else:
        config["dataset"].pop("synthetic_case_names", None)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/debug_synthetic_generalization.yaml")
    parser.add_argument("--output-root", default="outputs/debug_synthetic_curriculum")
    parser.add_argument("--stages", default="copy_sub,copy_ins,copy_del,mixed_hard,full_harder")
    parser.add_argument("--runs", default="target_only,full")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base = read_config(args.base_config)
    output_root = ensure_dir(Path(args.output_root))
    selected_stages = [stage for stage in args.stages.split(",") if stage]
    selected_runs = [run_name for run_name in args.runs.split(",") if run_name]
    summaries = {}
    for stage in selected_stages:
        if stage not in STAGES:
            raise KeyError(f"Unknown stage {stage}. Expected one of {sorted(STAGES)}")
        config = stage_config(base, stage, output_root)
        config_path = output_root / stage / "config.yaml"
        ensure_dir(config_path.parent)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print_config(config)
        run(project_root, "scripts/preprocess_dataset.py", "--config", str(config_path))
        run(project_root, "scripts/run_baselines.py", "--config", str(config_path))
        for run_name in selected_runs:
            run(project_root, "scripts/train_model.py", "--config", str(config_path), "--run-name", run_name)
        run(project_root, "scripts/export_summary.py", "--config", str(config_path))
        summaries[stage] = {
            "config": str(config_path),
            "summary": str(Path(config["train"]["output_dir"]) / "benchmark_summary.json"),
        }
    save_json(summaries, output_root / "curriculum_summary_index.json")


if __name__ == "__main__":
    main()
