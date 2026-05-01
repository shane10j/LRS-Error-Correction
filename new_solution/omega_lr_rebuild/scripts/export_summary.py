#!/usr/bin/env python
"""Export a benchmark summary across baselines and model runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omega_lr.utils import print_config, read_config, save_json
from omega_lr.eval.summaries import benchmark_record


def load_summary_if_exists(path: Path):
    if not path.exists():
        return None
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_benchmark_record(path: Path, run_name: str):
    record = load_summary_if_exists(path)
    if record is None:
        return None
    if all(key in record for key in ["identity", "edit_distance", "overcorrection_rate", "hard_edit_false_positive_rate"]):
        return record
    return benchmark_record(run_name, record.get("test_summary", {}))


def ensure_required_runs(config_path: str, output_root: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    required_paths = {
        "no_edit": output_root / "baselines/no_edit/benchmark_summary.json",
        "support_rule": output_root / "baselines/support_rule/benchmark_summary.json",
        "consensus": output_root / "baselines/consensus/benchmark_summary.json",
        "target_only": output_root / "target_only/benchmark_summary.json",
        "full_hybrid": output_root / "full/benchmark_summary.json",
        "full_neural_only": output_root / "full/full_neural_only_benchmark_summary.json",
    }
    if any(not required_paths[name].exists() for name in ["no_edit", "support_rule", "consensus"]):
        subprocess.run([sys.executable, "scripts/run_baselines.py", "--config", config_path], cwd=project_root, check=True, env=env)
    for run_name in ("target_only", "full"):
        required = [output_root / f"{run_name}/benchmark_summary.json"]
        if run_name == "full":
            required.append(output_root / "full/full_neural_only_benchmark_summary.json")
        if any(not path.exists() for path in required):
            subprocess.run([sys.executable, "scripts/train_model.py", "--config", config_path, "--run-name", run_name], cwd=project_root, check=True, env=env)


def check_regression_targets(config: dict, summary: dict) -> dict:
    """Fail loudly when named precision-first baselines regress."""
    runs = summary.get("runs", {})
    checks = []
    failures = []
    for target_name, target in config.get("regression_targets", {}).items():
        run_name = target.get("run_name", "full_hybrid")
        record = runs.get(run_name) or {}
        for key, expected in target.items():
            if key == "run_name":
                continue
            if key.endswith("_min"):
                metric = key[: -len("_min")]
                observed = record.get(metric, 0.0)
                passed = observed >= expected
                comparator = ">="
            elif key.endswith("_max"):
                metric = key[: -len("_max")]
                observed = record.get(metric, 0.0)
                passed = observed <= expected
                comparator = "<="
            else:
                continue
            row = {
                "target": target_name,
                "run_name": run_name,
                "metric": metric,
                "observed": observed,
                "comparator": comparator,
                "expected": expected,
                "passed": passed,
            }
            checks.append(row)
            if not passed:
                failures.append(row)
    return {"passed": not failures, "checks": checks, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--warn-on-regression-failure",
        action="store_true",
        help="Write summaries and warn instead of exiting nonzero when configured regression targets fail.",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    print_config(config)
    output_root = Path(config["train"]["output_dir"])
    ensure_required_runs(str(Path(config["resolved_config_path"])), output_root)
    summary = {
        "preset": config["name"],
        "runs": {
            "no_edit": load_benchmark_record(output_root / "baselines/no_edit/benchmark_summary.json", "no_edit"),
            "support_rule": load_benchmark_record(output_root / "baselines/support_rule/benchmark_summary.json", "support_rule"),
            "consensus": load_benchmark_record(output_root / "baselines/consensus/benchmark_summary.json", "consensus"),
            "target_only": load_benchmark_record(output_root / "target_only/benchmark_summary.json", "target_only"),
            "full_hybrid": load_benchmark_record(output_root / "full/benchmark_summary.json", "full_hybrid"),
            "full_neural_only": load_benchmark_record(output_root / "full/full_neural_only_benchmark_summary.json", "full_neural_only"),
            "full": load_benchmark_record(output_root / "full/benchmark_summary.json", "full_hybrid"),
        },
    }
    regression = check_regression_targets(config, summary)
    summary["regression_targets"] = regression
    save_json(summary, output_root / "benchmark_summary.json")
    save_json(regression, output_root / "regression_targets.json")
    if not regression["passed"]:
        if args.warn_on_regression_failure:
            print(f"[WARN] Regression targets failed: {regression['failures']}")
            return
        raise SystemExit(f"Regression targets failed: {regression['failures']}")


if __name__ == "__main__":
    main()
