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
from omega_lr.constants import ID_TO_EDIT


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


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def correction_counts(config: dict, output_root: Path, run_name: str) -> dict:
    """Count corrected, missed, and false hard edits for regression gates."""
    if run_name in {"full_hybrid", "full"}:
        predictions_path = output_root / "full/test_predictions.jsonl"
    elif run_name == "target_only":
        predictions_path = output_root / "target_only/test_predictions.jsonl"
    else:
        predictions_path = output_root / f"baselines/{run_name}/test_predictions.jsonl"
    examples = load_jsonl(Path(config["dataset"]["output_dir"]) / "test.jsonl")
    predictions = load_jsonl(predictions_path)
    corrected_edits = 0
    missed_edits = 0
    false_edits = 0
    for example, prediction in zip(examples, predictions):
        gold_labels = [ID_TO_EDIT[int(label)] for label in example["edit_labels"]]
        predicted_labels = prediction.get("predicted_labels", [])
        for pos, gold in enumerate(gold_labels):
            pred = predicted_labels[pos] if pos < len(predicted_labels) else "COPY"
            if gold != "COPY" and pred == gold:
                corrected_edits += 1
            elif gold != "COPY" and pred != gold:
                missed_edits += 1
            elif gold == "COPY" and pred != "COPY":
                false_edits += 1
    return {
        "corrected_edits": corrected_edits,
        "missed_edits": missed_edits,
        "false_edits": false_edits,
    }


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


def check_regression_targets(config: dict, summary: dict, output_root: Path) -> dict:
    """Fail loudly when named precision-first baselines regress."""
    runs = summary.get("runs", {})
    checks = []
    failures = []
    for target_name, target in config.get("regression_targets", {}).items():
        run_name = target.get("run_name", "full_hybrid")
        record = runs.get(run_name) or {}
        count_record = None
        for key, expected in target.items():
            if key == "run_name":
                continue
            if key in {"corrected_edits_min", "missed_edits_max", "false_edits_max"}:
                if count_record is None:
                    count_record = correction_counts(config, output_root, run_name)
                metric = key.rsplit("_", 1)[0]
                observed = count_record.get(metric, 0)
                comparator = ">=" if key.endswith("_min") else "<="
                passed = observed >= expected if key.endswith("_min") else observed <= expected
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
        for key, other_run_name in target.items():
            if not key.endswith("_gt_run"):
                continue
            metric = key[: -len("_gt_run")]
            observed = record.get(metric, 0.0)
            other_record = runs.get(other_run_name) or {}
            expected = other_record.get(metric, 0.0)
            passed = observed > expected
            row = {
                "target": target_name,
                "run_name": run_name,
                "metric": metric,
                "observed": observed,
                "comparator": ">",
                "expected_run": other_run_name,
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
    regression = check_regression_targets(config, summary, output_root)
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
