"""Prediction summaries."""

from __future__ import annotations

from omega_lr.eval.core import aggregate_predictions
from omega_lr.eval.score import usable_score
from omega_lr.eval.stratified import stratified_metrics


def build_summary(rows: list[dict]) -> dict:
    summary = aggregate_predictions(rows)
    summary["stratified"] = stratified_metrics(rows)
    summary["usable_score"] = usable_score(summary)
    summary["threshold_tables"] = {}
    return summary


def benchmark_record(run_name: str, summary: dict) -> dict:
    """Small flat record for notebook/export comparisons."""
    sequence = summary.get("sequence", {})
    safety = summary.get("safety", {})
    return {
        "run_name": run_name,
        "identity": sequence.get("identity", 0.0),
        "edit_distance": sequence.get("edit_distance", 0.0),
        "normalized_edit_distance": sequence.get("normalized_edit_distance", 0.0),
        "overcorrection_rate": safety.get("overcorrection_rate", 0.0),
        "hard_edit_false_positive_rate": safety.get("hard_edit_false_positive_rate", 0.0),
        "usable_score": summary.get("usable_score", 0.0),
        "test_summary": summary,
    }
