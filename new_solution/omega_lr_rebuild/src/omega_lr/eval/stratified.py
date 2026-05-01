"""Region-stratified metrics."""

from __future__ import annotations

from omega_lr.eval.core import aggregate_predictions


def _subset(rows: list[dict], predicate) -> list[dict]:
    subset = []
    for row in rows:
        indices = [idx for idx, _ in enumerate(row["predicted_labels"]) if idx < len(row["predicted_labels"]) and predicate(row, idx)]
        if not indices:
            continue
        subset.append(
            {
                **row,
                "predicted_labels": [row["predicted_labels"][idx] for idx in indices],
                "gold_edit_labels": [row["gold_edit_labels"][idx] for idx in indices],
                "trust": [row["trust"][idx] for idx in indices],
                "features": {
                    **row["features"],
                    "support_agreement": [row["features"]["support_agreement"][idx] for idx in indices],
                },
                "prediction": row["prediction"],
                "truth_seq": row["truth_seq"],
            }
        )
    return subset


def stratified_metrics(rows: list[dict]) -> dict:
    subsets = {
        "homopolymer": _subset(rows, lambda row, idx: row["masks"]["homopolymer_mask"][idx] == 1),
        "tandem_repeat": _subset(rows, lambda row, idx: row["masks"]["tandem_repeat_mask"][idx] == 1),
        "high_agreement": _subset(rows, lambda row, idx: row["features"]["support_agreement"][idx] >= 0.7),
        "low_agreement": _subset(rows, lambda row, idx: row["features"]["support_agreement"][idx] < 0.7),
        "high_entropy": _subset(rows, lambda row, idx: row["features"]["support_entropy"][idx] >= 1.0),
        "low_entropy": _subset(rows, lambda row, idx: row["features"]["support_entropy"][idx] < 1.0),
    }
    return {
        name: {"subset_size": len(subset), **aggregate_predictions(subset)}
        for name, subset in subsets.items()
    }

