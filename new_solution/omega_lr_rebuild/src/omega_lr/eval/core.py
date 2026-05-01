"""Core sequence and edit metrics."""

from __future__ import annotations

from collections import Counter, defaultdict

from omega_lr.alignment import global_align
from omega_lr.constants import EDIT_LABELS, HARD_EDIT_LABELS, ID_TO_EDIT


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def sequence_metrics(prediction: str, truth: str) -> dict[str, float]:
    aligned = global_align(prediction, truth)
    matches = 0
    edits = 0
    for pred_base, truth_base in zip(aligned.aligned_target, aligned.aligned_truth):
        if pred_base == truth_base:
            matches += 1
        else:
            edits += 1
    total = max(len(aligned.aligned_target), 1)
    return {
        "identity": matches / total,
        "edit_distance": edits,
        "normalized_edit_distance": edits / total,
        "predicted_length_ratio": len(prediction) / max(len(truth), 1),
    }


def label_names(edit_labels: list[int]) -> list[str]:
    return [ID_TO_EDIT[label] for label in edit_labels]


def edit_metrics(predicted_labels: list[str], gold_labels: list[str]) -> dict:
    edit_types = {
        "substitution": lambda value: value.startswith("SUB_"),
        "deletion": lambda value: value == "DEL",
        "insertion": lambda value: value.startswith("INS_"),
    }
    metrics = {}
    hard_fp = 0
    predicted_hard = 0
    for name, matcher in edit_types.items():
        tp = sum(matcher(pred) and matcher(gold) and pred == gold for pred, gold in zip(predicted_labels, gold_labels))
        fp = sum(matcher(pred) and not matcher(gold) for pred, gold in zip(predicted_labels, gold_labels))
        fn = sum((not matcher(pred)) and matcher(gold) for pred, gold in zip(predicted_labels, gold_labels))
        metrics[name] = {
            "precision": safe_divide(tp, tp + fp),
            "recall": safe_divide(tp, tp + fn),
            "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        hard_fp += fp
        predicted_hard += tp + fp
    total_positions = max(len(gold_labels), 1)
    metrics["overcorrection_rate"] = hard_fp / total_positions
    metrics["hard_edit_false_positive_rate"] = safe_divide(hard_fp, predicted_hard)
    return metrics


def confusion_matrix(predicted_labels: list[str], gold_labels: list[str]) -> dict:
    matrix = {gold: Counter() for gold in EDIT_LABELS}
    for pred, gold in zip(predicted_labels, gold_labels):
        matrix[gold][pred] += 1
    return {gold: dict(counts) for gold, counts in matrix.items()}


def aggregate_predictions(rows: list[dict]) -> dict:
    sequence_totals = defaultdict(float)
    edit_totals = defaultdict(float)
    type_totals = defaultdict(lambda: defaultdict(float))
    payload_recall = {
        label: {"count": 0, "correct": 0}
        for label in ["SUB_A", "SUB_C", "SUB_G", "SUB_T", "INS_A", "INS_C", "INS_G", "INS_T"]
    }
    confusion = {label: Counter() for label in EDIT_LABELS}
    gate_values = []
    agreement_high = []
    agreement_low = []
    for row in rows:
        seq = sequence_metrics(row["prediction"], row["truth_seq"])
        gold_labels = label_names(row["gold_edit_labels"])
        edits = edit_metrics(row["predicted_labels"], gold_labels)
        for pred, gold in zip(row["predicted_labels"], gold_labels):
            if gold in payload_recall:
                payload_recall[gold]["count"] += 1
                payload_recall[gold]["correct"] += int(pred == gold)
        for key, value in seq.items():
            sequence_totals[key] += value
        for key in ["overcorrection_rate", "hard_edit_false_positive_rate"]:
            edit_totals[key] += edits[key]
        for edit_name in ["substitution", "deletion", "insertion"]:
            for metric_name, value in edits[edit_name].items():
                type_totals[edit_name][metric_name] += value
        row_confusion = confusion_matrix(row["predicted_labels"], gold_labels)
        for gold, pred_counts in row_confusion.items():
            confusion[gold].update(pred_counts)
        gate_values.extend(row.get("trust", []))
        agreement = row["features"]["support_agreement"]
        for idx, trust in enumerate(row.get("trust", [])):
            if agreement[idx] >= 0.7:
                agreement_high.append(trust)
            else:
                agreement_low.append(trust)
    count = max(len(rows), 1)
    sequence_keys = ["identity", "edit_distance", "normalized_edit_distance", "predicted_length_ratio"]
    safety_keys = ["overcorrection_rate", "hard_edit_false_positive_rate"]
    edit_metric_keys = ["precision", "recall", "f1", "tp", "fp", "fn"]
    summary = {
        "num_examples": len(rows),
        "sequence": {key: sequence_totals.get(key, 0.0) / count for key in sequence_keys},
        "safety": {key: edit_totals.get(key, 0.0) / count for key in safety_keys},
        "edit_types": {
            edit_name: {
                metric: type_totals[edit_name].get(metric, 0.0) / count
                for metric in edit_metric_keys
            }
            for edit_name in ["substitution", "deletion", "insertion"]
        },
        "confusion_matrix": {gold: dict(counts) for gold, counts in confusion.items()},
        "per_base_edit_recall": {
            label: {
                "count": counts["count"],
                "correct": counts["correct"],
                "recall": safe_divide(counts["correct"], counts["count"]),
            }
            for label, counts in payload_recall.items()
        },
        "gate_statistics": {
            "mean_trust": sum(gate_values) / max(len(gate_values), 1),
            "trust_in_high_agreement_regions": sum(agreement_high) / max(len(agreement_high), 1),
            "trust_in_low_agreement_regions": sum(agreement_low) / max(len(agreement_low), 1),
        },
    }
    return summary
