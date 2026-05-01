"""Deterministic support-rule baseline for synthetic generalization sanity checks."""

from __future__ import annotations

from omega_lr.constants import BASES


def _passes(value: float, depth: float, threshold: float) -> bool:
    if threshold <= 1.0:
        return value / max(depth, 1.0) >= threshold
    return value >= threshold


def _majority_base(counts: list[int], fallback: str) -> tuple[str, int]:
    if sum(counts) <= 0:
        return fallback, 0
    idx = max(range(len(BASES)), key=lambda base_idx: counts[base_idx])
    return BASES[idx], counts[idx]


def _insertion_base(features: dict, pos: int, fallback: str) -> str:
    base_counts = features.get("support_ins_base_counts", [])
    if pos >= len(base_counts) or sum(base_counts[pos]) <= 0:
        return fallback
    return BASES[max(range(len(BASES)), key=lambda idx: base_counts[pos][idx])]


def predict(
    example: dict,
    agreement_threshold: float = 0.60,
    insertion_threshold: float = 0.50,
    deletion_threshold: float = 0.50,
) -> dict:
    features = example["features"]
    prediction = []
    labels = []
    for pos, target_base in enumerate(example["target_seq"]):
        counts = features["support_base_counts"][pos]
        depth = max(float(features["support_depth"][pos]), 1.0)
        agreement = float(features["support_agreement"][pos])
        base, base_count = _majority_base(counts, target_base)
        insertion_count = float(features["support_ins_count"][pos])
        deletion_count = float(features["support_del_count"][pos])

        if base != target_base and agreement >= agreement_threshold and _passes(base_count, depth, agreement_threshold):
            labels.append(f"SUB_{base}")
            prediction.append(base)
            continue
        if _passes(insertion_count, depth, insertion_threshold):
            ins_base = _insertion_base(features, pos, target_base)
            labels.append(f"INS_{ins_base}")
            prediction.append(target_base)
            prediction.append(ins_base)
            continue
        if _passes(deletion_count, depth, deletion_threshold):
            labels.append("DEL")
            continue
        labels.append("COPY")
        prediction.append(target_base)
    return {
        "prediction": "".join(prediction),
        "predicted_labels": labels,
        "trust": features["support_agreement"],
    }
