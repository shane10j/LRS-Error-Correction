#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.io_utils import read_jsonl, write_json


FEATURES = [
    "support_fraction",
    "support_margin_fraction",
    "entropy_inverse",
    "payload_prob",
    "top_inserted_base_fraction",
    "inserted_base_margin_fraction",
    "type_prob_sub",
    "type_prob_del",
    "allow_prob",
    "repeat_flag",
    "tandem_repeat_flag",
    "homopolymer_flag",
    "neighbor_rule_flag",
    "boundary_flag",
    "truth_vcf_overlap",
    "variant_proximity_flag",
    "local_variant_density",
    "variant_rich_flag",
    "low_confidence_or_preserve",
    "local_rule_candidate_density",
    "local_neural_candidate_density",
    "local_chosen_edit_density",
    "local_mismatch_density",
    "nearby_indel_density",
    "window_relative_position",
    "support_forward_fraction",
    "support_forward_count_fraction",
    "support_reverse_count_fraction",
    "support_strand_bias",
    "support_same_haplotype_fraction",
    "support_match_fraction",
    "left_support_match_fraction",
    "right_support_match_fraction",
    "repeat_strength",
    "mapping_quality_available",
    "mapping_quality_mean",
    "reference_kmer_uniqueness_available",
    "reference_kmer_uniqueness",
    "sub_local_gain",
    "sub_local_rule_density",
    "indel_local_gain",
    "indel_local_rule_density",
]


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _vector(row: dict) -> list[float]:
    depth = max(_float(row, "support_depth", 1.0), 1.0)
    values = {
        "support_fraction": _float(row, "support_fraction"),
        "support_margin_fraction": min(_float(row, "support_margin") / depth, 1.0),
        "entropy_inverse": 1.0 - min(_float(row, "entropy"), 1.0),
        "payload_prob": _float(row, "payload_prob", 0.5),
        "top_inserted_base_fraction": _float(row, "top_inserted_base_fraction"),
        "inserted_base_margin_fraction": min(_float(row, "inserted_base_margin") / depth, 1.0),
        "type_prob_sub": _float(row, "type_prob_sub"),
        "type_prob_del": _float(row, "type_prob_del"),
        "allow_prob": _float(row, "allow_prob", 0.5),
        "repeat_flag": _float(row, "repeat_flag"),
        "tandem_repeat_flag": _float(row, "tandem_repeat_flag"),
        "homopolymer_flag": _float(row, "homopolymer_flag"),
        "neighbor_rule_flag": _float(row, "neighbor_rule_flag"),
        "boundary_flag": _float(row, "boundary_flag"),
        "truth_vcf_overlap": _float(row, "truth_vcf_overlap"),
        "variant_proximity_flag": _float(row, "variant_proximity_flag"),
        "local_variant_density": _float(row, "local_variant_density"),
        "variant_rich_flag": _float(row, "variant_rich_flag"),
        "low_confidence_or_preserve": _float(row, "low_confidence_or_preserve"),
        "local_rule_candidate_density": min(_float(row, "local_rule_candidate_density") / 4.0, 1.0),
        "local_neural_candidate_density": min(_float(row, "local_neural_candidate_density") / 4.0, 1.0),
        "local_chosen_edit_density": min(_float(row, "local_chosen_edit_density") / 4.0, 1.0),
        "local_mismatch_density": _float(row, "local_mismatch_density"),
        "nearby_indel_density": _float(row, "nearby_indel_density"),
        "window_relative_position": _float(row, "window_relative_position"),
        "support_forward_fraction": _float(row, "support_forward_fraction", 0.5),
        "support_forward_count_fraction": min(_float(row, "support_forward_count") / depth, 1.0),
        "support_reverse_count_fraction": min(_float(row, "support_reverse_count") / depth, 1.0),
        "support_strand_bias": min(max(_float(row, "support_strand_bias"), 0.0), 1.0),
        "support_same_haplotype_fraction": _float(row, "support_same_haplotype_fraction"),
        "support_match_fraction": _float(row, "support_match_fraction"),
        "left_support_match_fraction": _float(row, "left_support_match_fraction"),
        "right_support_match_fraction": _float(row, "right_support_match_fraction"),
        "repeat_strength": _float(row, "repeat_strength"),
        "mapping_quality_available": _float(row, "mapping_quality_available"),
        "mapping_quality_mean": min(_float(row, "mapping_quality_mean") / 60.0, 1.0),
        "reference_kmer_uniqueness_available": _float(row, "reference_kmer_uniqueness_available"),
        "reference_kmer_uniqueness": _float(row, "reference_kmer_uniqueness"),
        "sub_local_gain": max(min(_float(row, "sub_local_gain") + 0.5, 1.0), 0.0),
        "sub_local_rule_density": min(_float(row, "sub_local_rule_density") / 4.0, 1.0),
        "indel_local_gain": max(min(_float(row, "indel_local_gain") + 0.5, 1.0), 0.0),
        "indel_local_rule_density": min(_float(row, "indel_local_rule_density") / 4.0, 1.0),
    }
    return [values[name] for name in FEATURES]


def _sigmoid(value: float) -> float:
    if value >= 30:
        return 1.0
    if value <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _predict(weights: list[float], bias: float, row: dict) -> float:
    xs = _vector(row)
    return _sigmoid(sum(w * x for w, x in zip(weights, xs)) + bias)


def _split(rows: list[dict], validation_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * validation_fraction)) if len(shuffled) > 2 else 0
    return shuffled[n_val:], shuffled[:n_val] if n_val else shuffled


def _train_one(rows: list[dict], args: argparse.Namespace) -> dict:
    if not rows:
        return {"weights": [0.0] * len(FEATURES), "bias": 0.0, "threshold": 1.0, "num_rows": 0}
    train_rows, val_rows = _split(rows, args.validation_fraction, args.seed)
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    positives = sum(int(row["safe_to_apply"]) for row in train_rows)
    negatives = max(len(train_rows) - positives, 1)
    positives = max(positives, 1)
    pos_weight = min(negatives / positives, args.max_pos_weight)
    for _ in range(args.epochs):
        random.Random(args.seed + _).shuffle(train_rows)
        for row in train_rows:
            xs = _vector(row)
            y = float(row["safe_to_apply"])
            pred = _sigmoid(sum(w * x for w, x in zip(weights, xs)) + bias)
            weight = pos_weight if y else 1.0
            grad = (pred - y) * weight
            for idx, value in enumerate(xs):
                weights[idx] -= args.lr * (grad * value + args.l2 * weights[idx])
            bias -= args.lr * grad
    scored = [(row, _predict(weights, bias, row)) for row in val_rows]
    thresholds = sorted({round(score, 6) for _, score in scored}, reverse=True) + [1.01]
    best = {"threshold": 1.01, "true_positives": 0, "false_positives": 0, "allowed": 0}
    max_fp_count = args.max_false_positives
    max_fp_rate = args.max_false_positive_rate
    safe_total = max(sum(1 for row in val_rows if row["safe_to_apply"]), 1)
    unsafe_total = max(sum(1 for row in val_rows if not row["safe_to_apply"]), 1)
    frontier = []
    for threshold in thresholds:
        allowed = [row for row, score in scored if score >= threshold]
        fp = sum(1 for row in allowed if not row["safe_to_apply"])
        tp = sum(1 for row in allowed if row["safe_to_apply"])
        frontier_row = {
            "threshold": threshold,
            "true_positives": tp,
            "false_positives": fp,
            "allowed": len(allowed),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / safe_total,
            "false_positive_rate": fp / unsafe_total,
            "usable_score": (tp - args.frontier_false_positive_penalty * fp) / max(len(val_rows), 1),
            "identity": (tp - fp) / max(len(val_rows), 1),
        }
        frontier.append(frontier_row)
        if fp <= max_fp_count and (fp / unsafe_total) <= max_fp_rate and (tp, -fp) > (best["true_positives"], -best["false_positives"]):
            best = frontier_row
    zero_fp_small = [row for row in frontier if row["false_positives"] == 0 and 1 <= row["true_positives"] <= 5]
    twenty_tp_one_fp = [row for row in frontier if row["true_positives"] >= 20 and row["false_positives"] <= 1]
    operating_points = []
    for target_tp in [0, 1, 5, 10, 20, 50, 100]:
        eligible = [row for row in frontier if row["true_positives"] >= target_tp]
        if not eligible:
            operating_points.append({"target_true_subs": target_tp, "available": False})
            continue
        best_point = min(eligible, key=lambda row: (row["false_positives"], -row["true_positives"], -row["usable_score"]))
        operating_points.append(
            {
                "target_true_subs": target_tp,
                "available": True,
                "threshold": best_point["threshold"],
                "allowed_true_subs": best_point["true_positives"],
                "false_subs": best_point["false_positives"],
                "usable_score": best_point["usable_score"],
                "identity": best_point["identity"],
            }
        )
    return {
        "weights": weights,
        "bias": bias,
        "threshold": best["threshold"],
        "validation": best,
        "frontier": frontier,
        "frontier_metric_note": "usable_score and identity are candidate-level proxy scores for threshold selection, not sequence-level benchmark metrics.",
        "zero_fp_1_to_5_true_edit_thresholds": zero_fp_small,
        "twenty_true_edits_le_one_false_positive_thresholds": twenty_tp_one_fp,
        "pareto_operating_points": operating_points,
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_safe": sum(int(row["safe_to_apply"]) for row in rows),
        "num_unsafe": sum(1 for row in rows if not row["safe_to_apply"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train small edit-type-specific real-data allow gates.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-false-positives", type=int, default=0)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.001)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--candidate-source", default="support_rule", help="Use a candidate source, or 'all'")
    parser.add_argument("--edit-types", default="SUB,INS,DEL", help="Comma-separated edit families to train")
    parser.add_argument("--frontier-false-positive-penalty", type=float, default=5.0)
    args = parser.parse_args()
    rows = read_jsonl(args.candidates)
    if args.candidate_source != "all":
        rows = [row for row in rows if row.get("candidate_source") == args.candidate_source]
    edit_types = [item.strip() for item in args.edit_types.split(",") if item.strip()]
    models = {
        edit_type: _train_one([row for row in rows if row.get("candidate_type") == edit_type], args)
        for edit_type in edit_types
    }
    payload = {
        "description": "Tiny per-edit-family logistic allow gates trained from real candidate labels.",
        "features": FEATURES,
        "objective": "maximize validation recall subject to near-zero false positives",
        "max_false_positives": args.max_false_positives,
        "max_false_positive_rate": args.max_false_positive_rate,
        "candidate_source": args.candidate_source,
        "edit_types": edit_types,
        "models": models,
    }
    write_json(args.output, payload)
    print(payload)


if __name__ == "__main__":
    main()
