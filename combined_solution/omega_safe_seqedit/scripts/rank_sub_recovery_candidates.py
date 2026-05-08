#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl


FEATURES = [
    "conservative_sub_safety_score",
    "sub_local_gain",
    "support_fraction",
    "support_margin_fraction",
    "entropy_inverse",
    "payload_prob",
    "type_prob_sub",
    "left_support_match_fraction",
    "right_support_match_fraction",
    "support_match_fraction",
    "strand_balance",
    "repeat_penalty",
    "neighbor_penalty",
    "local_mismatch_penalty",
    "nearby_indel_penalty",
    "low_confidence_penalty",
]


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score(row: dict) -> float:
    depth = max(_float(row, "support_depth", 1.0), 1.0)
    score = 0.0
    score += 1.5 * _float(row, "conservative_sub_safety_score")
    score += 1.0 * _float(row, "sub_local_gain")
    score += 0.5 * min(_float(row, "support_margin") / depth, 1.0)
    score += 0.5 * _float(row, "payload_prob")
    score += 0.3 * (1.0 - min(_float(row, "entropy"), 1.0))
    score -= 0.8 * _float(row, "local_mismatch_density")
    score -= 0.5 * _float(row, "nearby_indel_density")
    score -= 0.5 * _float(row, "repeat_strength")
    score -= 0.5 * min(max(_float(row, "support_strand_bias"), 0.0), 1.0)
    return score


def _feature_values(row: dict) -> dict[str, float]:
    depth = max(_float(row, "support_depth", 1.0), 1.0)
    return {
        "conservative_sub_safety_score": _float(row, "conservative_sub_safety_score"),
        "sub_local_gain": _float(row, "sub_local_gain"),
        "support_fraction": _float(row, "support_fraction"),
        "support_margin_fraction": min(_float(row, "support_margin") / depth, 1.0),
        "entropy_inverse": 1.0 - min(_float(row, "entropy"), 1.0),
        "payload_prob": _float(row, "payload_prob"),
        "type_prob_sub": _float(row, "type_prob_sub"),
        "left_support_match_fraction": _float(row, "left_support_match_fraction"),
        "right_support_match_fraction": _float(row, "right_support_match_fraction"),
        "support_match_fraction": _float(row, "support_match_fraction"),
        "strand_balance": 1.0 - min(max(_float(row, "support_strand_bias"), 0.0), 1.0),
        "repeat_penalty": -_float(row, "repeat_strength"),
        "neighbor_penalty": -_float(row, "neighbor_rule_flag"),
        "local_mismatch_penalty": -_float(row, "local_mismatch_density"),
        "nearby_indel_penalty": -_float(row, "nearby_indel_density"),
        "low_confidence_penalty": -_float(row, "low_confidence_or_preserve"),
    }


def _dot(weights: dict[str, float], row: dict) -> float:
    values = _feature_values(row)
    return sum(weights.get(name, 0.0) * values[name] for name in FEATURES)


def _train_pairwise_ranker(rows: list[dict], epochs: int, lr: float, l2: float, seed: int, max_pairs: int) -> dict:
    positives = [row for row in rows if row.get("gold_safe_label")]
    negatives = [row for row in rows if not row.get("gold_safe_label")]
    weights = {name: 0.0 for name in FEATURES}
    if not positives or not negatives:
        return {"weights": weights, "num_positive": len(positives), "num_negative": len(negatives), "num_pairs": 0}
    rng = random.Random(seed)
    pairs = [(rng.choice(positives), rng.choice(negatives)) for _ in range(min(max_pairs, len(positives) * len(negatives)))]
    for epoch in range(epochs):
        rng.shuffle(pairs)
        for positive, negative in pairs:
            pos_values = _feature_values(positive)
            neg_values = _feature_values(negative)
            margin = sum(weights[name] * (pos_values[name] - neg_values[name]) for name in FEATURES)
            # Logistic pairwise loss: -log(sigmoid(score_true - score_false)).
            grad_scale = -1.0 / (1.0 + math.exp(min(margin, 30.0)))
            for name in FEATURES:
                diff = pos_values[name] - neg_values[name]
                weights[name] -= lr * (grad_scale * diff + l2 * weights[name])
    sampled_accuracy = sum(1 for positive, negative in pairs if _dot(weights, positive) > _dot(weights, negative)) / max(len(pairs), 1)
    return {
        "weights": weights,
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "num_pairs": len(pairs),
        "sampled_pairwise_accuracy": sampled_accuracy,
    }


def _passes_local_validation(row: dict, min_local_gain: float) -> bool:
    return _float(row, "sub_local_gain", -999.0) >= min_local_gain


def _component_scores(row: dict, score: float, pairwise_score: float | None = None) -> dict:
    depth = max(_float(row, "support_depth", 1.0), 1.0)
    return {
        "ranked_score": score,
        "pairwise_score": pairwise_score,
        "conservative_sub_safety_score": _float(row, "conservative_sub_safety_score"),
        "sub_local_gain": _float(row, "sub_local_gain"),
        "support_fraction": _float(row, "support_fraction"),
        "support_margin_fraction": min(_float(row, "support_margin") / depth, 1.0),
        "entropy_inverse": 1.0 - min(_float(row, "entropy"), 1.0),
        "payload_prob": _float(row, "payload_prob"),
        "type_prob_sub": _float(row, "type_prob_sub"),
        "strand_balance": 1.0 - min(max(_float(row, "support_strand_bias"), 0.0), 1.0),
        "local_mismatch_penalty": -_float(row, "local_mismatch_density"),
        "repeat_penalty": -_float(row, "repeat_strength"),
        "neighbor_penalty": -_float(row, "neighbor_rule_flag"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank support-rule SUB candidates and emit top-k recovery allowlists.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--top-k", default="1,2,5,10,20")
    parser.add_argument("--min-local-gain", type=float, default=0.25)
    parser.add_argument("--false-penalty", type=float, default=5.0)
    parser.add_argument("--ranking-mode", choices=["heuristic", "pairwise"], default="pairwise")
    parser.add_argument("--pairwise-epochs", type=int, default=80)
    parser.add_argument("--pairwise-lr", type=float, default=0.05)
    parser.add_argument("--pairwise-l2", type=float, default=0.001)
    parser.add_argument("--pairwise-max-pairs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    rows = [
        row
        for row in read_jsonl(args.candidates)
        if row.get("candidate_source") == "support_rule"
        and row.get("candidate_type") == "SUB"
        and _passes_local_validation(row, args.min_local_gain)
    ]
    pairwise = _train_pairwise_ranker(
        rows,
        args.pairwise_epochs,
        args.pairwise_lr,
        args.pairwise_l2,
        args.seed,
        args.pairwise_max_pairs,
    ) if args.ranking_mode == "pairwise" else None
    if pairwise:
        score_fn = lambda row: _dot(pairwise["weights"], row)
    else:
        score_fn = _score
    ranked = sorted(rows, key=lambda row: (score_fn(row), _float(row, "support_fraction"), _float(row, "support_margin")), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["ranked_recovery_rank"] = rank
        row["ranked_recovery_score"] = score_fn(row)
        row["heuristic_recovery_score"] = _score(row)
        row["rank_score_components"] = _component_scores(row, score_fn(row), _dot(pairwise["weights"], row) if pairwise else None)
    write_jsonl(out_dir / "ranked_sub_candidates.jsonl", ranked)

    reports = []
    for k in [int(item.strip()) for item in args.top_k.split(",") if item.strip()]:
        selected = ranked[:k]
        tp = sum(1 for row in selected if row.get("gold_safe_label"))
        fp = sum(1 for row in selected if not row.get("gold_safe_label"))
        allowlist_path = out_dir / f"top_{k}_sub_allowlist.json"
        write_json(
            allowlist_path,
            {
                "description": "Exact support-rule SUB candidates allowed for ranked recovery.",
                "top_k": k,
                "min_local_gain": args.min_local_gain,
                "candidate_ids": [row["candidate_id"] for row in selected],
            },
        )
        reports.append(
            {
                "top_k": k,
                "allowlist": str(allowlist_path),
                "allowed_true_subs": tp,
                "false_subs": fp,
                "usable_score_proxy": tp - args.false_penalty * fp,
                "selected_preview": [
                    {
                        "rank": row["ranked_recovery_rank"],
                        "candidate_id": row["candidate_id"],
                        "gold_safe_label": row.get("gold_safe_label"),
                        "score": row["ranked_recovery_score"],
                        "sub_local_gain": row.get("sub_local_gain"),
                        "support_fraction": row.get("support_fraction"),
                        "support_margin": row.get("support_margin"),
                        "payload_prob": row.get("payload_prob"),
                    }
                    for row in selected[:10]
                ],
            }
        )
    summary = {
        "num_ranked_candidates": len(ranked),
        "ranked_candidates": str(out_dir / "ranked_sub_candidates.jsonl"),
        "ranking_mode": args.ranking_mode,
        "pairwise_ranker": pairwise,
        "active_policy_note": "ranked_sub_top_1 is the first nonzero safe real-data correction candidate if it keeps zero false edits; it is not SOTA-competitive evidence.",
        "local_validation": {"min_local_gain": args.min_local_gain},
        "metric_note": "usable_score_proxy is candidate-level: true SUBs - false_penalty * false SUBs.",
        "top_k_reports": reports,
    }
    write_json(args.summary_output, summary)
    print(summary)


if __name__ == "__main__":
    main()
