#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.io_utils import read_jsonl, write_json


def _score(row: dict) -> float:
    support_fraction = float(row.get("support_fraction") or 0.0)
    support_margin = float(row.get("support_margin") or 0.0)
    depth = max(float(row.get("support_depth") or 1.0), 1.0)
    entropy = float(row.get("entropy") or 0.0)
    payload = row.get("payload_prob")
    payload = float(payload) if payload is not None else 0.5
    type_prob = {
        "SUB": float(row.get("type_prob_sub") or 0.0),
        "DEL": float(row.get("type_prob_del") or 0.0),
        "INS": payload,
    }.get(row.get("candidate_type"), 0.0)
    ambiguity_penalty = 0.0
    for key in ["neighbor_rule_flag", "repeat_flag", "truth_vcf_overlap", "low_confidence_or_preserve", "variant_rich_flag"]:
        ambiguity_penalty += 0.12 if row.get(key) else 0.0
    return (
        0.35 * support_fraction
        + 0.20 * min(support_margin / depth, 1.0)
        + 0.20 * payload
        + 0.15 * type_prob
        + 0.10 * (1.0 - min(entropy, 1.0))
        - ambiguity_penalty
    )


def _best_threshold(rows: list[dict], max_false_positives: int) -> dict:
    if not rows:
        return {"threshold": 1.0, "true_positives": 0, "false_positives": 0, "candidates": 0}
    scored = [(row, _score(row)) for row in rows]
    thresholds = sorted({round(score, 6) for _, score in scored}, reverse=True) + [1.01]
    best = {"threshold": 1.01, "true_positives": 0, "false_positives": 0, "candidates": len(rows)}
    for threshold in thresholds:
        allowed = [row for row, score in scored if score >= threshold]
        fp = sum(1 for row in allowed if not row["safe_to_apply"])
        tp = sum(1 for row in allowed if row["safe_to_apply"])
        if fp <= max_false_positives and (tp, -fp) > (best["true_positives"], -best["false_positives"]):
            best = {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "candidates": len(rows),
                "allowed": len(allowed),
            }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune simple edit-family allow thresholds from real candidate-edit rows.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-false-positives", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.candidates)
    thresholds = {
        edit_type: _best_threshold(
            [row for row in rows if row.get("candidate_type") == edit_type],
            args.max_false_positives,
        )
        for edit_type in ["SUB", "INS", "DEL"]
    }
    payload = {
        "description": "Simple real-data allow/edit gate calibration. Higher threshold is safer.",
        "max_false_positives": args.max_false_positives,
        "thresholds": thresholds,
    }
    write_json(args.output, payload)
    print(payload)


if __name__ == "__main__":
    main()
