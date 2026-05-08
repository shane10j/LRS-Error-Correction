#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.io_utils import read_jsonl, write_json, write_jsonl


SUB_COLUMNS = [
    "candidate_id",
    "gold_safe_label",
    "applied",
    "false_if_applied",
    "example_id",
    "contig",
    "window_start",
    "window_end",
    "position",
    "target_base",
    "truth_base",
    "support_base_counts",
    "support_rule_label",
    "rule_label",
    "neural_label",
    "type_prob_sub",
    "payload_prob",
    "support_depth",
    "support_fraction",
    "support_margin",
    "entropy",
    "repeat_flag",
    "tandem_repeat_flag",
    "homopolymer_run_length",
    "neighbor_rule_flag",
    "boundary_flag",
    "variant_mask",
    "truth_vcf_overlap",
    "variant_proximity_flag",
    "local_variant_density",
    "confident_bed_status",
    "low_confidence_or_preserve",
    "variant_rich_flag",
    "local_rule_candidate_density",
    "local_rule_density",
    "local_mismatch_density",
    "nearby_indel_density",
    "window_relative_position",
    "support_forward_fraction",
    "support_forward_count",
    "support_reverse_count",
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
    "conservative_sub_safety_score",
]


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safety_score(row: dict) -> float:
    score = 0.0
    score += min(_float(row, "support_fraction"), 1.0)
    depth = max(_float(row, "support_depth", 1.0), 1.0)
    score += min(_float(row, "support_margin") / depth, 1.0)
    score += 1.0 - min(_float(row, "entropy"), 1.0)
    score += _float(row, "payload_prob")
    score += 0.5 * _float(row, "support_match_fraction")
    score += 0.25 * (1.0 - min(max(_float(row, "support_strand_bias"), 0.0), 1.0))
    for key in [
        "repeat_flag",
        "tandem_repeat_flag",
        "neighbor_rule_flag",
        "boundary_flag",
        "truth_vcf_overlap",
        "variant_proximity_flag",
        "low_confidence_or_preserve",
    ]:
        if row.get(key):
            score -= 0.75
    score -= _float(row, "nearby_indel_density")
    score -= _float(row, "local_mismatch_density")
    score -= _float(row, "repeat_strength")
    return score


def _project(row: dict) -> dict:
    projected = {key: row.get(key) for key in SUB_COLUMNS if key != "conservative_sub_safety_score"}
    projected["conservative_sub_safety_score"] = _safety_score(projected)
    return projected


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _feature_means(rows: list[dict]) -> dict:
    keys = [
        "support_depth",
        "support_fraction",
        "support_margin",
        "entropy",
        "payload_prob",
        "type_prob_sub",
        "local_rule_density",
        "local_mismatch_density",
        "nearby_indel_density",
        "left_support_match_fraction",
        "right_support_match_fraction",
        "support_forward_fraction",
        "support_strand_bias",
        "support_match_fraction",
        "repeat_strength",
    ]
    return {key: _mean(rows, key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export true-vs-false HG002 support-rule SUB candidates.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    rows = [
        _project(row)
        for row in read_jsonl(args.candidates)
        if row.get("candidate_source") == "support_rule" and row.get("candidate_type") == "SUB"
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("gold_safe_label") or 0),
            float(row.get("support_fraction") or 0.0),
            float(row.get("support_margin") or 0.0),
            -float(row.get("entropy") or 0.0),
        ),
        reverse=True,
    )
    true_rows = [row for row in rows if row.get("gold_safe_label")]
    false_rows = [row for row in rows if not row.get("gold_safe_label")]
    true_recovered = [row for row in true_rows if row.get("applied")]
    false_recovered = [row for row in false_rows if row.get("applied")]
    summary = {
        "num_support_rule_sub_candidates": len(rows),
        "true_sub_candidates": len(true_rows),
        "false_sub_candidates": len(false_rows),
        "applied_true_sub_candidates": sum(1 for row in true_rows if row.get("applied")),
        "applied_false_sub_candidates": sum(1 for row in false_rows if row.get("applied")),
        "comparison_target": "true recovered SUBs vs false recovered SUBs",
        "false_context_counts": dict(Counter(
            key
            for row in false_rows
            for key in ["repeat_flag", "neighbor_rule_flag", "truth_vcf_overlap", "low_confidence_or_preserve", "variant_rich_flag"]
            if row.get(key)
        )),
        "true_feature_means": _feature_means(true_rows),
        "false_feature_means": _feature_means(false_rows),
        "true_recovered_feature_means": _feature_means(true_recovered),
        "false_recovered_feature_means": _feature_means(false_recovered),
        "true_recovered_sub_candidates": true_recovered,
        "false_recovered_sub_candidates": false_recovered,
        "top_50_safest_true_sub_candidates": sorted(true_rows, key=lambda row: row.get("conservative_sub_safety_score") or 0.0, reverse=True)[:50],
        "top_50_safest_false_sub_candidates": sorted(false_rows, key=lambda row: row.get("conservative_sub_safety_score") or 0.0, reverse=True)[:50],
        "columns": SUB_COLUMNS,
    }
    write_jsonl(args.output, rows)
    write_json(args.summary_output, summary)
    print(summary)


if __name__ == "__main__":
    main()
