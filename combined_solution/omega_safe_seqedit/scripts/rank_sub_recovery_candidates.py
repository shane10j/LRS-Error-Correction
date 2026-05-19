#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.config import load_config
from omega_safe_seqedit.candidate_evidence import build_candidate_evidence
from omega_safe_seqedit.constants import BASES, ID_TO_RULE
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
    "candidate_indel_evidence_penalty",
    "local_window_delta_score",
    "left_flank_delta_score",
    "right_flank_delta_score",
    "competing_indel_penalty",
    "candidate_cluster_fraction",
    "target_cluster_penalty",
    "cluster_margin",
    "cluster_entropy_inverse",
    "supporting_read_fraction",
    "opposing_read_penalty",
    "supporting_strand_balance",
    "supporting_flank_match",
    "opposing_flank_mismatch",
    "supporting_local_mismatch_penalty",
    "reads_prefer_sub_fraction",
    "reads_prefer_copy_penalty",
    "mean_edit_distance_delta",
    "variant_distance_score",
    "known_variant_abstain_penalty",
    "candidate_matches_reference_penalty",
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


def _candidate_base(row: dict) -> str:
    label = row.get("candidate_label") or row.get("support_rule_label") or row.get("rule_label") or ""
    if "_" in label:
        return label.split("_", 1)[1]
    return row.get("candidate_base") or row.get("truth_base") or "A"


def _candidate_seq(target: str, rel_pos: int, base: str, start: int, end: int) -> str:
    chars = list(target[start:end])
    idx = rel_pos - start
    if 0 <= idx < len(chars):
        chars[idx] = base
    return "".join(chars)


def _seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / max(n, 1)


def _hamming_distance(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(1 for idx in range(n) if a[idx] != b[idx]) + abs(len(a) - len(b))


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts:
        if count:
            p = count / total
            out -= p * math.log2(p)
    return out


def _local_window_features(row: dict, record: dict, radius: int) -> dict:
    pos = int(row["position"])
    base = _candidate_base(row)
    target = record.get("target_seq", "")
    features = record.get("features", {})
    start = max(0, pos - radius)
    end = min(len(target), pos + radius + 1)
    copy_score = 0.0
    sub_score = 0.0
    left_copy = left_sub = right_copy = right_sub = 0.0
    nearby_indels = 0
    nearby_subs = 0
    rule_types = features.get("support_rule_type", [])
    for idx in range(start, end):
        depth = max(float(features.get("support_depth", [1])[idx]), 1.0)
        target_base = target[idx]
        counts = features.get("support_base_counts", [[0, 0, 0, 0]])[idx]
        copy = float(counts[BASES.index(target_base)]) / depth
        scoring_base = base if idx == pos else target_base
        sub = float(counts[BASES.index(scoring_base)]) / depth
        copy_score += copy
        sub_score += sub
        if idx < pos:
            left_copy += copy
            left_sub += sub
        elif idx > pos:
            right_copy += copy
            right_sub += sub
        if idx != pos and idx < len(rule_types):
            rule = ID_TO_RULE[rule_types[idx]]
            nearby_indels += int(rule in {"INS", "DEL"})
            nearby_subs += int(rule == "SUB")
    width = max(end - start, 1)
    return {
        "copy_window_score": copy_score,
        "sub_window_score": sub_score,
        "local_window_delta_score": sub_score - copy_score,
        "left_flank_delta_score": left_sub - left_copy,
        "right_flank_delta_score": right_sub - right_copy,
        "competing_indel_count": nearby_indels,
        "nearby_sub_count": nearby_subs,
        "competing_indel_penalty": min(nearby_indels / width, 1.0),
    }


def local_window_rerank_sub(candidate: dict, record: dict, window_radius: int = 20) -> dict:
    """Score COPY vs a candidate SUB over a local window around the candidate site."""
    return _local_window_features(candidate, record, window_radius)


def _strand_value(strands: list, read_idx: int, pos: int):
    if read_idx >= len(strands) or not isinstance(strands[read_idx], list) or pos >= len(strands[read_idx]):
        return None
    return strands[read_idx][pos]


def _read_level_features(row: dict, record: dict, radius: int) -> dict:
    pos = int(row["position"])
    base = _candidate_base(row)
    target = record.get("target_seq", "")
    seqs = record.get("support_aligned_seqs", [])
    strands = record.get("support_strand_tracks", [])
    mapqs = record.get("support_mapping_qualities", [])
    start = max(0, pos - radius)
    end = min(len(target), pos + radius + 1)
    target_local = target[start:end]
    candidate_local = _candidate_seq(target, pos, base, start, end)
    supporting = []
    opposing = []
    candidate_like = target_like = ambiguous = 0
    supporting_flank = []
    opposing_flank = []
    supporting_mismatch = []
    opposing_mismatch = []
    supporting_strands = []
    supporting_mapq = []
    opposing_mapq = []
    copy_distances = []
    sub_distances = []
    reads_prefer_sub = 0
    reads_prefer_copy = 0
    for idx, seq in enumerate(seqs):
        if pos >= len(seq):
            continue
        local = seq[start:min(end, len(seq))]
        if len(local) != len(target_local):
            ambiguous += 1
            continue
        target_sim = _seq_identity(local, target_local)
        candidate_sim = _seq_identity(local, candidate_local)
        copy_distance = _hamming_distance(local, target_local)
        sub_distance = _hamming_distance(local, candidate_local)
        copy_distances.append(copy_distance)
        sub_distances.append(sub_distance)
        reads_prefer_sub += int(sub_distance < copy_distance)
        reads_prefer_copy += int(copy_distance < sub_distance)
        if candidate_sim > target_sim + 0.02:
            candidate_like += 1
        elif target_sim > candidate_sim + 0.02:
            target_like += 1
        else:
            ambiguous += 1
        strand = _strand_value(strands, idx, pos)
        mapq = mapqs[idx] if idx < len(mapqs) and isinstance(mapqs[idx], (int, float)) else None
        left = _seq_identity(local[:max(pos - start, 0)], candidate_local[:max(pos - start, 0)])
        right = _seq_identity(local[pos - start + 1:], candidate_local[pos - start + 1:])
        mismatch = 1.0 - _seq_identity(local, candidate_local)
        if seq[pos] == base:
            supporting.append(idx)
            supporting_flank.append((left + right) / 2.0)
            supporting_mismatch.append(mismatch)
            if strand is not None:
                supporting_strands.append(strand)
            if mapq is not None:
                supporting_mapq.append(float(mapq))
        elif seq[pos] == row.get("target_base"):
            opposing.append(idx)
            opposing_flank.append((left + right) / 2.0)
            opposing_mismatch.append(mismatch)
            if mapq is not None:
                opposing_mapq.append(float(mapq))
    total = max(candidate_like + target_like + ambiguous, 1)
    reads_scored = max(len(copy_distances), 1)
    mean_copy_distance = sum(copy_distances) / reads_scored
    mean_sub_distance = sum(sub_distances) / reads_scored
    strand_counts = Counter("fwd" if float(x) >= 0 else "rev" for x in supporting_strands)
    strand_total = max(sum(strand_counts.values()), 1)
    strand_balance = 1.0 - abs(strand_counts.get("fwd", 0) - strand_counts.get("rev", 0)) / strand_total
    return {
        "candidate_like_support_count": candidate_like,
        "target_like_support_count": target_like,
        "ambiguous_support_count": ambiguous,
        "candidate_cluster_fraction": candidate_like / total,
        "target_cluster_fraction": target_like / total,
        "ambiguous_cluster_fraction": ambiguous / total,
        "cluster_margin": (candidate_like - target_like) / total,
        "cluster_entropy": _entropy([candidate_like, target_like, ambiguous]),
        "cluster_entropy_inverse": 1.0 - min(_entropy([candidate_like, target_like, ambiguous]) / math.log2(3), 1.0),
        "supporting_read_count": len(supporting),
        "opposing_read_count": len(opposing),
        "supporting_read_fraction": len(supporting) / max(len(supporting) + len(opposing), 1),
        "opposing_read_fraction": len(opposing) / max(len(supporting) + len(opposing), 1),
        "supporting_strand_balance": strand_balance,
        "supporting_avg_mapq": sum(supporting_mapq) / len(supporting_mapq) if supporting_mapq else 0.0,
        "opposing_avg_mapq": sum(opposing_mapq) / len(opposing_mapq) if opposing_mapq else 0.0,
        "supporting_flank_match": sum(supporting_flank) / len(supporting_flank) if supporting_flank else 0.0,
        "opposing_flank_match": sum(opposing_flank) / len(opposing_flank) if opposing_flank else 0.0,
        "supporting_local_mismatch_density": sum(supporting_mismatch) / len(supporting_mismatch) if supporting_mismatch else 1.0,
        "opposing_local_mismatch_density": sum(opposing_mismatch) / len(opposing_mismatch) if opposing_mismatch else 1.0,
        "reads_prefer_sub_count": reads_prefer_sub,
        "reads_prefer_copy_count": reads_prefer_copy,
        "reads_prefer_sub_fraction": reads_prefer_sub / reads_scored,
        "reads_prefer_copy_fraction": reads_prefer_copy / reads_scored,
        "mean_copy_edit_distance_to_support": mean_copy_distance,
        "mean_sub_edit_distance_to_support": mean_sub_distance,
        "mean_edit_distance_delta": mean_copy_distance - mean_sub_distance,
    }


def candidate_read_support_features(candidate: dict, record: dict, window_radius: int = 20) -> dict:
    """Summarize whole-read support, not only the single pileup column."""
    return _read_level_features(candidate, record, window_radius)


def _maybe_ref_vcf(config_path: str | None):
    if not config_path:
        return None, None
    config = load_config(config_path)
    ref = vcf = None
    try:
        import pysam
        ref_path = config.get("data", {}).get("reference_fasta")
        vcf_path = config.get("data", {}).get("truth_vcf")
        if ref_path and Path(ref_path).exists():
            ref = pysam.FastaFile(str(ref_path))
        if vcf_path and Path(vcf_path).exists():
            vcf = pysam.VariantFile(str(vcf_path))
    except Exception:
        return None, None
    return ref, vcf


def _variant_context(row: dict, record: dict, ref, vcf, radius: int = 20) -> dict:
    contig = record.get("contig")
    ref_pos0 = int(record.get("window_start", 0)) + int(row["position"])
    candidate = _candidate_base(row)
    target = row.get("target_base")
    out = {
        "distance_to_nearest_variant": 10**9,
        "candidate_matches_reference": 0,
        "target_matches_reference": 0,
        "candidate_matches_alt": 0,
        "target_matches_alt": 0,
        "local_variant_density_real": 0.0,
        "known_variant_like_target": 0,
        "reference_kmer_uniqueness_available": 0,
        "reference_kmer_uniqueness": _float(row, "reference_kmer_uniqueness"),
    }
    if ref is not None and contig:
        try:
            ref_base = ref.fetch(contig, ref_pos0, ref_pos0 + 1).upper()
            out["candidate_matches_reference"] = int(candidate == ref_base)
            out["target_matches_reference"] = int(target == ref_base)
            kstart = max(0, ref_pos0 - 10)
            kend = ref_pos0 + 11
            local = ref.fetch(contig, kstart, kend).upper()
            kmer = ref.fetch(contig, max(0, ref_pos0 - 5), ref_pos0 + 6).upper()
            out["reference_kmer_uniqueness_available"] = 1
            out["reference_kmer_uniqueness"] = 1.0 / max(local.count(kmer), 1) if kmer else 0.0
        except Exception:
            pass
    if vcf is not None and contig:
        variants = []
        try:
            variants = list(vcf.fetch(contig, max(0, ref_pos0 - radius), ref_pos0 + radius + 1))
        except Exception:
            variants = []
        out["local_variant_density_real"] = len(variants) / max(2 * radius + 1, 1)
        for rec in variants:
            dist = abs((int(rec.pos) - 1) - ref_pos0)
            out["distance_to_nearest_variant"] = min(out["distance_to_nearest_variant"], dist)
            if dist == 0:
                alts = set(rec.alts or [])
                out["candidate_matches_alt"] = int(candidate in alts)
                out["target_matches_alt"] = int(target in alts)
        out["known_variant_like_target"] = int(out["target_matches_alt"] and out["candidate_matches_reference"])
    out["variant_distance_score"] = 1.0 if out["distance_to_nearest_variant"] == 10**9 else min(out["distance_to_nearest_variant"] / radius, 1.0)
    return out


def _enrich_rows(rows: list[dict], predictions: str | None, config: str | None, window_radius: int) -> list[dict]:
    if not predictions:
        return rows
    records = {record["example_id"]: record for record in read_jsonl(predictions)}
    ref, vcf = _maybe_ref_vcf(config)
    enriched = []
    for row in rows:
        record = records.get(row.get("example_id"))
        if not record:
            enriched.append(row)
            continue
        updated = dict(row)
        updated.update(build_candidate_evidence(row, record, ref, vcf, window_radius, max_snippets=4))
        updated.update(local_window_rerank_sub(row, record, window_radius))
        updated.update(candidate_read_support_features(row, record, window_radius))
        updated.update(_variant_context(row, record, ref, vcf, window_radius))
        enriched.append(updated)
    return enriched


def _score(row: dict) -> float:
    if row.get("candidate_evidence_safety_score") is not None:
        return float(row["candidate_evidence_safety_score"])
    depth = max(_float(row, "support_depth", 1.0), 1.0)
    score = 0.0
    score += 1.5 * _float(row, "conservative_sub_safety_score")
    score += 1.0 * _float(row, "sub_local_gain")
    score += 0.5 * min(_float(row, "support_margin") / depth, 1.0)
    score += 0.5 * _float(row, "payload_prob")
    score += 0.3 * (1.0 - min(_float(row, "entropy"), 1.0))
    score -= 0.8 * _float(row, "local_mismatch_density")
    score -= 0.5 * _float(row, "nearby_indel_density")
    score -= 1.2 * min((_float(row, "support_ins_count") + _float(row, "support_del_count")) / depth, 1.0)
    score += 1.2 * _float(row, "local_window_delta_score")
    score += 0.4 * min(_float(row, "left_flank_delta_score"), 0.25)
    score += 0.4 * min(_float(row, "right_flank_delta_score"), 0.25)
    score -= 1.0 * _float(row, "competing_indel_penalty")
    score += 0.8 * _float(row, "cluster_margin")
    score += 0.3 * _float(row, "cluster_entropy_inverse")
    score += 0.4 * _float(row, "supporting_read_fraction")
    score -= 0.4 * _float(row, "opposing_read_fraction")
    score += 0.3 * _float(row, "supporting_strand_balance")
    score += 0.4 * _float(row, "supporting_flank_match")
    score -= 0.5 * _float(row, "supporting_local_mismatch_density")
    score += 1.0 * _float(row, "reads_prefer_sub_fraction")
    score -= 1.0 * _float(row, "reads_prefer_copy_fraction")
    score += 0.6 * _float(row, "mean_edit_distance_delta")
    score -= 2.0 * _float(row, "known_variant_like_target")
    score -= 0.5 * _float(row, "repeat_strength")
    score -= 0.5 * min(max(_float(row, "support_strand_bias"), 0.0), 1.0)
    if _float(row, "reference_kmer_uniqueness_available"):
        score += 0.2 * _float(row, "reference_kmer_uniqueness")
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
        "candidate_indel_evidence_penalty": -min((_float(row, "support_ins_count") + _float(row, "support_del_count")) / depth, 1.0),
        "local_window_delta_score": _float(row, "local_window_delta_score"),
        "left_flank_delta_score": _float(row, "left_flank_delta_score"),
        "right_flank_delta_score": _float(row, "right_flank_delta_score"),
        "competing_indel_penalty": -_float(row, "competing_indel_penalty"),
        "candidate_cluster_fraction": _float(row, "candidate_cluster_fraction"),
        "target_cluster_penalty": -_float(row, "target_cluster_fraction"),
        "cluster_margin": _float(row, "cluster_margin"),
        "cluster_entropy_inverse": _float(row, "cluster_entropy_inverse"),
        "supporting_read_fraction": _float(row, "supporting_read_fraction"),
        "opposing_read_penalty": -_float(row, "opposing_read_fraction"),
        "supporting_strand_balance": _float(row, "supporting_strand_balance"),
        "supporting_flank_match": _float(row, "supporting_flank_match"),
        "opposing_flank_mismatch": 1.0 - _float(row, "opposing_flank_match"),
        "supporting_local_mismatch_penalty": -_float(row, "supporting_local_mismatch_density"),
        "reads_prefer_sub_fraction": _float(row, "reads_prefer_sub_fraction"),
        "reads_prefer_copy_penalty": -_float(row, "reads_prefer_copy_fraction"),
        "mean_edit_distance_delta": _float(row, "mean_edit_distance_delta"),
        "variant_distance_score": _float(row, "variant_distance_score", 1.0),
        "known_variant_abstain_penalty": -_float(row, "known_variant_like_target"),
        "candidate_matches_reference_penalty": -_float(row, "candidate_matches_reference"),
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
    hard_negatives = [
        row for row in negatives
        if _float(row, "support_fraction") >= 0.9
        and _float(row, "payload_prob") >= 0.8
        and _float(row, "entropy") <= 0.2
    ]
    negative_pool = hard_negatives or negatives
    weights = {name: 0.0 for name in FEATURES}
    if not positives or not negatives:
        return {"weights": weights, "num_positive": len(positives), "num_negative": len(negatives), "num_pairs": 0}
    rng = random.Random(seed)
    pairs = [(rng.choice(positives), rng.choice(negative_pool)) for _ in range(min(max_pairs, len(positives) * len(negative_pool)))]
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
        "num_hard_negative": len(hard_negatives),
        "num_pairs": len(pairs),
        "sampled_pairwise_accuracy": sampled_accuracy,
    }


def _split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    if not rows:
        return {"train": [], "calib": [], "test": []}
    sorted_rows = sorted(rows, key=lambda row: (str(row.get("contig") or ""), int(row.get("window_start") or 0), int(row.get("position") or 0)))
    n = len(sorted_rows)
    return {
        "train": sorted_rows[: int(0.6 * n)],
        "calib": sorted_rows[int(0.6 * n): int(0.8 * n)],
        "test": sorted_rows[int(0.8 * n):],
    }


def _topk_metrics(ranked: list[dict], top_ks: list[int], false_penalty: float) -> list[dict]:
    reports = []
    for k in top_ks:
        selected = ranked[:k]
        tp = sum(1 for row in selected if row.get("gold_safe_label"))
        fp = sum(1 for row in selected if not row.get("gold_safe_label"))
        reports.append({
            "top_k": k,
            "allowed_true_subs": tp,
            "false_subs": fp,
            "precision_at_k": tp / max(k, 1),
            "usable_score_proxy": tp - false_penalty * fp,
        })
    return reports


def _ranker_sanity(ranked: list[dict]) -> dict:
    out = {}
    for k in [1, 2, 5, 10, 20]:
        selected = ranked[:k]
        out[f"precision_at_{k}"] = sum(1 for row in selected if row.get("gold_safe_label")) / max(k, 1)
    first_false = None
    first_true = None
    true_before_first_false = 0
    for idx, row in enumerate(ranked, start=1):
        is_true = bool(row.get("gold_safe_label"))
        if is_true and first_true is None:
            first_true = idx
        if not is_true and first_false is None:
            first_false = idx
        if is_true and first_false is None:
            true_before_first_false += 1
    out.update({
        "first_false_rank": first_false,
        "first_true_rank": first_true,
        "max_true_before_first_false": true_before_first_false,
        "active_policy_allowed": out["precision_at_1"] > 0.0,
    })
    return out


def _safe_retrieval_frontier(ranked: list[dict]) -> dict:
    corrected = 0
    false = 0
    best_zero_false = 0
    best_one_false = 0
    first_false_rank = None
    for idx, row in enumerate(ranked, start=1):
        if row.get("gold_safe_label"):
            corrected += 1
        else:
            false += 1
            if first_false_rank is None:
                first_false_rank = idx
        if false == 0:
            best_zero_false = corrected
        if false <= 1:
            best_one_false = corrected
    return {
        "max_corrected_edits_with_0_false": best_zero_false,
        "max_corrected_edits_with_le_1_false": best_one_false,
        "first_false_positive_rank": first_false_rank,
    }


def _passes_local_validation(row: dict, min_local_gain: float) -> bool:
    if _float(row, "known_variant_like_target"):
        return False
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
        "candidate_indel_evidence_penalty": -min((_float(row, "support_ins_count") + _float(row, "support_del_count")) / depth, 1.0),
        "local_window_delta_score": _float(row, "local_window_delta_score"),
        "left_flank_delta_score": _float(row, "left_flank_delta_score"),
        "right_flank_delta_score": _float(row, "right_flank_delta_score"),
        "competing_indel_penalty": -_float(row, "competing_indel_penalty"),
        "cluster_margin": _float(row, "cluster_margin"),
        "candidate_cluster_fraction": _float(row, "candidate_cluster_fraction"),
        "target_cluster_fraction": _float(row, "target_cluster_fraction"),
        "supporting_read_fraction": _float(row, "supporting_read_fraction"),
        "opposing_read_fraction": _float(row, "opposing_read_fraction"),
        "supporting_strand_balance": _float(row, "supporting_strand_balance"),
        "supporting_flank_match": _float(row, "supporting_flank_match"),
        "supporting_local_mismatch_density": _float(row, "supporting_local_mismatch_density"),
        "reads_prefer_sub_fraction": _float(row, "reads_prefer_sub_fraction"),
        "reads_prefer_copy_fraction": _float(row, "reads_prefer_copy_fraction"),
        "mean_edit_distance_delta": _float(row, "mean_edit_distance_delta"),
        "known_variant_like_target": _float(row, "known_variant_like_target"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preview(row: dict) -> dict:
    return {
        "rank": row.get("ranked_recovery_rank"),
        "candidate_id": row["candidate_id"],
        "gold_safe_label": row.get("gold_safe_label"),
        "safety_score": row["safety_score"],
        "risk_score": row["risk_score"],
        "sub_local_gain": row.get("sub_local_gain"),
        "local_window_delta_score": row.get("local_window_delta_score"),
        "support_fraction": row.get("support_fraction"),
        "support_margin": row.get("support_margin"),
        "payload_prob": row.get("payload_prob"),
        "old_score": row.get("conservative_sub_safety_score"),
        "new_local_window_score": row.get("candidate_evidence_safety_score"),
        "new_pairwise_score": row.get("pairwise_recovery_score"),
        "local_window_delta_score": row.get("local_window_delta_score"),
        "reads_prefer_sub_fraction": row.get("reads_prefer_sub_fraction"),
        "reads_prefer_copy_fraction": row.get("reads_prefer_copy_fraction"),
        "mean_edit_distance_delta": row.get("mean_edit_distance_delta"),
        "support_ins_count": row.get("support_ins_count"),
        "support_del_count": row.get("support_del_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank support-rule SUB candidates and emit top-k recovery allowlists.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--predictions", default=None, help="Optional decoded predictions JSONL for local-window/read-level features.")
    parser.add_argument("--config", default=None, help="Optional config for reference/VCF context features.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--top-k", default="1,2,5,10,20")
    parser.add_argument("--min-local-gain", type=float, default=0.25)
    parser.add_argument("--window-radius", type=int, default=20)
    parser.add_argument("--false-penalty", type=float, default=5.0)
    parser.add_argument("--ranking-mode", choices=["heuristic", "pairwise"], default="heuristic")
    parser.add_argument("--compare-pairwise", action="store_true", help="Train/report pairwise ranker diagnostics without using it as the active recovery policy.")
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
    ]
    rows = _enrich_rows(rows, args.predictions, args.config, args.window_radius)
    rows = [row for row in rows if _passes_local_validation(row, args.min_local_gain)]
    split_rows = _split_rows(rows)
    top_ks = [int(item.strip()) for item in args.top_k.split(",") if item.strip()]
    pairwise = _train_pairwise_ranker(
        split_rows["train"] or rows,
        args.pairwise_epochs,
        args.pairwise_lr,
        args.pairwise_l2,
        args.seed,
        args.pairwise_max_pairs,
    ) if args.ranking_mode == "pairwise" or args.compare_pairwise else None
    heuristic_score_fn = _score
    pairwise_score_fn = (lambda row: _dot(pairwise["weights"], row)) if pairwise else None
    score_fn = pairwise_score_fn if args.ranking_mode == "pairwise" and pairwise_score_fn else heuristic_score_fn
    ranked = sorted(rows, key=lambda row: (score_fn(row), _float(row, "support_fraction"), _float(row, "support_margin")), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["ranked_recovery_rank"] = rank
        row["safety_score"] = score_fn(row)
        row["risk_score"] = -row["safety_score"]
        row["ranked_recovery_score"] = row["safety_score"]
        row["heuristic_recovery_score"] = heuristic_score_fn(row)
        row["pairwise_recovery_score"] = pairwise_score_fn(row) if pairwise_score_fn else None
        row["rank_score_components"] = _component_scores(row, row["safety_score"], row["pairwise_recovery_score"])
    write_jsonl(out_dir / "ranked_sub_candidates.jsonl", ranked)
    pairwise_ranked = sorted(
        rows,
        key=lambda row: (pairwise_score_fn(row), _float(row, "support_fraction"), _float(row, "support_margin")),
        reverse=True,
    ) if pairwise_score_fn else []
    pairwise_rank_by_id = {row["candidate_id"]: rank for rank, row in enumerate(pairwise_ranked, start=1)}
    old_vs_pairwise = []
    for rank, row in enumerate(sorted(rows, key=lambda row: float(row.get("conservative_sub_safety_score") or 0.0), reverse=True)[:50], start=1):
        old_vs_pairwise.append({
            "rank": rank,
            "old_rank": rank,
            "new_pairwise_rank": pairwise_rank_by_id.get(row["candidate_id"]),
            "old_score": row.get("conservative_sub_safety_score"),
            "new_local_window_score": heuristic_score_fn(row),
            "new_pairwise_score": pairwise_score_fn(row) if pairwise_score_fn else None,
            "gold_safe_label": row.get("gold_safe_label"),
            "candidate_id": row["candidate_id"],
            "support_fraction": row.get("support_fraction"),
            "payload_prob": row.get("payload_prob"),
            "local_window_delta_score": row.get("local_window_delta_score"),
            "reads_prefer_sub_fraction": row.get("reads_prefer_sub_fraction"),
            "reads_prefer_copy_fraction": row.get("reads_prefer_copy_fraction"),
            "mean_edit_distance_delta": row.get("mean_edit_distance_delta"),
        })
    write_jsonl(out_dir / "old_vs_pairwise_top50.jsonl", old_vs_pairwise)

    reports = []
    created_at = datetime.now(timezone.utc).isoformat()
    for k in top_ks:
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
                "created_at": created_at,
                "score_name": "safety_score",
                "sort_direction": "descending",
                "active_ranking_mode": args.ranking_mode,
                "num_allowlisted": len(selected),
                "num_true_in_allowlist": tp,
                "num_false_in_allowlist": fp,
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "gold_safe_label": int(row.get("gold_safe_label") or 0),
                        "rank": row["ranked_recovery_rank"],
                        "safety_score": row["safety_score"],
                        "risk_score": row["risk_score"],
                    }
                    for row in selected
                ],
                "candidate_ids": [row["candidate_id"] for row in selected],
            },
        )
        allowlist_sha256 = _sha256(allowlist_path)
        reports.append(
            {
                "top_k": k,
                "allowlist": str(allowlist_path),
                "allowlist_sha256": allowlist_sha256,
                "allowlist_created_at": created_at,
                "num_allowlisted": len(selected),
                "allowed_true_subs": tp,
                "false_subs": fp,
                "usable_score_proxy": tp - args.false_penalty * fp,
                "selected_preview": [_preview(row) for row in selected[:10]],
            }
        )
    top_20 = ranked[:20]
    bottom_20 = list(reversed(ranked[-20:]))
    summary = {
        "num_ranked_candidates": len(ranked),
        "ranked_candidates": str(out_dir / "ranked_sub_candidates.jsonl"),
        "ranking_mode": args.ranking_mode,
        "pairwise_active_policy_disabled": args.ranking_mode != "pairwise",
        "pairwise_diagnostic_only": bool(pairwise) and args.ranking_mode != "pairwise",
        "old_vs_pairwise_top50": str(out_dir / "old_vs_pairwise_top50.jsonl"),
        "old_vs_pairwise_top50_preview": old_vs_pairwise[:20],
        "score_contract": {
            "safety_score": "higher is safer; candidates are sorted descending by this value",
            "risk_score": "higher is more dangerous; currently risk_score = -safety_score",
        },
        "ranking_direction_check": {
            "top_20_true": sum(1 for row in top_20 if row.get("gold_safe_label")),
            "top_20_false": sum(1 for row in top_20 if not row.get("gold_safe_label")),
            "bottom_20_true": sum(1 for row in bottom_20 if row.get("gold_safe_label")),
            "bottom_20_false": sum(1 for row in bottom_20 if not row.get("gold_safe_label")),
            "warning": "If bottom candidates are mostly true and top candidates are mostly false, the score direction is wrong.",
        },
        "top_20_by_safety_score": [_preview(row) for row in top_20],
        "bottom_20_by_safety_score": [_preview(row) for row in bottom_20],
        "pairwise_ranker": pairwise,
        "active_ranker_sanity": _ranker_sanity(ranked),
        "pairwise_ranker_sanity": _ranker_sanity(pairwise_ranked) if pairwise_ranked else None,
        "pairwise_recovery_warning": (
            "Pairwise ranker is diagnostic only and must not be evaluated as a correction policy when precision_at_1 is 0."
            if pairwise_ranked and _ranker_sanity(pairwise_ranked)["precision_at_1"] == 0.0
            else None
        ),
        "ranker_split_note": "Interval-style split by contig/window_start: train first 60%, calibration next 20%, test final 20%.",
        "ranker_train_topk": _topk_metrics(sorted(split_rows["train"], key=lambda row: score_fn(row), reverse=True), top_ks, args.false_penalty),
        "ranker_calib_topk": _topk_metrics(sorted(split_rows["calib"], key=lambda row: score_fn(row), reverse=True), top_ks, args.false_penalty),
        "ranker_test_topk": _topk_metrics(sorted(split_rows["test"], key=lambda row: score_fn(row), reverse=True), top_ks, args.false_penalty),
        "safe_retrieval_frontier_all_candidates": _safe_retrieval_frontier(ranked),
        "safe_retrieval_frontier_train": _safe_retrieval_frontier(
            sorted(split_rows["train"], key=lambda row: score_fn(row), reverse=True)
        ),
        "safe_retrieval_frontier_calib": _safe_retrieval_frontier(
            sorted(split_rows["calib"], key=lambda row: score_fn(row), reverse=True)
        ),
        "safe_retrieval_frontier_test": _safe_retrieval_frontier(
            sorted(split_rows["test"], key=lambda row: score_fn(row), reverse=True)
        ),
        "safe_topk_metric_note": "Development target is max corrected SUBs at 0 false: 1 -> 5 -> 10 -> 20, not identity.",
        "active_policy_note": "ranked_sub_top_1 is the first nonzero safe real-data correction candidate if it keeps zero false edits; it is not SOTA-competitive evidence.",
        "local_validation": {"min_local_gain": args.min_local_gain},
        "metric_note": "usable_score_proxy is candidate-level: true SUBs - false_penalty * false SUBs.",
        "top_k_reports": reports,
    }
    write_json(args.summary_output, summary)
    print(summary)


if __name__ == "__main__":
    main()
