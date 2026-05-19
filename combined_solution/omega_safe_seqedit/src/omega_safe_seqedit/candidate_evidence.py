"""Local-window and read-level evidence for candidate edit decisions."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from statistics import mean

from omega_safe_seqedit.config import load_config
from omega_safe_seqedit.constants import BASES, ID_TO_RULE


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_base(row: dict) -> str:
    label = row.get("candidate_label") or row.get("support_rule_label") or row.get("rule_label") or ""
    if "_" in label:
        return label.split("_", 1)[1]
    return row.get("candidate_base") or row.get("truth_base") or "A"


def safe_window(seq: str, pos: int, radius: int) -> tuple[int, int, str]:
    start = max(0, pos - radius)
    end = min(len(seq), pos + radius + 1)
    return start, end, seq[start:end]


def candidate_window(target: str, pos: int, base: str, start: int, end: int) -> str:
    chars = list(target[start:end])
    idx = pos - start
    if 0 <= idx < len(chars):
        chars[idx] = base
    return "".join(chars)


def seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(1 for idx in range(n) if a[idx] == b[idx]) / max(n, 1)


def hamming_distance(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(1 for idx in range(n) if a[idx] != b[idx]) + abs(len(a) - len(b))


def entropy(counts: list[int | float]) -> float:
    total = sum(float(x) for x in counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts:
        if count:
            p = float(count) / total
            out -= p * math.log2(p)
    return out


def maybe_reference_and_vcf(config_path: str | None):
    if not config_path:
        return None, None
    config = load_config(config_path)
    try:
        import pysam
    except ImportError:
        return None, None
    ref = vcf = None
    ref_path = config.get("data", {}).get("reference_fasta") or config.get("data", {}).get("reference")
    vcf_path = config.get("data", {}).get("truth_vcf")
    try:
        if ref_path and Path(ref_path).exists():
            ref = pysam.FastaFile(str(ref_path))
        if vcf_path and Path(vcf_path).exists():
            vcf = pysam.VariantFile(str(vcf_path))
    except Exception:
        return None, None
    return ref, vcf


def variant_context(row: dict, record: dict, ref=None, vcf=None, radius: int = 20) -> dict:
    contig = record.get("contig")
    ref_pos0 = int(record.get("window_start", 0)) + int(row["position"])
    cand_base = candidate_base(row)
    target_base = row.get("target_base")
    out = {
        "distance_to_nearest_variant": None,
        "local_variant_density_real": 0.0,
        "candidate_matches_reference": 0,
        "target_matches_reference": 0,
        "candidate_matches_alt": 0,
        "target_matches_alt": 0,
        "known_variant_like_target": 0,
        "reference_kmer_uniqueness_available": 0,
        "reference_kmer_uniqueness": as_float(row, "reference_kmer_uniqueness"),
    }
    if ref is not None and contig:
        try:
            ref_base = ref.fetch(contig, ref_pos0, ref_pos0 + 1).upper()
            out["candidate_matches_reference"] = int(cand_base == ref_base)
            out["target_matches_reference"] = int(target_base == ref_base)
            local = ref.fetch(contig, max(0, ref_pos0 - 10), ref_pos0 + 11).upper()
            kmer = ref.fetch(contig, max(0, ref_pos0 - 5), ref_pos0 + 6).upper()
            out["reference_kmer_uniqueness_available"] = 1
            out["reference_kmer_uniqueness"] = 1.0 / max(local.count(kmer), 1) if kmer else 0.0
        except Exception:
            pass
    if vcf is not None and contig:
        nearest = 10**9
        try:
            variants = list(vcf.fetch(contig, max(0, ref_pos0 - radius), ref_pos0 + radius + 1))
        except Exception:
            variants = []
        out["local_variant_density_real"] = len(variants) / max(2 * radius + 1, 1)
        for rec in variants:
            dist = abs((int(rec.pos) - 1) - ref_pos0)
            nearest = min(nearest, dist)
            if dist == 0:
                alts = set(rec.alts or [])
                out["candidate_matches_alt"] = int(cand_base in alts)
                out["target_matches_alt"] = int(target_base in alts)
        out["distance_to_nearest_variant"] = None if nearest == 10**9 else nearest
        out["known_variant_like_target"] = int(out["target_matches_alt"] and out["candidate_matches_reference"])
    return out


def pileup_window(record: dict, start: int, end: int) -> list[dict]:
    features = record.get("features", {})
    target = record.get("target_seq", "")
    rows = []
    for pos in range(start, end):
        base_counts = features.get("support_base_counts", [[0, 0, 0, 0]])[pos]
        rows.append(
            {
                "position": pos,
                "target_base": target[pos] if pos < len(target) else None,
                "base_counts": dict(zip(BASES, base_counts)),
                "del_count": features.get("support_del_count", [0])[pos],
                "ins_count": features.get("support_ins_count", [0])[pos],
                "depth": features.get("support_depth", [0])[pos],
                "agreement": features.get("support_agreement", [0.0])[pos],
                "entropy": features.get("support_entropy", [0.0])[pos],
                "rule_type": ID_TO_RULE[features.get("support_rule_type", [0])[pos]]
                if pos < len(features.get("support_rule_type", []))
                else "COPY",
            }
        )
    return rows


def _strand_at(strands: list, read_idx: int, pos: int):
    if read_idx >= len(strands) or not isinstance(strands[read_idx], list) or pos >= len(strands[read_idx]):
        return None
    return strands[read_idx][pos]


def support_read_evidence(row: dict, record: dict, radius: int = 20, max_snippets: int = 12) -> dict:
    pos = int(row["position"])
    cand_base = candidate_base(row)
    target = record.get("target_seq", "")
    start, end, target_win = safe_window(target, pos, radius)
    cand_win = candidate_window(target, pos, cand_base, start, end)
    seqs = record.get("support_aligned_seqs", [])
    ids = record.get("support_read_ids", [])
    strands = record.get("support_strand_tracks", [])
    mapqs = record.get("support_mapping_qualities", [])
    cigars = record.get("support_cigar_snippets", [])
    snippets = []
    copy_distances = []
    sub_distances = []
    candidate_like = target_like = ambiguous = 0
    support_strands = []
    support_mapqs = []
    left_deltas = []
    right_deltas = []
    for idx, seq in enumerate(seqs):
        if pos >= len(seq):
            continue
        local = seq[start:min(end, len(seq))]
        if len(local) != len(target_win):
            ambiguous += 1
            continue
        copy_distance = hamming_distance(local, target_win)
        sub_distance = hamming_distance(local, cand_win)
        copy_distances.append(copy_distance)
        sub_distances.append(sub_distance)
        if sub_distance < copy_distance:
            candidate_like += 1
        elif copy_distance < sub_distance:
            target_like += 1
        else:
            ambiguous += 1
        left_len = max(pos - start, 0)
        right_start = left_len + 1
        left_deltas.append(seq_identity(local[:left_len], cand_win[:left_len]) - seq_identity(local[:left_len], target_win[:left_len]))
        right_deltas.append(seq_identity(local[right_start:], cand_win[right_start:]) - seq_identity(local[right_start:], target_win[right_start:]))
        if seq[pos] == cand_base:
            strand = _strand_at(strands, idx, pos)
            if strand is not None:
                support_strands.append(strand)
            if idx < len(mapqs) and isinstance(mapqs[idx], (int, float)):
                support_mapqs.append(float(mapqs[idx]))
        if len(snippets) < max_snippets:
            snippets.append(
                {
                    "read_id": ids[idx] if idx < len(ids) else f"support_{idx}",
                    "snippet": local,
                    "base_at_candidate": seq[pos],
                    "copy_distance": copy_distance,
                    "sub_distance": sub_distance,
                    "prefers": "SUB" if sub_distance < copy_distance else "COPY" if copy_distance < sub_distance else "AMBIG",
                    "strand": _strand_at(strands, idx, pos),
                    "mapping_quality": mapqs[idx] if idx < len(mapqs) else None,
                    "cigar_snippet": cigars[idx] if idx < len(cigars) else None,
                }
            )
    total = max(candidate_like + target_like + ambiguous, 1)
    strand_counts = Counter("fwd" if float(x) >= 0 else "rev" for x in support_strands)
    strand_total = max(sum(strand_counts.values()), 1)
    return {
        "support_read_snippets": snippets,
        "candidate_like_count": candidate_like,
        "target_like_count": target_like,
        "ambiguous_count": ambiguous,
        "candidate_cluster_fraction": candidate_like / total,
        "target_cluster_fraction": target_like / total,
        "cluster_margin": (candidate_like - target_like) / total,
        "cluster_entropy": entropy([candidate_like, target_like, ambiguous]),
        "reads_prefer_sub_fraction": candidate_like / total,
        "reads_prefer_copy_fraction": target_like / total,
        "mean_copy_edit_distance_to_support": mean(copy_distances) if copy_distances else None,
        "mean_sub_edit_distance_to_support": mean(sub_distances) if sub_distances else None,
        "mean_edit_distance_delta": (mean(copy_distances) - mean(sub_distances)) if copy_distances else 0.0,
        "left_flank_delta": mean(left_deltas) if left_deltas else 0.0,
        "right_flank_delta": mean(right_deltas) if right_deltas else 0.0,
        "strand_balance": 1.0 - abs(strand_counts.get("fwd", 0) - strand_counts.get("rev", 0)) / strand_total,
        "supporting_strand_counts": dict(strand_counts),
        "mapq_summary": {
            "available": bool(support_mapqs),
            "mean": mean(support_mapqs) if support_mapqs else None,
            "min": min(support_mapqs) if support_mapqs else None,
            "max": max(support_mapqs) if support_mapqs else None,
        },
    }


def pileup_hypothesis_score(row: dict, record: dict, radius: int = 20) -> dict:
    pos = int(row["position"])
    cand_base = candidate_base(row)
    target = record.get("target_seq", "")
    features = record.get("features", {})
    start, end, target_win = safe_window(target, pos, radius)
    cand_win = candidate_window(target, pos, cand_base, start, end)
    copy_score = 0.0
    sub_score = 0.0
    nearby_indels = 0
    nearby_mismatch = 0
    rule_types = features.get("support_rule_type", [])
    for idx in range(start, end):
        depth = max(float(features.get("support_depth", [1])[idx]), 1.0)
        counts = features.get("support_base_counts", [[0, 0, 0, 0]])[idx]
        copy_base = target_win[idx - start]
        sub_base = cand_win[idx - start]
        copy_score += float(counts[BASES.index(copy_base)]) / depth
        sub_score += float(counts[BASES.index(sub_base)]) / depth
        if idx != pos and idx < len(rule_types):
            rule = ID_TO_RULE[rule_types[idx]]
            nearby_indels += int(rule in {"INS", "DEL"})
            nearby_mismatch += int(rule == "SUB")
    width = max(end - start, 1)
    return {
        "copy_window_score": copy_score,
        "sub_window_score": sub_score,
        "delta_window_score": sub_score - copy_score,
        "nearby_indel_density": nearby_indels / width,
        "nearby_mismatch_density": nearby_mismatch / width,
    }


def candidate_safety_score(evidence: dict) -> float:
    left_right = (float(evidence.get("left_flank_delta") or 0.0) + float(evidence.get("right_flank_delta") or 0.0)) / 2.0
    score = 0.0
    score += float(evidence.get("delta_window_score") or 0.0)
    score += float(evidence.get("cluster_margin") or 0.0)
    score += left_right
    score += 0.25 * float(evidence.get("strand_balance") or 0.0)
    score += 0.75 * float(evidence.get("mean_edit_distance_delta") or 0.0)
    score -= 0.75 * float(evidence.get("ambiguous_count") or 0.0) / max(float(evidence.get("support_depth") or 1.0), 1.0)
    score -= 1.5 * float(evidence.get("nearby_indel_density") or 0.0)
    score -= 1.0 * int(bool(evidence.get("known_variant_like_target")))
    score -= 0.5 * int(bool(evidence.get("repeat_flag")))
    score -= 0.35 * float(evidence.get("repeat_strength") or 0.0)
    return score


def build_candidate_evidence(row: dict, record: dict, ref=None, vcf=None, radius: int = 20, max_snippets: int = 12) -> dict:
    pos = int(row["position"])
    cand_base = candidate_base(row)
    target = record.get("target_seq", "")
    start, end, target_win = safe_window(target, pos, radius)
    truth_start, truth_end, truth_win = safe_window(record.get("truth_seq", ""), pos, radius)
    evidence = {
        **row,
        "candidate_id": row["candidate_id"],
        "window_start": record.get("window_start"),
        "window_end": record.get("window_end"),
        "evidence_window_start": start,
        "evidence_window_end": end,
        "target_window": target_win,
        "truth_window": truth_win if truth_start == start and truth_end == end else truth_win,
        "candidate_window": candidate_window(target, pos, cand_base, start, end),
        "candidate_allele": cand_base,
        "support_pileup_window": pileup_window(record, start, end),
    }
    evidence.update(pileup_hypothesis_score(row, record, radius))
    evidence.update(support_read_evidence(row, record, radius, max_snippets))
    evidence.update(variant_context(row, record, ref, vcf, radius))
    for key in [
        "support_depth",
        "support_fraction",
        "support_margin",
        "entropy",
        "payload_prob",
        "type_prob_sub",
        "repeat_flag",
        "tandem_repeat_flag",
        "homopolymer_run_length",
        "neighbor_rule_flag",
        "boundary_flag",
        "variant_mask",
        "truth_vcf_overlap",
        "confident_bed_status",
        "low_confidence_or_preserve",
        "repeat_strength",
    ]:
        evidence.setdefault(key, row.get(key))
    evidence["candidate_evidence_safety_score"] = candidate_safety_score(evidence)
    evidence["candidate_evidence_risk_score"] = -evidence["candidate_evidence_safety_score"]
    return evidence

