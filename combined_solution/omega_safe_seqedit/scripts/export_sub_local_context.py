#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.config import load_config
from omega_safe_seqedit.constants import BASES
from omega_safe_seqedit.io_utils import read_jsonl, write_json, write_jsonl


def _maybe_fasta(config: dict):
    ref_path = config.get("data", {}).get("reference_fasta") or config.get("data", {}).get("reference")
    if not ref_path or not Path(ref_path).exists():
        return None
    try:
        import pysam
    except ImportError:
        return None
    return pysam.FastaFile(str(ref_path))


def _nearby_variants(config: dict, contig: str, start: int, end: int) -> list[dict]:
    vcf_path = config.get("data", {}).get("truth_vcf")
    if not vcf_path or not Path(vcf_path).exists():
        return []
    try:
        import pysam
    except ImportError:
        return []
    rows = []
    try:
        vcf = pysam.VariantFile(str(vcf_path))
        for rec in vcf.fetch(contig, max(start, 0), max(end, start + 1)):
            rows.append({"pos": int(rec.pos), "ref": rec.ref, "alts": list(rec.alts or [])})
    except Exception:
        return []
    return rows


def _candidate_key(row: dict) -> tuple[str, int]:
    return str(row["example_id"]), int(row["position"])


def _safe_slice(seq: str, pos: int, flank: int) -> tuple[int, int, str]:
    start = max(0, pos - flank)
    end = min(len(seq), pos + flank + 1)
    return start, end, seq[start:end]


def _label_from_row(row: dict) -> str:
    return row.get("candidate_label") or row.get("support_rule_label") or row.get("rule_label") or "SUB"


def _pileup_window(record: dict, start: int, end: int) -> list[dict]:
    f = record.get("features", {})
    rows = []
    target = record.get("target_seq", "")
    for pos in range(start, end):
        base_counts = f.get("support_base_counts", [[0, 0, 0, 0]])[pos]
        rows.append(
            {
                "position": pos,
                "target_base": target[pos] if pos < len(target) else None,
                "base_counts": dict(zip(BASES, base_counts)),
                "del_count": f.get("support_del_count", [0])[pos],
                "ins_count": f.get("support_ins_count", [0])[pos],
                "depth": f.get("support_depth", [0])[pos],
                "agreement": f.get("support_agreement", [0.0])[pos],
                "entropy": f.get("support_entropy", [0.0])[pos],
                "rule_type": f.get("support_rule_type", [0])[pos],
            }
        )
    return rows


def _strand_at(track, pos: int):
    if isinstance(track, list) and pos < len(track):
        return track[pos]
    return None


def _supporting_reads(record: dict, pos: int, base: str) -> list[dict]:
    ids = record.get("support_read_ids", [])
    seqs = record.get("support_aligned_seqs", [])
    strands = record.get("support_strand_tracks", [])
    cigars = record.get("support_cigar_snippets", [])
    mapqs = record.get("support_mapping_qualities", [])
    rows = []
    for idx, seq in enumerate(seqs):
        if pos >= len(seq) or seq[pos] != base:
            continue
        rows.append(
            {
                "read_id": ids[idx] if idx < len(ids) else f"support_{idx}",
                "base": seq[pos],
                "strand": _strand_at(strands[idx], pos) if idx < len(strands) else None,
                "mapping_quality": mapqs[idx] if idx < len(mapqs) else None,
                "cigar_snippet": cigars[idx] if idx < len(cigars) else None,
            }
        )
    return rows


def _local_scores(record: dict, pos: int, sub_base: str, radius: int) -> dict:
    f = record.get("features", {})
    target = record.get("target_seq", "")
    start = max(0, pos - radius)
    end = min(len(target), pos + radius + 1)
    copy_score = 0.0
    sub_score = 0.0
    for idx in range(start, end):
        depth = max(float(f.get("support_depth", [1])[idx]), 1.0)
        target_base = target[idx]
        copy_score += float(f.get("support_base_counts", [[0, 0, 0, 0]])[idx][BASES.index(target_base)]) / depth
        scoring_base = sub_base if idx == pos else target_base
        sub_score += float(f.get("support_base_counts", [[0, 0, 0, 0]])[idx][BASES.index(scoring_base)]) / depth
    return {
        "radius": radius,
        "copy_support_consistency": copy_score,
        "sub_support_consistency": sub_score,
        "sub_minus_copy": sub_score - copy_score,
    }


def _context_row(row: dict, record: dict, fasta, config: dict, flank: int, score_radius: int) -> dict:
    pos = int(row["position"])
    label = _label_from_row(row)
    sub_base = label.split("_", 1)[1] if "_" in label else row.get("candidate_base")
    start, end, target_context = _safe_slice(record.get("target_seq", ""), pos, flank)
    _, _, truth_context = _safe_slice(record.get("truth_seq", ""), pos, flank)
    contig = record.get("contig")
    ref_start = int(record.get("window_start", 0)) + start
    ref_end = int(record.get("window_start", 0)) + end
    reference_context = None
    if fasta is not None and contig:
        try:
            reference_context = fasta.fetch(contig, ref_start, ref_end).upper()
        except Exception:
            reference_context = None
    supporting_reads = _supporting_reads(record, pos, sub_base or "N")
    strand_counts = Counter(str(item.get("strand")) for item in supporting_reads)
    return {
        "candidate_id": row.get("candidate_id"),
        "example_id": row.get("example_id"),
        "gold_safe_label": row.get("gold_safe_label"),
        "applied": row.get("applied"),
        "false_if_applied": row.get("false_if_applied"),
        "support_rule_label": row.get("support_rule_label") or row.get("rule_label"),
        "neural_label": row.get("neural_label"),
        "target_base": row.get("target_base"),
        "truth_base": row.get("truth_base"),
        "candidate_base": sub_base,
        "contig": contig,
        "window_start": record.get("window_start"),
        "window_end": record.get("window_end"),
        "position": pos,
        "reference_position_0based": int(record.get("window_start", 0)) + pos,
        "target_context_pm20": target_context,
        "truth_context_pm20": truth_context,
        "reference_context_pm20": reference_context,
        "support_pileup_pm20": _pileup_window(record, start, end),
        "supporting_majority_reads": supporting_reads,
        "supporting_majority_read_count": len(supporting_reads),
        "supporting_majority_strand_counts": dict(strand_counts),
        "nearby_variants": _nearby_variants(config, contig, ref_start, ref_end) if contig else [],
        "repeat_low_complexity_annotation": {
            "repeat_flag": row.get("repeat_flag"),
            "tandem_repeat_flag": row.get("tandem_repeat_flag"),
            "homopolymer_run_length": row.get("homopolymer_run_length"),
            "repeat_strength": row.get("repeat_strength"),
            "low_confidence_or_preserve": row.get("low_confidence_or_preserve"),
        },
        "scalar_features": {
            key: row.get(key)
            for key in [
                "support_depth",
                "support_fraction",
                "support_margin",
                "entropy",
                "type_prob_sub",
                "payload_prob",
                "local_rule_density",
                "local_mismatch_density",
                "left_support_match_fraction",
                "right_support_match_fraction",
                "support_forward_fraction",
                "support_strand_bias",
                "truth_vcf_overlap",
                "variant_proximity_flag",
                "confident_bed_status",
            ]
        },
        "local_window_scores": _local_scores(record, pos, sub_base or "A", score_radius),
        "ranked_recovery_rank": row.get("ranked_recovery_rank"),
        "ranked_recovery_score": row.get("ranked_recovery_score"),
        "heuristic_recovery_score": row.get("heuristic_recovery_score"),
        "rank_score_components": row.get("rank_score_components"),
        "availability_notes": {
            "cigar_snippets": "available only if source JSONL preserved support CIGARs",
            "mapping_qualities": "available only if source JSONL preserved support MAPQ",
            "reference_context": "requires indexed reference FASTA from config",
            "nearby_variants": "requires indexed truth VCF from config",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paired true-vs-false SUB local sequence/pileup inspection rows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--limit-per-class", type=int, default=50)
    parser.add_argument("--flank", type=int, default=20)
    parser.add_argument("--score-radius", type=int, default=5)
    parser.add_argument("--ranked-candidates", default=None, help="Optional ranked_sub_candidates.jsonl to inspect by rank.")
    parser.add_argument("--ranks", default=None, help="Comma-separated ranks to export from --ranked-candidates, e.g. 1,2.")
    args = parser.parse_args()

    config = load_config(args.config)
    records = {record["example_id"]: record for record in read_jsonl(args.predictions)}
    if args.ranked_candidates and args.ranks:
        wanted = {int(item.strip()) for item in args.ranks.split(",") if item.strip()}
        rows = [row for row in read_jsonl(args.ranked_candidates) if int(row.get("ranked_recovery_rank", -1)) in wanted]
        selected = sorted(rows, key=lambda row: int(row.get("ranked_recovery_rank", 10**9)))
    else:
        rows = [
            row
            for row in read_jsonl(args.candidates)
            if row.get("candidate_source") == "support_rule" and row.get("candidate_type") == "SUB"
        ]
        true_rows = [row for row in rows if row.get("gold_safe_label")]
        false_rows = [row for row in rows if not row.get("gold_safe_label")]
        rank_key = lambda row: (
            int(row.get("applied") or 0),
            float(row.get("conservative_sub_safety_score") or 0.0),
            float(row.get("support_fraction") or 0.0),
            float(row.get("support_margin") or 0.0),
        )
        selected = (
            sorted(true_rows, key=rank_key, reverse=True)[: args.limit_per_class]
            + sorted(false_rows, key=rank_key, reverse=True)[: args.limit_per_class]
        )
    fasta = _maybe_fasta(config)
    output_rows = [
        _context_row(row, records[row["example_id"]], fasta, config, args.flank, args.score_radius)
        for row in selected
        if row.get("example_id") in records
    ]
    write_jsonl(args.output, output_rows)
    summary = {
        "num_candidate_sub_rows": len(rows),
        "selected_true_rows": sum(1 for row in output_rows if row.get("gold_safe_label")),
        "selected_false_rows": sum(1 for row in output_rows if not row.get("gold_safe_label")),
        "output": args.output,
        "rank_1_vs_rank_2_question": "Compare the first true positive and first false positive ranks to find the next discriminative feature.",
        "columns": sorted(output_rows[0].keys()) if output_rows else [],
    }
    write_json(args.summary_output, summary)
    print(summary)


if __name__ == "__main__":
    main()
