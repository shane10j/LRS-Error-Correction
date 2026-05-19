#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.candidate_evidence import build_candidate_evidence, maybe_reference_and_vcf
from omega_safe_seqedit.io_utils import read_jsonl, write_json, write_jsonl


def _is_sub_candidate(row: dict) -> bool:
    return row.get("candidate_source") == "support_rule" and row.get("candidate_type") == "SUB"


def _sort_old(row: dict) -> float:
    return float(row.get("conservative_sub_safety_score") or 0.0)


def _sort_new(row: dict) -> float:
    return float(row.get("candidate_evidence_safety_score") or 0.0)


def _compact(row: dict) -> dict:
    keys = [
        "candidate_id",
        "gold_safe_label",
        "false_if_applied",
        "applied",
        "target_base",
        "truth_base",
        "candidate_allele",
        "support_fraction",
        "payload_prob",
        "conservative_sub_safety_score",
        "candidate_evidence_safety_score",
        "copy_window_score",
        "sub_window_score",
        "delta_window_score",
        "candidate_like_count",
        "target_like_count",
        "ambiguous_count",
        "cluster_margin",
        "left_flank_delta",
        "right_flank_delta",
        "strand_balance",
        "nearby_indel_density",
        "nearby_mismatch_density",
        "known_variant_like_target",
        "repeat_flag",
        "neighbor_rule_flag",
    ]
    return {key: row.get(key) for key in keys}


def _ranker_sanity(rows: list[dict], top_ks: list[int]) -> dict:
    out = {}
    for k in top_ks:
        selected = rows[:k]
        out[f"precision_at_{k}"] = sum(1 for row in selected if row.get("gold_safe_label")) / max(k, 1)
    first_false = None
    first_true = None
    true_before_false = 0
    for idx, row in enumerate(rows, start=1):
        is_true = bool(row.get("gold_safe_label"))
        if is_true and first_true is None:
            first_true = idx
        if not is_true and first_false is None:
            first_false = idx
        if is_true and first_false is None:
            true_before_false += 1
    out.update({
        "first_false_rank": first_false,
        "first_true_rank": first_true,
        "max_true_before_first_false": true_before_false,
    })
    return out


def _md_table(rows: list[dict]) -> str:
    columns = [
        "rank",
        "gold",
        "old",
        "new",
        "delta",
        "cluster",
        "copy_reads",
        "sub_reads",
        "ambig",
        "candidate_id",
    ]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "|"
            + "|".join(
                [
                    str(rank),
                    str(row.get("gold_safe_label")),
                    f"{float(row.get('conservative_sub_safety_score') or 0.0):.3f}",
                    f"{float(row.get('candidate_evidence_safety_score') or 0.0):.3f}",
                    f"{float(row.get('delta_window_score') or 0.0):.3f}",
                    f"{float(row.get('cluster_margin') or 0.0):.3f}",
                    str(row.get("target_like_count")),
                    str(row.get("candidate_like_count")),
                    str(row.get("ambiguous_count")),
                    "`" + str(row.get("candidate_id")) + "`",
                ]
            )
            + "|"
        )
    return "\n".join(lines)


def _candidate_block(row: dict) -> str:
    pileup = row.get("support_pileup_window", [])
    snippets = row.get("support_read_snippets", [])
    lines = [
        f"### {html.escape(str(row.get('candidate_id')))}",
        "",
        f"- gold_safe_label: `{row.get('gold_safe_label')}`",
        f"- candidate allele: `{row.get('candidate_allele')}`",
        f"- old score: `{float(row.get('conservative_sub_safety_score') or 0.0):.4f}`",
        f"- new safety score: `{float(row.get('candidate_evidence_safety_score') or 0.0):.4f}`",
        f"- copy/sub/delta: `{row.get('copy_window_score')}` / `{row.get('sub_window_score')}` / `{row.get('delta_window_score')}`",
        f"- read clusters candidate/target/ambiguous: `{row.get('candidate_like_count')}` / `{row.get('target_like_count')}` / `{row.get('ambiguous_count')}`",
        f"- strand balance: `{row.get('strand_balance')}`, repeat: `{row.get('repeat_flag')}`, variant-like-target: `{row.get('known_variant_like_target')}`",
        "",
        "```text",
        f"target:    {row.get('target_window')}",
        f"candidate: {row.get('candidate_window')}",
        f"truth:     {row.get('truth_window')}",
        "```",
        "",
        "**Support pileup window preview**",
        "",
        "```text",
    ]
    for item in pileup[:15]:
        lines.append(
            f"{item.get('position'):>5} {item.get('target_base')} "
            f"A:{item.get('base_counts', {}).get('A')} C:{item.get('base_counts', {}).get('C')} "
            f"G:{item.get('base_counts', {}).get('G')} T:{item.get('base_counts', {}).get('T')} "
            f"del:{item.get('del_count')} ins:{item.get('ins_count')} rule:{item.get('rule_type')}"
        )
    lines.extend(["```", "", "**Support read snippets**", "", "```text"])
    for item in snippets[:8]:
        lines.append(
            f"{item.get('prefers'):>5} {item.get('base_at_candidate')} "
            f"copy_d={item.get('copy_distance')} sub_d={item.get('sub_distance')} "
            f"strand={item.get('strand')} mapq={item.get('mapping_quality')} {item.get('snippet')}"
        )
    lines.extend(["```", ""])
    return "\n".join(lines)


def _write_report(path: str | Path, sections: dict[str, list[dict]], summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SUB Candidate Forensic Report",
        "",
        "This report is generated from local-window and whole-read evidence. It is diagnostic; it does not run correction.",
        "",
        "## Ranker Sanity",
        "",
        "```json",
        str(summary.get("ranker_sanity")),
        "```",
        "",
    ]
    for title, rows in sections.items():
        lines.extend([f"## {title}", "", _md_table(rows), ""])
        for row in rows[:5]:
            lines.append(_candidate_block(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export whole-read local-window evidence for support-rule SUB candidates.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--unsafe-candidates", default=None)
    parser.add_argument("--unsafe-predictions", default=None)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--max-snippets", type=int, default=12)
    args = parser.parse_args()

    records = {record["example_id"]: record for record in read_jsonl(args.predictions)}
    ref, vcf = maybe_reference_and_vcf(args.config)
    rows = []
    for row in read_jsonl(args.candidates):
        if _is_sub_candidate(row) and row.get("example_id") in records:
            rows.append(build_candidate_evidence(row, records[row["example_id"]], ref, vcf, args.radius, args.max_snippets))

    old_ranked = sorted(rows, key=_sort_old, reverse=True)
    new_ranked = sorted(rows, key=_sort_new, reverse=True)
    false_from_unsafe = []
    true_vetoed = [row for row in rows if row.get("gold_safe_label") and not row.get("applied")]
    if args.unsafe_candidates and args.unsafe_predictions:
        unsafe_records = {record["example_id"]: record for record in read_jsonl(args.unsafe_predictions)}
        for row in read_jsonl(args.unsafe_candidates):
            if _is_sub_candidate(row) and row.get("false_if_applied") and row.get("example_id") in unsafe_records:
                false_from_unsafe.append(
                    build_candidate_evidence(row, unsafe_records[row["example_id"]], ref, vcf, args.radius, args.max_snippets)
                )
    if not false_from_unsafe:
        false_from_unsafe = [row for row in rows if not row.get("gold_safe_label")]

    sections = {
        "Top 20 by old score": old_ranked[:20],
        "Top 20 by new local-window score": new_ranked[:20],
        "Top 20 false positives from unsafe recovery": sorted(false_from_unsafe, key=_sort_new, reverse=True)[:20],
        "Top 20 true positives previously vetoed": sorted(true_vetoed, key=_sort_new, reverse=True)[:20],
    }
    summary = {
        "num_support_rule_sub_candidates": len(rows),
        "num_true_sub_candidates": sum(1 for row in rows if row.get("gold_safe_label")),
        "num_false_sub_candidates": sum(1 for row in rows if not row.get("gold_safe_label")),
        "evidence_columns": sorted(rows[0].keys()) if rows else [],
        "ranker_sanity": {
            "old_score": _ranker_sanity(old_ranked, [1, 2, 5, 10, 20]),
            "candidate_evidence_safety_score": _ranker_sanity(new_ranked, [1, 2, 5, 10, 20]),
        },
        "top_20_old_score": [_compact(row) for row in old_ranked[:20]],
        "top_20_new_score": [_compact(row) for row in new_ranked[:20]],
        "report": str(args.report_output),
    }
    write_jsonl(args.output, rows)
    write_json(args.summary_output, summary)
    _write_report(args.report_output, sections, summary)
    print(summary)


if __name__ == "__main__":
    main()

