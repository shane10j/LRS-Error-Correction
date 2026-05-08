#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.constants import BASES
from omega_safe_seqedit.io_utils import read_jsonl, write_json, write_jsonl
from omega_safe_seqedit.labels import label_events
from omega_safe_seqedit.metrics import _event_key, _feature_context


def _label(edit_type: str, base: str | None = None) -> str:
    return f"{edit_type}_{base}" if edit_type in {"SUB", "INS"} else edit_type


def _prob_at(values: list[float] | None, index: int) -> float | None:
    if values is None or index < 0 or index >= len(values):
        return None
    return float(values[index])


def _candidate_density(trace: list[dict], pos: int, radius: int = 2) -> dict:
    start = max(0, pos - radius)
    end = min(len(trace), pos + radius + 1)
    rule_hard = 0
    neural_hard = 0
    chosen_hard = 0
    for idx in range(start, end):
        if idx == pos:
            continue
        row = trace[idx]
        if row.get("rule_type") not in {None, "COPY"}:
            rule_hard += 1
        if row.get("neural_main") in {"SUB", "DEL"} or row.get("neural_ins_base"):
            neural_hard += 1
        if row.get("chosen_main") in {"SUB", "DEL"} or row.get("chosen_ins_base"):
            chosen_hard += 1
    return {
        "local_rule_candidate_density": rule_hard,
        "local_neural_candidate_density": neural_hard,
        "local_chosen_edit_density": chosen_hard,
    }


def _neural_label(trace: dict, target_base: str) -> str:
    if trace.get("neural_ins_base"):
        return _label("INS", trace.get("neural_ins_base"))
    if trace.get("neural_main") == "SUB":
        return _label("SUB", trace.get("neural_sub_base"))
    if trace.get("neural_main") == "DEL":
        return _label("DEL", target_base)
    return "COPY"


def _normalized_ins_payload_counts(context: dict) -> dict:
    ins_counts = [float(x) for x in (context.get("support_ins_base_counts") or [0, 0, 0, 0])]
    support_ins_count = max(float(context.get("support_ins_count") or 0.0), 0.0)
    raw_total = sum(ins_counts)
    if raw_total > 0.0 and support_ins_count > 0.0 and raw_total > support_ins_count:
        scale = support_ins_count / raw_total
        effective = [x * scale for x in ins_counts]
    else:
        effective = ins_counts
    ordered = sorted(effective, reverse=True)
    top = ordered[0] if ordered else 0.0
    second = ordered[1] if len(ordered) > 1 else 0.0
    return {
        "normalized_support_ins_base_counts": effective,
        "top_inserted_base_count": top,
        "top_inserted_base_fraction": top / max(support_ins_count, 1.0),
        "inserted_base_margin": top - second,
        "raw_inserted_base_total": raw_total,
    }


def _candidate_rows(record: dict) -> list[dict]:
    gold = {_event_key(event) for event in label_events(record["labels"], record["target_seq"])}
    applied = {_event_key(event) for event in record.get("pred_events", [])}
    rows = []
    trace_rows = record.get("trace", [])
    for trace in trace_rows:
        pos = int(trace["pos"])
        context = _feature_context(record, pos)
        density = _candidate_density(trace_rows, pos)
        sources: list[tuple[str, str, str | None]] = []
        rule_type = trace.get("rule_type")
        rule_base = trace.get("rule_base")
        if rule_type in {"SUB", "DEL", "INS"}:
            if rule_type == "DEL":
                rule_base = record["target_seq"][pos]
            sources.append(("support_rule", rule_type, rule_base))
        neural_main = trace.get("neural_main")
        if neural_main == "SUB":
            sources.append(("neural", "SUB", trace.get("neural_sub_base")))
        elif neural_main == "DEL":
            sources.append(("neural", "DEL", record["target_seq"][pos]))
        if trace.get("neural_ins_base"):
            sources.append(("neural", "INS", trace.get("neural_ins_base")))
        chosen_main = trace.get("chosen_main")
        if chosen_main == "SUB":
            sources.append(("hybrid_chosen", "SUB", trace.get("rule_base") or trace.get("neural_sub_base")))
        elif chosen_main == "DEL":
            sources.append(("hybrid_chosen", "DEL", record["target_seq"][pos]))
        if trace.get("chosen_ins_base"):
            sources.append(("hybrid_chosen", "INS", trace.get("chosen_ins_base")))

        seen = set()
        for source, edit_type, base in sources:
            key = (source, edit_type, base)
            if key in seen:
                continue
            seen.add(key)
            event_key = (pos, edit_type, base)
            main_probs = trace.get("main_probs") or []
            sub_probs = trace.get("sub_probs") or []
            ins_probs = trace.get("ins_probs") or []
            payload_prob = None
            if edit_type == "SUB" and base in BASES:
                payload_prob = _prob_at(sub_probs, BASES.index(base))
            elif edit_type == "INS" and base in BASES:
                payload_prob = _prob_at(ins_probs, BASES.index(base) + 1)
            is_safe = int(event_key in gold)
            is_applied = int(event_key in applied)
            ins_stats = _normalized_ins_payload_counts(context)
            rows.append(
                {
                    "candidate_id": f"{record['example_id']}:{pos}:{source}:{_label(edit_type, base)}",
                    "example_id": record["example_id"],
                    "case_type": record.get("case_type"),
                    "contig": record.get("contig"),
                    "window_start": record.get("window_start"),
                    "window_end": record.get("window_end"),
                    "position": pos,
                    "candidate_source": source,
                    "candidate_type": edit_type,
                    "candidate_base": base,
                    "candidate_label": _label(edit_type, base),
                    "gold_safe_label": is_safe,
                    "safe_to_apply": is_safe,
                    "applied": is_applied,
                    "false_if_applied": int(is_applied and not is_safe),
                    "target_base": record["target_seq"][pos],
                    "truth_base": record["truth_seq"][pos] if pos < len(record["truth_seq"]) else None,
                    "support_rule_label": _label(rule_type, rule_base) if rule_type in {"SUB", "INS"} else rule_type,
                    "rule_label": _label(rule_type, rule_base) if rule_type in {"SUB", "INS"} else rule_type,
                    "neural_label": _neural_label(trace, record["target_seq"][pos]),
                    "neural_main": neural_main,
                    "chosen_main": chosen_main,
                    "forced_by_rule": trace.get("forced_by_rule"),
                    "accepted_sub_candidate": trace.get("accepted_sub_candidate"),
                    "accepted_indel_candidate": trace.get("accepted_indel_candidate"),
                    "vetoed": trace.get("vetoed"),
                    "reasons": trace.get("reasons"),
                    "sub_local_gain": (trace.get("sub_candidate_details") or {}).get("local_gain"),
                    "sub_local_rule_density": (trace.get("sub_candidate_details") or {}).get("local_rule_density"),
                    "indel_local_gain": (trace.get("indel_candidate_details") or {}).get("local_gain"),
                    "indel_local_rule_density": (trace.get("indel_candidate_details") or {}).get("local_rule_density"),
                    "indel_allow_gate_score": (trace.get("indel_candidate_details") or {}).get("allow_gate_score"),
                    "type_prob_copy": _prob_at(main_probs, 0),
                    "type_prob_sub": _prob_at(main_probs, 1),
                    "type_prob_del": _prob_at(main_probs, 2),
                    "payload_prob": payload_prob,
                    "allow_prob": trace.get("allow_prob"),
                    **ins_stats,
                    **density,
                    **context,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export real-data candidate edits for allow/edit gate calibration.")
    parser.add_argument("--predictions", required=True, help="Decoded predictions JSONL, e.g. test_hybrid_predictions.jsonl")
    parser.add_argument("--output", required=True, help="Candidate JSONL output path")
    parser.add_argument("--summary-output", default=None, help="Optional candidate summary JSON")
    args = parser.parse_args()

    rows = []
    for record in read_jsonl(args.predictions):
        rows.extend(_candidate_rows(record))
    write_jsonl(args.output, rows)
    summary = {
        "num_candidates": len(rows),
        "num_safe": sum(row["safe_to_apply"] for row in rows),
        "num_unsafe": sum(1 for row in rows if not row["safe_to_apply"]),
        "by_type": {},
    }
    for edit_type in ["SUB", "INS", "DEL"]:
        typed = [row for row in rows if row["candidate_type"] == edit_type]
        summary["by_type"][edit_type] = {
            "candidates": len(typed),
            "safe": sum(row["safe_to_apply"] for row in typed),
            "unsafe": sum(1 for row in typed if not row["safe_to_apply"]),
        }
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(summary)


if __name__ == "__main__":
    main()
