#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.config import load_config, print_resolved_config
from omega_safe_seqedit.io_utils import read_jsonl, write_json
from omega_safe_seqedit.trainer import evaluate_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ranked_sub_candidate_id(record: dict, trace: dict) -> str | None:
    if trace.get("chosen_main") != "SUB" or trace.get("rule_type") != "SUB" or not trace.get("rule_base"):
        return None
    return f"{record['example_id']}:{int(trace['pos'])}:support_rule:SUB_{trace['rule_base']}"


def _ranked_allowlist_metadata(allowlist_path: str, predictions_path: Path) -> dict:
    path = Path(allowlist_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_rows = payload.get("candidates") or [
        {"candidate_id": candidate_id, "gold_safe_label": None}
        for candidate_id in payload.get("candidate_ids", [])
    ]
    label_by_id = {str(row["candidate_id"]): row.get("gold_safe_label") for row in candidate_rows}
    allowlisted = set(label_by_id)
    num_true = sum(1 for value in label_by_id.values() if value in {1, True})
    num_false = sum(1 for value in label_by_id.values() if value in {0, False})
    applied_ids = set()
    for record in read_jsonl(predictions_path):
        for trace in record.get("trace", []):
            candidate_id = _ranked_sub_candidate_id(record, trace)
            if candidate_id and candidate_id in allowlisted:
                applied_ids.add(candidate_id)
    evaluated_true = sum(1 for candidate_id in applied_ids if label_by_id.get(candidate_id) in {1, True})
    evaluated_false = sum(1 for candidate_id in applied_ids if label_by_id.get(candidate_id) in {0, False})
    missing = sorted(allowlisted - applied_ids)
    unexpected = sorted(applied_ids - allowlisted)
    counts_match = (
        evaluated_true == num_true
        and evaluated_false == num_false
        and not missing
        and not unexpected
    )
    return {
        "allowlist_path": str(path),
        "allowlist_sha256": _sha256(path),
        "allowlist_created_at": payload.get("created_at"),
        "num_allowlisted": len(allowlisted),
        "num_true_in_allowlist": num_true,
        "num_false_in_allowlist": num_false,
        "evaluated_true_from_allowlist": evaluated_true,
        "evaluated_false_from_allowlist": evaluated_false,
        "evaluated_allowlisted_count": len(applied_ids),
        "allowlist_counts_match_evaluation": counts_match,
        "allowlisted_but_not_applied": missing[:50],
        "unexpected_applied_ranked_sub_candidates": unexpected[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", choices=["target_only", "full"], default="full")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["neural", "rule", "hybrid"], default="hybrid")
    parser.add_argument("--allow-gate", default=None, help="Optional learned allow gate JSON to use during hybrid decoding")
    parser.add_argument("--output-tag", default=None, help="Optional output tag, e.g. hybrid_new or hybrid_old")
    parser.add_argument("--disable-hard-indel-veto", action="store_true", help="Evaluate legacy hybrid without the HG002 hard indel veto")
    parser.add_argument("--enable-safe-recovery", action="store_true", help="Enable narrow safe true-edit recovery for an explicit A/B run")
    parser.add_argument("--disable-safe-recovery", action="store_true", help="Evaluate without narrow safe true-edit recovery")
    parser.add_argument("--safe-recovery-edit-types", default=None, help="Comma-separated recovery edit types, e.g. SUB or SUB,INS")
    parser.add_argument("--force-support-sub", action="store_true", help="Evaluate legacy behavior that force-applies support-rule SUB after thresholds")
    parser.add_argument("--ultra-safe-sub-recovery", action="store_true", help="Enable an intentionally tiny SUB recovery policy targeting 1-5 zero-FP true edits")
    parser.add_argument("--ranked-sub-recovery-allowlist", default=None, help="JSON allowlist from rank_sub_recovery_candidates.py")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.allow_gate:
        config.setdefault("decode", {})["learned_allow_gate_path"] = args.allow_gate
    if args.disable_hard_indel_veto:
        config.setdefault("decode", {})["hard_veto_indels"] = False
    if args.enable_safe_recovery:
        config.setdefault("decode", {})["safe_true_edit_recovery"] = True
    if args.disable_safe_recovery:
        config.setdefault("decode", {})["safe_true_edit_recovery"] = False
    if args.safe_recovery_edit_types:
        config.setdefault("decode", {})["safe_recovery_edit_types"] = [
            item.strip() for item in args.safe_recovery_edit_types.split(",") if item.strip()
        ]
    if args.force_support_sub:
        config.setdefault("decode", {})["force_support_sub"] = True
    if args.ultra_safe_sub_recovery:
        decode = config.setdefault("decode", {})
        decode["safe_true_edit_recovery"] = True
        decode["safe_recovery_edit_types"] = ["SUB"]
        decode["safe_recovery_min_depth"] = 12
        decode["safe_recovery_min_fraction"] = 1.0
        decode["safe_recovery_min_margin"] = 12
        decode["safe_recovery_max_entropy"] = 0.0
        decode["safe_recovery_sub_min_payload_prob"] = 0.93
        decode["safe_recovery_sub_min_type_prob"] = 0.0
        decode["safe_recovery_min_local_gain"] = 0.25
        decode["safe_recovery_max_local_mismatch_density"] = 0.15
        decode["safe_recovery_min_flank_match_fraction"] = 0.85
        decode["safe_recovery_max_strand_bias"] = 0.85
        decode["sub_candidate_min_type_prob"] = 1.01
        decode["sub_candidate_min_payload_prob"] = 1.01
        decode["sub_candidate_min_fraction"] = 1.0
        decode["sub_recovery_require_allow_gate"] = bool(args.allow_gate)
    if args.ranked_sub_recovery_allowlist:
        decode = config.setdefault("decode", {})
        decode["safe_true_edit_recovery"] = True
        decode["safe_recovery_edit_types"] = ["SUB"]
        decode["ranked_sub_recovery_mode"] = True
        decode["ranked_sub_recovery_allowlist_path"] = args.ranked_sub_recovery_allowlist
        decode["ranked_sub_recovery_min_local_gain"] = float(decode.get("ranked_sub_recovery_min_local_gain", 0.25))
        decode["ranked_sub_max_site_indel_evidence"] = float(decode.get("ranked_sub_max_site_indel_evidence", 0.0))
        decode["ranked_sub_min_payload_prob"] = float(decode.get("ranked_sub_min_payload_prob", 0.0))
        decode["sub_candidate_min_type_prob"] = 1.01
        decode["sub_candidate_min_payload_prob"] = 1.01
        decode["sub_candidate_min_fraction"] = 1.0
    print_resolved_config(config)
    summary = evaluate_checkpoint(config, args.run, args.split, args.mode, output_tag=args.output_tag)
    if args.ranked_sub_recovery_allowlist:
        tag = args.output_tag or ("neural" if args.run == "target_only" else args.mode)
        run_dir = Path(config["paths"]["runs_dir"]) / args.run
        summary_path = run_dir / f"{args.split}_{tag}_summary.json"
        predictions_path = run_dir / f"{args.split}_{tag}_predictions.jsonl"
        metadata = _ranked_allowlist_metadata(args.ranked_sub_recovery_allowlist, predictions_path)
        summary = {**summary, **metadata, "ranked_sub_allowlist_audit": metadata}
        write_json(summary_path, summary)
        assert metadata["allowlist_counts_match_evaluation"], (
            "Ranked SUB allowlist/evaluation mismatch: "
            f"{json.dumps(metadata, sort_keys=True)}"
        )
    print(summary)


if __name__ == "__main__":
    main()
