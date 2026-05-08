"""Correction and safety metrics."""

from __future__ import annotations

from collections import Counter, defaultdict

from omega_safe_seqedit.constants import BASES
from omega_safe_seqedit.dna import edit_distance, identity
from omega_safe_seqedit.labels import label_events


def _event_key(event: dict) -> tuple:
    return (event["pos"], event["type"], event.get("base"))


def sequence_metrics(prediction: str, target: str, truth: str) -> dict:
    pred_ed = edit_distance(prediction, truth)
    target_ed = edit_distance(target, truth)
    false_hard_proxy = max(0, pred_ed - target_ed)
    return {
        "identity": identity(prediction, truth),
        "edit_distance": pred_ed,
        "normalized_edit_distance": pred_ed / max(len(truth), len(prediction), 1),
        "target_edit_distance": target_ed,
        "predicted_length_ratio": len(prediction) / max(len(truth), 1),
        "overcorrection_rate": false_hard_proxy / max(len(truth), 1),
    }


def event_metrics(pred_events: list[dict], gold_events: list[dict]) -> dict:
    pred = {_event_key(e) for e in pred_events}
    gold = {_event_key(e) for e in gold_events}
    true_pos = pred & gold
    false_pos = pred - gold
    false_neg = gold - pred
    out = {
        "hard_edit_precision": len(true_pos) / max(len(pred), 1),
        "hard_edit_recall": len(true_pos) / max(len(gold), 1),
        "hard_edit_f1": 2 * len(true_pos) / max(len(pred) + len(gold), 1),
        "hard_edit_false_positive_rate": len(false_pos) / max(len(pred), 1),
        "corrected_edits": len(true_pos),
        "missed_edits": len(false_neg),
        "false_edits": len(false_pos),
    }
    for edit_type in ["SUB", "INS", "DEL"]:
        p = {k for k in pred if k[1] == edit_type}
        g = {k for k in gold if k[1] == edit_type}
        tp = p & g
        out[f"{edit_type.lower()}_precision"] = len(tp) / max(len(p), 1)
        out[f"{edit_type.lower()}_recall"] = len(tp) / max(len(g), 1)
        out[f"{edit_type.lower()}_f1"] = 2 * len(tp) / max(len(p) + len(g), 1)
    return out


def _support_rule_events(record: dict) -> list[dict]:
    events = []
    features = record["features"]
    for pos, rule_id in enumerate(features["support_rule_type"]):
        rule = ["COPY", "SUB", "DEL", "INS"][rule_id]
        if rule == "SUB":
            events.append({"pos": pos, "type": "SUB", "base": BASES[features["support_rule_sub_base"][pos]]})
        elif rule == "DEL":
            events.append({"pos": pos, "type": "DEL", "base": record["target_seq"][pos]})
        elif rule == "INS":
            events.append({"pos": pos, "type": "INS", "base": BASES[features["support_rule_ins_base"][pos]]})
    return events


def _neural_events_from_trace(record: dict) -> list[dict]:
    events = []
    for item in record.get("trace", []):
        pos = int(item["pos"])
        if item.get("neural_ins_base"):
            events.append({"pos": pos, "type": "INS", "base": item["neural_ins_base"]})
        if item.get("neural_main") == "SUB":
            events.append({"pos": pos, "type": "SUB", "base": item.get("neural_sub_base")})
        elif item.get("neural_main") == "DEL":
            events.append({"pos": pos, "type": "DEL", "base": record["target_seq"][pos]})
    return events


def _feature_context(record: dict, pos: int) -> dict:
    features = record["features"]
    depth = max(float(features["support_depth"][pos]), 1.0)
    def flag(key: str, default: int = 0) -> int | None:
        values = features.get(key)
        if values is None or pos >= len(values):
            return default
        return int(values[pos])
    def scalar(key: str, default: float = 0.0) -> float:
        values = features.get(key)
        if values is None or pos >= len(values):
            return default
        return float(values[pos])
    homopolymer_run = features["homopolymer_run_length"][pos]
    tandem_repeat = int(features["tandem_repeat_flag"][pos]) or flag("region_tandem_repeat_flag", 0)
    region_homopolymer = flag("region_homopolymer_flag", 0)
    preserve = flag("preserve_mask", 0)
    uncertainty = flag("uncertainty_label", 0)
    confident = flag("confident_mask", 1)
    variant = flag("variant_mask", 0)
    phased_variant = flag("phased_variant_mask", 0)
    variant_rich = flag("variant_rich_flag", 0)
    ins_base_counts = features["support_ins_base_counts"][pos]
    ins_total = max(float(features["support_ins_count"][pos]), 0.0)
    raw_ins_total = sum(float(x) for x in ins_base_counts)
    if raw_ins_total > 0.0 and ins_total > 0.0 and raw_ins_total > ins_total:
        scale = ins_total / raw_ins_total
        effective_ins_counts = [float(x) * scale for x in ins_base_counts]
    else:
        effective_ins_counts = [float(x) for x in ins_base_counts]
    sorted_ins = sorted(effective_ins_counts, reverse=True)
    top_inserted_base_count = sorted_ins[0] if sorted_ins else 0.0
    second_inserted_base_count = sorted_ins[1] if len(sorted_ins) > 1 else 0.0
    return {
        "support_base_counts": features["support_base_counts"][pos],
        "support_ins_base_counts": features["support_ins_base_counts"][pos],
        "support_del_count": features["support_del_count"][pos],
        "support_ins_count": features["support_ins_count"][pos],
        "normalized_support_ins_base_counts": effective_ins_counts,
        "top_inserted_base_count": top_inserted_base_count,
        "top_inserted_base_fraction": top_inserted_base_count / max(ins_total, 1.0),
        "inserted_base_margin": top_inserted_base_count - second_inserted_base_count,
        "raw_inserted_base_total": raw_ins_total,
        "support_depth": features["support_depth"][pos],
        "consensus_agreement": features["support_agreement"][pos],
        "support_fraction": max(
            max(features["support_base_counts"][pos]),
            features["support_del_count"][pos],
            features["support_ins_count"][pos],
        )
        / depth,
        "support_margin": features["support_margin"][pos],
        "entropy": features["support_entropy"][pos],
        "homopolymer_run_length": homopolymer_run,
        "homopolymer_flag": int(homopolymer_run >= 4 or bool(region_homopolymer)),
        "tandem_repeat_flag": tandem_repeat,
        "repeat_flag": int(bool(tandem_repeat) or homopolymer_run >= 4 or bool(region_homopolymer)),
        "neighbor_rule_flag": features["neighbor_rule_flag"][pos],
        "neighbor_edit_distance": 1 if features["neighbor_rule_flag"][pos] else None,
        "boundary_flag": features["boundary_flag"][pos],
        "variant_mask": variant,
        "phased_variant_mask": phased_variant,
        "truth_vcf_overlap": int(bool(variant) or bool(phased_variant)),
        "preserve_mask": preserve,
        "uncertainty_label": uncertainty,
        "variant_rich_flag": variant_rich,
        "confident_mask": confident,
        "confident_bed_status": confident,
        "low_confidence_or_preserve": int(bool(preserve) or bool(uncertainty) or not bool(confident)),
        "local_rule_density": flag("local_rule_density", 0),
        "local_mismatch_density": scalar("local_mismatch_density"),
        "local_variant_density": scalar("local_variant_density"),
        "nearby_indel_density": scalar("nearby_indel_density"),
        "variant_proximity_flag": flag("variant_proximity_flag", 0),
        "window_relative_position": scalar("window_relative_position"),
        "support_forward_fraction": scalar("support_forward_fraction", 0.5),
        "support_forward_count": scalar("support_forward_count"),
        "support_reverse_count": scalar("support_reverse_count"),
        "support_strand_bias": min(max(scalar("support_strand_bias"), 0.0), 1.0),
        "support_same_haplotype_fraction": scalar("support_same_haplotype_fraction"),
        "support_match_fraction": scalar("support_match_fraction"),
        "left_support_match_fraction": scalar("left_support_match_fraction"),
        "right_support_match_fraction": scalar("right_support_match_fraction"),
        "repeat_strength": scalar("repeat_strength"),
        "mapping_quality_mean": scalar("mapping_quality_mean"),
        "mapping_quality_available": flag("mapping_quality_available", 0),
        "reference_kmer_uniqueness": scalar("reference_kmer_uniqueness"),
        "reference_kmer_uniqueness_available": flag("reference_kmer_uniqueness_available", 0),
    }


def _mechanism_counts(false_rows: list[dict]) -> dict:
    return {
        "possible_true_variant_or_haplotype": sum(
            1
            for row in false_rows
            if row.get("truth_vcf_overlap") or row.get("phased_variant_mask") or row.get("preserve_mask")
        ),
        "repeat_or_homopolymer_ambiguity": sum(
            1 for row in false_rows if row.get("repeat_flag") or row.get("homopolymer_flag") or row.get("tandem_repeat_flag")
        ),
        "neighbor_alignment_ambiguity": sum(1 for row in false_rows if row.get("neighbor_rule_flag")),
        "support_rule_false_positive": sum(
            1
            for row in false_rows
            if row.get("support_rule") not in {None, "COPY"}
            and (row.get("forced_by_rule") or row.get("accepted_sub_candidate") or row.get("accepted_indel_candidate"))
        ),
        "neural_hallucination_or_rescue": sum(
            1
            for row in false_rows
            if row.get("support_rule") in {None, "COPY"} or "neural_rescue" in (row.get("reasons") or [])
        ),
        "low_confidence_or_preserve": sum(1 for row in false_rows if row.get("low_confidence_or_preserve")),
        "high_entropy": sum(1 for row in false_rows if float(row.get("entropy", 0.0)) >= 0.5),
        "low_support_fraction": sum(1 for row in false_rows if float(row.get("support_fraction", 1.0)) < 0.9),
    }


def _safe_recovery_score(row: dict) -> float:
    """Rank vetoed true edits by conservative real-data safety signals."""
    depth = max(float(row.get("support_depth") or 0.0), 0.0)
    margin = float(row.get("support_margin") or 0.0)
    entropy = float(row.get("entropy") or 0.0)
    score = 0.0
    score += min(depth / 12.0, 1.0)
    score += min(float(row.get("support_fraction") or 0.0), 1.0)
    score += min(margin / max(depth, 1.0), 1.0)
    score += 1.0 - min(entropy, 1.0)
    for key in ["repeat_flag", "tandem_repeat_flag", "neighbor_rule_flag", "boundary_flag", "truth_vcf_overlap", "low_confidence_or_preserve"]:
        if row.get(key):
            score -= 0.75
    if int(row.get("homopolymer_run_length") or 1) >= 2:
        score -= 0.35
    return score


def summarize_predictions(records: list[dict]) -> dict:
    sums: defaultdict[str, float] = defaultdict(float)
    false_rows = []
    support_rule_gap_rows = []
    would_have_corrected_rows = []
    vetoed_true_rows = []
    per_base = Counter()
    per_base_hit = Counter()
    false_by_type = Counter()
    support_rule_true_positive = Counter()
    support_rule_false_positive = Counter()
    neural_true_positive = Counter()
    neural_false_positive = Counter()
    neural_vs_rule_confusion = Counter()
    contribution = Counter()
    rule_payload_counts = Counter()
    rule_payload_hits = Counter()
    for record in records:
        gold = label_events(record["labels"], record["target_seq"])
        rule_events = _support_rule_events(record)
        neural_events = _neural_events_from_trace(record)
        seq = sequence_metrics(record["prediction"], record["target_seq"], record["truth_seq"])
        evt = event_metrics(record.get("pred_events", []), gold)
        for key, value in {**seq, **evt}.items():
            sums[key] += float(value)
        gold_keys = {_event_key(e) for e in gold}
        pred_keys = {_event_key(e) for e in record.get("pred_events", [])}
        rule_keys = {_event_key(e) for e in rule_events}
        neural_keys = {_event_key(e) for e in neural_events}
        for key in rule_keys & gold_keys:
            support_rule_true_positive[key[1]] += 1
        for key in rule_keys - gold_keys:
            support_rule_false_positive[key[1]] += 1
        for key in neural_keys & gold_keys:
            neural_true_positive[key[1]] += 1
        for key in neural_keys - gold_keys:
            neural_false_positive[key[1]] += 1
        for pos, typ, base in gold_keys:
            if typ in {"SUB", "INS"}:
                per_base[f"{typ}_{base}"] += 1
                if (pos, typ, base) in pred_keys:
                    per_base_hit[f"{typ}_{base}"] += 1
        for event in rule_events:
            pos = event["pos"]
            trace = record.get("trace", [])
            pos_trace = trace[pos] if pos < len(trace) else {}
            if event["type"] == "SUB":
                rule_payload_counts["SUB"] += 1
                if pos_trace.get("neural_sub_base") == event.get("base"):
                    rule_payload_hits["SUB"] += 1
            elif event["type"] == "INS":
                rule_payload_counts["INS"] += 1
                if pos_trace.get("neural_ins_base") == event.get("base"):
                    rule_payload_hits["INS"] += 1
        for key in sorted((rule_keys & gold_keys) - pred_keys):
            pos, typ, base = key
            trace = record.get("trace", [])
            pos_trace = trace[pos] if pos < len(trace) else {}
            row = {
                "example_id": record["example_id"],
                "case_type": record.get("case_type"),
                "pos": pos,
                "position": pos,
                "edit_type": typ,
                "gold_type": typ,
                "gold_base": base,
                "gold_label": f"{typ}_{base}" if typ != "DEL" else "DEL",
                "support_rule_label": f"{typ}_{base}" if typ != "DEL" else "DEL",
                "hybrid_label": (
                    f"INS_{pos_trace.get('chosen_ins_base')}"
                    if pos_trace.get("chosen_ins_base")
                    else f"SUB_{base}"
                    if pos_trace.get("chosen_main") == "SUB"
                    else pos_trace.get("chosen_main")
                ),
                "neural_label": (
                    f"INS_{pos_trace.get('neural_ins_base')}"
                    if pos_trace.get("neural_ins_base")
                    else f"SUB_{pos_trace.get('neural_sub_base')}"
                    if pos_trace.get("neural_main") == "SUB"
                    else pos_trace.get("neural_main")
                ),
                "chosen_main": pos_trace.get("chosen_main"),
                "chosen_ins_base": pos_trace.get("chosen_ins_base"),
                "neural_main": pos_trace.get("neural_main"),
                "neural_sub_base": pos_trace.get("neural_sub_base"),
                "neural_ins_base": pos_trace.get("neural_ins_base"),
                "vetoed": pos_trace.get("vetoed"),
                "forced_by_rule": pos_trace.get("forced_by_rule"),
                "accepted_sub_candidate": pos_trace.get("accepted_sub_candidate"),
                "accepted_indel_candidate": pos_trace.get("accepted_indel_candidate"),
                "veto_reason": pos_trace.get("reasons"),
                "reasons": pos_trace.get("reasons"),
                "neural_main_probs": pos_trace.get("main_probs"),
                "payload_probs": pos_trace.get("sub_probs") if typ == "SUB" else pos_trace.get("ins_probs") if typ == "INS" else None,
                "main_probs": pos_trace.get("main_probs"),
                "sub_probs": pos_trace.get("sub_probs"),
                "ins_probs": pos_trace.get("ins_probs"),
                "sub_candidate_details": pos_trace.get("sub_candidate_details"),
                "indel_candidate_details": pos_trace.get("indel_candidate_details"),
                "candidate_allow_score": (pos_trace.get("indel_candidate_details") or {}).get("allow_gate_score"),
                **_feature_context(record, pos),
            }
            row["safe_recovery_score"] = _safe_recovery_score(row)
            support_rule_gap_rows.append(row)
            would_have_corrected_rows.append(row)
        for key in sorted((gold_keys & neural_keys) - pred_keys):
            pos, typ, base = key
            trace = record.get("trace", [])
            pos_trace = trace[pos] if pos < len(trace) else {}
            vetoed_true_rows.append(
                {
                    "example_id": record["example_id"],
                    "case_type": record.get("case_type"),
                    "pos": pos,
                    "gold_type": typ,
                    "gold_base": base,
                    "neural_main": pos_trace.get("neural_main"),
                    "neural_sub_base": pos_trace.get("neural_sub_base"),
                    "neural_ins_base": pos_trace.get("neural_ins_base"),
                    "chosen_main": pos_trace.get("chosen_main"),
                    "chosen_ins_base": pos_trace.get("chosen_ins_base"),
                    "vetoed": pos_trace.get("vetoed"),
                    "reasons": pos_trace.get("reasons"),
                    "main_probs": pos_trace.get("main_probs"),
                    "sub_probs": pos_trace.get("sub_probs"),
                    "ins_probs": pos_trace.get("ins_probs"),
                    **_feature_context(record, pos),
                }
            )
        for item in record.get("trace", []):
            neural_vs_rule_confusion[f"{item.get('rule_type')}->{item.get('neural_main')}"] += 1
        for event in record.get("pred_events", []):
            if _event_key(event) not in gold_keys:
                trace = record.get("trace", [])
                pos_trace = trace[event["pos"]] if event["pos"] < len(trace) else {}
                false_by_type[event["type"]] += 1
                false_rows.append(
                    {
                        "example_id": record["example_id"],
                        "case_type": record.get("case_type"),
                        "pos": event["pos"],
                        "position": event["pos"],
                        "gold_label": "COPY",
                        "predicted_label": f"{event['type']}_{event.get('base')}" if event["type"] != "DEL" else "DEL",
                        "predicted_type": event["type"],
                        "edit_type": event["type"],
                        "predicted_base": event.get("base"),
                        "target_base": record["target_seq"][event["pos"]] if event["pos"] < len(record["target_seq"]) else "",
                        "truth_base": record["truth_seq"][event["pos"]] if event["pos"] < len(record["truth_seq"]) else "",
                        "support_rule": pos_trace.get("rule_type"),
                        "support_rule_label": (
                            f"{pos_trace.get('rule_type')}_{pos_trace.get('rule_base')}"
                            if pos_trace.get("rule_type") in {"SUB", "INS"}
                            else pos_trace.get("rule_type")
                        ),
                        "neural_main": pos_trace.get("neural_main"),
                        "neural_only_label": (
                            f"INS_{pos_trace.get('neural_ins_base')}"
                            if pos_trace.get("neural_ins_base")
                            else f"SUB_{pos_trace.get('neural_sub_base')}"
                            if pos_trace.get("neural_main") == "SUB"
                            else pos_trace.get("neural_main")
                        ),
                        "forced_by_rule": pos_trace.get("forced_by_rule"),
                        "accepted_sub_candidate": pos_trace.get("accepted_sub_candidate"),
                        "accepted_indel_candidate": pos_trace.get("accepted_indel_candidate"),
                        "vetoed": pos_trace.get("vetoed"),
                        "veto_or_rescue_reason": pos_trace.get("reasons"),
                        "reasons": pos_trace.get("reasons"),
                        "sub_candidate_details": pos_trace.get("sub_candidate_details"),
                        "indel_candidate_details": pos_trace.get("indel_candidate_details"),
                        "main_probs": pos_trace.get("main_probs"),
                        "sub_probs": pos_trace.get("sub_probs"),
                        "ins_probs": pos_trace.get("ins_probs"),
                        **(_feature_context(record, event["pos"]) if event["pos"] < len(record["target_seq"]) else {}),
                    }
                )
        for event in record.get("pred_events", []):
            key = _event_key(event)
            if key in gold_keys:
                trace = record.get("trace", [])
                pos_trace = trace[event["pos"]] if event["pos"] < len(trace) else {}
                if pos_trace.get("forced_by_rule"):
                    contribution["forced_by_rule_correct"] += 1
                if key in neural_keys:
                    contribution["neural_agreed_correct"] += 1
                if "neural_rescue" in pos_trace.get("reasons", []):
                    contribution["rescued_by_neural_correct"] += 1
            else:
                trace = record.get("trace", [])
                pos_trace = trace[event["pos"]] if event["pos"] < len(trace) else {}
                if pos_trace.get("forced_by_rule"):
                    contribution["forced_by_rule_wrong"] += 1
                if key in neural_keys:
                    contribution["neural_agreed_wrong"] += 1
                if "neural_rescue" in pos_trace.get("reasons", []):
                    contribution["rescued_by_neural_wrong"] += 1
    n = max(len(records), 1)
    summary = {key: value / n for key, value in sums.items()}
    summary["total_corrected_edits"] = sums.get("corrected_edits", 0.0)
    summary["total_missed_edits"] = sums.get("missed_edits", 0.0)
    summary["total_false_edits"] = sums.get("false_edits", 0.0)
    summary["false_edits_per_example"] = sums.get("false_edits", 0.0) / n
    summary["usable_score"] = (
        summary.get("identity", 0.0)
        - 0.5 * summary.get("overcorrection_rate", 0.0)
        - 0.5 * summary.get("hard_edit_false_positive_rate", 0.0)
    )
    summary["num_examples"] = len(records)
    summary["false_edit_table"] = false_rows
    summary["support_rule_gap_table"] = support_rule_gap_rows
    summary["would_have_corrected_table"] = would_have_corrected_rows
    summary["vetoed_true_support_rule_table"] = sorted(
        would_have_corrected_rows,
        key=lambda row: row.get("safe_recovery_score", -999.0),
        reverse=True,
    )
    summary["vetoed_true_edit_table"] = vetoed_true_rows
    summary["false_edit_counts_by_type"] = dict(false_by_type)
    summary["false_del_count"] = false_by_type["DEL"]
    summary["false_sub_count"] = false_by_type["SUB"]
    summary["false_ins_count"] = false_by_type["INS"]
    summary["neighbor_induced_false_edits"] = sum(1 for row in false_rows if row.get("neighbor_rule_flag"))
    summary["homopolymer_false_del_count"] = sum(
        1 for row in false_rows if row.get("predicted_type") == "DEL" and row.get("homopolymer_run_length", 1) >= 4
    )
    summary["false_edit_context_counts"] = {
        "homopolymer": sum(1 for row in false_rows if row.get("homopolymer_flag")),
        "tandem_repeat": sum(1 for row in false_rows if row.get("tandem_repeat_flag")),
        "repeat": sum(1 for row in false_rows if row.get("repeat_flag")),
        "neighbor": sum(1 for row in false_rows if row.get("neighbor_rule_flag")),
        "boundary": sum(1 for row in false_rows if row.get("boundary_flag")),
        "variant": sum(1 for row in false_rows if row.get("variant_mask")),
        "phased_variant": sum(1 for row in false_rows if row.get("phased_variant_mask")),
        "truth_vcf_overlap": sum(1 for row in false_rows if row.get("truth_vcf_overlap")),
        "variant_rich": sum(1 for row in false_rows if row.get("variant_rich_flag")),
        "low_confidence_or_preserve": sum(1 for row in false_rows if row.get("low_confidence_or_preserve")),
        "high_entropy": sum(1 for row in false_rows if float(row.get("entropy", 0.0)) >= 0.5),
        "low_support_fraction": sum(1 for row in false_rows if float(row.get("support_fraction", 1.0)) < 0.9),
    }
    summary["false_edit_mechanism_hypotheses"] = _mechanism_counts(false_rows)
    summary["support_rule_true_positive_counts"] = dict(support_rule_true_positive)
    summary["support_rule_false_positive_counts"] = dict(support_rule_false_positive)
    summary["neural_true_positive_counts"] = dict(neural_true_positive)
    summary["neural_false_positive_counts"] = dict(neural_false_positive)
    summary["neural_hard_edit_recall"] = sum(neural_true_positive.values()) / max(
        sum(1 for record in records for event in label_events(record["labels"], record["target_seq"])),
        1,
    )
    summary["neural_vs_rule_confusion"] = dict(neural_vs_rule_confusion)
    summary["rule_positive_payload_accuracy"] = {
        key: rule_payload_hits[key] / max(rule_payload_counts[key], 1)
        for key in sorted(rule_payload_counts)
    }
    summary["hybrid_contribution"] = dict(contribution)
    summary["per_base_recall"] = {
        key: per_base_hit[key] / max(per_base[key], 1)
        for key in sorted(per_base)
    }
    return summary
