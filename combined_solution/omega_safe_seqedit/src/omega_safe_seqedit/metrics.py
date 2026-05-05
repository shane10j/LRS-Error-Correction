"""Correction and safety metrics."""

from __future__ import annotations

from collections import Counter, defaultdict

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


def summarize_predictions(records: list[dict]) -> dict:
    sums: defaultdict[str, float] = defaultdict(float)
    false_rows = []
    per_base = Counter()
    per_base_hit = Counter()
    for record in records:
        gold = label_events(record["labels"], record["target_seq"])
        seq = sequence_metrics(record["prediction"], record["target_seq"], record["truth_seq"])
        evt = event_metrics(record.get("pred_events", []), gold)
        for key, value in {**seq, **evt}.items():
            sums[key] += float(value)
        gold_keys = {_event_key(e) for e in gold}
        pred_keys = {_event_key(e) for e in record.get("pred_events", [])}
        for pos, typ, base in gold_keys:
            if typ in {"SUB", "INS"}:
                per_base[f"{typ}_{base}"] += 1
                if (pos, typ, base) in pred_keys:
                    per_base_hit[f"{typ}_{base}"] += 1
        for event in record.get("pred_events", []):
            if _event_key(event) not in gold_keys:
                trace = record.get("trace", [])
                pos_trace = trace[event["pos"]] if event["pos"] < len(trace) else {}
                false_rows.append(
                    {
                        "example_id": record["example_id"],
                        "pos": event["pos"],
                        "predicted_type": event["type"],
                        "predicted_base": event.get("base"),
                        "target_base": record["target_seq"][event["pos"]] if event["pos"] < len(record["target_seq"]) else "",
                        "truth_base": record["truth_seq"][event["pos"]] if event["pos"] < len(record["truth_seq"]) else "",
                        "support_rule": pos_trace.get("rule_type"),
                        "neural_main": pos_trace.get("neural_main"),
                        "forced_by_rule": pos_trace.get("forced_by_rule"),
                        "vetoed": pos_trace.get("vetoed"),
                        "reasons": pos_trace.get("reasons"),
                    }
                )
    n = max(len(records), 1)
    summary = {key: value / n for key, value in sums.items()}
    summary["usable_score"] = (
        summary.get("identity", 0.0)
        - 0.5 * summary.get("overcorrection_rate", 0.0)
        - 0.5 * summary.get("hard_edit_false_positive_rate", 0.0)
    )
    summary["num_examples"] = len(records)
    summary["false_edit_table"] = false_rows
    summary["per_base_recall"] = {
        key: per_base_hit[key] / max(per_base[key], 1)
        for key in sorted(per_base)
    }
    return summary
