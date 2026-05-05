"""Simple correction baselines."""

from __future__ import annotations

from omega_safe_seqedit.constants import BASES, ID_TO_RULE
from omega_safe_seqedit.io_utils import read_fastx


def no_edit(record: dict) -> dict:
    return {"prediction": record["target_seq"], "pred_events": [], "trace": []}


def support_rule(record: dict) -> dict:
    out = []
    events = []
    f = record["features"]
    for pos, base in enumerate(record["target_seq"]):
        rule = ID_TO_RULE[f["support_rule_type"][pos]]
        if rule == "INS":
            ins_base = BASES[f["support_rule_ins_base"][pos]]
            out.append(ins_base)
            events.append({"pos": pos, "type": "INS", "base": ins_base})
        if rule == "SUB":
            sub_base = BASES[f["support_rule_sub_base"][pos]]
            out.append(sub_base)
            events.append({"pos": pos, "type": "SUB", "base": sub_base})
        elif rule == "DEL":
            events.append({"pos": pos, "type": "DEL", "base": base})
            continue
        else:
            out.append(base)
    return {"prediction": "".join(out), "pred_events": events, "trace": []}


def conservative_consensus(record: dict, agreement_threshold: float = 0.85) -> dict:
    out = []
    events = []
    f = record["features"]
    for pos, base in enumerate(record["target_seq"]):
        depth = max(float(f["support_depth"][pos]), 1.0)
        best_id = max(range(4), key=lambda idx: f["support_base_counts"][pos][idx])
        frac = f["support_base_counts"][pos][best_id] / depth
        del_frac = f["support_del_count"][pos] / depth
        ins_frac = f["support_ins_count"][pos] / depth
        if ins_frac >= agreement_threshold:
            ins_id = max(range(4), key=lambda idx: f["support_ins_base_counts"][pos][idx])
            out.append(BASES[ins_id])
            events.append({"pos": pos, "type": "INS", "base": BASES[ins_id]})
        if del_frac >= agreement_threshold:
            events.append({"pos": pos, "type": "DEL", "base": base})
            continue
        if BASES[best_id] != base and frac >= agreement_threshold:
            out.append(BASES[best_id])
            events.append({"pos": pos, "type": "SUB", "base": BASES[best_id]})
        else:
            out.append(base)
    return {"prediction": "".join(out), "pred_events": events, "trace": []}


def external_predictions(path: str, records: list[dict]) -> list[dict]:
    fastx = read_fastx(path)
    out = []
    for record in records:
        pred = fastx.get(record["example_id"]) or fastx.get(record["target_read_id"]) or record["target_seq"]
        out.append({**record, "prediction": pred, "pred_events": [], "trace": []})
    return out
