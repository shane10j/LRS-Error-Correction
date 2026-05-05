"""Neural, rule, and hybrid conservative decoding."""

from __future__ import annotations

import torch

from omega_safe_seqedit.constants import BASES, ID_TO_MAIN, ID_TO_RULE, INS_TO_ID, MAIN_TO_ID, RULE_TO_ID


def _rule_label(example: dict, pos: int) -> tuple[str, str | None]:
    rule_type = ID_TO_RULE[example["features"]["support_rule_type"][pos]]
    if rule_type == "SUB":
        return "SUB", BASES[example["features"]["support_rule_sub_base"][pos]]
    if rule_type == "INS":
        return "INS", BASES[example["features"]["support_rule_ins_base"][pos]]
    return rule_type, None


def _confidence(example: dict, pos: int, edit_type: str) -> dict:
    f = example["features"]
    depth = max(float(f["support_depth"][pos]), 1.0)
    support_value = max(f["support_base_counts"][pos]) if edit_type == "SUB" else f["support_ins_count"][pos] if edit_type == "INS" else f["support_del_count"][pos]
    return {
        "fraction": float(support_value) / depth,
        "margin": float(f["support_margin"][pos]) / depth,
        "entropy": float(f["support_entropy"][pos]),
        "depth": depth,
        "neighbor": bool(f["neighbor_rule_flag"][pos]),
        "boundary": bool(f["boundary_flag"][pos]),
        "homopolymer": f["homopolymer_run_length"][pos] >= 4,
    }


def _passes_rule_force(conf: dict, edit_type: str, config: dict) -> tuple[bool, list[str]]:
    decode = config["decode"]
    min_frac = float(decode.get("rule_force_min_fraction", {}).get(edit_type, 0.85))
    max_entropy = float(decode.get("rule_force_max_entropy", {}).get(edit_type, 0.85))
    if conf["neighbor"] and decode.get("neighbor_abstention", True):
        min_frac = max(min_frac, float(decode.get("neighbor_min_fraction", 0.90)))
    if edit_type == "DEL" and conf["homopolymer"]:
        min_frac = max(min_frac, float(decode.get("homopolymer_del_min_fraction", 0.95)))
    reasons = []
    if conf["fraction"] < min_frac:
        reasons.append("low_support_fraction")
    if conf["entropy"] > max_entropy:
        reasons.append("high_entropy")
    if edit_type in {"SUB", "DEL"} and conf["neighbor"] and conf["fraction"] < min_frac:
        reasons.append("neighbor_abstention")
    return not reasons, reasons


def decode_record(outputs: dict, index: int, example: dict, config: dict, mode: str = "hybrid") -> dict:
    main_probs = torch.softmax(outputs["main_logits"][index], dim=-1).detach().cpu()
    sub_probs = torch.softmax(outputs["sub_logits"][index], dim=-1).detach().cpu()
    ins_probs = torch.softmax(outputs["insert_logits"][index], dim=-1).detach().cpu()
    allow_probs = torch.sigmoid(outputs["allow_logits"][index]).detach().cpu()
    out = []
    trace = []
    pred_events = []
    target = example["target_seq"]
    for pos, base in enumerate(target):
        rule_type, rule_base = _rule_label(example, pos)
        neural_main_id = int(main_probs[pos].argmax())
        neural_main = ID_TO_MAIN[neural_main_id]
        neural_sub_base = BASES[int(sub_probs[pos].argmax())]
        neural_ins_id = int(ins_probs[pos].argmax())
        neural_ins_base = BASES[neural_ins_id - 1] if neural_ins_id > 0 else None
        chosen_main = neural_main
        chosen_sub_base = neural_sub_base
        chosen_ins_base = neural_ins_base
        reasons: list[str] = []
        forced = False
        vetoed = False

        if mode == "rule":
            chosen_main = "COPY"
            chosen_ins_base = None
            if rule_type == "SUB":
                chosen_main = "SUB"
                chosen_sub_base = rule_base or neural_sub_base
            elif rule_type == "DEL":
                chosen_main = "DEL"
            elif rule_type == "INS":
                chosen_ins_base = rule_base
        elif mode == "hybrid":
            if rule_type != "COPY":
                conf = _confidence(example, pos, rule_type)
                passes, rule_reasons = _passes_rule_force(conf, rule_type, config)
                if passes:
                    forced = True
                    if rule_type == "SUB":
                        chosen_main = "SUB"
                        chosen_sub_base = rule_base or neural_sub_base
                    elif rule_type == "DEL":
                        chosen_main = "DEL"
                        chosen_ins_base = None
                    elif rule_type == "INS":
                        chosen_main = "COPY"
                        chosen_ins_base = rule_base if config["decode"].get("use_support_payload_for_insertions", True) else neural_ins_base
                else:
                    reasons.extend(rule_reasons)
                    type_min = float(config["decode"].get("neural_rescue_min_type_prob", {}).get(rule_type, 0.99))
                    payload_min = float(config["decode"].get("neural_rescue_min_payload_prob", 0.97))
                    if rule_type == "SUB":
                        type_ok = float(main_probs[pos, MAIN_TO_ID["SUB"]]) >= type_min
                        payload_ok = float(sub_probs[pos, BASES.index(rule_base or "A")]) >= payload_min
                    elif rule_type == "INS":
                        type_ok = float(ins_probs[pos].max()) >= payload_min
                        payload_ok = True
                    else:
                        type_ok = float(main_probs[pos, MAIN_TO_ID["DEL"]]) >= type_min
                        payload_ok = True
                    if type_ok and payload_ok and not conf["neighbor"]:
                        reasons.append("neural_rescue")
                    else:
                        chosen_main = "COPY"
                        chosen_ins_base = None
                        vetoed = True
            elif config["decode"].get("rule_negative_veto", True):
                if chosen_main != "COPY" or chosen_ins_base is not None:
                    chosen_main = "COPY"
                    chosen_ins_base = None
                    vetoed = True
                    reasons.append("rule_negative_veto")

        if chosen_ins_base is not None:
            out.append(chosen_ins_base)
            pred_events.append({"pos": pos, "type": "INS", "base": chosen_ins_base})
        if chosen_main == "COPY":
            out.append(base)
        elif chosen_main == "SUB":
            out.append(chosen_sub_base)
            pred_events.append({"pos": pos, "type": "SUB", "base": chosen_sub_base})
        elif chosen_main == "DEL":
            pred_events.append({"pos": pos, "type": "DEL", "base": base})
        trace.append(
            {
                "pos": pos,
                "target_base": base,
                "rule_type": rule_type,
                "rule_base": rule_base,
                "neural_main": neural_main,
                "neural_sub_base": neural_sub_base,
                "neural_ins_base": neural_ins_base,
                "chosen_main": chosen_main,
                "chosen_ins_base": chosen_ins_base,
                "main_probs": [float(x) for x in main_probs[pos]],
                "sub_probs": [float(x) for x in sub_probs[pos]],
                "ins_probs": [float(x) for x in ins_probs[pos]],
                "allow_prob": float(allow_probs[pos]),
                "forced_by_rule": forced,
                "vetoed": vetoed,
                "reasons": reasons,
            }
        )
    return {"prediction": "".join(out), "pred_events": pred_events, "trace": trace}
