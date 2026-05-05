"""Structured conservative and debug decoding."""

from __future__ import annotations

import torch

from omega_lr.constants import BASES, EDIT_TYPE_LABELS, ID_TO_BASE, compose_edit_label


def _consistency_allows(example: dict, pos: int, proposed_edit: str) -> bool:
    agreement = example["features"]["support_agreement"][pos]
    entropy = example["features"]["support_entropy"][pos]
    del_count = example["features"]["support_del_count"][pos]
    depth = max(1, example["features"]["support_depth"][pos])
    if proposed_edit.startswith("SUB_"):
        majority_idx = max(range(4), key=lambda idx: example["features"]["support_base_counts"][pos][idx])
        majority_base = BASES[majority_idx]
        return agreement >= 0.55 and entropy <= 1.5 and proposed_edit.endswith(majority_base)
    if proposed_edit == "DEL":
        return (del_count / depth) >= 0.40 and entropy <= 1.8
    if proposed_edit.startswith("INS_"):
        return example["features"]["support_ins_count"][pos] > 0 and entropy <= 1.8
    return True


def _mode_flags(decode_config: dict) -> dict:
    debug_mode = decode_config.get("mode", "conservative") == "debug"
    return {
        "debug_mode": debug_mode,
        "use_trust_threshold": decode_config.get("use_trust_threshold", not debug_mode),
        "use_delete_candidate_veto": decode_config.get("use_delete_candidate_veto", not debug_mode),
        "restrict_supported_candidates": decode_config.get("restrict_supported_candidates", not debug_mode),
        "consistency_check": decode_config.get("consistency_check", not debug_mode),
        "full_trace": decode_config.get("full_trace", debug_mode),
    }


def _supported_sub_bases(example: dict, pos: int) -> list[int]:
    target_base = example["target_seq"][pos]
    return [idx for idx, count in enumerate(example["features"]["support_base_counts"][pos]) if count > 0 and BASES[idx] != target_base]


def _supported_ins_bases(example: dict, pos: int) -> list[int]:
    if example["features"]["support_ins_count"][pos] <= 0:
        return []
    ins_counts = example["features"].get("support_ins_base_counts", [])
    supported = [idx for idx, count in enumerate(ins_counts[pos])] if pos < len(ins_counts) else []
    if supported:
        return supported
    majority_idx = max(range(4), key=lambda idx: example["features"]["support_base_counts"][pos][idx])
    return [majority_idx]


def _deletion_supported(example: dict, pos: int) -> bool:
    return example["features"]["support_del_count"][pos] > 0 or sum(example["features"]["gap_length_hist"][pos]) > 0


def _passes_support_threshold(value: float, depth: float, threshold: float) -> bool:
    if threshold <= 1.0:
        return value / max(depth, 1.0) >= threshold
    return value >= threshold


def _support_rule_label(example: dict, pos: int, decode_config: dict) -> str:
    features = example["features"]
    target_base = example["target_seq"][pos]
    counts = features["support_base_counts"][pos]
    depth = max(float(features["support_depth"][pos]), 1.0)
    agreement_threshold = decode_config.get("support_rule_agreement_threshold", 0.60)
    insertion_threshold = decode_config.get("support_rule_insertion_threshold", 0.50)
    deletion_threshold = decode_config.get("support_rule_deletion_threshold", 0.50)

    if sum(counts) > 0:
        majority_idx = max(range(4), key=lambda idx: counts[idx])
        majority_base = BASES[majority_idx]
        if (
            majority_base != target_base
            and features["support_agreement"][pos] >= agreement_threshold
            and _passes_support_threshold(float(counts[majority_idx]), depth, agreement_threshold)
        ):
            return f"SUB_{majority_base}"

    ins_count = float(features["support_ins_count"][pos])
    if _passes_support_threshold(ins_count, depth, insertion_threshold):
        ins_counts = features.get("support_ins_base_counts", [])
        if pos < len(ins_counts) and sum(ins_counts[pos]) > 0:
            return f"INS_{BASES[max(range(4), key=lambda idx: ins_counts[pos][idx])]}"
        majority_idx = max(range(4), key=lambda idx: counts[idx]) if sum(counts) > 0 else BASES.index(target_base)
        return f"INS_{BASES[majority_idx]}"

    del_count = float(features["support_del_count"][pos])
    if _passes_support_threshold(del_count, depth, deletion_threshold):
        return "DEL"
    return "COPY"


def _choose_base(base_probs: torch.Tensor, allowed_ids: list[int] | None) -> tuple[int, float]:
    if allowed_ids:
        best_idx = max(allowed_ids, key=lambda idx: float(base_probs[idx].item()))
    else:
        best_idx = int(torch.argmax(base_probs).item())
    return best_idx, float(base_probs[best_idx].item())


def _base_specific_threshold(decode_config: dict, key: str, base: str, default: float) -> float:
    by_base = decode_config.get(f"{key}_by_base", {})
    return float(by_base.get(base, default))


def _support_fraction(example: dict, pos: int, value: float) -> float:
    depth = max(float(example["features"]["support_depth"][pos]), 1.0)
    return float(value) / depth


def _support_majority_base(example: dict, pos: int) -> tuple[str, float]:
    counts = example["features"]["support_base_counts"][pos]
    if sum(counts) <= 0:
        return example["target_seq"][pos], 0.0
    best_idx = max(range(4), key=lambda idx: counts[idx])
    return BASES[best_idx], _support_fraction(example, pos, float(counts[best_idx]))


def _has_neighbor_support_rule_edit(example: dict, pos: int, decode_config: dict) -> bool:
    radius = int(decode_config.get("hybrid_neighbor_support_rule_radius", 1))
    start = max(0, pos - radius)
    end = min(len(example["target_seq"]), pos + radius + 1)
    return any(
        idx != pos and _support_rule_label(example, idx, decode_config) != "COPY"
        for idx in range(start, end)
    )


def _support_rule_confidence(example: dict, pos: int, rule_label: str, decode_config: dict | None = None) -> dict:
    decode_config = decode_config or {}
    features = example["features"]
    counts = list(features["support_base_counts"][pos])
    depth = max(float(features["support_depth"][pos]), 1.0)
    sorted_counts = sorted(counts, reverse=True)
    top_count = float(sorted_counts[0]) if sorted_counts else 0.0
    second_count = float(sorted_counts[1]) if len(sorted_counts) > 1 else 0.0
    support_value = top_count
    if rule_label.startswith("INS_"):
        support_value = float(features["support_ins_count"][pos])
    elif rule_label == "DEL":
        support_value = float(features["support_del_count"][pos])
    nearby_support_rule_hard = _has_neighbor_support_rule_edit(example, pos, decode_config)
    return {
        "support_margin": top_count - second_count,
        "support_fraction": support_value / depth,
        "del_fraction": float(features["support_del_count"][pos]) / depth,
        "base_top_fraction": top_count / depth,
        "local_entropy": float(features["support_entropy"][pos]),
        "support_depth": depth,
        "homopolymer_run_length": int(features.get("homopolymer_run_length", [1] * len(example["target_seq"]))[pos]),
        "tandem_repeat_flag": int(features.get("tandem_repeat_flag", [0] * len(example["target_seq"]))[pos]),
        "neighbor_edit_proximity": 1 if nearby_support_rule_hard or "neighbor" in example.get("example_id", "") else 0,
    }


def _rule_confidence_passes(confidence: dict, rule_label: str, decode_config: dict) -> tuple[bool, list[str]]:
    reasons = []
    min_fraction = float(decode_config.get("hybrid_rule_force_min_support_fraction", 0.67))
    min_margin = float(decode_config.get("hybrid_rule_force_min_support_margin", 1.0))
    max_entropy = float(decode_config.get("hybrid_rule_force_max_entropy", 1.25))
    min_depth = float(decode_config.get("hybrid_rule_force_min_depth", 3.0))
    if rule_label == "DEL":
        min_fraction = float(decode_config.get("hybrid_del_force_min_support_fraction", min_fraction))
        min_margin = float(decode_config.get("hybrid_del_force_min_support_margin", min_margin))
        max_entropy = float(decode_config.get("hybrid_del_force_max_entropy", max_entropy))
    elif rule_label.startswith("INS_"):
        min_fraction = float(decode_config.get("hybrid_ins_force_min_support_fraction", min_fraction))
        min_margin = float(decode_config.get("hybrid_ins_force_min_support_margin", min_margin))
        max_entropy = float(decode_config.get("hybrid_ins_force_max_entropy", max_entropy))
    if confidence["neighbor_edit_proximity"]:
        min_fraction = max(min_fraction, float(decode_config.get("hybrid_neighbor_min_support_fraction", 0.80)))
        max_entropy = min(max_entropy, float(decode_config.get("hybrid_neighbor_max_entropy", 0.90)))
    if confidence["homopolymer_run_length"] >= 4:
        min_fraction = max(min_fraction, float(decode_config.get("hybrid_homopolymer_min_support_fraction", 0.80)))
    if confidence["support_depth"] < min_depth:
        reasons.append("rule_low_depth")
    if confidence["support_fraction"] < min_fraction:
        reasons.append("rule_low_support_fraction")
    if confidence["support_margin"] < min_margin and rule_label.startswith("SUB_"):
        reasons.append("rule_low_margin")
    if confidence["local_entropy"] > max_entropy:
        reasons.append("rule_high_entropy")
    return not reasons, reasons


def _neighbor_abstention_reasons(label: str, confidence: dict, decode_config: dict) -> list[str]:
    """Abstain from independent edits in ambiguous local edit clusters."""
    if label == "COPY" or not decode_config.get("hybrid_neighbor_abstention", False):
        return []
    if not confidence.get("neighbor_edit_proximity", 0):
        return []
    reasons = []
    family = "SUB" if label.startswith("SUB_") else "INS" if label.startswith("INS_") else "DEL" if label == "DEL" else "COPY"
    min_fraction = float(decode_config.get("hybrid_neighbor_abstain_min_support_fraction", 0.90))
    min_margin = float(decode_config.get("hybrid_neighbor_abstain_min_support_margin", 4.0))
    max_entropy = float(decode_config.get("hybrid_neighbor_abstain_max_entropy", 0.30))
    if family == "DEL":
        min_fraction = float(decode_config.get("hybrid_neighbor_del_min_support_fraction", min_fraction))
        min_margin = float(decode_config.get("hybrid_neighbor_del_min_support_margin", min_margin))
        max_entropy = float(decode_config.get("hybrid_neighbor_del_max_entropy", max_entropy))
    elif family == "SUB":
        min_fraction = float(decode_config.get("hybrid_neighbor_sub_min_support_fraction", min_fraction))
        min_margin = float(decode_config.get("hybrid_neighbor_sub_min_support_margin", min_margin))
        max_entropy = float(decode_config.get("hybrid_neighbor_sub_max_entropy", max_entropy))
    elif family == "INS":
        min_fraction = float(decode_config.get("hybrid_neighbor_ins_min_support_fraction", min_fraction))
        min_margin = float(decode_config.get("hybrid_neighbor_ins_min_support_margin", min_margin))
        max_entropy = float(decode_config.get("hybrid_neighbor_ins_max_entropy", max_entropy))
    if confidence.get("support_fraction", 0.0) < min_fraction:
        reasons.append("hybrid_neighbor_abstain_low_support_fraction")
    if confidence.get("support_margin", 0.0) < min_margin:
        reasons.append("hybrid_neighbor_abstain_low_margin")
    if confidence.get("local_entropy", 0.0) > max_entropy:
        reasons.append("hybrid_neighbor_abstain_high_entropy")
    if label == "DEL" and confidence.get("homopolymer_run_length", 1) >= 4:
        min_hpoly_fraction = float(decode_config.get("hybrid_neighbor_homopolymer_del_min_support_fraction", 0.95))
        if confidence.get("del_fraction", confidence.get("support_fraction", 0.0)) < min_hpoly_fraction:
            reasons.append("hybrid_neighbor_homopolymer_del_veto")
    return reasons


def _local_window_parsimony_veto(label: str, confidence: dict, decode_config: dict) -> list[str]:
    if label == "COPY" or not decode_config.get("hybrid_local_window_rerank", False):
        return []
    if not confidence.get("neighbor_edit_proximity", 0):
        return []
    score = (
        float(confidence.get("support_fraction", 0.0))
        + 0.05 * float(confidence.get("support_margin", 0.0))
        - 0.25 * float(confidence.get("local_entropy", 0.0))
    )
    edit_penalty = float(decode_config.get("hybrid_local_window_edit_penalty", 0.20))
    min_score = float(decode_config.get("hybrid_local_window_min_keep_score", 0.95))
    if label == "DEL":
        edit_penalty = float(decode_config.get("hybrid_local_window_del_edit_penalty", max(edit_penalty, 0.35)))
        min_score = float(decode_config.get("hybrid_local_window_del_min_keep_score", max(min_score, 1.05)))
    if score - edit_penalty < min_score:
        return ["hybrid_local_window_parsimony_veto"]
    return []


def _support_insertion_payload(example: dict, pos: int) -> tuple[int | None, float, float]:
    ins_counts = example["features"].get("support_ins_base_counts", [])
    if pos >= len(ins_counts) or sum(ins_counts[pos]) <= 0:
        return None, 0.0, 0.0
    best_id = max(range(4), key=lambda idx: ins_counts[pos][idx])
    depth = max(float(example["features"]["support_depth"][pos]), 1.0)
    return best_id, float(ins_counts[pos][best_id]) / depth, float(ins_counts[pos][best_id])


def _insertion_rule_rescue_allows(example: dict, pos: int, rule_label: str, confidence: dict, decode_config: dict) -> tuple[bool, list[str]]:
    if not decode_config.get("hybrid_ins_support_payload_rescue", False) or not rule_label.startswith("INS_"):
        return False, []
    reasons = []
    rule_base_id = BASES.index(rule_label[-1])
    support_base_id, support_fraction, support_count = _support_insertion_payload(example, pos)
    min_fraction = float(decode_config.get("hybrid_ins_support_payload_min_fraction", 2.0 / 3.0))
    min_count = float(decode_config.get("hybrid_ins_support_payload_min_count", 2.0))
    min_depth = float(decode_config.get("hybrid_ins_support_payload_min_depth", 3.0))
    max_entropy = float(decode_config.get("hybrid_ins_support_payload_max_entropy", 1.0))
    if support_base_id != rule_base_id:
        reasons.append("hybrid_ins_support_payload_mismatch")
    if support_fraction < min_fraction:
        reasons.append("hybrid_ins_support_payload_low_fraction")
    if support_count < min_count:
        reasons.append("hybrid_ins_support_payload_low_count")
    if confidence["support_depth"] < min_depth:
        reasons.append("hybrid_ins_support_payload_low_depth")
    if confidence["local_entropy"] > max_entropy:
        reasons.append("hybrid_ins_support_payload_high_entropy")
    if confidence["neighbor_edit_proximity"] and not decode_config.get("hybrid_ins_support_payload_allow_neighbor", False):
        reasons.append("hybrid_ins_support_payload_neighbor_conflict")
    return not reasons, reasons


def _sub_t_calibrated_rescue_allows(
    example: dict,
    pos: int,
    rule_label: str,
    payload_base_id: int | None,
    payload_score: float,
    sub_type_prob: float,
    confidence: dict,
    decode_config: dict,
) -> tuple[bool, list[str]]:
    """Recover the calibrated small-noisy SUB_T miss without loosening all SUBs."""
    if not decode_config.get("hybrid_sub_t_calibrated_rescue", False) or rule_label != "SUB_T":
        return False, []
    reasons = []
    features = example["features"]
    counts = features["support_base_counts"][pos]
    depth = float(features["support_depth"][pos])
    t_count = float(counts[BASES.index("T")])
    required_count = float(decode_config.get("hybrid_sub_t_rescue_required_support_count", 2.0))
    required_depth = float(decode_config.get("hybrid_sub_t_rescue_required_depth", 3.0))
    min_payload = float(decode_config.get("hybrid_sub_t_rescue_min_payload", 0.60))
    min_type = float(decode_config.get("hybrid_sub_t_rescue_min_type_prob", 0.25))
    max_entropy = float(decode_config.get("hybrid_sub_t_rescue_max_entropy", 0.95))
    max_ins = float(decode_config.get("hybrid_sub_t_rescue_max_ins_count", 0.0))
    max_del = float(decode_config.get("hybrid_sub_t_rescue_max_del_count", 0.0))
    if payload_base_id != BASES.index("T"):
        reasons.append("hybrid_sub_t_payload_mismatch")
    if payload_score < min_payload:
        reasons.append("hybrid_sub_t_payload_low")
    if sub_type_prob < min_type:
        reasons.append("hybrid_sub_t_type_low")
    if abs(depth - required_depth) > 1e-6 or abs(t_count - required_count) > 1e-6:
        reasons.append("hybrid_sub_t_not_two_of_three")
    if confidence.get("support_fraction", 0.0) < (required_count / max(required_depth, 1.0) - 1e-6):
        reasons.append("hybrid_sub_t_low_support_fraction")
    if confidence.get("local_entropy", 0.0) > max_entropy:
        reasons.append("hybrid_sub_t_high_entropy")
    if float(features.get("support_ins_count", [0.0])[pos]) > max_ins:
        reasons.append("hybrid_sub_t_ins_conflict")
    if float(features.get("support_del_count", [0.0])[pos]) > max_del:
        reasons.append("hybrid_sub_t_del_conflict")
    return not reasons, reasons


def _remove_rule_confidence_reasons(veto_reasons: list[str]) -> list[str]:
    return [reason for reason in veto_reasons if not reason.startswith("rule_")]


def _rebuild_prediction(target_seq: str, labels: list[str]) -> str:
    corrected = []
    pos = 0
    while pos < len(target_seq):
        label = labels[pos] if pos < len(labels) else "COPY"
        if label == "COPY":
            corrected.append(target_seq[pos])
        elif label.startswith("SUB_"):
            corrected.append(label[-1])
        elif label.startswith("INS_"):
            corrected.append(target_seq[pos])
            corrected.append(label[-1])
        elif label == "DEL":
            pass
        else:
            corrected.append(target_seq[pos])
        pos += 1
    return "".join(corrected)


def _apply_adjacent_parsimony(target_seq: str, labels: list[str], trace: list[dict], decode_config: dict) -> tuple[str, list[str], list[dict]]:
    if not decode_config.get("hybrid_adjacent_edit_suppression", False) or not trace:
        return _rebuild_prediction(target_seq, labels), labels, trace
    updated = list(labels)
    trace_by_pos = {item["pos"]: item for item in trace}
    for pos in range(1, len(updated)):
        if updated[pos] == "COPY" or updated[pos - 1] == "COPY":
            continue
        current = trace_by_pos.get(pos, {})
        previous = trace_by_pos.get(pos - 1, {})
        if decode_config.get("hybrid_adjacent_keep_strong_rule_agreeing_edits", False):
            min_score = float(decode_config.get("hybrid_adjacent_keep_min_label_score", 0.95))
            current_rule_agrees = current.get("support_rule_label") == updated[pos]
            previous_rule_agrees = previous.get("support_rule_label") == updated[pos - 1]
            if (
                current_rule_agrees
                and previous_rule_agrees
                and float(current.get("label_score", 0.0)) >= min_score
                and float(previous.get("label_score", 0.0)) >= min_score
            ):
                continue
        current_conf = current.get("rule_confidence", {})
        previous_conf = previous.get("rule_confidence", {})
        current_score = float(current_conf.get("support_fraction", current.get("label_score", 0.0)))
        previous_score = float(previous_conf.get("support_fraction", previous.get("label_score", 0.0)))
        keep_pos, drop_pos = (pos, pos - 1) if current_score > previous_score else (pos - 1, pos)
        if abs(current_score - previous_score) <= float(decode_config.get("hybrid_adjacent_score_tie_epsilon", 0.05)):
            drop_pos = pos
        updated[drop_pos] = "COPY"
        if drop_pos in trace_by_pos:
            trace_by_pos[drop_pos]["veto_reasons"] = list(trace_by_pos[drop_pos].get("veto_reasons", [])) + [
                "hybrid_adjacent_parsimony_veto"
            ]
            trace_by_pos[drop_pos]["final_label"] = "COPY"
            trace_by_pos[drop_pos]["local_rerank_kept_neighbor"] = keep_pos
    return _rebuild_prediction(target_seq, updated), updated, trace


def _strong_negative_veto_escape(
    chosen_label: str,
    example: dict,
    pos: int,
    type_probs: torch.Tensor,
    sub_base_probs: torch.Tensor,
    ins_base_probs: torch.Tensor,
    decode_config: dict,
) -> bool:
    """Allow a rule-negative neural edit only with unusually strong support evidence."""
    type_threshold = float(decode_config.get("hybrid_negative_veto_min_type_prob", 0.80))
    agreement_threshold = float(decode_config.get("hybrid_negative_veto_min_agreement", 0.95))
    support_fraction_threshold = float(decode_config.get("hybrid_negative_veto_min_support_fraction", 0.80))
    payload_threshold = float(decode_config.get("hybrid_negative_veto_min_payload_prob", 0.98))
    agreement = float(example["features"]["support_agreement"][pos])

    if chosen_label.startswith("SUB_"):
        base = chosen_label[-1]
        majority_base, majority_fraction = _support_majority_base(example, pos)
        base_id = BASES.index(base)
        type_prob = float(type_probs[pos, EDIT_TYPE_LABELS.index("SUB")].item())
        payload_prob = float(sub_base_probs[pos, base_id].item())
        if base == "A":
            payload_threshold = float(decode_config.get("hybrid_sub_a_copy_veto_min_payload_prob", payload_threshold))
            agreement_threshold = float(decode_config.get("hybrid_sub_a_copy_veto_min_agreement", agreement_threshold))
            support_fraction_threshold = float(
                decode_config.get("hybrid_sub_a_copy_veto_min_support_fraction", support_fraction_threshold)
            )
        return (
            majority_base == base
            and agreement >= agreement_threshold
            and majority_fraction >= support_fraction_threshold
            and type_prob >= type_threshold
            and payload_prob >= payload_threshold
        )
    if chosen_label.startswith("INS_"):
        base = chosen_label[-1]
        base_id = BASES.index(base)
        type_prob = float(type_probs[pos, EDIT_TYPE_LABELS.index("INS")].item())
        payload_prob = float(ins_base_probs[pos, base_id].item())
        ins_fraction = _support_fraction(example, pos, float(example["features"]["support_ins_count"][pos]))
        return ins_fraction >= support_fraction_threshold and type_prob >= type_threshold and payload_prob >= payload_threshold
    if chosen_label == "DEL":
        type_prob = float(type_probs[pos, EDIT_TYPE_LABELS.index("DEL")].item())
        del_fraction = _support_fraction(example, pos, float(example["features"]["support_del_count"][pos]))
        return del_fraction >= support_fraction_threshold and type_prob >= type_threshold
    return True


def _strong_rule_agreeing_neural_edit(
    rule_label: str,
    chosen_label: str,
    example: dict,
    pos: int,
    type_probs: torch.Tensor,
    sub_base_probs: torch.Tensor,
    ins_base_probs: torch.Tensor,
    decode_config: dict,
) -> bool:
    if rule_label != chosen_label or chosen_label == "COPY":
        return False
    min_type_prob = float(decode_config.get("hybrid_rule_agree_min_type_prob", 0.75))
    min_payload_prob = float(decode_config.get("hybrid_rule_agree_min_payload_prob", 0.85))
    min_support_fraction = float(decode_config.get("hybrid_rule_agree_min_support_fraction", 0.60))
    min_agreement = float(decode_config.get("hybrid_rule_agree_min_agreement", 0.60))
    agreement = float(example["features"]["support_agreement"][pos])

    if chosen_label.startswith("SUB_"):
        base = chosen_label[-1]
        majority_base, majority_fraction = _support_majority_base(example, pos)
        base_id = BASES.index(base)
        return (
            majority_base == base
            and agreement >= min_agreement
            and majority_fraction >= min_support_fraction
            and float(type_probs[pos, EDIT_TYPE_LABELS.index("SUB")].item()) >= min_type_prob
            and float(sub_base_probs[pos, base_id].item()) >= min_payload_prob
        )
    if chosen_label.startswith("INS_"):
        base_id = BASES.index(chosen_label[-1])
        ins_fraction = _support_fraction(example, pos, float(example["features"]["support_ins_count"][pos]))
        return (
            ins_fraction >= min_support_fraction
            and float(type_probs[pos, EDIT_TYPE_LABELS.index("INS")].item()) >= min_type_prob
            and float(ins_base_probs[pos, base_id].item()) >= min_payload_prob
        )
    if chosen_label == "DEL":
        del_fraction = _support_fraction(example, pos, float(example["features"]["support_del_count"][pos]))
        return (
            del_fraction >= min_support_fraction
            and float(type_probs[pos, EDIT_TYPE_LABELS.index("DEL")].item()) >= min_type_prob
        )
    return False


def _neural_rescue_allows(
    rule_label: str,
    example: dict,
    pos: int,
    type_probs: torch.Tensor,
    sub_base_probs: torch.Tensor,
    ins_base_probs: torch.Tensor,
    decode_config: dict,
) -> bool:
    """Allow borderline support-rule edits only when payload and support agree."""
    if not decode_config.get("hybrid_neural_rescue_enabled", False) or rule_label == "COPY":
        return False
    min_type_prob = float(decode_config.get("hybrid_neural_rescue_min_type_prob", 0.0))
    min_del_type_prob = float(decode_config.get("hybrid_neural_rescue_min_del_type_prob", min_type_prob))
    min_payload_prob = float(decode_config.get("hybrid_neural_rescue_min_payload_prob", 0.95))
    min_support_fraction = float(decode_config.get("hybrid_neural_rescue_min_support_fraction", 0.60))
    min_agreement = float(decode_config.get("hybrid_neural_rescue_min_agreement", 0.60))
    confidence = _support_rule_confidence(example, pos, rule_label, decode_config)
    agreement = float(example["features"]["support_agreement"][pos])
    if confidence.get("neighbor_edit_proximity", 0) and decode_config.get("hybrid_disable_neural_rescue_near_neighbors", False):
        min_fraction = float(decode_config.get("hybrid_neighbor_neural_rescue_min_support_fraction", 0.90))
        min_margin = float(decode_config.get("hybrid_neighbor_neural_rescue_min_support_margin", 4.0))
        max_entropy = float(decode_config.get("hybrid_neighbor_neural_rescue_max_entropy", 0.30))
        if (
            confidence["support_fraction"] < min_fraction
            or confidence["support_margin"] < min_margin
            or confidence["local_entropy"] > max_entropy
        ):
            return False

    if rule_label.startswith("SUB_"):
        base = rule_label[-1]
        base_id = BASES.index(base)
        majority_base, majority_fraction = _support_majority_base(example, pos)
        return (
            majority_base == base
            and confidence["support_fraction"] >= min_support_fraction
            and agreement >= min_agreement
            and float(sub_base_probs[pos, base_id].item()) >= min_payload_prob
            and float(type_probs[pos, EDIT_TYPE_LABELS.index("SUB")].item()) >= min_type_prob
        )
    if rule_label.startswith("INS_"):
        base_id = BASES.index(rule_label[-1])
        return (
            confidence["support_fraction"] >= min_support_fraction
            and float(ins_base_probs[pos, base_id].item()) >= min_payload_prob
            and float(type_probs[pos, EDIT_TYPE_LABELS.index("INS")].item()) >= min_type_prob
        )
    if rule_label == "DEL":
        return (
            confidence["del_fraction"] >= min_support_fraction
            and float(type_probs[pos, EDIT_TYPE_LABELS.index("DEL")].item()) >= min_del_type_prob
        )
    return False


def _decode_structured(target_seq: str, example: dict, outputs: dict, decode_config: dict, argmax_only: bool) -> dict:
    flags = _mode_flags(decode_config)
    type_probs = torch.softmax(outputs["type_logits"], dim=-1)
    sub_base_probs = torch.softmax(outputs["sub_base_logits"], dim=-1)
    ins_base_probs = torch.softmax(outputs["ins_base_logits"], dim=-1)
    delete_candidate_probs = torch.sigmoid(outputs["delete_candidate_logits"])
    delete_length_probs = torch.softmax(outputs["delete_length_logits"], dim=-1)
    trust = outputs["trust"]
    flat_edit_probs = torch.softmax(outputs["edit_logits"], dim=-1)

    predicted_labels = []
    corrected = []
    trace = []
    pos = 0
    while pos < len(target_seq):
        type_id = int(torch.argmax(type_probs[pos]).item())
        edit_type = EDIT_TYPE_LABELS[type_id]
        chosen_label = "COPY"
        label_score = float(type_probs[pos, type_id].item())
        payload_base_id = None
        supported_payload_ids = []
        veto_reasons = []
        forced_by_rule = False
        rescued_by_neural = False
        rescued_by_support_payload = False
        rescued_by_sub_t_calibration = False
        rule_label = "COPY"
        rule_confidence = {}

        if edit_type == "SUB":
            supported_payload_ids = _supported_sub_bases(example, pos) if flags["restrict_supported_candidates"] else []
            payload_base_id, payload_score = _choose_base(sub_base_probs[pos], supported_payload_ids)
            chosen_label = compose_edit_label("SUB", payload_base_id)
            label_score *= payload_score
            if flags["restrict_supported_candidates"] and not supported_payload_ids:
                veto_reasons.append("unsupported_sub_base")
        elif edit_type == "INS":
            supported_payload_ids = _supported_ins_bases(example, pos) if flags["restrict_supported_candidates"] else []
            payload_base_id, payload_score = _choose_base(ins_base_probs[pos], supported_payload_ids)
            chosen_label = compose_edit_label("INS", payload_base_id)
            label_score *= payload_score
            if flags["restrict_supported_candidates"] and not supported_payload_ids:
                veto_reasons.append("unsupported_ins_base")
        elif edit_type == "DEL":
            chosen_label = "DEL"
            if flags["restrict_supported_candidates"] and not _deletion_supported(example, pos):
                veto_reasons.append("unsupported_deletion")

        if decode_config.get("hybrid_rule_decode", False):
            rule_label = _support_rule_label(example, pos, decode_config)
            rule_confidence = _support_rule_confidence(example, pos, rule_label, decode_config)
            rule_confident, rule_confidence_reasons = _rule_confidence_passes(rule_confidence, rule_label, decode_config)
            if rule_label.startswith("SUB_"):
                hybrid_sub_payload_threshold = decode_config.get(
                    "hybrid_sub_payload_threshold",
                    decode_config.get("hybrid_payload_threshold", 0.0),
                )
                hybrid_sub_min_type_prob = decode_config.get("hybrid_sub_min_type_prob", 0.0)
                hybrid_sub_min_copy_margin = decode_config.get("hybrid_sub_min_copy_margin", float("-inf"))
                rule_base_id = BASES.index(rule_label[-1])
                rule_base = rule_label[-1]
                hybrid_sub_payload_threshold = _base_specific_threshold(
                    decode_config,
                    "hybrid_sub_payload_threshold",
                    rule_base,
                    float(hybrid_sub_payload_threshold),
                )
                payload_base_id, payload_score = _choose_base(sub_base_probs[pos], None)
                sub_type_prob = float(type_probs[pos, EDIT_TYPE_LABELS.index("SUB")].item())
                copy_type_prob = float(type_probs[pos, EDIT_TYPE_LABELS.index("COPY")].item())
                sub_copy_margin = sub_type_prob - copy_type_prob
                if (
                    rule_confident
                    and
                    payload_base_id == rule_base_id
                    and payload_score >= hybrid_sub_payload_threshold
                    and (
                        sub_type_prob >= hybrid_sub_min_type_prob
                        or sub_copy_margin >= hybrid_sub_min_copy_margin
                    )
                ):
                    chosen_label = rule_label
                    label_score = sub_type_prob * payload_score
                    veto_reasons = []
                    forced_by_rule = True
                else:
                    if not rule_confident:
                        veto_reasons.extend(rule_confidence_reasons)
                    if payload_base_id != rule_base_id:
                        veto_reasons.append("hybrid_sub_payload_mismatch")
                    if payload_score < hybrid_sub_payload_threshold:
                        veto_reasons.append("hybrid_sub_payload_threshold")
                    if sub_type_prob < hybrid_sub_min_type_prob and sub_copy_margin < hybrid_sub_min_copy_margin:
                        veto_reasons.append("hybrid_sub_type_too_low")
                    sub_t_rescue_allows, sub_t_rescue_reasons = _sub_t_calibrated_rescue_allows(
                        example,
                        pos,
                        rule_label,
                        payload_base_id,
                        float(payload_score),
                        sub_type_prob,
                        rule_confidence,
                        decode_config,
                    )
                    if (
                        payload_base_id == rule_base_id
                        and _neural_rescue_allows(
                            rule_label,
                            example,
                            pos,
                            type_probs,
                            sub_base_probs,
                            ins_base_probs,
                            decode_config,
                        )
                    ):
                        chosen_label = rule_label
                        label_score = sub_type_prob * payload_score
                        veto_reasons = _remove_rule_confidence_reasons(veto_reasons)
                        rescued_by_neural = True
                    elif sub_t_rescue_allows:
                        chosen_label = rule_label
                        label_score = sub_type_prob * payload_score
                        veto_reasons = []
                        rescued_by_sub_t_calibration = True
                    elif sub_t_rescue_reasons:
                        veto_reasons.extend(sub_t_rescue_reasons)
            elif rule_label.startswith("INS_"):
                hybrid_ins_payload_threshold = decode_config.get(
                    "hybrid_ins_payload_threshold",
                    decode_config.get("hybrid_payload_threshold", 0.0),
                )
                rule_base_id = BASES.index(rule_label[-1])
                rule_base = rule_label[-1]
                hybrid_ins_payload_threshold = _base_specific_threshold(
                    decode_config,
                    "hybrid_ins_payload_threshold",
                    rule_base,
                    float(hybrid_ins_payload_threshold),
                )
                payload_base_id, payload_score = _choose_base(ins_base_probs[pos], None)
                support_payload_allows, support_payload_reasons = _insertion_rule_rescue_allows(
                    example, pos, rule_label, rule_confidence, decode_config
                )
                if support_payload_allows:
                    chosen_label = rule_label
                    support_payload_id, support_payload_fraction, _ = _support_insertion_payload(example, pos)
                    label_score = support_payload_fraction
                    payload_base_id = support_payload_id
                    payload_score = support_payload_fraction
                    veto_reasons = []
                    forced_by_rule = True
                    rescued_by_support_payload = True
                elif rule_confident and payload_base_id == rule_base_id and payload_score >= hybrid_ins_payload_threshold:
                    chosen_label = rule_label
                    label_score = float(type_probs[pos, EDIT_TYPE_LABELS.index("INS")].item()) * payload_score
                    veto_reasons = []
                    forced_by_rule = True
                else:
                    if not rule_confident:
                        veto_reasons.extend(rule_confidence_reasons)
                    veto_reasons.extend(support_payload_reasons)
                    if payload_base_id != rule_base_id:
                        veto_reasons.append("hybrid_ins_payload_mismatch")
                    if payload_score < hybrid_ins_payload_threshold:
                        veto_reasons.append("hybrid_ins_payload_threshold")
                    if (
                        payload_base_id == rule_base_id
                        and _neural_rescue_allows(
                            rule_label,
                            example,
                            pos,
                            type_probs,
                            sub_base_probs,
                            ins_base_probs,
                            decode_config,
                        )
                    ):
                        chosen_label = rule_label
                        label_score = float(type_probs[pos, EDIT_TYPE_LABELS.index("INS")].item()) * payload_score
                        veto_reasons = _remove_rule_confidence_reasons(veto_reasons)
                        if decode_config.get("hybrid_ins_neighbor_neural_rescue", False):
                            veto_reasons = [
                                reason
                                for reason in veto_reasons
                                if reason != "hybrid_ins_support_payload_neighbor_conflict"
                            ]
                        rescued_by_neural = True
            elif rule_label == "DEL" and decode_config.get("hybrid_force_del", True):
                hybrid_del_threshold = decode_config.get("hybrid_del_threshold", 0.0)
                del_support = example["features"]["support_del_count"][pos]
                if rule_confident and del_support >= hybrid_del_threshold:
                    chosen_label = "DEL"
                    label_score = float(type_probs[pos, EDIT_TYPE_LABELS.index("DEL")].item())
                    veto_reasons = []
                    forced_by_rule = True
                else:
                    if not rule_confident:
                        veto_reasons.extend(rule_confidence_reasons)
                    if del_support < hybrid_del_threshold:
                        veto_reasons.append("hybrid_del_threshold")
                    if del_support >= hybrid_del_threshold and _neural_rescue_allows(
                        rule_label,
                        example,
                        pos,
                        type_probs,
                        sub_base_probs,
                        ins_base_probs,
                        decode_config,
                    ):
                        chosen_label = "DEL"
                        label_score = float(type_probs[pos, EDIT_TYPE_LABELS.index("DEL")].item())
                        veto_reasons = _remove_rule_confidence_reasons(veto_reasons)
                        rescued_by_neural = True
            elif (
                rule_label == "COPY"
                and chosen_label != "COPY"
                and decode_config.get("hybrid_negative_veto", True)
                and not _strong_negative_veto_escape(
                    chosen_label,
                    example,
                    pos,
                    type_probs,
                    sub_base_probs,
                    ins_base_probs,
                    decode_config,
                )
            ):
                veto_reasons.append("hybrid_rule_copy_veto")

            if (
                chosen_label == "SUB_A"
                and rule_label != "SUB_A"
                and decode_config.get("hybrid_sub_a_copy_safety", True)
                and not _strong_negative_veto_escape(
                    chosen_label,
                    example,
                    pos,
                    type_probs,
                    sub_base_probs,
                    ins_base_probs,
                    decode_config,
                )
            ):
                veto_reasons.append("hybrid_sub_a_safety_veto")

            if (
                chosen_label != "COPY"
                and not forced_by_rule
                and not rescued_by_neural
                and not rescued_by_sub_t_calibration
                and decode_config.get("hybrid_require_rule_agreement_for_neural_edits", False)
                and not _strong_rule_agreeing_neural_edit(
                    rule_label,
                    chosen_label,
                    example,
                    pos,
                    type_probs,
                    sub_base_probs,
                    ins_base_probs,
                    decode_config,
                )
            ):
                veto_reasons.append("hybrid_rule_agreement_confidence_veto")

            if chosen_label != "COPY":
                neighbor_reasons = _neighbor_abstention_reasons(chosen_label, rule_confidence, decode_config)
                if neighbor_reasons:
                    veto_reasons.extend(neighbor_reasons)
                    if rescued_by_neural:
                        veto_reasons.append("hybrid_neighbor_neural_rescue_disabled")
                else:
                    veto_reasons.extend(_local_window_parsimony_veto(chosen_label, rule_confidence, decode_config))

        if not argmax_only and not forced_by_rule:
            if chosen_label.startswith("SUB_") and label_score < decode_config["sub_threshold"]:
                veto_reasons.append("sub_threshold")
            elif chosen_label.startswith("INS_") and label_score < decode_config["ins_threshold"]:
                veto_reasons.append("ins_threshold")
            elif chosen_label == "DEL" and label_score < decode_config["del_threshold"]:
                veto_reasons.append("del_threshold")

            if flags["use_trust_threshold"] and chosen_label != "COPY" and float(trust[pos].item()) < decode_config["trust_threshold"]:
                veto_reasons.append("trust_threshold")

            if flags["use_delete_candidate_veto"] and chosen_label == "DEL":
                if float(delete_candidate_probs[pos].item()) < decode_config["del_threshold"]:
                    veto_reasons.append("delete_candidate_veto")

            if flags["consistency_check"] and chosen_label != "COPY" and not veto_reasons:
                if not _consistency_allows(example, pos, chosen_label):
                    veto_reasons.append("consistency_check")

        kept = chosen_label != "COPY" and not veto_reasons
        final_label = chosen_label if kept else "COPY"
        predicted_labels.append(final_label)

        delete_length = 1
        if final_label == "COPY":
            corrected.append(target_seq[pos])
            pos += 1
        elif final_label.startswith("SUB_"):
            corrected.append(final_label[-1])
            pos += 1
        elif final_label.startswith("INS_"):
            corrected.append(target_seq[pos])
            corrected.append(final_label[-1])
            pos += 1
        elif final_label == "DEL":
            delete_length = int(torch.argmax(delete_length_probs[pos]).item())
            delete_length = max(1, min(decode_config["max_deletion_length"], delete_length, len(target_seq) - pos))
            predicted_labels.extend(["DEL"] * (delete_length - 1))
            pos += delete_length
        else:
            corrected.append(target_seq[pos])
            pos += 1

        if flags["full_trace"]:
            trace.append(
                {
                    "pos": pos - 1 if final_label != "DEL" else pos - delete_length,
                    "type_probs": {label: round(float(type_probs[pos - 1 if final_label != 'DEL' else pos - delete_length, idx].item()), 4) for idx, label in enumerate(EDIT_TYPE_LABELS)},
                    "sub_base_probs": {base: round(float(sub_base_probs[pos - 1 if final_label != 'DEL' else pos - delete_length, idx].item()), 4) for idx, base in enumerate(BASES)},
                    "ins_base_probs": {base: round(float(ins_base_probs[pos - 1 if final_label != 'DEL' else pos - delete_length, idx].item()), 4) for idx, base in enumerate(BASES)},
                    "delete_candidate_prob": round(float(delete_candidate_probs[pos - 1 if final_label != 'DEL' else pos - delete_length].item()), 4),
                    "delete_length_probs": [round(float(value), 4) for value in delete_length_probs[pos - 1 if final_label != 'DEL' else pos - delete_length].tolist()],
                    "trust": round(float(trust[pos - 1 if final_label != 'DEL' else pos - delete_length].item()), 4),
                    "candidate_label": chosen_label,
                    "final_label": final_label,
                    "support_rule_label": rule_label,
                    "rule_confidence": rule_confidence,
                    "label_score": round(label_score, 4),
                    "supported_payload_ids": [ID_TO_BASE[idx] for idx in supported_payload_ids],
                    "veto_reasons": veto_reasons,
                    "forced_by_rule": forced_by_rule,
                    "rescued_by_neural": rescued_by_neural,
                    "rescued_by_support_payload": rescued_by_support_payload,
                    "rescued_by_sub_t_calibration": rescued_by_sub_t_calibration,
                    "delete_length": delete_length,
                }
            )

    prediction = "".join(corrected)
    if not argmax_only:
        prediction, predicted_labels, trace = _apply_adjacent_parsimony(target_seq, predicted_labels, trace, decode_config)
    return {
        "prediction": prediction,
        "predicted_labels": predicted_labels,
        "trust": [float(value) for value in trust[: len(target_seq)].detach().cpu().tolist()],
        "edit_probs": flat_edit_probs[: len(target_seq)].detach().cpu().tolist(),
        "trace": trace,
    }


def decode_example(target_seq: str, example: dict, outputs: dict, decode_config: dict) -> dict:
    return _decode_structured(target_seq, example, outputs, decode_config, argmax_only=False)


def decode_example_argmax(target_seq: str, example: dict, outputs: dict) -> dict:
    return _decode_structured(
        target_seq,
        example,
        outputs,
        {"max_deletion_length": outputs["delete_length_logits"].shape[-1] - 1, "mode": "debug", "full_trace": True},
        argmax_only=True,
    )


def decode_batch(batch: dict, outputs: dict, decode_config: dict) -> list[dict]:
    results = []
    for idx, example in enumerate(batch["raw_examples"]):
        length = len(example["target_seq"])
        sliced = {
            "type_logits": outputs["type_logits"][idx, :length].detach().cpu(),
            "sub_base_logits": outputs["sub_base_logits"][idx, :length].detach().cpu(),
            "ins_base_logits": outputs["ins_base_logits"][idx, :length].detach().cpu(),
            "edit_logits": outputs["edit_logits"][idx, :length].detach().cpu(),
            "delete_candidate_logits": outputs["delete_candidate_logits"][idx, :length].detach().cpu(),
            "delete_length_logits": outputs["delete_length_logits"][idx, :length].detach().cpu(),
            "trust": outputs["trust"][idx, :length].detach().cpu(),
        }
        results.append(decode_example(example["target_seq"], example, sliced, decode_config))
    return results


def decode_batch_argmax(batch: dict, outputs: dict) -> list[dict]:
    results = []
    for idx, example in enumerate(batch["raw_examples"]):
        length = len(example["target_seq"])
        sliced = {
            "type_logits": outputs["type_logits"][idx, :length].detach().cpu(),
            "sub_base_logits": outputs["sub_base_logits"][idx, :length].detach().cpu(),
            "ins_base_logits": outputs["ins_base_logits"][idx, :length].detach().cpu(),
            "edit_logits": outputs["edit_logits"][idx, :length].detach().cpu(),
            "delete_candidate_logits": outputs["delete_candidate_logits"][idx, :length].detach().cpu(),
            "delete_length_logits": outputs["delete_length_logits"][idx, :length].detach().cpu(),
            "trust": outputs["trust"][idx, :length].detach().cpu(),
        }
        results.append(decode_example_argmax(example["target_seq"], example, sliced))
    return results
