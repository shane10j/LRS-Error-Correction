"""Structured conservative and debug decoding."""

from __future__ import annotations

import torch

from omega_lr.constants import BASES, EDIT_LABELS, EDIT_TYPE_LABELS, ID_TO_BASE, compose_edit_label


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
    supported = [idx for idx, count in enumerate(example["features"]["support_base_counts"][pos]) if count > 0]
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
        rule_label = "COPY"

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
                    if payload_base_id != rule_base_id:
                        veto_reasons.append("hybrid_sub_payload_mismatch")
                    if payload_score < hybrid_sub_payload_threshold:
                        veto_reasons.append("hybrid_sub_payload_threshold")
                    if sub_type_prob < hybrid_sub_min_type_prob and sub_copy_margin < hybrid_sub_min_copy_margin:
                        veto_reasons.append("hybrid_sub_type_too_low")
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
                if payload_base_id == rule_base_id and payload_score >= hybrid_ins_payload_threshold:
                    chosen_label = rule_label
                    label_score = float(type_probs[pos, EDIT_TYPE_LABELS.index("INS")].item()) * payload_score
                    veto_reasons = []
                    forced_by_rule = True
                else:
                    if payload_base_id != rule_base_id:
                        veto_reasons.append("hybrid_ins_payload_mismatch")
                    if payload_score < hybrid_ins_payload_threshold:
                        veto_reasons.append("hybrid_ins_payload_threshold")
            elif rule_label == "DEL" and decode_config.get("hybrid_force_del", True):
                hybrid_del_threshold = decode_config.get("hybrid_del_threshold", 0.0)
                del_support = example["features"]["support_del_count"][pos]
                if del_support >= hybrid_del_threshold:
                    chosen_label = "DEL"
                    label_score = float(type_probs[pos, EDIT_TYPE_LABELS.index("DEL")].item())
                    veto_reasons = []
                    forced_by_rule = True
                else:
                    veto_reasons.append("hybrid_del_threshold")
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
                    "label_score": round(label_score, 4),
                    "supported_payload_ids": [ID_TO_BASE[idx] for idx in supported_payload_ids],
                    "veto_reasons": veto_reasons,
                    "forced_by_rule": forced_by_rule,
                    "delete_length": delete_length,
                }
            )

    return {
        "prediction": "".join(corrected),
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
