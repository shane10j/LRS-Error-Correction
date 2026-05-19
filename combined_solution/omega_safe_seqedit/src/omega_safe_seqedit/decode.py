"""Neural, rule, and hybrid conservative decoding."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from omega_safe_seqedit.constants import BASES, ID_TO_MAIN, ID_TO_RULE, INS_TO_ID, MAIN_TO_ID, RULE_TO_ID


_ALLOW_GATE_CACHE: dict[str, dict] = {}
_SUB_ALLOWLIST_CACHE: dict[str, set[str]] = {}


def _rule_label(example: dict, pos: int) -> tuple[str, str | None]:
    rule_type = ID_TO_RULE[example["features"]["support_rule_type"][pos]]
    if rule_type == "SUB":
        return "SUB", BASES[example["features"]["support_rule_sub_base"][pos]]
    if rule_type == "INS":
        return "INS", BASES[example["features"]["support_rule_ins_base"][pos]]
    return rule_type, None


def _insertion_payload_stats(ins_base_counts: list[int | float], support_ins_count: float) -> dict:
    """Normalize insertion payload evidence so reported fractions are bounded by 1."""
    raw = [max(float(x), 0.0) for x in ins_base_counts]
    support_total = max(float(support_ins_count), 0.0)
    raw_total = sum(raw)
    if raw_total <= 0.0 or support_total <= 0.0:
        effective = [0.0, 0.0, 0.0, 0.0]
    elif raw_total > support_total:
        scale = support_total / raw_total
        effective = [x * scale for x in raw]
    else:
        effective = raw
    ordered = sorted(effective, reverse=True)
    top = ordered[0] if ordered else 0.0
    second = ordered[1] if len(ordered) > 1 else 0.0
    return {
        "effective_counts": effective,
        "top_count": top,
        "top_fraction": top / max(support_total, 1.0),
        "margin": top - second,
        "raw_total": raw_total,
    }


def _confidence(example: dict, pos: int, edit_type: str) -> dict:
    f = example["features"]
    depth = max(float(f["support_depth"][pos]), 1.0)
    support_value = max(f["support_base_counts"][pos]) if edit_type == "SUB" else f["support_ins_count"][pos] if edit_type == "INS" else f["support_del_count"][pos]
    def get_flag(key: str, default: int = 0) -> int:
        values = f.get(key)
        if values is None or pos >= len(values):
            return default
        return int(values[pos])
    def get_scalar(key: str, default: float = 0.0) -> float:
        values = f.get(key)
        if values is None or pos >= len(values):
            return default
        return float(values[pos])
    homopolymer = f["homopolymer_run_length"][pos] >= 4 or bool(get_flag("region_homopolymer_flag"))
    tandem = bool(f["tandem_repeat_flag"][pos]) or bool(get_flag("region_tandem_repeat_flag"))
    repeat = tandem or homopolymer
    low_confidence = bool(get_flag("preserve_mask")) or bool(get_flag("uncertainty_label")) or not bool(get_flag("confident_mask", 1))
    ins_base_counts = f.get("support_ins_base_counts", [[0, 0, 0, 0] for _ in example["target_seq"]])[pos]
    ins_stats = _insertion_payload_stats(ins_base_counts, float(f["support_ins_count"][pos]))
    return {
        "fraction": float(support_value) / depth,
        "margin": float(f["support_margin"][pos]) / depth,
        "raw_margin": float(f["support_margin"][pos]),
        "entropy": float(f["support_entropy"][pos]),
        "depth": depth,
        "support_del_count": float(f["support_del_count"][pos]),
        "support_ins_count": float(f["support_ins_count"][pos]),
        "top_inserted_base_count": ins_stats["top_count"],
        "top_inserted_base_fraction": min(ins_stats["top_fraction"], 1.0),
        "inserted_base_margin": ins_stats["margin"],
        "raw_inserted_base_total": ins_stats["raw_total"],
        "neighbor": bool(f["neighbor_rule_flag"][pos]),
        "boundary": bool(f["boundary_flag"][pos]),
        "homopolymer": homopolymer,
        "homopolymer_run_length": int(f["homopolymer_run_length"][pos]),
        "tandem_repeat": tandem,
        "repeat": repeat,
        "variant": bool(get_flag("variant_mask")),
        "phased_variant": bool(get_flag("phased_variant_mask")),
        "preserve": bool(get_flag("preserve_mask")),
        "variant_rich": bool(get_flag("variant_rich_flag")),
        "low_confidence": low_confidence,
        "variant_proximity": bool(get_flag("variant_proximity_flag")),
        "local_variant_density": get_scalar("local_variant_density"),
        "window_relative_position": get_scalar("window_relative_position"),
        "local_mismatch_density": get_scalar("local_mismatch_density"),
        "nearby_indel_density": get_scalar("nearby_indel_density"),
        "left_support_match_fraction": get_scalar("left_support_match_fraction"),
        "right_support_match_fraction": get_scalar("right_support_match_fraction"),
        "support_forward_fraction": get_scalar("support_forward_fraction", 0.5),
        "support_forward_count": get_scalar("support_forward_count"),
        "support_reverse_count": get_scalar("support_reverse_count"),
        "support_strand_bias": min(max(get_scalar("support_strand_bias"), 0.0), 1.0),
        "support_same_haplotype_fraction": get_scalar("support_same_haplotype_fraction"),
        "support_match_fraction": get_scalar("support_match_fraction"),
        "repeat_strength": get_scalar("repeat_strength"),
        "mapping_quality_mean": get_scalar("mapping_quality_mean"),
        "mapping_quality_available": bool(get_flag("mapping_quality_available")),
        "reference_kmer_uniqueness": get_scalar("reference_kmer_uniqueness"),
        "reference_kmer_uniqueness_available": bool(get_flag("reference_kmer_uniqueness_available")),
    }


def _passes_rule_force(conf: dict, edit_type: str, config: dict) -> tuple[bool, list[str]]:
    decode = config["decode"]
    min_frac = float(decode.get("rule_force_min_fraction", {}).get(edit_type, 0.85))
    max_entropy = float(decode.get("rule_force_max_entropy", {}).get(edit_type, 0.85))
    if conf["neighbor"] and decode.get("neighbor_abstention", True):
        min_frac = max(min_frac, float(decode.get("neighbor_min_fraction", 0.90)))
    if edit_type == "DEL" and conf["homopolymer"]:
        min_frac = max(min_frac, float(decode.get("homopolymer_del_min_fraction", 0.95)))
    if decode.get("variant_aware_abstention", False) and (
        conf["variant"] or conf["phased_variant"] or conf["preserve"] or conf["variant_rich"] or conf["low_confidence"]
    ):
        min_frac = max(min_frac, float(decode.get("variant_min_fraction", 0.98)))
        max_entropy = min(max_entropy, float(decode.get("variant_max_entropy", 0.35)))
    if decode.get("repeat_aware_abstention", False) and (conf["tandem_repeat"] or conf["homopolymer"]):
        min_frac = max(min_frac, float(decode.get("repeat_min_fraction", 0.95)))
        max_entropy = min(max_entropy, float(decode.get("repeat_max_entropy", 0.45)))
    if edit_type == "DEL" and decode.get("strict_del_in_ambiguous_context", True):
        if conf["neighbor"] or conf["homopolymer"] or conf["tandem_repeat"] or conf["variant_rich"]:
            min_frac = max(min_frac, float(decode.get("ambiguous_del_min_fraction", 0.985)))
    reasons = []
    if conf["fraction"] < min_frac:
        reasons.append("low_support_fraction")
    if conf["entropy"] > max_entropy:
        reasons.append("high_entropy")
    if edit_type in {"SUB", "DEL"} and conf["neighbor"] and conf["fraction"] < min_frac:
        reasons.append("neighbor_abstention")
    return not reasons, reasons


def _local_rule_density(example: dict, pos: int, radius: int) -> int:
    rules = example["features"]["support_rule_type"]
    start = max(0, pos - radius)
    end = min(len(rules), pos + radius + 1)
    return sum(1 for idx in range(start, end) if idx != pos and ID_TO_RULE[rules[idx]] != "COPY")


def _support_base_fraction(example: dict, pos: int, base: str) -> float:
    f = example["features"]
    depth = max(float(f["support_depth"][pos]), 1.0)
    return float(f["support_base_counts"][pos][BASES.index(base)]) / depth


def _sub_local_window_accepts(example: dict, pos: int, sub_base: str, config: dict) -> tuple[bool, dict]:
    """Parsimony-biased local support score for a candidate substitution."""
    decode = config["decode"]
    radius = int(decode.get("sub_local_rerank_radius", 2))
    edit_penalty = float(decode.get("sub_local_edit_penalty", 0.08))
    min_gain = float(decode.get("sub_local_min_gain", 0.05))
    f = example["features"]
    target = example["target_seq"]
    start = max(0, pos - radius)
    end = min(len(target), pos + radius + 1)
    before = 0.0
    after = 0.0
    for idx in range(start, end):
        depth = max(float(f["support_depth"][idx]), 1.0)
        before += float(f["support_base_counts"][idx][BASES.index(target[idx])]) / depth
        if idx == pos:
            after += float(f["support_base_counts"][idx][BASES.index(sub_base)]) / depth
        else:
            after += float(f["support_base_counts"][idx][BASES.index(target[idx])]) / depth
    gain = after - before - edit_penalty
    density = _local_rule_density(example, pos, radius)
    if density:
        gain -= float(decode.get("sub_neighbor_density_penalty", 0.08)) * density
    return gain > min_gain, {"local_gain": gain, "local_rule_density": density, "radius": radius}


def _safe_true_sub_recovery_passes(
    example: dict,
    pos: int,
    rule_base: str,
    conf: dict,
    main_probs: torch.Tensor,
    sub_probs: torch.Tensor,
    details: dict,
    config: dict,
) -> tuple[bool, list[str]]:
    decode = config["decode"]
    if not decode.get("safe_true_edit_recovery", False):
        return False, []
    reasons: list[str] = []
    candidate_id = _sub_candidate_id(example, pos, rule_base)
    details["safe_recovery_candidate_id"] = candidate_id
    allowlist = _load_sub_recovery_allowlist(config)
    if allowlist is not None:
        details["ranked_sub_recovery_allowlisted"] = candidate_id in allowlist
        if candidate_id not in allowlist:
            reasons.append("ranked_sub_recovery_not_allowlisted")
    if "SUB" not in set(decode.get("safe_recovery_edit_types", ["SUB"])):
        reasons.append("recovery_sub_disabled")
    type_prob = float(main_probs[pos, MAIN_TO_ID["SUB"]])
    payload_prob = float(sub_probs[pos, BASES.index(rule_base)])
    details["safe_recovery_sub_type_prob"] = type_prob
    details["safe_recovery_sub_payload_prob"] = payload_prob
    if decode.get("ranked_sub_recovery_mode", False):
        if (conf.get("support_ins_count", 0.0) + conf.get("support_del_count", 0.0)) > float(
            decode.get("ranked_sub_max_site_indel_evidence", 0.0)
        ):
            reasons.append("ranked_sub_site_indel_evidence")
        # Ranked recovery allowlists are produced by scripts/rank_sub_recovery_candidates.py
        # with a wider local-window scorer. Avoid silently re-vetoing those candidates
        # with the older small-radius decoder reranker unless explicitly requested.
        if decode.get("ranked_sub_runtime_local_gain_check", False) and details.get("local_gain", 0.0) < float(
            decode.get("ranked_sub_recovery_min_local_gain", 0.25)
        ):
            reasons.append("ranked_sub_local_gain_too_low")
        if int(sub_probs[pos].argmax()) != BASES.index(rule_base):
            reasons.append("ranked_sub_payload_argmax_disagrees")
        if payload_prob < float(decode.get("ranked_sub_min_payload_prob", 0.0)):
            reasons.append("ranked_sub_payload_prob_too_low")
        return not reasons, reasons
    if conf["repeat"] or conf["tandem_repeat"] or conf["homopolymer"]:
        reasons.append("recovery_repeat_or_homopolymer")
    if conf["neighbor"]:
        reasons.append("recovery_neighbor")
    if conf["boundary"]:
        reasons.append("recovery_boundary")
    if conf["variant"] or conf["phased_variant"] or conf["variant_rich"] or conf["low_confidence"] or conf["preserve"]:
        reasons.append("recovery_variant_or_low_confidence")
    if conf["depth"] < float(decode.get("safe_recovery_min_depth", 8)):
        reasons.append("recovery_depth_too_low")
    if conf["fraction"] < float(decode.get("safe_recovery_min_fraction", 0.99)):
        reasons.append("recovery_fraction_too_low")
    if conf["raw_margin"] < float(decode.get("safe_recovery_min_margin", 6)):
        reasons.append("recovery_margin_too_low")
    if conf["entropy"] > float(decode.get("safe_recovery_max_entropy", 0.02)):
        reasons.append("recovery_entropy_too_high")
    if details.get("local_rule_density", 0) > 0:
        reasons.append("recovery_local_density")
    if details.get("local_gain", 0.0) < float(decode.get("safe_recovery_min_local_gain", -999.0)):
        reasons.append("recovery_local_gain_too_low")
    if conf.get("local_mismatch_density", 0.0) > float(decode.get("safe_recovery_max_local_mismatch_density", 1.0)):
        reasons.append("recovery_local_mismatch_density_too_high")
    min_flank = float(decode.get("safe_recovery_min_flank_match_fraction", 0.0))
    if conf.get("left_support_match_fraction", 0.0) < min_flank or conf.get("right_support_match_fraction", 0.0) < min_flank:
        reasons.append("recovery_flank_match_fraction_too_low")
    if conf.get("support_strand_bias", 0.0) > float(decode.get("safe_recovery_max_strand_bias", 1.0)):
        reasons.append("recovery_strand_bias_too_high")
    if type_prob < float(decode.get("safe_recovery_sub_min_type_prob", 0.45)):
        reasons.append("recovery_type_prob_too_low")
    if payload_prob < float(decode.get("safe_recovery_sub_min_payload_prob", 0.80)):
        reasons.append("recovery_payload_prob_too_low")
    if int(sub_probs[pos].argmax()) != BASES.index(rule_base):
        reasons.append("recovery_payload_argmax_disagrees")
    gate_ok, gate_score = _allow_gate_says_safe(
        config,
        "SUB",
        conf,
        {
            "type_prob_sub": type_prob,
            "payload_prob": payload_prob,
        },
        details,
    )
    details["safe_recovery_allow_gate_score"] = gate_score
    if decode.get("sub_recovery_require_allow_gate", True) and not gate_ok:
        reasons.append("recovery_sub_allow_gate_reject")
    return not reasons, reasons


def _indel_local_window_accepts(example: dict, pos: int, edit_type: str, config: dict) -> tuple[bool, dict]:
    """Small indel-only reranker for ambiguous repeat/neighbor/boundary sites."""
    decode = config["decode"]
    radius = int(decode.get("indel_local_rerank_radius", 2))
    min_gain = float(decode.get("indel_local_min_gain", 0.18))
    density_penalty = float(decode.get("indel_neighbor_density_penalty", 0.08))
    edit_penalty = float(decode.get("indel_local_edit_penalty", 0.06))
    f = example["features"]
    start = max(0, pos - radius)
    end = min(len(example["target_seq"]), pos + radius + 1)
    density = _local_rule_density(example, pos, radius)
    local_entropy = sum(float(f["support_entropy"][idx]) for idx in range(start, end)) / max(end - start, 1)
    depth = max(float(f["support_depth"][pos]), 1.0)
    if edit_type == "INS":
        support_gain = float(f["support_ins_count"][pos]) / depth
    else:
        support_gain = float(f["support_del_count"][pos]) / depth
    gain = support_gain - edit_penalty - density_penalty * density - 0.05 * local_entropy
    return gain >= min_gain, {
        "local_gain": gain,
        "local_rule_density": density,
        "local_entropy": local_entropy,
        "radius": radius,
    }


def _load_allow_gate(config: dict) -> dict | None:
    path = config.get("decode", {}).get("learned_allow_gate_path")
    if not path:
        return None
    gate_path = Path(path)
    if not gate_path.exists():
        return None
    key = str(gate_path.resolve())
    if key not in _ALLOW_GATE_CACHE:
        _ALLOW_GATE_CACHE[key] = json.loads(gate_path.read_text(encoding="utf-8"))
    return _ALLOW_GATE_CACHE[key]


def _load_sub_recovery_allowlist(config: dict) -> set[str] | None:
    path = config.get("decode", {}).get("ranked_sub_recovery_allowlist_path")
    if not path:
        return None
    allowlist_path = Path(path)
    if not allowlist_path.exists():
        return set()
    key = str(allowlist_path.resolve())
    if key not in _SUB_ALLOWLIST_CACHE:
        payload = json.loads(allowlist_path.read_text(encoding="utf-8"))
        ids = payload.get("candidate_ids", payload if isinstance(payload, list) else [])
        _SUB_ALLOWLIST_CACHE[key] = {str(item) for item in ids}
    return _SUB_ALLOWLIST_CACHE[key]


def _sub_candidate_id(example: dict, pos: int, base: str) -> str:
    return f"{example['example_id']}:{pos}:support_rule:SUB_{base}"


def _gate_feature_value(name: str, conf: dict, probs: dict, details: dict) -> float:
    depth = max(float(conf.get("depth", 1.0)), 1.0)
    if name == "support_fraction":
        return float(conf.get("fraction", 0.0))
    if name == "support_margin_fraction":
        return min(float(conf.get("raw_margin", 0.0)) / depth, 1.0)
    if name == "entropy_inverse":
        return 1.0 - min(float(conf.get("entropy", 0.0)), 1.0)
    if name == "payload_prob":
        return float(probs.get("payload_prob", 0.5))
    if name == "top_inserted_base_fraction":
        return float(conf.get("top_inserted_base_fraction", 0.0))
    if name == "inserted_base_margin_fraction":
        return min(float(conf.get("inserted_base_margin", 0.0)) / depth, 1.0)
    if name == "type_prob_sub":
        return float(probs.get("type_prob_sub", 0.0))
    if name == "type_prob_del":
        return float(probs.get("type_prob_del", 0.0))
    if name == "allow_prob":
        return float(probs.get("allow_prob", 0.5))
    if name in {"repeat_flag", "tandem_repeat_flag", "homopolymer_flag", "neighbor_rule_flag", "boundary_flag"}:
        key = {
            "repeat_flag": "repeat",
            "tandem_repeat_flag": "tandem_repeat",
            "homopolymer_flag": "homopolymer",
            "neighbor_rule_flag": "neighbor",
            "boundary_flag": "boundary",
        }[name]
        return 1.0 if conf.get(key) else 0.0
    if name == "truth_vcf_overlap":
        return 1.0 if conf.get("variant") or conf.get("phased_variant") else 0.0
    if name == "variant_rich_flag":
        return 1.0 if conf.get("variant_rich") else 0.0
    if name == "low_confidence_or_preserve":
        return 1.0 if conf.get("low_confidence") or conf.get("preserve") else 0.0
    if name == "variant_proximity_flag":
        return 1.0 if conf.get("variant_proximity") else 0.0
    if name == "local_variant_density":
        return float(conf.get("local_variant_density", 0.0))
    if name == "local_rule_candidate_density":
        return min(float(details.get("local_rule_density", 0.0)) / 4.0, 1.0)
    if name == "local_neural_candidate_density":
        return 0.0
    if name == "local_chosen_edit_density":
        return 0.0
    if name == "local_mismatch_density":
        return float(conf.get("local_mismatch_density", 0.0))
    if name == "nearby_indel_density":
        return float(conf.get("nearby_indel_density", 0.0))
    if name == "window_relative_position":
        return float(conf.get("window_relative_position", 0.0))
    if name == "support_forward_fraction":
        return float(conf.get("support_forward_fraction", 0.5))
    if name == "support_forward_count_fraction":
        return min(float(conf.get("support_forward_count", 0.0)) / depth, 1.0)
    if name == "support_reverse_count_fraction":
        return min(float(conf.get("support_reverse_count", 0.0)) / depth, 1.0)
    if name == "support_strand_bias":
        return float(conf.get("support_strand_bias", 0.0))
    if name == "support_same_haplotype_fraction":
        return float(conf.get("support_same_haplotype_fraction", 0.0))
    if name == "support_match_fraction":
        return float(conf.get("support_match_fraction", 0.0))
    if name == "left_support_match_fraction":
        return float(conf.get("left_support_match_fraction", 0.0))
    if name == "right_support_match_fraction":
        return float(conf.get("right_support_match_fraction", 0.0))
    if name == "repeat_strength":
        return float(conf.get("repeat_strength", 0.0))
    if name == "mapping_quality_available":
        return 1.0 if conf.get("mapping_quality_available") else 0.0
    if name == "mapping_quality_mean":
        return min(float(conf.get("mapping_quality_mean", 0.0)) / 60.0, 1.0)
    if name == "reference_kmer_uniqueness_available":
        return 1.0 if conf.get("reference_kmer_uniqueness_available") else 0.0
    if name == "reference_kmer_uniqueness":
        return float(conf.get("reference_kmer_uniqueness", 0.0))
    if name == "sub_local_gain":
        return max(min(float(details.get("local_gain", 0.0)) + 0.5, 1.0), 0.0)
    if name == "sub_local_rule_density":
        return min(float(details.get("local_rule_density", 0.0)) / 4.0, 1.0)
    return 0.0


def _sigmoid(value: float) -> float:
    if value >= 30:
        return 1.0
    if value <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _allow_gate_says_safe(config: dict, edit_type: str, conf: dict, probs: dict, details: dict) -> tuple[bool, float | None]:
    gate = _load_allow_gate(config)
    if not gate:
        return False, None
    model = gate.get("models", {}).get(edit_type)
    if not model:
        return False, None
    features = gate.get("features", [])
    weights = model.get("weights", [])
    bias = float(model.get("bias", 0.0))
    threshold = float(model.get("threshold", 1.01))
    score = _sigmoid(sum(float(w) * _gate_feature_value(name, conf, probs, details) for w, name in zip(weights, features)) + bias)
    return score >= threshold, score


def _sub_candidate_passes(
    example: dict,
    pos: int,
    rule_base: str,
    conf: dict,
    main_probs: torch.Tensor,
    sub_probs: torch.Tensor,
    config: dict,
) -> tuple[bool, list[str], dict]:
    decode = config["decode"]
    reasons: list[str] = []
    details: dict = {}
    if conf["tandem_repeat"] and decode.get("sub_abstain_in_tandem_repeat", False):
        reasons.append("sub_tandem_repeat_candidate_only")
    if conf["neighbor"] and decode.get("sub_abstain_near_neighbor", False):
        reasons.append("sub_neighbor_candidate_only")
    if (conf["variant"] or conf["phased_variant"] or conf["variant_rich"] or conf["low_confidence"]) and decode.get(
        "sub_abstain_in_variant_or_low_confidence", True
    ):
        reasons.append("sub_variant_or_low_confidence_candidate_only")

    payload_prob = float(sub_probs[pos, BASES.index(rule_base)])
    type_prob = float(main_probs[pos, MAIN_TO_ID["SUB"]])
    details["sub_type_prob"] = type_prob
    details["sub_payload_prob"] = payload_prob
    model_ok = (
        type_prob >= float(decode.get("sub_candidate_min_type_prob", 0.80))
        and payload_prob >= float(decode.get("sub_candidate_min_payload_prob", 0.90))
    )
    if decode.get("sub_require_model_agreement", True) and not model_ok:
        reasons.append("sub_model_agreement_too_low")

    if decode.get("sub_require_local_rerank", True):
        rerank_ok, rerank_details = _sub_local_window_accepts(example, pos, rule_base, config)
        details.update(rerank_details)
        if not rerank_ok:
            reasons.append("sub_local_rerank_reject")

    min_fraction = float(decode.get("sub_candidate_min_fraction", 0.95))
    if conf["fraction"] < min_fraction:
        reasons.append("sub_support_fraction_too_low")

    if reasons and not (decode.get("allow_ambiguous_sub_with_strong_model_and_rerank", False) and model_ok and details.get("local_gain", -1.0) > 0.20):
        recovery_ok, recovery_reasons = _safe_true_sub_recovery_passes(example, pos, rule_base, conf, main_probs, sub_probs, details, config)
        if recovery_ok:
            details["safe_true_recovery"] = True
            return True, ["safe_true_sub_recovery"], details
        details["safe_true_recovery_reject_reasons"] = recovery_reasons
        return False, reasons, details
    return True, reasons, details


def _ins_candidate_passes(
    example: dict,
    pos: int,
    rule_base: str,
    conf: dict,
    ins_probs: torch.Tensor,
    allow_probs: torch.Tensor,
    config: dict,
) -> tuple[bool, list[str], dict]:
    decode = config["decode"]
    reasons: list[str] = []
    details: dict = {
        "top_inserted_base_fraction": conf["top_inserted_base_fraction"],
        "inserted_base_margin": conf["inserted_base_margin"],
    }
    if conf["top_inserted_base_fraction"] < float(decode.get("ins_top_base_min_fraction", 0.90)):
        reasons.append("ins_top_base_fraction_too_low")
    if conf["inserted_base_margin"] < float(decode.get("ins_inserted_base_min_margin", 2.0)):
        reasons.append("ins_inserted_base_margin_too_low")
    if conf["repeat"] or conf["tandem_repeat"]:
        reasons.append("ins_repeat_veto")
    repeat_like = conf["repeat"] or conf["tandem_repeat"]
    ambiguous = repeat_like or conf["homopolymer"] or conf["neighbor"] or conf["boundary"]
    probs = {
        "payload_prob": float(ins_probs[pos, BASES.index(rule_base) + 1]),
        "allow_prob": float(allow_probs[pos]),
    }
    if ambiguous:
        rerank_ok = True
        if decode.get("indel_local_rerank_in_ambiguous_context", True):
            rerank_ok, rerank_details = _indel_local_window_accepts(example, pos, "INS", config)
            details.update(rerank_details)
            if not rerank_ok:
                reasons.append("ins_local_rerank_reject")
    gate_ok, gate_score = _allow_gate_says_safe(config, "INS", conf, probs, details)
    details["allow_gate_score"] = gate_score
    if decode.get("indel_require_allow_gate", False) and not gate_ok:
        reasons.append("ins_requires_allow_gate")
    return not reasons, reasons, details


def _del_candidate_passes(
    example: dict,
    pos: int,
    conf: dict,
    main_probs: torch.Tensor,
    allow_probs: torch.Tensor,
    config: dict,
) -> tuple[bool, list[str], dict]:
    decode = config["decode"]
    reasons: list[str] = []
    details: dict = {"del_fraction": conf["fraction"], "del_count": conf["support_del_count"]}
    if conf["repeat"] or conf["tandem_repeat"]:
        reasons.append("del_repeat_veto")
    if conf["homopolymer_run_length"] >= int(decode.get("del_homopolymer_run_veto_min", 2)):
        reasons.append("del_homopolymer_veto")
    if conf["neighbor"]:
        reasons.append("del_neighbor_veto")
    if conf["fraction"] < float(decode.get("del_hard_min_fraction", 0.95)):
        reasons.append("del_fraction_too_low")
    if decode.get("del_veto_zero_base_support", False) and sum(example["features"]["support_base_counts"][pos]) == 0:
        reasons.append("del_zero_base_support_veto")
    ambiguous = conf["repeat"] or conf["homopolymer_run_length"] >= 2 or conf["neighbor"] or conf["boundary"]
    if ambiguous and decode.get("indel_local_rerank_in_ambiguous_context", True):
        rerank_ok, rerank_details = _indel_local_window_accepts(example, pos, "DEL", config)
        details.update(rerank_details)
        if not rerank_ok:
            reasons.append("del_local_rerank_reject")
    probs = {
        "type_prob_del": float(main_probs[pos, MAIN_TO_ID["DEL"]]),
        "allow_prob": float(allow_probs[pos]),
    }
    gate_ok, gate_score = _allow_gate_says_safe(config, "DEL", conf, probs, details)
    details["allow_gate_score"] = gate_score
    if decode.get("indel_require_allow_gate", False) and not gate_ok:
        reasons.append("del_requires_allow_gate")
    return not reasons, reasons, details


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
        sub_details = {}
        indel_details = {}
        accepted_sub_candidate = False
        accepted_indel_candidate = False

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
                if rule_type == "SUB" and not config["decode"].get("force_support_sub", True):
                    passes, sub_rule_reasons, sub_details = _sub_candidate_passes(
                        example, pos, rule_base or neural_sub_base, conf, main_probs, sub_probs, config
                    )
                    rule_reasons.extend(sub_rule_reasons)
                    accepted_sub_candidate = passes
                elif rule_type == "INS" and config["decode"].get("hard_veto_indels", False):
                    passes, indel_reasons, indel_details = _ins_candidate_passes(
                        example, pos, rule_base or neural_ins_base or "A", conf, ins_probs, allow_probs, config
                    )
                    rule_reasons.extend(indel_reasons)
                    accepted_indel_candidate = passes
                elif rule_type == "DEL" and config["decode"].get("hard_veto_indels", False):
                    passes, indel_reasons, indel_details = _del_candidate_passes(
                        example, pos, conf, main_probs, allow_probs, config
                    )
                    rule_reasons.extend(indel_reasons)
                    accepted_indel_candidate = passes
                if passes:
                    forced = not (accepted_sub_candidate or accepted_indel_candidate)
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
                    if rule_type in {"INS", "DEL"} and config["decode"].get("indel_require_allow_gate", False):
                        chosen_main = "COPY"
                        chosen_ins_base = None
                        vetoed = True
                        reasons.append("indel_gate_required_no_neural_rescue")
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
                                "accepted_sub_candidate": accepted_sub_candidate,
                                "accepted_indel_candidate": accepted_indel_candidate,
                                "vetoed": vetoed,
                                "reasons": reasons,
                                "sub_candidate_details": {},
                                "indel_candidate_details": indel_details,
                            }
                        )
                        out.append(base)
                        continue
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
                    ambiguous = (
                        conf["neighbor"]
                        or conf["variant"]
                        or conf["phased_variant"]
                        or conf["preserve"]
                        or conf["variant_rich"]
                        or conf["low_confidence"]
                    )
                    if type_ok and payload_ok and not ambiguous:
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
                "accepted_sub_candidate": accepted_sub_candidate,
                "accepted_indel_candidate": accepted_indel_candidate,
                "vetoed": vetoed,
                "reasons": reasons,
                "sub_candidate_details": sub_details if rule_type == "SUB" else {},
                "indel_candidate_details": indel_details if rule_type in {"INS", "DEL"} else {},
            }
        )
    return {"prediction": "".join(out), "pred_events": pred_events, "trace": trace}
