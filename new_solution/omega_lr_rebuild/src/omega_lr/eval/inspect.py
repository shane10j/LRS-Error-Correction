"""Notebook-friendly inspection helpers for surgical debug passes."""

from __future__ import annotations

from copy import deepcopy
from math import log
from pathlib import Path

import torch

from omega_lr.constants import BASES, EDIT_LABELS, EDIT_TYPE_LABELS, ID_TO_EDIT
from omega_lr.model.decode import decode_batch_argmax, decode_example
from omega_lr.paths import resolve_path
from omega_lr.train.checkpointing import load_checkpoint
from omega_lr.train.trainer import build_model, choose_device, hybrid_gap_diagnostics, make_loader


def lowered_debug_config(config: dict) -> dict:
    updated = deepcopy(config)
    updated["name"] = f"{config['name']}_full_low_threshold_probe"
    updated["train"]["output_dir"] = "outputs/debug_tiny_low_threshold_probe"
    updated["decode"]["sub_threshold"] = 0.20
    updated["decode"]["del_threshold"] = 0.20
    updated["decode"]["ins_threshold"] = 0.20
    updated["decode"]["trust_threshold"] = 0.05
    updated["decode"]["mode"] = "debug"
    updated["decode"]["use_trust_threshold"] = False
    updated["decode"]["use_delete_candidate_veto"] = False
    updated["decode"]["restrict_supported_candidates"] = False
    updated["decode"]["consistency_check"] = False
    updated["decode"]["full_trace"] = True
    return updated


def overfit_debug_config(
    config: dict,
    run_name: str = "target_only",
    num_examples: int = 4,
    synthetic_case_names: list[str] | None = None,
    remove_trust_gate: bool = True,
    remove_delete_length_head: bool = True,
) -> dict:
    updated = deepcopy(config)
    updated["name"] = f"{config['name']}_{run_name}_overfit_probe"
    updated["dataset"]["kind"] = "synthetic"
    updated["dataset"]["output_dir"] = "outputs/debug_tiny_overfit_probe/dataset"
    updated["dataset"]["splits"] = {"train": num_examples, "val": num_examples, "test": num_examples}
    updated["dataset"]["shared_examples_across_splits"] = True
    updated["dataset"]["synthetic_case_names"] = synthetic_case_names or ["sub", "del", "ins", "copy"][:num_examples]
    updated["train"]["output_dir"] = "outputs/debug_tiny_overfit_probe"
    updated["train"]["epochs"] = 200
    updated["train"]["patience"] = 200
    updated["train"]["batch_size"] = 1
    updated["train"]["lr"] = 0.003
    updated["train"]["class_weights"] = {"COPY": 1.0, "SUB": 3.5, "DEL": 1.0, "INS": 2.5}
    updated["train"]["delete_candidate_aux_weight"] = 0.05
    updated["train"]["type_loss_weight"] = 4.0
    updated["train"]["hard_type_loss_weight"] = 10.0
    updated["train"]["sub_payload_loss_weight"] = 1.0
    updated["train"]["ins_payload_loss_weight"] = 5.0
    updated["train"]["type_margin"] = 0.50
    updated["train"]["type_margin_weight"] = 1.00
    updated["train"]["sub_copy_margin"] = 1.00
    updated["train"]["sub_copy_margin_weight"] = 1.50
    updated["train"]["sub_del_margin"] = 1.00
    updated["train"]["sub_del_margin_weight"] = 1.50
    updated["train"]["ins_del_margin"] = 0.75
    updated["train"]["ins_del_margin_weight"] = 1.00
    updated["train"]["non_del_margin"] = 0.75
    updated["train"]["non_del_margin_weight"] = 1.00
    updated["train"]["del_fallback_penalty_weight"] = 1.00
    updated["train"]["hard_copy_penalty_weight"] = 1.00
    updated["train"]["encourage_edits_schedule"] = {
        "warmup_epochs": 20,
        "early_false_positive_penalty_weight": 0.02,
        "late_false_positive_penalty_weight": 0.35,
        "early_copy_to_sub_penalty_weight": 0.00,
        "late_copy_to_sub_penalty_weight": 0.60,
        "early_copy_to_del_penalty_weight": 0.00,
        "late_copy_to_del_penalty_weight": 0.90,
        "early_copy_to_ins_penalty_weight": 0.00,
        "late_copy_to_ins_penalty_weight": 0.60,
        "early_positive_hard_edit_reward_weight": 1.20,
        "late_positive_hard_edit_reward_weight": 0.70,
        "early_gate_open_bias": 0.0,
        "late_gate_open_bias": 0.0,
        "early_trust_regularization_weight": 0.0,
        "late_trust_regularization_weight": 0.0,
        "early_curriculum_fraction": 1.0,
        "late_curriculum_fraction": 1.0,
        "soft_decode_thresholds": {
            "sub_threshold": updated["decode"]["sub_threshold"],
            "del_threshold": updated["decode"]["del_threshold"],
            "ins_threshold": updated["decode"]["ins_threshold"],
            "trust_threshold": updated["decode"]["trust_threshold"],
        },
    }
    updated["decode"]["mode"] = "debug"
    updated["decode"]["use_trust_threshold"] = False
    updated["decode"]["use_delete_candidate_veto"] = False
    updated["decode"]["restrict_supported_candidates"] = False
    updated["decode"]["consistency_check"] = False
    updated["decode"]["full_trace"] = True
    updated["model_debug"] = {
        "use_trust_gate": not remove_trust_gate,
        "use_delete_length_head": not remove_delete_length_head,
    }
    updated["debug_run_name"] = run_name
    return updated


def load_model_for_inspection(config: dict, checkpoint_path: str | Path):
    checkpoint = load_checkpoint(checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    dataset, _ = make_loader(dataset_dir / "test.jsonl", batch_size=1, shuffle=False)
    support_input_dim = len(dataset[0]["pileup_features"][0])
    model = build_model(config, checkpoint["run_name"], support_input_dim)
    model.load_state_dict(checkpoint["model_state"])
    device = choose_device()
    model.to(device)
    model.eval()
    return model, device


def _gold_label_name(label: int) -> str:
    return ID_TO_EDIT[int(label)]


def _top_k_labels(logits_row: torch.Tensor, probs_row: torch.Tensor, k: int = 3) -> list[dict]:
    indices = torch.argsort(probs_row, descending=True)[:k].tolist()
    return [
        {
            "label": ID_TO_EDIT[int(idx)],
            "logit": round(float(logits_row[int(idx)].item()), 4),
            "prob": round(float(probs_row[int(idx)].item()), 4),
        }
        for idx in indices
    ]


def _named_probs(labels: list[str], probs_row: torch.Tensor) -> dict[str, float]:
    return {label: round(float(probs_row[idx].item()), 4) for idx, label in enumerate(labels)}


def _ins_payload_diagnostic(gold_label: str, ins_probs_row: torch.Tensor) -> dict | None:
    if not gold_label.startswith("INS_"):
        return None
    gold_base = gold_label.split("_", 1)[1]
    gold_idx = BASES.index(gold_base)
    gold_prob = float(ins_probs_row[gold_idx].item())
    rank = 1 + sum(float(value.item()) > gold_prob for value in ins_probs_row)
    return {
        "gold_inserted_base": gold_base,
        "gold_base_rank": int(rank),
        "gold_base_prob": round(gold_prob, 4),
        "payload_ce": round(-log(max(gold_prob, 1e-8)), 4),
        "ins_base_probs": _named_probs(BASES, ins_probs_row),
    }


def _edit_family(label: str) -> str:
    if label.startswith("SUB_"):
        return "SUB"
    if label.startswith("INS_"):
        return "INS"
    if label == "DEL":
        return "DEL"
    return "COPY"


def _payload_confidence(label: str, sub_probs_row: torch.Tensor, ins_probs_row: torch.Tensor) -> float:
    if label.startswith("SUB_"):
        return float(sub_probs_row[BASES.index(label[-1])].item())
    if label.startswith("INS_"):
        return float(ins_probs_row[BASES.index(label[-1])].item())
    if label == "DEL":
        return 1.0
    return 0.0


def _nearest_other_hard_distance(gold_labels: list[str], pos: int) -> int | None:
    distances = [abs(pos - idx) for idx, label in enumerate(gold_labels) if idx != pos and label != "COPY"]
    return min(distances) if distances else None


def _support_evidence(example: dict, pos: int) -> dict:
    features = example["features"]
    base_counts = features["support_base_counts"][pos]
    majority_idx = max(range(4), key=lambda idx: base_counts[idx]) if sum(base_counts) > 0 else BASES.index(example["target_seq"][pos])
    return {
        "target_base": example["target_seq"][pos],
        "truth_base": example["truth_seq"][pos] if pos < len(example["truth_seq"]) else None,
        "support_base_counts": dict(zip(BASES, base_counts)),
        "majority_base": BASES[majority_idx],
        "support_ins_base_counts": dict(zip(BASES, features.get("support_ins_base_counts", [[0, 0, 0, 0] for _ in example["target_seq"]])[pos])),
        "support_ins_count": features["support_ins_count"][pos],
        "support_del_count": features["support_del_count"][pos],
        "support_agreement": round(float(features["support_agreement"][pos]), 4),
        "support_entropy": round(float(features["support_entropy"][pos]), 4),
    }


def _support_rule_row(
    example: dict,
    pos: int,
    gold_label: str,
    hybrid_label: str | None,
    neural_label: str | None,
    trace: dict,
    type_probs_row: torch.Tensor,
    sub_probs_row: torch.Tensor,
    ins_probs_row: torch.Tensor,
    gold_labels: list[str],
) -> dict:
    rule_label = trace.get("support_rule_label", "COPY")
    rule_family = _edit_family(rule_label)
    rule_type_prob = float(type_probs_row[EDIT_TYPE_LABELS.index(rule_family)].item()) if rule_family != "COPY" else 0.0
    payload_confidence = _payload_confidence(rule_label, sub_probs_row, ins_probs_row)
    confidence = trace.get("rule_confidence", {})
    nearest_distance = _nearest_other_hard_distance(gold_labels, pos)
    outcome = "true_positive" if rule_label == gold_label else "false_positive"
    return {
        "example_id": example["example_id"],
        "pos": pos,
        "read_boundary_position": pos == 0 or pos == len(example["target_seq"]) - 1,
        "target_base": example["target_seq"][pos],
        "gold_label": gold_label,
        "support_rule_label": rule_label,
        "hybrid_label": hybrid_label,
        "neural_only_label": neural_label,
        "outcome": outcome,
        "is_true_positive": outcome == "true_positive",
        "is_corrected_by_hybrid": hybrid_label == gold_label,
        "forced_by_rule": bool(trace.get("forced_by_rule", False)),
        "rescued_by_neural": bool(trace.get("rescued_by_neural", False)),
        "veto_reasons": trace.get("veto_reasons", []),
        "support_evidence": _support_evidence(example, pos),
        "support_fraction": round(float(confidence.get("support_fraction", 0.0)), 4),
        "support_margin": round(float(confidence.get("support_margin", 0.0)), 4),
        "base_top_fraction": round(float(confidence.get("base_top_fraction", 0.0)), 4),
        "del_fraction": round(float(confidence.get("del_fraction", 0.0)), 4),
        "entropy": round(float(confidence.get("local_entropy", 0.0)), 4),
        "support_depth": round(float(confidence.get("support_depth", 0.0)), 4),
        "homopolymer_run_length": int(confidence.get("homopolymer_run_length", 1)),
        "tandem_repeat_flag": int(confidence.get("tandem_repeat_flag", 0)),
        "neighboring_edit_distance": nearest_distance,
        "neighbor_edit_proximity": int(confidence.get("neighbor_edit_proximity", 0)),
        "neural_type_prob": round(rule_type_prob, 4),
        "payload_confidence": round(payload_confidence, 4),
        "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs_row),
        "sub_base_probs": _named_probs(BASES, sub_probs_row),
        "ins_base_probs": _named_probs(BASES, ins_probs_row),
    }


def diagnose_example(gold_labels: list[str], argmax_labels: list[str], decoded_labels: list[str], hard_positions: list[int]) -> dict:
    if not hard_positions:
        return {"summary": "no_hard_edits_in_gold"}
    argmax_hits = sum(argmax_labels[pos] == gold_labels[pos] for pos in hard_positions)
    decoded_hits = sum(
        pos < len(decoded_labels) and decoded_labels[pos] == gold_labels[pos] for pos in hard_positions
    )
    if argmax_hits == 0:
        summary = "training_or_representation_failure"
    elif decoded_hits < argmax_hits:
        summary = "decoding_is_suppressing_some_correct_raw_predictions"
    else:
        summary = "raw_predictions_and_decoding_agree_on_failures"
    return {
        "summary": summary,
        "hard_positions": len(hard_positions),
        "argmax_hits": argmax_hits,
        "decoded_hits": decoded_hits,
    }


def inspect_debug_examples(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
    example_filters: tuple[str, ...] = ("sub_", "del_", "ins_"),
) -> list[dict]:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    reports = []
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            if not any(token in example["example_id"] for token in example_filters):
                continue
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            length = len(example["target_seq"])
            sliced = {
                "type_logits": outputs["type_logits"][0, :length].detach().cpu(),
                "sub_base_logits": outputs["sub_base_logits"][0, :length].detach().cpu(),
                "ins_base_logits": outputs["ins_base_logits"][0, :length].detach().cpu(),
                "edit_logits": outputs["edit_logits"][0, :length].detach().cpu(),
                "delete_candidate_logits": outputs["delete_candidate_logits"][0, :length].detach().cpu(),
                "delete_length_logits": outputs["delete_length_logits"][0, :length].detach().cpu(),
                "trust": outputs["trust"][0, :length].detach().cpu(),
            }
            decoded = decode_example(example["target_seq"], example, sliced, config["decode"])
            argmax_decoded = decode_batch_argmax(
                {"raw_examples": [example]},
                {
                    "type_logits": outputs["type_logits"][0:1, :length],
                    "sub_base_logits": outputs["sub_base_logits"][0:1, :length],
                    "ins_base_logits": outputs["ins_base_logits"][0:1, :length],
                    "edit_logits": outputs["edit_logits"][0:1, :length],
                    "delete_candidate_logits": outputs["delete_candidate_logits"][0:1, :length],
                    "delete_length_logits": outputs["delete_length_logits"][0:1, :length],
                    "trust": outputs["trust"][0:1, :length],
                },
            )[0]
            edit_probs = torch.softmax(sliced["edit_logits"], dim=-1)
            type_probs = torch.softmax(sliced["type_logits"], dim=-1)
            sub_probs = torch.softmax(sliced["sub_base_logits"], dim=-1)
            ins_probs = torch.softmax(sliced["ins_base_logits"], dim=-1)
            delete_probs = torch.sigmoid(sliced["delete_candidate_logits"])
            delete_length_probs = torch.softmax(sliced["delete_length_logits"], dim=-1)
            gold_labels = [ID_TO_EDIT[label] for label in example["edit_labels"]]
            argmax_labels = argmax_decoded["predicted_labels"]
            hard_positions = [idx for idx, label in enumerate(gold_labels) if label != "COPY"]
            reports.append(
                {
                    "example_id": example["example_id"],
                    "target_seq": example["target_seq"],
                    "truth_seq": example["truth_seq"],
                    "prediction": decoded["prediction"],
                    "gold_labels": gold_labels,
                    "argmax_labels": argmax_labels,
                    "decoded_labels": decoded["predicted_labels"],
                    "hard_positions": hard_positions,
                    "positions": [
                        {
                            "pos": pos,
                            "target_base": example["target_seq"][pos],
                            "support_evidence": _support_evidence(example, pos),
                            "gold_label": gold_labels[pos],
                            "argmax_label": argmax_labels[pos],
                            "decoded_label": decoded["predicted_labels"][pos] if pos < len(decoded["predicted_labels"]) else None,
                            "top_edit_candidates": _top_k_labels(sliced["edit_logits"][pos], edit_probs[pos]),
                            "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs[pos]),
                            "sub_base_probs": _named_probs(BASES, sub_probs[pos]),
                            "ins_base_probs": _named_probs(BASES, ins_probs[pos]),
                            "ins_payload_diagnostic": _ins_payload_diagnostic(gold_labels[pos], ins_probs[pos]),
                            "delete_candidate_prob": round(float(delete_probs[pos].item()), 4),
                            "delete_length_probs": [round(float(value), 4) for value in delete_length_probs[pos].tolist()],
                            "trust": round(float(decoded["trust"][pos]), 4),
                            "support_agreement": round(float(example["features"]["support_agreement"][pos]), 4),
                            "support_entropy": round(float(example["features"]["support_entropy"][pos]), 4),
                            "decoder_trace": next((trace for trace in decoded["trace"] if trace["pos"] == pos), None),
                        }
                        for pos in hard_positions
                    ],
                    "diagnosis": diagnose_example(gold_labels, argmax_labels, decoded["predicted_labels"], hard_positions),
                }
            )
    return reports


def argmax_decode_reports(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
    example_filters: tuple[str, ...] = ("sub_", "del_", "ins_"),
) -> list[dict]:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    reports = []
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            if not any(token in example["example_id"] for token in example_filters):
                continue
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            argmax_decoded = decode_batch_argmax(batch, outputs)[0]
            edit_logits = outputs["edit_logits"][0, : len(example["target_seq"])].detach().cpu()
            edit_probs = torch.softmax(edit_logits, dim=-1)
            type_probs = torch.softmax(outputs["type_logits"][0, : len(example["target_seq"])].detach().cpu(), dim=-1)
            ins_probs = torch.softmax(outputs["ins_base_logits"][0, : len(example["target_seq"])].detach().cpu(), dim=-1)
            gold_labels = [_gold_label_name(label) for label in example["edit_labels"]]
            argmax_labels = argmax_decoded["predicted_labels"]
            reports.append(
                {
                    "example_id": example["example_id"],
                    "target_seq": example["target_seq"],
                    "truth_seq": example["truth_seq"],
                    "prediction": argmax_decoded["prediction"],
                    "gold_labels": gold_labels,
                    "argmax_labels": argmax_labels,
                    "delete_length_labels": example["delete_length_labels"],
                    "hard_positions": [idx for idx, label in enumerate(gold_labels) if label != "COPY"],
                    "positions": [
                        {
                            "pos": pos,
                            "gold_label": gold_labels[pos],
                            "argmax_label": argmax_labels[pos],
                            "top_edit_candidates": _top_k_labels(edit_logits[pos], edit_probs[pos]),
                            "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs[pos]),
                            "ins_base_probs": _named_probs(BASES, ins_probs[pos]),
                            "ins_payload_diagnostic": _ins_payload_diagnostic(gold_labels[pos], ins_probs[pos]),
                            "decoder_trace": next((trace for trace in argmax_decoded["trace"] if trace["pos"] == pos), None),
                        }
                        for pos in range(len(gold_labels))
                        if gold_labels[pos] != "COPY"
                    ],
                }
            )
    return reports


def classwise_edit_statistics(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    stats = {
        label: {
            "count": 0,
            "avg_gold_logit": 0.0,
            "avg_gold_prob": 0.0,
            "avg_argmax_prob": 0.0,
            "argmax_match_rate": 0.0,
        }
        for label in EDIT_LABELS
    }
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            edit_logits = outputs["edit_logits"][0, : len(example["target_seq"])].detach().cpu()
            edit_probs = torch.softmax(edit_logits, dim=-1)
            argmax_decoded = decode_batch_argmax(batch, outputs)[0]
            for pos, gold_id in enumerate(example["edit_labels"]):
                label = _gold_label_name(gold_id)
                stats[label]["count"] += 1
                stats[label]["avg_gold_logit"] += float(edit_logits[pos, gold_id].item())
                stats[label]["avg_gold_prob"] += float(edit_probs[pos, gold_id].item())
                argmax_label = argmax_decoded["predicted_labels"][pos]
                argmax_id = EDIT_LABELS.index(argmax_label)
                stats[label]["avg_argmax_prob"] += float(edit_probs[pos, argmax_id].item())
                stats[label]["argmax_match_rate"] += float(int(argmax_label == label))
    for label, values in stats.items():
        count = max(values["count"], 1)
        values["avg_gold_logit"] = round(values["avg_gold_logit"] / count, 4)
        values["avg_gold_prob"] = round(values["avg_gold_prob"] / count, 4)
        values["avg_argmax_prob"] = round(values["avg_argmax_prob"] / count, 4)
        values["argmax_match_rate"] = round(values["argmax_match_rate"] / count, 4)
    return stats


def missed_hard_edit_evidence(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> list[dict]:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    reports = []
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            decoded = decode_batch_argmax(batch, outputs)[0]
            length = len(example["target_seq"])
            type_probs = torch.softmax(outputs["type_logits"][0, :length].detach().cpu(), dim=-1)
            sub_probs = torch.softmax(outputs["sub_base_logits"][0, :length].detach().cpu(), dim=-1)
            ins_probs = torch.softmax(outputs["ins_base_logits"][0, :length].detach().cpu(), dim=-1)
            gold_labels = [_gold_label_name(label) for label in example["edit_labels"]]
            for pos, gold_label in enumerate(gold_labels):
                if gold_label == "COPY":
                    continue
                predicted_label = decoded["predicted_labels"][pos]
                if predicted_label == gold_label:
                    continue
                reports.append(
                    {
                        "example_id": example["example_id"],
                        "pos": pos,
                        "gold_label": gold_label,
                        "predicted_label": predicted_label,
                        "support_evidence": _support_evidence(example, pos),
                        "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs[pos]),
                        "sub_base_probs": _named_probs(BASES, sub_probs[pos]),
                        "ins_base_probs": _named_probs(BASES, ins_probs[pos]),
                    }
                )
    return reports


def false_sub_diagnostics(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> list[dict]:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    reports = []
    neural_config = deepcopy(config["decode"])
    neural_config["hybrid_rule_decode"] = False
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            length = len(example["target_seq"])
            sliced = {
                "type_logits": outputs["type_logits"][0, :length].detach().cpu(),
                "sub_base_logits": outputs["sub_base_logits"][0, :length].detach().cpu(),
                "ins_base_logits": outputs["ins_base_logits"][0, :length].detach().cpu(),
                "edit_logits": outputs["edit_logits"][0, :length].detach().cpu(),
                "delete_candidate_logits": outputs["delete_candidate_logits"][0, :length].detach().cpu(),
                "delete_length_logits": outputs["delete_length_logits"][0, :length].detach().cpu(),
                "trust": outputs["trust"][0, :length].detach().cpu(),
            }
            hybrid_decoded = decode_example(example["target_seq"], example, sliced, config["decode"])
            neural_decoded = decode_example(example["target_seq"], example, sliced, neural_config)
            type_probs = torch.softmax(sliced["type_logits"], dim=-1)
            sub_probs = torch.softmax(sliced["sub_base_logits"], dim=-1)
            gold_labels = [_gold_label_name(label) for label in example["edit_labels"]]
            for pos, gold_label in enumerate(gold_labels):
                predicted = hybrid_decoded["predicted_labels"][pos] if pos < len(hybrid_decoded["predicted_labels"]) else None
                if gold_label != "COPY" or not (predicted or "").startswith("SUB_"):
                    continue
                trace = next((item for item in hybrid_decoded["trace"] if item["pos"] == pos), {})
                reports.append(
                    {
                        "example_id": example["example_id"],
                        "pos": pos,
                        "gold_label": gold_label,
                        "hybrid_label": predicted,
                        "neural_only_label": neural_decoded["predicted_labels"][pos] if pos < len(neural_decoded["predicted_labels"]) else None,
                        "target_base": example["target_seq"][pos],
                        "support_evidence": _support_evidence(example, pos),
                        "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs[pos]),
                        "sub_base_probs": _named_probs(BASES, sub_probs[pos]),
                        "forced_by_rule": bool(trace.get("forced_by_rule", False)),
                        "veto_reasons": trace.get("veto_reasons", []),
                        "candidate_label": trace.get("candidate_label"),
                    }
                )
    return reports


def insertion_payload_diagnostics(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> list[dict]:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    reports = []
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            length = len(example["target_seq"])
            sliced = {
                "type_logits": outputs["type_logits"][0, :length].detach().cpu(),
                "sub_base_logits": outputs["sub_base_logits"][0, :length].detach().cpu(),
                "ins_base_logits": outputs["ins_base_logits"][0, :length].detach().cpu(),
                "edit_logits": outputs["edit_logits"][0, :length].detach().cpu(),
                "delete_candidate_logits": outputs["delete_candidate_logits"][0, :length].detach().cpu(),
                "delete_length_logits": outputs["delete_length_logits"][0, :length].detach().cpu(),
                "trust": outputs["trust"][0, :length].detach().cpu(),
            }
            decoded = decode_example(example["target_seq"], example, sliced, config["decode"])
            type_probs = torch.softmax(sliced["type_logits"], dim=-1)
            ins_probs = torch.softmax(sliced["ins_base_logits"], dim=-1)
            gold_labels = [_gold_label_name(label) for label in example["edit_labels"]]
            for pos, gold_label in enumerate(gold_labels):
                if not gold_label.startswith("INS_"):
                    continue
                predicted = decoded["predicted_labels"][pos] if pos < len(decoded["predicted_labels"]) else None
                trace = next((item for item in decoded["trace"] if item["pos"] == pos), {})
                reports.append(
                    {
                        "example_id": example["example_id"],
                        "pos": pos,
                        "gold_label": gold_label,
                        "decoded_label": predicted,
                        "is_correct": predicted == gold_label,
                        "support_evidence": _support_evidence(example, pos),
                        "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs[pos]),
                        "ins_payload_diagnostic": _ins_payload_diagnostic(gold_label, ins_probs[pos]),
                        "forced_by_rule": bool(trace.get("forced_by_rule", False)),
                        "veto_reasons": trace.get("veto_reasons", []),
                    }
                )
    return reports


def support_rule_positive_audit(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict:
    """Audit every support-rule-positive position as true/false support evidence."""
    rows = _support_rule_positive_rows(config, checkpoint_path, split)
    by_type = {}
    for row in rows:
        family = _edit_family(row["support_rule_label"])
        bucket = by_type.setdefault(
            family,
            {"total": 0, "true_positive": 0, "false_positive": 0, "hybrid_corrected": 0},
        )
        bucket["total"] += 1
        bucket[row["outcome"]] += 1
        bucket["hybrid_corrected"] += int(row["is_corrected_by_hybrid"])
    false_rows = [row for row in rows if not row["is_true_positive"]]
    true_rows = [row for row in rows if row["is_true_positive"]]
    summary = {
        "split": split,
        "support_rule_positive_count": len(rows),
        "true_positive_count": len(true_rows),
        "false_positive_count": len(false_rows),
        "hybrid_corrected_true_positive_count": sum(int(row["is_corrected_by_hybrid"]) for row in true_rows),
        "by_type": by_type,
    }
    return {
        "summary": summary,
        "rows": rows,
        "false_positive_rows": false_rows,
        "true_positive_rows": true_rows,
    }


def _support_rule_positive_rows(config: dict, checkpoint_path: str | Path, split: str) -> list[dict]:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    rows = []
    neural_config = deepcopy(config["decode"])
    neural_config["hybrid_rule_decode"] = False
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            length = len(example["target_seq"])
            sliced = {
                "type_logits": outputs["type_logits"][0, :length].detach().cpu(),
                "sub_base_logits": outputs["sub_base_logits"][0, :length].detach().cpu(),
                "ins_base_logits": outputs["ins_base_logits"][0, :length].detach().cpu(),
                "edit_logits": outputs["edit_logits"][0, :length].detach().cpu(),
                "delete_candidate_logits": outputs["delete_candidate_logits"][0, :length].detach().cpu(),
                "delete_length_logits": outputs["delete_length_logits"][0, :length].detach().cpu(),
                "trust": outputs["trust"][0, :length].detach().cpu(),
            }
            hybrid_decoded = decode_example(example["target_seq"], example, sliced, config["decode"])
            neural_decoded = decode_example(example["target_seq"], example, sliced, neural_config)
            type_probs = torch.softmax(sliced["type_logits"], dim=-1)
            sub_probs = torch.softmax(sliced["sub_base_logits"], dim=-1)
            ins_probs = torch.softmax(sliced["ins_base_logits"], dim=-1)
            gold_labels = [_gold_label_name(label) for label in example["edit_labels"]]
            for pos, gold_label in enumerate(gold_labels):
                trace = next((item for item in hybrid_decoded["trace"] if item["pos"] == pos), {})
                if trace.get("support_rule_label", "COPY") == "COPY":
                    continue
                rows.append(
                    _support_rule_row(
                        example=example,
                        pos=pos,
                        gold_label=gold_label,
                        hybrid_label=hybrid_decoded["predicted_labels"][pos] if pos < len(hybrid_decoded["predicted_labels"]) else None,
                        neural_label=neural_decoded["predicted_labels"][pos] if pos < len(neural_decoded["predicted_labels"]) else None,
                        trace=trace,
                        type_probs_row=type_probs[pos],
                        sub_probs_row=sub_probs[pos],
                        ins_probs_row=ins_probs[pos],
                        gold_labels=gold_labels,
                    )
                )
    return rows


def support_rule_calibration_report(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict:
    """Fit a tiny logistic allow/abstain model for support-rule-positive edits."""
    train_rows = _support_rule_positive_rows(config, checkpoint_path, "train")
    eval_rows = _support_rule_positive_rows(config, checkpoint_path, split)
    feature_names = [
        "support_fraction",
        "support_margin",
        "base_top_fraction",
        "del_fraction",
        "entropy",
        "support_depth",
        "homopolymer_run_length",
        "tandem_repeat_flag",
        "neighbor_edit_proximity",
        "read_boundary_position",
        "neural_type_prob",
        "payload_confidence",
        "rule_is_sub",
        "rule_is_ins",
        "rule_is_del",
    ]
    if len(train_rows) < 2 or len({row["is_true_positive"] for row in train_rows}) < 2:
        return {
            "status": "not_enough_mixed_support_rule_examples",
            "feature_names": feature_names,
            "train_summary": _support_rule_rows_summary(train_rows),
            "eval_summary": _support_rule_rows_summary(eval_rows),
            "eval_rows": eval_rows,
        }
    train_x, train_y = _calibration_tensors(train_rows)
    eval_x, eval_y = _calibration_tensors(eval_rows)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp(min=1e-6)
    train_x = (train_x - mean) / std
    eval_x = (eval_x - mean) / std if len(eval_rows) else eval_x
    torch.manual_seed(7)
    model = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train_x).squeeze(-1), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_scores = torch.sigmoid(model(train_x).squeeze(-1))
        eval_scores = torch.sigmoid(model(eval_x).squeeze(-1)) if len(eval_rows) else torch.empty(0)
    threshold = _zero_fp_threshold(train_scores, train_y)
    eval_threshold = _zero_fp_threshold(eval_scores, eval_y) if len(eval_rows) else threshold
    return {
        "status": "ok",
        "feature_names": feature_names,
        "threshold_selected_for_zero_train_fp": round(float(threshold), 4),
        "threshold_selected_for_zero_eval_fp": round(float(eval_threshold), 4),
        "train_summary": _calibration_summary(train_rows, train_scores, train_y, threshold),
        "eval_summary": _calibration_summary(eval_rows, eval_scores, eval_y, threshold),
        "eval_tuned_train_summary": _calibration_summary(train_rows, train_scores, train_y, eval_threshold),
        "eval_tuned_eval_summary": _calibration_summary(eval_rows, eval_scores, eval_y, eval_threshold),
        "per_family": _family_calibration_summaries(eval_rows, eval_scores, eval_y, eval_threshold),
        "weights": {
            name: round(float(weight), 4)
            for name, weight in zip(feature_names, model.weight.detach().squeeze(0).tolist())
        },
        "bias": round(float(model.bias.detach().item()), 4),
        "eval_rows": _attach_calibration_scores(eval_rows, eval_scores, threshold, eval_threshold),
    }


def _support_rule_rows_summary(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "true_positive": sum(int(row["is_true_positive"]) for row in rows),
        "false_positive": sum(int(not row["is_true_positive"]) for row in rows),
    }


def _row_features(row: dict) -> list[float]:
    family = _edit_family(row["support_rule_label"])
    return [
        float(row["support_fraction"]),
        float(row["support_margin"]),
        float(row["base_top_fraction"]),
        float(row["del_fraction"]),
        float(row["entropy"]),
        float(row["support_depth"]),
        float(row["homopolymer_run_length"]),
        float(row["tandem_repeat_flag"]),
        float(row["neighbor_edit_proximity"]),
        float(row["read_boundary_position"]),
        float(row["neural_type_prob"]),
        float(row["payload_confidence"]),
        float(family == "SUB"),
        float(family == "INS"),
        float(family == "DEL"),
    ]


def _calibration_tensors(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        return torch.empty(0, 15), torch.empty(0)
    return (
        torch.tensor([_row_features(row) for row in rows], dtype=torch.float32),
        torch.tensor([float(row["is_true_positive"]) for row in rows], dtype=torch.float32),
    )


def _zero_fp_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    thresholds = sorted({float(score.item()) for score in scores}, reverse=True) + [1.01]
    best_threshold = 1.01
    best_tp = -1
    for threshold in thresholds:
        allowed = scores >= threshold
        fp = int(((allowed == 1) & (labels == 0)).sum().item())
        tp = int(((allowed == 1) & (labels == 1)).sum().item())
        if fp == 0 and tp > best_tp:
            best_tp = tp
            best_threshold = threshold
    return best_threshold


def _calibration_summary(rows: list[dict], scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict:
    if not rows:
        return {"total": 0, "allowed": 0, "allowed_true_positive": 0, "allowed_false_positive": 0}
    allowed = scores >= threshold
    return {
        "total": len(rows),
        "true_positive": int(labels.sum().item()),
        "false_positive": int((labels == 0).sum().item()),
        "allowed": int(allowed.sum().item()),
        "allowed_true_positive": int(((allowed == 1) & (labels == 1)).sum().item()),
        "allowed_false_positive": int(((allowed == 1) & (labels == 0)).sum().item()),
        "blocked_true_positive": int(((allowed == 0) & (labels == 1)).sum().item()),
        "blocked_false_positive": int(((allowed == 0) & (labels == 0)).sum().item()),
    }


def _family_calibration_summaries(rows: list[dict], scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict:
    summaries = {}
    for family in ["SUB", "INS", "DEL"]:
        indices = [idx for idx, row in enumerate(rows) if _edit_family(row["support_rule_label"]) == family]
        if not indices:
            summaries[family] = {"total": 0, "allowed": 0, "allowed_true_positive": 0, "allowed_false_positive": 0}
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        summaries[family] = _calibration_summary(
            [rows[idx] for idx in indices],
            scores[idx_tensor],
            labels[idx_tensor],
            threshold,
        )
    return summaries


def _attach_calibration_scores(rows: list[dict], scores: torch.Tensor, threshold: float, eval_threshold: float | None = None) -> list[dict]:
    output = []
    for row, score in zip(rows, scores.tolist()):
        updated = dict(row)
        updated["allow_edit_probability"] = round(float(score), 4)
        updated["calibration_allows"] = float(score) >= threshold
        updated["eval_zero_fp_calibration_allows"] = float(score) >= (threshold if eval_threshold is None else eval_threshold)
        output.append(updated)
    return output


def calibration_allowed_missed_edits(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> list[dict]:
    """Rows that eval-zero-FP calibration would allow but hybrid did not correct."""
    report = support_rule_calibration_report(config, checkpoint_path, split)
    rows = report.get("eval_rows", [])
    missed = [
        row
        for row in rows
        if row.get("eval_zero_fp_calibration_allows")
        and row.get("is_true_positive")
        and not row.get("is_corrected_by_hybrid")
    ]
    return [
        {
            "example_id": row["example_id"],
            "pos": row["pos"],
            "support_rule_label": row["support_rule_label"],
            "gold_label": row["gold_label"],
            "hybrid_label": row["hybrid_label"],
            "neural_only_label": row["neural_only_label"],
            "argmax_label": row["argmax_label"],
            "allow_edit_probability": row["allow_edit_probability"],
            "veto_reasons": row["veto_reasons"],
            "support_fraction": row["support_fraction"],
            "del_fraction": row["del_fraction"],
            "support_margin": row["support_margin"],
            "entropy": row["entropy"],
            "neighbor_edit_proximity": row["neighbor_edit_proximity"],
            "read_boundary_position": row["read_boundary_position"],
            "neural_type_prob": row["neural_type_prob"],
            "payload_confidence": row["payload_confidence"],
            "support_evidence": row["support_evidence"],
            "type_probs": row["type_probs"],
            "sub_base_probs": row["sub_base_probs"],
            "ins_base_probs": row["ins_base_probs"],
            "decoder_trace": row["decoder_trace"],
        }
        for row in missed
    ]


def hybrid_miss_diagnostics(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> list[dict]:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    reports = []
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            length = len(example["target_seq"])
            sliced = {
                "type_logits": outputs["type_logits"][0, :length].detach().cpu(),
                "sub_base_logits": outputs["sub_base_logits"][0, :length].detach().cpu(),
                "ins_base_logits": outputs["ins_base_logits"][0, :length].detach().cpu(),
                "edit_logits": outputs["edit_logits"][0, :length].detach().cpu(),
                "delete_candidate_logits": outputs["delete_candidate_logits"][0, :length].detach().cpu(),
                "delete_length_logits": outputs["delete_length_logits"][0, :length].detach().cpu(),
                "trust": outputs["trust"][0, :length].detach().cpu(),
            }
            decoded = decode_example(example["target_seq"], example, sliced, config["decode"])
            neural_config = deepcopy(config["decode"])
            neural_config["hybrid_rule_decode"] = False
            neural_decoded = decode_example(example["target_seq"], example, sliced, neural_config)
            argmax_decoded = decode_batch_argmax(batch, outputs)[0]
            type_probs = torch.softmax(sliced["type_logits"], dim=-1)
            sub_probs = torch.softmax(sliced["sub_base_logits"], dim=-1)
            ins_probs = torch.softmax(sliced["ins_base_logits"], dim=-1)
            gold_labels = [_gold_label_name(label) for label in example["edit_labels"]]
            for pos, gold_label in enumerate(gold_labels):
                if gold_label == "COPY":
                    continue
                decoded_label = decoded["predicted_labels"][pos] if pos < len(decoded["predicted_labels"]) else None
                if decoded_label == gold_label:
                    continue
                trace = next((item for item in decoded["trace"] if item["pos"] == pos), {})
                rule_confidence = trace.get("rule_confidence", {})
                support_rule_label = trace.get("support_rule_label")
                payload_confidence = _payload_confidence(
                    support_rule_label or gold_label,
                    sub_probs[pos],
                    ins_probs[pos],
                )
                type_family = _edit_family(support_rule_label or gold_label)
                type_probability = (
                    float(type_probs[pos, EDIT_TYPE_LABELS.index(type_family)].item())
                    if type_family != "COPY"
                    else float(type_probs[pos, EDIT_TYPE_LABELS.index(_edit_family(gold_label))].item())
                )
                reports.append(
                    {
                        "example_id": example["example_id"],
                        "pos": pos,
                        "is_boundary_position": pos == 0 or pos == len(example["target_seq"]) - 1,
                        "target_prefix": example["target_seq"][: min(len(example["target_seq"]), 8)],
                        "truth_prefix": example["truth_seq"][: min(len(example["truth_seq"]), 10)],
                        "gold_label": gold_label,
                        "hybrid_label": decoded_label,
                        "neural_only_label": neural_decoded["predicted_labels"][pos] if pos < len(neural_decoded["predicted_labels"]) else None,
                        "argmax_label": argmax_decoded["predicted_labels"][pos] if pos < len(argmax_decoded["predicted_labels"]) else None,
                        "support_rule_label": support_rule_label,
                        "forced_by_rule": bool(trace.get("forced_by_rule", False)),
                        "veto_reasons": trace.get("veto_reasons", []),
                        "candidate_label": trace.get("candidate_label"),
                        "rescue_diagnostic": {
                            "support_rule_label": support_rule_label,
                            "payload_confidence": round(payload_confidence, 4),
                            "type_probability": round(type_probability, 4),
                            "veto_reasons": trace.get("veto_reasons", []),
                            "support_fraction": round(float(rule_confidence.get("support_fraction", 0.0)), 4),
                            "del_fraction": round(float(rule_confidence.get("del_fraction", 0.0)), 4),
                            "entropy": round(float(rule_confidence.get("local_entropy", 0.0)), 4),
                            "support_margin": round(float(rule_confidence.get("support_margin", 0.0)), 4),
                            "support_depth": round(float(rule_confidence.get("support_depth", 0.0)), 4),
                            "homopolymer_run_length": int(rule_confidence.get("homopolymer_run_length", 1)),
                            "tandem_repeat_flag": int(rule_confidence.get("tandem_repeat_flag", 0)),
                            "neighbor_edit_proximity": int(rule_confidence.get("neighbor_edit_proximity", 0)),
                            "is_boundary_position": pos == 0 or pos == len(example["target_seq"]) - 1,
                        },
                        "support_evidence": _support_evidence(example, pos),
                        "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs[pos]),
                        "sub_base_probs": _named_probs(BASES, sub_probs[pos]),
                        "ins_base_probs": _named_probs(BASES, ins_probs[pos]),
                        "ins_payload_diagnostic": _ins_payload_diagnostic(gold_label, ins_probs[pos]),
                        "decoder_trace": trace,
                    }
                )
    return reports


def false_hard_edit_diagnostics(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> list[dict]:
    """Dump every COPY position where the model emits a hard edit."""
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    reports = []
    neural_config = deepcopy(config["decode"])
    neural_config["hybrid_rule_decode"] = False
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            length = len(example["target_seq"])
            sliced = {
                "type_logits": outputs["type_logits"][0, :length].detach().cpu(),
                "sub_base_logits": outputs["sub_base_logits"][0, :length].detach().cpu(),
                "ins_base_logits": outputs["ins_base_logits"][0, :length].detach().cpu(),
                "edit_logits": outputs["edit_logits"][0, :length].detach().cpu(),
                "delete_candidate_logits": outputs["delete_candidate_logits"][0, :length].detach().cpu(),
                "delete_length_logits": outputs["delete_length_logits"][0, :length].detach().cpu(),
                "trust": outputs["trust"][0, :length].detach().cpu(),
            }
            hybrid_decoded = decode_example(example["target_seq"], example, sliced, config["decode"])
            neural_decoded = decode_example(example["target_seq"], example, sliced, neural_config)
            type_probs = torch.softmax(sliced["type_logits"], dim=-1)
            sub_probs = torch.softmax(sliced["sub_base_logits"], dim=-1)
            ins_probs = torch.softmax(sliced["ins_base_logits"], dim=-1)
            gold_labels = [_gold_label_name(label) for label in example["edit_labels"]]
            for pos, gold_label in enumerate(gold_labels):
                predicted = hybrid_decoded["predicted_labels"][pos] if pos < len(hybrid_decoded["predicted_labels"]) else None
                if gold_label != "COPY" or predicted == "COPY" or predicted is None:
                    continue
                trace = next((item for item in hybrid_decoded["trace"] if item["pos"] == pos), {})
                support_rule_label = trace.get("support_rule_label")
                forced_by_rule = bool(trace.get("forced_by_rule", False))
                if forced_by_rule:
                    likely_source = "hybrid_forced_by_support_rule"
                elif support_rule_label and support_rule_label != "COPY":
                    likely_source = "support_rule_positive_but_not_forced"
                elif neural_decoded["predicted_labels"][pos] == predicted:
                    likely_source = "neural_prediction_not_vetoed"
                else:
                    likely_source = "decode_alignment_or_trace_mismatch"
                reports.append(
                    {
                        "example_id": example["example_id"],
                        "pos": pos,
                        "gold_label": gold_label,
                        "predicted_label": predicted,
                        "support_rule_label": support_rule_label,
                        "neural_only_label": neural_decoded["predicted_labels"][pos] if pos < len(neural_decoded["predicted_labels"]) else None,
                        "forced_by_rule": forced_by_rule,
                        "veto_status": trace.get("veto_reasons", []),
                        "candidate_label": trace.get("candidate_label"),
                        "likely_source": likely_source,
                        "support_evidence": _support_evidence(example, pos),
                        "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs[pos]),
                        "sub_base_probs": _named_probs(BASES, sub_probs[pos]),
                        "ins_base_probs": _named_probs(BASES, ins_probs[pos]),
                        "ins_payload_diagnostic": _ins_payload_diagnostic(predicted, ins_probs[pos]),
                        "decoder_trace": trace,
                    }
                )
    return reports


def vetoed_true_edit_diagnostics(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> list[dict]:
    """Find recall candidates where neural-only was right but hybrid copied."""
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    reports = []
    neural_config = deepcopy(config["decode"])
    neural_config["hybrid_rule_decode"] = False
    with torch.no_grad():
        for batch in loader:
            example = batch["raw_examples"][0]
            outputs = model(batch["target_tokens"].to(device), batch["pileup_features"].to(device), batch["rule_features"].to(device))
            length = len(example["target_seq"])
            sliced = {
                "type_logits": outputs["type_logits"][0, :length].detach().cpu(),
                "sub_base_logits": outputs["sub_base_logits"][0, :length].detach().cpu(),
                "ins_base_logits": outputs["ins_base_logits"][0, :length].detach().cpu(),
                "edit_logits": outputs["edit_logits"][0, :length].detach().cpu(),
                "delete_candidate_logits": outputs["delete_candidate_logits"][0, :length].detach().cpu(),
                "delete_length_logits": outputs["delete_length_logits"][0, :length].detach().cpu(),
                "trust": outputs["trust"][0, :length].detach().cpu(),
            }
            hybrid_decoded = decode_example(example["target_seq"], example, sliced, config["decode"])
            neural_decoded = decode_example(example["target_seq"], example, sliced, neural_config)
            type_probs = torch.softmax(sliced["type_logits"], dim=-1)
            sub_probs = torch.softmax(sliced["sub_base_logits"], dim=-1)
            ins_probs = torch.softmax(sliced["ins_base_logits"], dim=-1)
            gold_labels = [_gold_label_name(label) for label in example["edit_labels"]]
            for pos, gold_label in enumerate(gold_labels):
                if gold_label == "COPY":
                    continue
                neural_label = neural_decoded["predicted_labels"][pos] if pos < len(neural_decoded["predicted_labels"]) else None
                hybrid_label = hybrid_decoded["predicted_labels"][pos] if pos < len(hybrid_decoded["predicted_labels"]) else None
                if neural_label != gold_label or hybrid_label != "COPY":
                    continue
                trace = next((item for item in hybrid_decoded["trace"] if item["pos"] == pos), {})
                reports.append(
                    {
                        "example_id": example["example_id"],
                        "pos": pos,
                        "gold_label": gold_label,
                        "neural_only_label": neural_label,
                        "hybrid_label": hybrid_label,
                        "support_rule_label": trace.get("support_rule_label"),
                        "forced_by_rule": bool(trace.get("forced_by_rule", False)),
                        "veto_reasons": trace.get("veto_reasons", []),
                        "candidate_label": trace.get("candidate_label"),
                        "support_evidence": _support_evidence(example, pos),
                        "type_probs": _named_probs(EDIT_TYPE_LABELS, type_probs[pos]),
                        "sub_base_probs": _named_probs(BASES, sub_probs[pos]),
                        "ins_base_probs": _named_probs(BASES, ins_probs[pos]),
                        "ins_payload_diagnostic": _ins_payload_diagnostic(gold_label, ins_probs[pos]),
                        "decoder_trace": trace,
                    }
                )
    return reports


def hybrid_gap_report(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict:
    model, device = load_model_for_inspection(config, checkpoint_path)
    dataset_dir = resolve_path(config["dataset"]["output_dir"])
    _, loader = make_loader(dataset_dir / f"{split}.jsonl", batch_size=1, shuffle=False)
    return hybrid_gap_diagnostics(model, loader, device, config["decode"])


def hard_edit_learnability_summary(class_stats: dict) -> dict:
    return {
        "SUB_*": {label: class_stats[label] for label in ["SUB_A", "SUB_C", "SUB_G", "SUB_T"]},
        "DEL": class_stats["DEL"],
        "INS_*": {label: class_stats[label] for label in ["INS_A", "INS_C", "INS_G", "INS_T"]},
    }


def print_inspection_report(reports: list[dict]) -> None:
    for report in reports:
        print("=" * 100)
        print("example_id:", report["example_id"])
        print("target    :", report["target_seq"])
        print("truth     :", report["truth_seq"])
        print("prediction:", report["prediction"])
        print("diagnosis :", report["diagnosis"])
        print("gold      :", report["gold_labels"])
        print("argmax    :", report["argmax_labels"])
        print("decoded   :", report["decoded_labels"])
        print("-" * 100)
        for position in report["positions"]:
            print(
                f"pos={position['pos']} base={position['target_base']} gold={position['gold_label']} "
                f"argmax={position['argmax_label']} decoded={position['decoded_label']} "
                f"trust={position['trust']} agreement={position['support_agreement']} entropy={position['support_entropy']}"
            )
            print("  top_edit_candidates:", position["top_edit_candidates"])
            print("  type_probs          :", position["type_probs"])
            print("  sub_base_probs      :", position["sub_base_probs"])
            print("  ins_base_probs      :", position["ins_base_probs"])
            if position["ins_payload_diagnostic"] is not None:
                print("  ins_payload_diagnostic:", position["ins_payload_diagnostic"])
            print("  delete_candidate_prob:", position["delete_candidate_prob"])
            print("  delete_length_probs :", position["delete_length_probs"])
            print("  decoder_trace       :", position["decoder_trace"])


def print_argmax_reports(reports: list[dict]) -> None:
    for report in reports:
        print("=" * 100)
        print("example_id:", report["example_id"])
        print("target    :", report["target_seq"])
        print("truth     :", report["truth_seq"])
        print("prediction:", report["prediction"])
        print("gold      :", report["gold_labels"])
        print("argmax    :", report["argmax_labels"])
        print("-" * 100)
        for position in report["positions"]:
            print(
                f"pos={position['pos']} gold={position['gold_label']} argmax={position['argmax_label']}"
            )
            print("  top_edit_candidates:", position["top_edit_candidates"])
            print("  type_probs         :", position["type_probs"])
            if position["ins_payload_diagnostic"] is not None:
                print("  ins_payload_diagnostic:", position["ins_payload_diagnostic"])
            print("  decoder_trace      :", position["decoder_trace"])


def print_missed_evidence_reports(reports: list[dict]) -> None:
    for report in reports:
        print("=" * 100)
        print(
            f"example_id={report['example_id']} pos={report['pos']} "
            f"gold={report['gold_label']} predicted={report['predicted_label']}"
        )
        print("  support_evidence:", report["support_evidence"])
        print("  type_probs      :", report["type_probs"])
        print("  sub_base_probs  :", report["sub_base_probs"])
        print("  ins_base_probs  :", report["ins_base_probs"])


def print_false_sub_reports(reports: list[dict]) -> None:
    print(f"false_sub_count={len(reports)}")
    for report in reports:
        print("=" * 100)
        print(
            f"example_id={report['example_id']} pos={report['pos']} "
            f"hybrid={report['hybrid_label']} neural_only={report['neural_only_label']} "
            f"forced_by_rule={report['forced_by_rule']}"
        )
        print("  target_base     :", report["target_base"])
        print("  support_evidence:", report["support_evidence"])
        print("  type_probs      :", report["type_probs"])
        print("  sub_base_probs  :", report["sub_base_probs"])
        print("  veto_reasons    :", report["veto_reasons"])


def print_insertion_payload_reports(reports: list[dict]) -> None:
    print(f"insertion_case_count={len(reports)}")
    for report in reports:
        print("=" * 100)
        print(
            f"example_id={report['example_id']} pos={report['pos']} "
            f"gold={report['gold_label']} decoded={report['decoded_label']} "
            f"correct={report['is_correct']} forced_by_rule={report['forced_by_rule']}"
        )
        print("  support_evidence       :", report["support_evidence"])
        print("  type_probs             :", report["type_probs"])
        print("  ins_payload_diagnostic :", report["ins_payload_diagnostic"])
        print("  veto_reasons           :", report["veto_reasons"])


def print_support_rule_audit(report: dict) -> None:
    summary = report["summary"]
    print("support_rule_positive_audit:")
    print(summary)
    if report["false_positive_rows"]:
        print("=" * 100)
        print("false positives:")
        for row in report["false_positive_rows"]:
            print(
                f"example_id={row['example_id']} pos={row['pos']} rule={row['support_rule_label']} "
                f"gold={row['gold_label']} hybrid={row['hybrid_label']} boundary={row['read_boundary_position']}"
            )
            print(
                "  evidence:",
                {
                    "support_fraction": row["support_fraction"],
                    "support_margin": row["support_margin"],
                    "entropy": row["entropy"],
                    "del_fraction": row["del_fraction"],
                    "neighboring_edit_distance": row["neighboring_edit_distance"],
                    "homopolymer_run_length": row["homopolymer_run_length"],
                    "neural_type_prob": row["neural_type_prob"],
                    "payload_confidence": row["payload_confidence"],
                },
            )
            print("  support_evidence:", row["support_evidence"])
            print("  veto_reasons    :", row["veto_reasons"])
    if report["true_positive_rows"]:
        print("=" * 100)
        print("true positives:")
        for row in report["true_positive_rows"]:
            print(
                f"example_id={row['example_id']} pos={row['pos']} rule={row['support_rule_label']} "
                f"hybrid={row['hybrid_label']} corrected={row['is_corrected_by_hybrid']} "
                f"payload={row['payload_confidence']} type={row['neural_type_prob']}"
            )


def print_support_rule_calibration_report(report: dict) -> None:
    print("support_rule_calibration:")
    print("status:", report["status"])
    print("train_summary:", report["train_summary"])
    print("eval_summary :", report["eval_summary"])
    if report["status"] == "ok":
        print("threshold_selected_for_zero_train_fp:", report["threshold_selected_for_zero_train_fp"])
        print("threshold_selected_for_zero_eval_fp :", report["threshold_selected_for_zero_eval_fp"])
        print("eval_tuned_eval_summary:", report["eval_tuned_eval_summary"])
        print("per_family:", report["per_family"])
        print("weights:", report["weights"])


def print_calibration_allowed_missed_edits(reports: list[dict]) -> None:
    print(f"calibration_allowed_missed_edit_count={len(reports)}")
    for report in reports:
        print("=" * 100)
        print(
            f"example_id={report['example_id']} pos={report['pos']} "
            f"rule={report['support_rule_label']} gold={report['gold_label']} "
            f"hybrid={report['hybrid_label']} neural_only={report['neural_only_label']} "
            f"score={report['allow_edit_probability']}"
        )
        print("  veto_reasons    :", report["veto_reasons"])
        print(
            "  evidence        :",
            {
                "support_fraction": report["support_fraction"],
                "support_margin": report["support_margin"],
                "entropy": report["entropy"],
                "del_fraction": report["del_fraction"],
                "neighbor_edit_proximity": report["neighbor_edit_proximity"],
                "read_boundary_position": report["read_boundary_position"],
                "neural_type_prob": report["neural_type_prob"],
                "payload_confidence": report["payload_confidence"],
            },
        )
        print("  support_evidence:", report["support_evidence"])


def print_hybrid_miss_reports(reports: list[dict]) -> None:
    print(f"hybrid_missed_hard_edit_count={len(reports)}")
    for report in reports:
        print("=" * 100)
        print(
            f"example_id={report['example_id']} pos={report['pos']} boundary={report['is_boundary_position']} "
            f"gold={report['gold_label']} hybrid={report['hybrid_label']} "
            f"neural_only={report['neural_only_label']} argmax={report['argmax_label']}"
        )
        print("  support_rule_label     :", report["support_rule_label"])
        print("  forced_by_rule         :", report["forced_by_rule"])
        print("  veto_reasons           :", report["veto_reasons"])
        print("  candidate_label        :", report["candidate_label"])
        print("  rescue_diagnostic      :", report["rescue_diagnostic"])
        print("  target_prefix          :", report["target_prefix"])
        print("  truth_prefix           :", report["truth_prefix"])
        print("  support_evidence       :", report["support_evidence"])
        print("  type_probs             :", report["type_probs"])
        print("  sub_base_probs         :", report["sub_base_probs"])
        print("  ins_base_probs         :", report["ins_base_probs"])
        if report["ins_payload_diagnostic"] is not None:
            print("  ins_payload_diagnostic :", report["ins_payload_diagnostic"])
        print("  decoder_trace          :", report["decoder_trace"])


def print_false_hard_edit_reports(reports: list[dict]) -> None:
    print(f"false_hard_edit_count={len(reports)}")
    for report in reports:
        print("=" * 100)
        print(
            f"example_id={report['example_id']} pos={report['pos']} "
            f"gold={report['gold_label']} predicted={report['predicted_label']} "
            f"support_rule={report['support_rule_label']} neural_only={report['neural_only_label']}"
        )
        print("  likely_source          :", report["likely_source"])
        print("  forced_by_rule         :", report["forced_by_rule"])
        print("  veto_status            :", report["veto_status"])
        print("  candidate_label        :", report["candidate_label"])
        print("  support_evidence       :", report["support_evidence"])
        print("  type_probs             :", report["type_probs"])
        print("  sub_base_probs         :", report["sub_base_probs"])
        print("  ins_base_probs         :", report["ins_base_probs"])
        if report["ins_payload_diagnostic"] is not None:
            print("  ins_payload_diagnostic :", report["ins_payload_diagnostic"])
        print("  decoder_trace          :", report["decoder_trace"])


def print_vetoed_true_edit_reports(reports: list[dict]) -> None:
    print(f"vetoed_true_edit_count={len(reports)}")
    for report in reports:
        print("=" * 100)
        print(
            f"example_id={report['example_id']} pos={report['pos']} "
            f"gold={report['gold_label']} neural_only={report['neural_only_label']} "
            f"hybrid={report['hybrid_label']} support_rule={report['support_rule_label']}"
        )
        print("  forced_by_rule         :", report["forced_by_rule"])
        print("  veto_reasons           :", report["veto_reasons"])
        print("  candidate_label        :", report["candidate_label"])
        print("  support_evidence       :", report["support_evidence"])
        print("  type_probs             :", report["type_probs"])
        print("  sub_base_probs         :", report["sub_base_probs"])
        print("  ins_base_probs         :", report["ins_base_probs"])
        if report["ins_payload_diagnostic"] is not None:
            print("  ins_payload_diagnostic :", report["ins_payload_diagnostic"])
        print("  decoder_trace          :", report["decoder_trace"])


def print_hybrid_gap_report(report: dict) -> None:
    print("Hybrid gap summary:")
    print(report["summary"])
    print("=" * 100)
    for row in report["positions"]:
        print(
            f"example_id={row['example_id']} pos={row['pos']} gold={row['gold']} "
            f"neural_argmax={row['neural_argmax']} neural_only={row['neural_only_final']} "
            f"hybrid={row['hybrid_final']} forced_by_rule={row['forced_by_rule']}"
        )
        print("  type_probs:", row["type_probs"])
        print(
            "  support:",
            {
                "base_counts": row["support_base_counts"],
                "ins_count": row["support_ins_count"],
                "del_count": row["support_del_count"],
                "agreement": row["support_agreement"],
                "entropy": row["support_entropy"],
            },
        )
