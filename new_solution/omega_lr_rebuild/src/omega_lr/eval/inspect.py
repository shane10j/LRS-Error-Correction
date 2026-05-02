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
