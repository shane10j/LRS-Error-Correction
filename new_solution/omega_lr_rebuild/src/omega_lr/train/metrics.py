"""Cheap but structured train-time metrics."""

from __future__ import annotations

import torch

from omega_lr.constants import BASES, EDIT_TYPE_LABELS, EDIT_TYPE_TO_ID
from omega_lr.train.losses import structured_targets


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(max(denominator, 1.0))


def batch_label_metrics(batch: dict, outputs: dict) -> dict[str, float]:
    mask = batch["attention_mask"] > 0.5
    edit_predictions = outputs["edit_logits"].argmax(dim=-1).detach().cpu()
    edit_labels = batch["edit_labels"].detach().cpu()
    type_labels, sub_base_labels, ins_base_labels = structured_targets(edit_labels)
    type_predictions = outputs["type_logits"].argmax(dim=-1).detach().cpu()
    sub_predictions = outputs["sub_base_logits"].argmax(dim=-1).detach().cpu()
    ins_predictions = outputs["ins_base_logits"].argmax(dim=-1).detach().cpu()
    ins_probs = torch.softmax(outputs["ins_base_logits"], dim=-1).detach().cpu()
    del_length_predictions = outputs["delete_length_logits"].argmax(dim=-1).detach().cpu()
    delete_length_labels = batch["delete_length_labels"].detach().cpu()
    type_logits = outputs["type_logits"].detach().cpu()

    metrics: dict[str, float] = {
        "total_tokens": float(mask.sum().item()),
        "correct_labels": float(((edit_predictions == edit_labels) & mask).sum().item()),
        "hard_pred_count": float(((type_predictions != EDIT_TYPE_TO_ID["COPY"]) & mask).sum().item()),
        "hard_true_count": float(((type_labels != EDIT_TYPE_TO_ID["COPY"]) & mask).sum().item()),
    }

    active_type_labels = type_labels[mask]
    active_type_predictions = type_predictions[mask]
    for gold_idx, gold_name in enumerate(EDIT_TYPE_LABELS):
        for pred_idx, pred_name in enumerate(EDIT_TYPE_LABELS):
            key = f"type_confusion_{gold_name}_to_{pred_name}"
            metrics[key] = float(((active_type_labels == gold_idx) & (active_type_predictions == pred_idx)).sum().item())
        tp = ((active_type_labels == gold_idx) & (active_type_predictions == gold_idx)).sum().item()
        fp = ((active_type_labels != gold_idx) & (active_type_predictions == gold_idx)).sum().item()
        fn = ((active_type_labels == gold_idx) & (active_type_predictions != gold_idx)).sum().item()
        metrics[f"type_{gold_name}_tp"] = float(tp)
        metrics[f"type_{gold_name}_fp"] = float(fp)
        metrics[f"type_{gold_name}_fn"] = float(fn)

    metrics["copy_to_del_false_positive_count"] = float(
        ((active_type_labels == EDIT_TYPE_TO_ID["COPY"]) & (active_type_predictions == EDIT_TYPE_TO_ID["DEL"])).sum().item()
    )
    metrics["copy_to_sub_false_positive_count"] = float(
        ((active_type_labels == EDIT_TYPE_TO_ID["COPY"]) & (active_type_predictions == EDIT_TYPE_TO_ID["SUB"])).sum().item()
    )
    metrics["copy_to_ins_false_positive_count"] = float(
        ((active_type_labels == EDIT_TYPE_TO_ID["COPY"]) & (active_type_predictions == EDIT_TYPE_TO_ID["INS"])).sum().item()
    )
    metrics["sub_to_copy_miss_count"] = float(
        ((active_type_labels == EDIT_TYPE_TO_ID["SUB"]) & (active_type_predictions == EDIT_TYPE_TO_ID["COPY"])).sum().item()
    )
    metrics["sub_to_del_confusion_count"] = float(
        ((active_type_labels == EDIT_TYPE_TO_ID["SUB"]) & (active_type_predictions == EDIT_TYPE_TO_ID["DEL"])).sum().item()
    )
    metrics["ins_to_copy_miss_count"] = float(
        ((active_type_labels == EDIT_TYPE_TO_ID["INS"]) & (active_type_predictions == EDIT_TYPE_TO_ID["COPY"])).sum().item()
    )
    metrics["ins_to_del_confusion_count"] = float(
        ((active_type_labels == EDIT_TYPE_TO_ID["INS"]) & (active_type_predictions == EDIT_TYPE_TO_ID["DEL"])).sum().item()
    )

    sub_type_correct = mask & (type_labels == EDIT_TYPE_TO_ID["SUB"]) & (type_predictions == EDIT_TYPE_TO_ID["SUB"])
    ins_type_correct = mask & (type_labels == EDIT_TYPE_TO_ID["INS"]) & (type_predictions == EDIT_TYPE_TO_ID["INS"])
    del_type_correct = mask & (type_labels == EDIT_TYPE_TO_ID["DEL"]) & (type_predictions == EDIT_TYPE_TO_ID["DEL"])
    del_gold = mask & (type_labels == EDIT_TYPE_TO_ID["DEL"])
    hard_gold = mask & (type_labels != EDIT_TYPE_TO_ID["COPY"])
    hard_copy_top = hard_gold & (type_predictions == EDIT_TYPE_TO_ID["COPY"])
    copy_logits = type_logits[..., EDIT_TYPE_TO_ID["COPY"]]
    gold_type_logits = type_logits.gather(-1, type_labels.unsqueeze(-1)).squeeze(-1)
    hard_margins = gold_type_logits[hard_gold] - copy_logits[hard_gold]
    metrics["sub_payload_type_correct_total"] = float(sub_type_correct.sum().item())
    metrics["sub_payload_type_correct_correct"] = float(
        (sub_predictions[sub_type_correct] == sub_base_labels[sub_type_correct]).sum().item()
    )
    metrics["ins_payload_type_correct_total"] = float(ins_type_correct.sum().item())
    metrics["ins_payload_type_correct_correct"] = float(
        (ins_predictions[ins_type_correct] == ins_base_labels[ins_type_correct]).sum().item()
    )
    ins_gold = mask & (type_labels == EDIT_TYPE_TO_ID["INS"])
    metrics["ins_gold_total"] = float(ins_gold.sum().item())
    metrics["ins_wrong_payload_count"] = float(
        ((ins_predictions != ins_base_labels) & ins_gold & (type_predictions == EDIT_TYPE_TO_ID["INS"])).sum().item()
    )
    if ins_gold.any():
        gold_ins_probs = ins_probs.gather(-1, ins_base_labels.unsqueeze(-1)).squeeze(-1)
        gold_ranks = (ins_probs > gold_ins_probs.unsqueeze(-1)).sum(dim=-1) + 1
        metrics["ins_payload_gold_rank_total"] = float(gold_ranks[ins_gold].sum().item())
    else:
        metrics["ins_payload_gold_rank_total"] = 0.0
    for base_idx, base in enumerate(BASES):
        sub_base_mask = mask & (type_labels == EDIT_TYPE_TO_ID["SUB"]) & (sub_base_labels == base_idx)
        metrics[f"sub_{base}_gold_total"] = float(sub_base_mask.sum().item())
        metrics[f"sub_{base}_exact_correct"] = float(
            (sub_base_mask & (type_predictions == EDIT_TYPE_TO_ID["SUB"]) & (sub_predictions == base_idx)).sum().item()
        )
        base_mask = ins_gold & (ins_base_labels == base_idx)
        metrics[f"ins_{base}_gold_total"] = float(base_mask.sum().item())
        metrics[f"ins_{base}_payload_correct"] = float((base_mask & (ins_predictions == base_idx)).sum().item())
        metrics[f"ins_{base}_exact_correct"] = float(
            (base_mask & (type_predictions == EDIT_TYPE_TO_ID["INS"]) & (ins_predictions == base_idx)).sum().item()
        )
        if base_mask.any():
            gold_ins_probs = ins_probs.gather(-1, ins_base_labels.unsqueeze(-1)).squeeze(-1)
            gold_ranks = (ins_probs > gold_ins_probs.unsqueeze(-1)).sum(dim=-1) + 1
            metrics[f"ins_{base}_gold_rank_total"] = float(gold_ranks[base_mask].sum().item())
        else:
            metrics[f"ins_{base}_gold_rank_total"] = 0.0
    metrics["del_length_gold_total"] = float(del_gold.sum().item())
    metrics["del_length_type_correct_total"] = float(del_type_correct.sum().item())
    metrics["del_length_type_correct_correct"] = float(
        (del_length_predictions[del_type_correct] == delete_length_labels[del_type_correct]).sum().item()
    )
    metrics["hard_copy_top_count"] = float(hard_copy_top.sum().item())
    metrics["hard_margin_total"] = float(hard_margins.sum().item()) if hard_margins.numel() else 0.0
    metrics["hard_margin_min"] = float(hard_margins.min().item()) if hard_margins.numel() else 0.0
    return metrics


def finalize_train_metrics(totals: dict[str, float]) -> dict[str, object]:
    total_tokens = totals.get("total_tokens", 0.0)
    metrics: dict[str, object] = {
        "label_accuracy": round(_safe_div(totals.get("correct_labels", 0.0), total_tokens), 4),
        "predicted_hard_edit_rate": round(_safe_div(totals.get("hard_pred_count", 0.0), total_tokens), 4),
        "true_hard_edit_rate": round(_safe_div(totals.get("hard_true_count", 0.0), total_tokens), 4),
        "copy_to_del_false_positive_count": int(totals.get("copy_to_del_false_positive_count", 0.0)),
        "copy_to_sub_false_positive_count": int(totals.get("copy_to_sub_false_positive_count", 0.0)),
        "copy_to_ins_false_positive_count": int(totals.get("copy_to_ins_false_positive_count", 0.0)),
        "sub_to_copy_miss_count": int(totals.get("sub_to_copy_miss_count", 0.0)),
        "sub_to_del_confusion_count": int(totals.get("sub_to_del_confusion_count", 0.0)),
        "ins_to_copy_miss_count": int(totals.get("ins_to_copy_miss_count", 0.0)),
        "ins_to_del_confusion_count": int(totals.get("ins_to_del_confusion_count", 0.0)),
        "ins_wrong_payload_count": int(totals.get("ins_wrong_payload_count", 0.0)),
        "ins_payload_avg_gold_rank": round(
            _safe_div(totals.get("ins_payload_gold_rank_total", 0.0), totals.get("ins_gold_total", 0.0)),
            4,
        ),
        "payload_when_type_correct": {
            "SUB": {
                "count": int(totals.get("sub_payload_type_correct_total", 0.0)),
                "accuracy": round(
                    _safe_div(
                        totals.get("sub_payload_type_correct_correct", 0.0),
                        totals.get("sub_payload_type_correct_total", 0.0),
                    ),
                    4,
                ),
            },
            "INS": {
                "count": int(totals.get("ins_payload_type_correct_total", 0.0)),
                "accuracy": round(
                    _safe_div(
                        totals.get("ins_payload_type_correct_correct", 0.0),
                        totals.get("ins_payload_type_correct_total", 0.0),
                    ),
                    4,
                ),
            },
            "DEL_LENGTH": {
                "gold_count": int(totals.get("del_length_gold_total", 0.0)),
                "type_correct_count": int(totals.get("del_length_type_correct_total", 0.0)),
                "accuracy_when_type_correct": round(
                    _safe_div(
                        totals.get("del_length_type_correct_correct", 0.0),
                        totals.get("del_length_type_correct_total", 0.0),
                    ),
                    4,
                ),
            },
        },
        "copy_vs_hard_margins": {
            "gold_hard_count": int(totals.get("hard_true_count", 0.0)),
            "copy_wins_on_gold_hard_count": int(totals.get("hard_copy_top_count", 0.0)),
            "avg_gold_minus_copy_logit": round(
                _safe_div(totals.get("hard_margin_total", 0.0), totals.get("hard_true_count", 0.0)),
                4,
            ),
            "min_gold_minus_copy_logit": round(totals.get("hard_margin_min", 0.0), 4),
        },
        "ins_payload_by_base": {
            base: {
                "count": int(totals.get(f"ins_{base}_gold_total", 0.0)),
                "accuracy": round(
                    _safe_div(
                        totals.get(f"ins_{base}_payload_correct", 0.0),
                        totals.get(f"ins_{base}_gold_total", 0.0),
                    ),
                    4,
                ),
                "avg_gold_rank": round(
                    _safe_div(
                        totals.get(f"ins_{base}_gold_rank_total", 0.0),
                        totals.get(f"ins_{base}_gold_total", 0.0),
                    ),
                    4,
                ),
            }
            for base in BASES
        },
        "per_base_edit_recall": {
            **{
                f"SUB_{base}": {
                    "count": int(totals.get(f"sub_{base}_gold_total", 0.0)),
                    "correct": int(totals.get(f"sub_{base}_exact_correct", 0.0)),
                    "recall": round(
                        _safe_div(
                            totals.get(f"sub_{base}_exact_correct", 0.0),
                            totals.get(f"sub_{base}_gold_total", 0.0),
                        ),
                        4,
                    ),
                }
                for base in BASES
            },
            **{
                f"INS_{base}": {
                    "count": int(totals.get(f"ins_{base}_gold_total", 0.0)),
                    "correct": int(totals.get(f"ins_{base}_exact_correct", 0.0)),
                    "recall": round(
                        _safe_div(
                            totals.get(f"ins_{base}_exact_correct", 0.0),
                            totals.get(f"ins_{base}_gold_total", 0.0),
                        ),
                        4,
                    ),
                }
                for base in BASES
            },
        },
    }

    confusion = {}
    per_type = {}
    for type_name in EDIT_TYPE_LABELS:
        confusion[type_name] = {
            pred_name: int(totals.get(f"type_confusion_{type_name}_to_{pred_name}", 0.0))
            for pred_name in EDIT_TYPE_LABELS
        }
        tp = totals.get(f"type_{type_name}_tp", 0.0)
        fp = totals.get(f"type_{type_name}_fp", 0.0)
        fn = totals.get(f"type_{type_name}_fn", 0.0)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall) if precision + recall > 0 else 0.0
        per_type[type_name] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    metrics["type_confusion"] = confusion
    metrics["per_type"] = per_type
    return metrics
