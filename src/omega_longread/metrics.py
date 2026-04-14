from __future__ import annotations

import math
from typing import Dict, List

import torch

from .decode import apply_edit_ops
from .support import compute_support_statistics
from .vocab import ID_TO_BASE
from .vocab import EDIT_TO_ID, PAD_EDIT_ID

_COPY_ID = EDIT_TO_ID["COPY"]
_DEL_ID = EDIT_TO_ID["DEL"]
_SUB_IDS = {EDIT_TO_ID["SUB_A"], EDIT_TO_ID["SUB_C"], EDIT_TO_ID["SUB_G"], EDIT_TO_ID["SUB_T"]}
_INS_IDS = {EDIT_TO_ID["INS_A"], EDIT_TO_ID["INS_C"], EDIT_TO_ID["INS_G"], EDIT_TO_ID["INS_T"]}
_HOMOPOLYMER_THRESHOLD = 3
_LOW_COMPLEXITY_THRESHOLD = 0.55
_REPEAT_RICH_THRESHOLD = 0.45
_LOW_SUPPORT_ENTROPY_THRESHOLD = 0.25
_HIGH_SUPPORT_ENTROPY_THRESHOLD = 0.75
_LOW_SUPPORT_AGREEMENT_THRESHOLD = 0.55
_HIGH_SUPPORT_AGREEMENT_THRESHOLD = 0.9


def _decode_base_ids(ids: torch.Tensor) -> str:
    return "".join(ID_TO_BASE.get(int(token), "N") for token in ids.tolist())


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _mean_burst_length(counts: List[int]) -> float:
    bursts: List[int] = []
    current = 0
    for count in counts:
        if count > 0:
            current += count
        elif current > 0:
            bursts.append(current)
            current = 0
    if current > 0:
        bursts.append(current)
    if not bursts:
        return 0.0
    return sum(bursts) / len(bursts)


def _base_complexity(seq: str) -> float:
    if not seq:
        return 0.0
    counts = {base: seq.count(base) for base in "ACGT"}
    probs = [count / len(seq) for count in counts.values() if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(4.0)


def _repeat_richness(seq: str, k: int = 3) -> float:
    if len(seq) < k:
        return 0.0
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    unique_ratio = len(set(kmers)) / max(len(kmers), 1)
    return 1.0 - unique_ratio


def summarize_edit_label_predictions(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    valid = labels.ne(PAD_EDIT_ID)
    correct = (preds == labels) & valid
    accuracy = correct.sum().float() / valid.sum().clamp_min(1)

    core_preds = preds[:, :, -1]
    core_labels = labels[:, :, -1]
    core_valid = core_labels.ne(PAD_EDIT_ID)
    core_accuracy = ((core_preds == core_labels) & core_valid).sum().float() / core_valid.sum().clamp_min(1)

    pred_valid = core_preds[core_valid]
    label_valid = core_labels[core_valid]
    ins_preds = preds[:, :, :-1][valid[:, :, :-1]]
    ins_labels = labels[:, :, :-1][valid[:, :, :-1]]
    out = {
        "edit_accuracy": float(accuracy.cpu()),
        "core_edit_accuracy": float(core_accuracy.cpu()),
    }
    for name, token_id in [("copy", _COPY_ID), ("delete", _DEL_ID)]:
        denom = (label_valid == token_id).sum().clamp_min(1)
        rec = ((pred_valid == token_id) & (label_valid == token_id)).sum().float() / denom
        out[f"{name}_recall"] = float(rec.cpu())
        pred_denom = (pred_valid == token_id).sum().clamp_min(1)
        prec = ((pred_valid == token_id) & (label_valid == token_id)).sum().float() / pred_denom
        out[f"{name}_precision"] = float(prec.cpu())

    for name, token_ids, pred_tensor, label_tensor in [
        ("sub", _SUB_IDS, pred_valid, label_valid),
        ("ins", _INS_IDS, ins_preds, ins_labels),
    ]:
        pred_mask = torch.zeros_like(pred_tensor, dtype=torch.bool)
        label_mask = torch.zeros_like(label_tensor, dtype=torch.bool)
        for token_id in token_ids:
            label_mask |= label_tensor == token_id
            pred_mask |= pred_tensor == token_id
        rec = (pred_mask & label_mask).sum().float() / label_mask.sum().clamp_min(1)
        prec = (pred_mask & label_mask).sum().float() / pred_mask.sum().clamp_min(1)
        out[f"{name}_recall"] = float(rec.cpu())
        out[f"{name}_precision"] = float(prec.cpu())
    return out


def summarize_hard_edit_precision_stratified(
    preds: torch.Tensor,
    labels: torch.Tensor,
    target_run_lengths: torch.Tensor,
    support_base_support: torch.Tensor,
    support_del_mask: torch.Tensor | None = None,
    support_ins_base_support: torch.Tensor | None = None,
) -> Dict[str, float]:
    preds = preds.detach()
    labels = labels.detach()
    target_run_lengths = target_run_lengths.detach()
    support_stats = compute_support_statistics(
        support_base_support.detach(),
        None if support_del_mask is None else support_del_mask.detach(),
        None if support_ins_base_support is None else support_ins_base_support.detach(),
    )

    core_preds = preds[:, :, -1]
    core_labels = labels[:, :, -1]
    core_valid = core_labels.ne(PAD_EDIT_ID)
    pred_sub_mask = torch.zeros_like(core_preds, dtype=torch.bool)
    label_sub_mask = torch.zeros_like(core_labels, dtype=torch.bool)
    for token_id in _SUB_IDS:
        pred_sub_mask |= core_preds == token_id
        label_sub_mask |= core_labels == token_id
    pred_del_mask = core_preds == _DEL_ID
    label_del_mask = core_labels == _DEL_ID

    low_entropy_mask = support_stats["entropy"] <= _LOW_SUPPORT_ENTROPY_THRESHOLD
    high_entropy_mask = support_stats["entropy"] >= _HIGH_SUPPORT_ENTROPY_THRESHOLD
    low_agreement_mask = support_stats["agreement"] <= _LOW_SUPPORT_AGREEMENT_THRESHOLD
    high_agreement_mask = support_stats["agreement"] >= _HIGH_SUPPORT_AGREEMENT_THRESHOLD
    homopolymer_mask = target_run_lengths >= _HOMOPOLYMER_THRESHOLD
    non_homopolymer_mask = ~homopolymer_mask

    def precision(pred_mask: torch.Tensor, label_mask: torch.Tensor, region_mask: torch.Tensor) -> float:
        mask = pred_mask & region_mask & core_valid
        if mask.sum() == 0:
            return 0.0
        correct = mask & label_mask
        return float((correct.sum().float() / mask.sum().float()).cpu())

    def pred_count(pred_mask: torch.Tensor, region_mask: torch.Tensor) -> float:
        return float((pred_mask & region_mask & core_valid).sum().cpu())

    out = {
        "sub_precision_low_entropy": precision(pred_sub_mask, label_sub_mask, low_entropy_mask),
        "sub_precision_high_entropy": precision(pred_sub_mask, label_sub_mask, high_entropy_mask),
        "sub_precision_low_agreement": precision(pred_sub_mask, label_sub_mask, low_agreement_mask),
        "sub_precision_high_agreement": precision(pred_sub_mask, label_sub_mask, high_agreement_mask),
        "sub_precision_homopolymer": precision(pred_sub_mask, label_sub_mask, homopolymer_mask),
        "sub_precision_non_homopolymer": precision(pred_sub_mask, label_sub_mask, non_homopolymer_mask),
        "delete_precision_low_entropy": precision(pred_del_mask, label_del_mask, low_entropy_mask),
        "delete_precision_high_entropy": precision(pred_del_mask, label_del_mask, high_entropy_mask),
        "delete_precision_low_agreement": precision(pred_del_mask, label_del_mask, low_agreement_mask),
        "delete_precision_high_agreement": precision(pred_del_mask, label_del_mask, high_agreement_mask),
        "delete_precision_homopolymer": precision(pred_del_mask, label_del_mask, homopolymer_mask),
        "delete_precision_non_homopolymer": precision(pred_del_mask, label_del_mask, non_homopolymer_mask),
        "sub_pred_count_low_entropy": pred_count(pred_sub_mask, low_entropy_mask),
        "sub_pred_count_high_entropy": pred_count(pred_sub_mask, high_entropy_mask),
        "sub_pred_count_low_agreement": pred_count(pred_sub_mask, low_agreement_mask),
        "sub_pred_count_high_agreement": pred_count(pred_sub_mask, high_agreement_mask),
        "delete_pred_count_low_entropy": pred_count(pred_del_mask, low_entropy_mask),
        "delete_pred_count_high_entropy": pred_count(pred_del_mask, high_entropy_mask),
        "delete_pred_count_low_agreement": pred_count(pred_del_mask, low_agreement_mask),
        "delete_pred_count_high_agreement": pred_count(pred_del_mask, high_agreement_mask),
    }
    return out


def summarize_edit_predictions(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = logits.argmax(dim=-1)
    return summarize_edit_label_predictions(preds, labels)


def summarize_sequence_label_predictions(
    preds: torch.Tensor,
    labels: torch.Tensor,
    target_bases: torch.Tensor,
    target_mask: torch.Tensor,
    target_run_lengths: torch.Tensor,
    metadata: List[Dict[str, object]],
    max_insertions_per_pos: int,
    support_base_support: torch.Tensor | None = None,
    support_del_mask: torch.Tensor | None = None,
    support_ins_base_support: torch.Tensor | None = None,
    trust_gate: torch.Tensor | None = None,
) -> Dict[str, float]:
    preds = preds.detach().cpu()
    labels_cpu = labels.detach().cpu()
    target_bases_cpu = target_bases.detach().cpu()
    target_mask_cpu = target_mask.detach().cpu()
    target_run_lengths_cpu = target_run_lengths.detach().cpu()

    seq_distances: List[float] = []
    seq_norm_distances: List[float] = []
    seq_identities: List[float] = []
    length_ratios: List[float] = []
    predicted_insertions_per_window: List[float] = []
    target_insertions_per_window: List[float] = []
    predicted_burst_lengths: List[float] = []
    target_burst_lengths: List[float] = []
    low_complexity_identities: List[float] = []
    repeat_rich_identities: List[float] = []
    low_support_entropy_identities: List[float] = []
    high_support_entropy_identities: List[float] = []
    low_support_agreement_identities: List[float] = []
    high_support_agreement_identities: List[float] = []

    support_entropy_by_window: List[float] = []
    support_agreement_by_window: List[float] = []
    trust_gate_by_window: List[float] = []
    low_support_entropy_window_count = 0
    high_support_entropy_window_count = 0
    low_support_agreement_window_count = 0
    high_support_agreement_window_count = 0
    if support_base_support is not None:
        support_stats = compute_support_statistics(
            support_base_support.detach(),
            None if support_del_mask is None else support_del_mask.detach(),
            None if support_ins_base_support is None else support_ins_base_support.detach(),
        )
        support_entropy_cpu = support_stats["entropy"].detach().cpu()
        support_agreement_cpu = support_stats["agreement"].detach().cpu()
    else:
        support_entropy_cpu = None
        support_agreement_cpu = None
    trust_gate_cpu = trust_gate.detach().cpu() if trust_gate is not None else None

    for i, meta in enumerate(metadata):
        valid_len = int(target_mask_cpu[i].sum().item())
        noisy_seq = _decode_base_ids(target_bases_cpu[i, :valid_len])
        pred_slots = preds[i, :valid_len].tolist()
        true_slots = labels_cpu[i, :valid_len].tolist()
        predicted_seq = apply_edit_ops(noisy_seq, pred_slots, max_insertions_per_pos=max_insertions_per_pos)
        truth_seq = meta.get("target_sequence", "") if isinstance(meta, dict) else ""
        if truth_seq:
            distance = levenshtein_distance(predicted_seq, truth_seq)
            denom = max(len(truth_seq), len(predicted_seq), 1)
            seq_distances.append(float(distance))
            seq_norm_distances.append(distance / denom)
            seq_identities.append(1.0 - (distance / denom))
            length_ratios.append(len(predicted_seq) / max(len(truth_seq), 1))
            complexity = _base_complexity(truth_seq)
            repeat_richness = _repeat_richness(truth_seq)
            if complexity <= _LOW_COMPLEXITY_THRESHOLD:
                low_complexity_identities.append(1.0 - (distance / denom))
            if repeat_richness >= _REPEAT_RICH_THRESHOLD:
                repeat_rich_identities.append(1.0 - (distance / denom))

        if support_entropy_cpu is not None:
            window_entropy = float(support_entropy_cpu[i, :valid_len].mean().item())
            support_entropy_by_window.append(window_entropy)
            if truth_seq:
                identity = seq_identities[-1]
                if window_entropy <= _LOW_SUPPORT_ENTROPY_THRESHOLD:
                    low_support_entropy_identities.append(identity)
                    low_support_entropy_window_count += 1
                if window_entropy >= _HIGH_SUPPORT_ENTROPY_THRESHOLD:
                    high_support_entropy_identities.append(identity)
                    high_support_entropy_window_count += 1
        if support_agreement_cpu is not None:
            window_agreement = float(support_agreement_cpu[i, :valid_len].mean().item())
            support_agreement_by_window.append(window_agreement)
            if truth_seq:
                identity = seq_identities[-1]
                if window_agreement <= _LOW_SUPPORT_AGREEMENT_THRESHOLD:
                    low_support_agreement_identities.append(identity)
                    low_support_agreement_window_count += 1
                if window_agreement >= _HIGH_SUPPORT_AGREEMENT_THRESHOLD:
                    high_support_agreement_identities.append(identity)
                    high_support_agreement_window_count += 1
        if trust_gate_cpu is not None:
            trust_gate_by_window.append(float(trust_gate_cpu[i, :valid_len].mean().item()))

        predicted_insert_counts = [
            sum(1 for token in slots[:-1] if token in _INS_IDS)
            for slots in pred_slots
        ]
        target_insert_counts = [
            sum(1 for token in slots[:-1] if token != PAD_EDIT_ID)
            for slots in true_slots
        ]
        predicted_insertions_per_window.append(float(sum(predicted_insert_counts)))
        target_insertions_per_window.append(float(sum(target_insert_counts)))
        predicted_burst_lengths.append(_mean_burst_length(predicted_insert_counts))
        target_burst_lengths.append(_mean_burst_length(target_insert_counts))

    core_preds = preds[:, :, -1]
    core_labels = labels_cpu[:, :, -1]
    core_valid = core_labels.ne(PAD_EDIT_ID)
    homopolymer_mask = target_run_lengths_cpu >= _HOMOPOLYMER_THRESHOLD
    non_homopolymer_mask = ~homopolymer_mask

    def masked_error_rate(mask: torch.Tensor) -> float:
        masked = mask & core_valid
        if masked.sum() == 0:
            return 0.0
        errors = (core_preds != core_labels) & masked
        return float((errors.sum().float() / masked.sum().float()).cpu())

    out = {
        "sequence_edit_distance": sum(seq_distances) / max(len(seq_distances), 1),
        "sequence_normalized_edit_distance": sum(seq_norm_distances) / max(len(seq_norm_distances), 1),
        "sequence_identity": sum(seq_identities) / max(len(seq_identities), 1),
        "predicted_length_ratio": sum(length_ratios) / max(len(length_ratios), 1),
        "predicted_insertions_per_window": sum(predicted_insertions_per_window) / max(len(predicted_insertions_per_window), 1),
        "target_insertions_per_window": sum(target_insertions_per_window) / max(len(target_insertions_per_window), 1),
        "predicted_insertion_burst_length": sum(predicted_burst_lengths) / max(len(predicted_burst_lengths), 1),
        "target_insertion_burst_length": sum(target_burst_lengths) / max(len(target_burst_lengths), 1),
        "homopolymer_core_error_rate": masked_error_rate(homopolymer_mask),
        "non_homopolymer_core_error_rate": masked_error_rate(non_homopolymer_mask),
        "low_complexity_sequence_identity": sum(low_complexity_identities) / max(len(low_complexity_identities), 1),
        "repeat_rich_sequence_identity": sum(repeat_rich_identities) / max(len(repeat_rich_identities), 1),
        "low_support_entropy_sequence_identity": sum(low_support_entropy_identities) / max(len(low_support_entropy_identities), 1),
        "high_support_entropy_sequence_identity": sum(high_support_entropy_identities) / max(len(high_support_entropy_identities), 1),
        "low_support_agreement_sequence_identity": sum(low_support_agreement_identities) / max(len(low_support_agreement_identities), 1),
        "high_support_agreement_sequence_identity": sum(high_support_agreement_identities) / max(len(high_support_agreement_identities), 1),
        "support_entropy_window_mean": sum(support_entropy_by_window) / max(len(support_entropy_by_window), 1),
        "support_agreement_window_mean": sum(support_agreement_by_window) / max(len(support_agreement_by_window), 1),
        "trust_gate_window_mean": sum(trust_gate_by_window) / max(len(trust_gate_by_window), 1),
        "low_support_entropy_window_count": float(low_support_entropy_window_count),
        "high_support_entropy_window_count": float(high_support_entropy_window_count),
        "low_support_agreement_window_count": float(low_support_agreement_window_count),
        "high_support_agreement_window_count": float(high_support_agreement_window_count),
    }
    return out


def summarize_sequence_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_bases: torch.Tensor,
    target_mask: torch.Tensor,
    target_run_lengths: torch.Tensor,
    metadata: List[Dict[str, object]],
    max_insertions_per_pos: int,
    support_base_support: torch.Tensor | None = None,
    support_del_mask: torch.Tensor | None = None,
    support_ins_base_support: torch.Tensor | None = None,
    trust_gate: torch.Tensor | None = None,
) -> Dict[str, float]:
    preds = logits.argmax(dim=-1)
    return summarize_sequence_label_predictions(
        preds,
        labels,
        target_bases,
        target_mask,
        target_run_lengths,
        metadata,
        max_insertions_per_pos,
        support_base_support=support_base_support,
        support_del_mask=support_del_mask,
        support_ins_base_support=support_ins_base_support,
        trust_gate=trust_gate,
    )


def summarize_support_trust(
    trust_gate: torch.Tensor,
    support_base_support: torch.Tensor,
    support_del_mask: torch.Tensor,
    support_ins_base_support: torch.Tensor,
    target_run_lengths: torch.Tensor,
    target_mask: torch.Tensor,
) -> Dict[str, float]:
    trust_gate = trust_gate.detach()
    target_run_lengths = target_run_lengths.detach()
    target_mask = target_mask.detach()
    support_stats = compute_support_statistics(
        support_base_support.detach(),
        support_del_mask.detach(),
        support_ins_base_support.detach(),
    )
    valid_mask = target_mask.bool()
    low_entropy_mask = support_stats["entropy"] <= _LOW_SUPPORT_ENTROPY_THRESHOLD
    high_entropy_mask = support_stats["entropy"] >= _HIGH_SUPPORT_ENTROPY_THRESHOLD
    low_agreement_mask = support_stats["agreement"] <= _LOW_SUPPORT_AGREEMENT_THRESHOLD
    high_agreement_mask = support_stats["agreement"] >= _HIGH_SUPPORT_AGREEMENT_THRESHOLD
    homopolymer_mask = target_run_lengths >= _HOMOPOLYMER_THRESHOLD

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        mask = mask & valid_mask
        if mask.sum() == 0:
            return 0.0
        return float(values[mask].mean().cpu())

    return {
        "trust_gate_mean": masked_mean(trust_gate, valid_mask),
        "trust_gate_low_entropy_mean": masked_mean(trust_gate, low_entropy_mask),
        "trust_gate_high_entropy_mean": masked_mean(trust_gate, high_entropy_mask),
        "trust_gate_low_agreement_mean": masked_mean(trust_gate, low_agreement_mask),
        "trust_gate_high_agreement_mean": masked_mean(trust_gate, high_agreement_mask),
        "trust_gate_homopolymer_mean": masked_mean(trust_gate, homopolymer_mask),
        "trust_gate_non_homopolymer_mean": masked_mean(trust_gate, ~homopolymer_mask),
        "support_confidence_mean": masked_mean(support_stats["confidence"], valid_mask),
        "support_uncertainty_mean": masked_mean(support_stats["uncertainty"], valid_mask),
    }


def estimate_overcorrection(logits: torch.Tensor, preserve_mask: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)[:, :, -1]
    valid = labels[:, :, -1].ne(PAD_EDIT_ID)
    risky = (preserve_mask > 0) & valid
    if risky.sum() == 0:
        return 0.0
    overcorrect = (preds != _COPY_ID) & risky
    return float((overcorrect.sum().float() / risky.sum().float()).cpu())


def aggregate_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metric_dicts[0].keys()
    return {k: sum(d[k] for d in metric_dicts) / len(metric_dicts) for k in keys}
