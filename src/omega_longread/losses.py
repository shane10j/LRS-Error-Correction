from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from .config import LossConfig
from .support import compute_support_statistics
from .vocab import BASE_TO_ID, EDIT_TO_ID, EDIT_TOKENS, PAD_EDIT_ID

_COPY_ID = EDIT_TO_ID["COPY"]
_SUB_IDS = torch.tensor([EDIT_TO_ID["SUB_A"], EDIT_TO_ID["SUB_C"], EDIT_TO_ID["SUB_G"], EDIT_TO_ID["SUB_T"]])
_INS_IDS = torch.tensor([EDIT_TO_ID["INS_A"], EDIT_TO_ID["INS_C"], EDIT_TO_ID["INS_G"], EDIT_TO_ID["INS_T"]])
_BASE_EMIT_BLANK_ID = len(BASE_TO_ID)
_INS_TOKEN_TO_BASE_ID = {
    EDIT_TO_ID["INS_A"]: BASE_TO_ID["A"],
    EDIT_TO_ID["INS_C"]: BASE_TO_ID["C"],
    EDIT_TO_ID["INS_G"]: BASE_TO_ID["G"],
    EDIT_TO_ID["INS_T"]: BASE_TO_ID["T"],
}
_SUB_TOKEN_TO_BASE_ID = {
    EDIT_TO_ID["SUB_A"]: BASE_TO_ID["A"],
    EDIT_TO_ID["SUB_C"]: BASE_TO_ID["C"],
    EDIT_TO_ID["SUB_G"]: BASE_TO_ID["G"],
    EDIT_TO_ID["SUB_T"]: BASE_TO_ID["T"],
}


def _support_core_distribution_to_edit_space(
    base_counts: torch.Tensor,
    del_counts: torch.Tensor,
    noisy_base_ids: torch.Tensor,
    edit_vocab_size: int,
) -> torch.Tensor:
    support_target = torch.zeros(
        *base_counts.shape[:2],
        edit_vocab_size,
        device=base_counts.device,
        dtype=base_counts.dtype,
    )
    support_target[..., EDIT_TO_ID["DEL"]] = del_counts
    for base_id, base_char in enumerate("ACGT"):
        counts = base_counts[..., base_id]
        copy_mask = noisy_base_ids == base_id
        support_target[..., _COPY_ID] += counts * copy_mask.float()
        support_target[..., EDIT_TO_ID[f"SUB_{base_char}"]] += counts * (~copy_mask).float()
    total = support_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return support_target / total


def _edit_logits_to_emission_log_probs(
    edit_logits: torch.Tensor,
    noisy_base_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(edit_logits, dim=-1)
    batch_size, seq_len, num_slots, _ = probs.shape
    num_emit_classes = len(BASE_TO_ID) + 1
    emission = torch.zeros(
        batch_size,
        seq_len,
        num_slots,
        num_emit_classes,
        device=edit_logits.device,
        dtype=edit_logits.dtype,
    )
    assigned = torch.zeros(batch_size, seq_len, num_slots, device=edit_logits.device, dtype=edit_logits.dtype)

    if num_slots > 1:
        insertion_probs = probs[:, :, :-1, :]
        for token_id, base_id in _INS_TOKEN_TO_BASE_ID.items():
            token_prob = insertion_probs[..., token_id]
            emission[:, :, :-1, base_id] += token_prob
            assigned[:, :, :-1] += token_prob

    core_probs = probs[:, :, -1, :]
    core_emission = emission[:, :, -1, :]
    core_assigned = assigned[:, :, -1]
    for token_id, base_id in _SUB_TOKEN_TO_BASE_ID.items():
        token_prob = core_probs[..., token_id]
        core_emission[..., base_id] += token_prob
        core_assigned += token_prob

    copy_prob = core_probs[..., _COPY_ID]
    core_emission.scatter_add_(-1, noisy_base_ids.unsqueeze(-1), copy_prob.unsqueeze(-1))
    core_assigned += copy_prob

    emission[..., _BASE_EMIT_BLANK_ID] = (1.0 - assigned).clamp_min(0.0)
    emission = emission / emission.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    flat_emission = emission.reshape(batch_size, seq_len * num_slots, num_emit_classes)
    log_probs = flat_emission.clamp_min(1e-8).log().transpose(0, 1)
    slot_nonblank_probs = 1.0 - emission[..., _BASE_EMIT_BLANK_ID]
    return log_probs, slot_nonblank_probs


def _build_label_scaling(labels: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    scale = torch.ones_like(labels, dtype=torch.float32)
    if cfg.substitution_loss_scale != 1.0:
        sub_mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in _SUB_IDS.tolist():
            sub_mask |= labels == token_id
        scale[sub_mask] *= cfg.substitution_loss_scale
    if cfg.deletion_loss_scale != 1.0:
        scale[labels == EDIT_TO_ID["DEL"]] *= cfg.deletion_loss_scale
    if cfg.insertion_loss_scale != 1.0:
        ins_mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in _INS_IDS.tolist():
            ins_mask |= labels == token_id
        scale[ins_mask] *= cfg.insertion_loss_scale
    return scale


class OmegaLoss:
    def __init__(self, cfg: LossConfig, edit_class_weights: torch.Tensor | None = None) -> None:
        self.cfg = cfg
        self.edit_class_weights = edit_class_weights.clone().float() if edit_class_weights is not None else None

    def _get_edit_class_weights(self, device: torch.device | str) -> torch.Tensor | None:
        if self.edit_class_weights is None:
            return None
        return self.edit_class_weights.to(device)

    def __call__(self, outputs: Dict[str, torch.Tensor], batch) -> tuple[torch.Tensor, Dict[str, float]]:
        valid_mask = batch.edit_labels.ne(PAD_EDIT_ID)
        core_logits = outputs["edit_logits"][:, :, -1, :]
        core_probs = F.softmax(core_logits, dim=-1)
        core_valid_mask = valid_mask[:, :, -1]
        edit_class_weights = self._get_edit_class_weights(outputs["edit_logits"].device)

        edit_loss = F.cross_entropy(
            outputs["edit_logits"].permute(0, 3, 1, 2),
            batch.edit_labels,
            weight=edit_class_weights,
            ignore_index=PAD_EDIT_ID,
            reduction="none",
            label_smoothing=self.cfg.label_smoothing,
        )
        run_weight = 1.0 + self.cfg.homopolymer_weight_scale * batch.target_run_lengths.float()
        slot_weight = valid_mask.float().clone()
        slot_weight[:, :, -1] *= run_weight
        slot_weight *= _build_label_scaling(batch.edit_labels, self.cfg).to(slot_weight.device)
        edit_loss = (edit_loss * slot_weight).sum() / valid_mask.float().sum().clamp_min(1.0)

        emit_log_probs, nonblank_probs = _edit_logits_to_emission_log_probs(
            outputs["edit_logits"],
            batch.target_bases,
        )
        input_lengths = batch.target_mask.sum(dim=-1).to(dtype=torch.long) * outputs["edit_logits"].shape[2]
        target_lengths = batch.target_sequence_mask.sum(dim=-1).to(dtype=torch.long)
        flat_targets = batch.target_sequence_ids[batch.target_sequence_mask]
        if flat_targets.numel() > 0:
            sequence_loss = F.ctc_loss(
                emit_log_probs,
                flat_targets,
                input_lengths,
                target_lengths,
                blank=_BASE_EMIT_BLANK_ID,
                reduction="mean",
                zero_infinity=True,
            )
        else:
            sequence_loss = outputs["edit_logits"].new_zeros(())

        predicted_lengths = nonblank_probs.sum(dim=(-1, -2))
        target_lengths_float = target_lengths.float()
        length_error = F.smooth_l1_loss(
            predicted_lengths,
            target_lengths_float,
            reduction="none",
        ) / target_lengths_float.clamp_min(1.0)
        length_loss = length_error.mean()

        predicted_insertions = nonblank_probs[:, :, :-1].sum(dim=(-1, -2))
        target_insertions = valid_mask[:, :, :-1].float().sum(dim=(-1, -2))
        insertion_count_error = F.smooth_l1_loss(
            predicted_insertions,
            target_insertions,
            reduction="none",
        ) / target_lengths_float.clamp_min(1.0)
        insertion_count_loss = insertion_count_error.mean()

        support_stats = compute_support_statistics(
            batch.support_base_support,
            batch.support_del_mask,
            batch.support_ins_base_support,
        )
        support_mask = support_stats["mask"]
        support_target = _support_core_distribution_to_edit_space(
            support_stats["base_counts"],
            support_stats["del_counts"],
            batch.target_bases,
            edit_vocab_size=core_logits.shape[-1],
        )
        model_support = core_probs
        support_loss_per_pos = F.kl_div(
            model_support.clamp_min(1e-8).log(),
            support_target.clamp_min(1e-8),
            reduction="none",
        ).sum(dim=-1)
        support_weight = core_valid_mask.float() * support_mask.float()
        support_loss = (support_loss_per_pos * support_weight).sum() / support_weight.sum().clamp_min(1.0)

        trust_gate = outputs["trust_gate"]
        trust_target = support_stats["confidence"].detach()
        trust_loss = F.binary_cross_entropy(
            trust_gate.clamp(1e-6, 1.0 - 1e-6),
            trust_target,
            reduction="none",
        )
        trust_loss = (trust_loss * core_valid_mask.float()).sum() / core_valid_mask.float().sum().clamp_min(1.0)

        hard_edit_prob = core_probs[..., EDIT_TO_ID["DEL"]] + core_probs.index_select(
            dim=-1,
            index=_SUB_IDS.to(core_probs.device),
        ).sum(dim=-1)
        entropy_excess = torch.relu(
            support_stats["entropy"] - self.cfg.hard_edit_entropy_threshold
        ) / max(1.0 - self.cfg.hard_edit_entropy_threshold, 1e-6)
        agreement_deficit = torch.relu(
            self.cfg.hard_edit_agreement_threshold - support_stats["agreement"]
        ) / max(self.cfg.hard_edit_agreement_threshold, 1e-6)
        unsupported_hard_edit_weight = torch.pow(
            support_stats["uncertainty"].clamp(0.0, 1.0),
            self.cfg.hard_edit_uncertainty_power,
        )
        unsupported_hard_edit_weight = unsupported_hard_edit_weight * (
            1.0
            + self.cfg.hard_edit_entropy_scale * entropy_excess
            + self.cfg.hard_edit_low_agreement_scale * agreement_deficit
        )
        hard_edit_loss = (
            hard_edit_prob
            * unsupported_hard_edit_weight
            * core_valid_mask.float()
        ).sum() / core_valid_mask.float().sum().clamp_min(1.0)

        hard_edit_label_mask = (
            (batch.edit_labels[:, :, -1] == EDIT_TO_ID["DEL"])
            | (batch.edit_labels[:, :, -1].unsqueeze(-1) == _SUB_IDS.to(batch.edit_labels.device)).any(dim=-1)
        )
        selective_mask = (
            support_stats["agreement"] >= self.cfg.selective_hard_edit_min_support_agreement
        ) & (
            support_stats["confidence"] >= self.cfg.selective_hard_edit_confidence_threshold
        ) & (
            support_stats["uncertainty"] <= self.cfg.selective_hard_edit_uncertainty_threshold
        ) & hard_edit_label_mask & core_valid_mask
        selective_hard_edit_ce = F.cross_entropy(
            core_logits.transpose(1, 2),
            batch.edit_labels[:, :, -1],
            reduction="none",
            label_smoothing=self.cfg.label_smoothing,
        )
        selective_hard_edit_loss = (
            selective_hard_edit_ce * selective_mask.float()
        ).sum() / selective_mask.float().sum().clamp_min(1.0)

        hard_edit_target = hard_edit_label_mask.float()
        hard_edit_precision_weight = torch.where(
            hard_edit_label_mask,
            torch.full_like(hard_edit_target, self.cfg.hard_edit_false_negative_weight),
            torch.full_like(hard_edit_target, self.cfg.hard_edit_false_positive_weight),
        )
        hard_edit_precision_weight = hard_edit_precision_weight * (
            1.0
            + self.cfg.hard_edit_entropy_scale * entropy_excess
            + self.cfg.hard_edit_low_agreement_scale * agreement_deficit
        )
        hard_edit_precision_loss = F.binary_cross_entropy(
            hard_edit_prob.clamp(1e-6, 1.0 - 1e-6),
            hard_edit_target,
            reduction="none",
        )
        hard_edit_precision_loss = (
            hard_edit_precision_loss
            * hard_edit_precision_weight
            * core_valid_mask.float()
        ).sum() / core_valid_mask.float().sum().clamp_min(1.0)

        preserve_labels = batch.preserve_mask
        preserve_penalty = 1.0 - core_probs[..., _COPY_ID]
        preserve_loss = (preserve_penalty * preserve_labels * core_valid_mask.float()).sum() / (
            (preserve_labels * core_valid_mask.float()).sum().clamp_min(1.0)
        )

        uncertainty_loss = F.cross_entropy(
            outputs["uncertainty_logits"].transpose(1, 2),
            batch.uncertainty_labels,
            reduction="none",
        )
        uncertainty_loss = (uncertainty_loss * core_valid_mask.float()).sum() / core_valid_mask.float().sum().clamp_min(1.0)

        total = (
            self.cfg.lambda_edit * edit_loss
            + self.cfg.lambda_sequence * sequence_loss
            + self.cfg.lambda_length * length_loss
            + self.cfg.lambda_insertion_count * insertion_count_loss
            + self.cfg.lambda_trust * trust_loss
            + self.cfg.lambda_hard_edit * hard_edit_loss
            + self.cfg.lambda_hard_edit_precision * hard_edit_precision_loss
            + self.cfg.lambda_selective_hard_edit * selective_hard_edit_loss
            + self.cfg.lambda_support * support_loss
            + self.cfg.lambda_preserve * preserve_loss
            + self.cfg.lambda_uncertainty * uncertainty_loss
        )
        metrics = {
            "loss": float(total.detach().cpu()),
            "edit_loss": float(edit_loss.detach().cpu()),
            "sequence_loss": float(sequence_loss.detach().cpu()),
            "length_loss": float(length_loss.detach().cpu()),
            "insertion_count_loss": float(insertion_count_loss.detach().cpu()),
            "trust_loss": float(trust_loss.detach().cpu()),
            "hard_edit_loss": float(hard_edit_loss.detach().cpu()),
            "hard_edit_precision_loss": float(hard_edit_precision_loss.detach().cpu()),
            "selective_hard_edit_loss": float(selective_hard_edit_loss.detach().cpu()),
            "hard_edit_entropy_excess_mean": float(entropy_excess.mean().detach().cpu()),
            "hard_edit_agreement_deficit_mean": float(agreement_deficit.mean().detach().cpu()),
            "support_loss": float(support_loss.detach().cpu()),
            "preserve_loss": float(preserve_loss.detach().cpu()),
            "uncertainty_loss": float(uncertainty_loss.detach().cpu()),
            "predicted_length": float(predicted_lengths.mean().detach().cpu()),
            "target_length": float(target_lengths_float.mean().detach().cpu()),
            "predicted_insertions": float(predicted_insertions.mean().detach().cpu()),
            "target_insertions": float(target_insertions.mean().detach().cpu()),
            "trust_gate_mean": float(trust_gate.mean().detach().cpu()),
            "support_confidence_mean": float(support_stats["confidence"].mean().detach().cpu()),
            "support_uncertainty_mean": float(support_stats["uncertainty"].mean().detach().cpu()),
            "selective_hard_edit_fraction": float(selective_mask.float().mean().detach().cpu()),
        }
        return total, metrics


def resolve_edit_class_weights(
    cfg: LossConfig,
    train_path: str | Path | None = None,
    edit_vocab_size: int | None = None,
) -> torch.Tensor | None:
    if cfg.edit_class_weights:
        weights = torch.tensor(cfg.edit_class_weights, dtype=torch.float32)
        if edit_vocab_size is not None and weights.numel() != edit_vocab_size:
            raise ValueError(
                f"Expected {edit_vocab_size} edit class weights, but received {weights.numel()}."
            )
        return weights
    if not cfg.auto_edit_class_weights:
        return None
    if train_path is None:
        raise ValueError("Automatic edit class weighting requires a training dataset path.")
    if edit_vocab_size is None:
        raise ValueError("Automatic edit class weighting requires the edit vocabulary size.")
    return compute_edit_class_weights_from_jsonl(
        train_path=train_path,
        edit_vocab_size=edit_vocab_size,
        power=cfg.edit_class_weight_power,
        count_smoothing=cfg.edit_class_weight_count_smoothing,
        min_weight=cfg.edit_class_weight_min,
        max_weight=cfg.edit_class_weight_max,
    )


def compute_edit_class_weights_from_jsonl(
    train_path: str | Path,
    edit_vocab_size: int,
    power: float = 1.0,
    count_smoothing: float = 1.0,
    min_weight: float | None = None,
    max_weight: float | None = None,
) -> torch.Tensor:
    counts = torch.zeros(edit_vocab_size, dtype=torch.float64)
    train_path = Path(train_path)
    with train_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            edit_labels = item["edit_labels"]
            if edit_labels and isinstance(edit_labels[0], int):
                edit_labels = [[label] for label in edit_labels]
            for label_slots in edit_labels:
                for token_id in label_slots:
                    token_id = int(token_id)
                    if token_id == PAD_EDIT_ID:
                        continue
                    counts[token_id] += 1

    non_pad_ids = [idx for idx in range(edit_vocab_size) if idx != PAD_EDIT_ID]
    smoothed_counts = counts[non_pad_ids] + float(count_smoothing)
    total = smoothed_counts.sum()
    if total <= 0:
        raise ValueError(f"No non-padding edit labels found in {train_path}.")

    raw_weights = torch.pow(total / (len(non_pad_ids) * smoothed_counts), float(power))
    empirical = counts[non_pad_ids] / counts[non_pad_ids].sum().clamp_min(1.0)
    normalization = (empirical * raw_weights).sum().clamp_min(1e-8)
    normalized = raw_weights / normalization

    if min_weight is not None or max_weight is not None:
        lower = float(min_weight) if min_weight is not None else float("-inf")
        upper = float(max_weight) if max_weight is not None else float("inf")
        normalized = normalized.clamp(min=lower, max=upper)

    weights = torch.ones(edit_vocab_size, dtype=torch.float32)
    for idx, class_id in enumerate(non_pad_ids):
        weights[class_id] = float(normalized[idx])
    weights[PAD_EDIT_ID] = 0.0
    return weights


def summarize_edit_class_weights(weights: torch.Tensor) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for token_id, token in enumerate(EDIT_TOKENS):
        rows.append(
            {
                "token_id": token_id,
                "token": token,
                "weight": float(weights[token_id]),
            }
        )
    return rows
