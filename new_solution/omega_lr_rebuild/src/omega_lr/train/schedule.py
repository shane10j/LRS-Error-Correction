"""Epoch-wise training schedule for edit encouragement and later calibration."""

from __future__ import annotations


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _progress(epoch_idx: int, total_epochs: int, warmup_epochs: int) -> float:
    if total_epochs <= warmup_epochs:
        return 1.0
    numerator = max(0, epoch_idx - warmup_epochs + 1)
    denominator = max(total_epochs - warmup_epochs, 1)
    return min(1.0, numerator / denominator)


def epoch_schedule(config: dict, epoch_idx: int) -> dict:
    train_cfg = config["train"]
    schedule_cfg = train_cfg.get("encourage_edits_schedule", {})
    total_epochs = train_cfg["epochs"]
    warmup_epochs = schedule_cfg.get("warmup_epochs", max(1, total_epochs // 3))
    late_progress = _progress(epoch_idx, total_epochs, warmup_epochs)
    decode_cfg = config["decode"]
    soft_cfg = schedule_cfg.get(
        "soft_decode_thresholds",
        {
            "sub_threshold": 0.10,
            "del_threshold": 0.10,
            "ins_threshold": 0.10,
            "trust_threshold": 0.00,
        },
    )
    return {
        "epoch_index": epoch_idx,
        "late_progress": late_progress,
        "false_positive_penalty_weight": _lerp(
            schedule_cfg.get("early_false_positive_penalty_weight", 0.10),
            schedule_cfg.get("late_false_positive_penalty_weight", 0.50),
            late_progress,
        ),
        "copy_to_sub_penalty_weight": _lerp(
            schedule_cfg.get("early_copy_to_sub_penalty_weight", 0.00),
            schedule_cfg.get("late_copy_to_sub_penalty_weight", 0.50),
            late_progress,
        ),
        "copy_to_del_penalty_weight": _lerp(
            schedule_cfg.get("early_copy_to_del_penalty_weight", 0.00),
            schedule_cfg.get("late_copy_to_del_penalty_weight", 0.75),
            late_progress,
        ),
        "copy_to_ins_penalty_weight": _lerp(
            schedule_cfg.get("early_copy_to_ins_penalty_weight", 0.00),
            schedule_cfg.get("late_copy_to_ins_penalty_weight", 0.50),
            late_progress,
        ),
        "positive_hard_edit_reward_weight": _lerp(
            schedule_cfg.get("early_positive_hard_edit_reward_weight", 0.80),
            schedule_cfg.get("late_positive_hard_edit_reward_weight", 0.35),
            late_progress,
        ),
        "trust_regularization_weight": _lerp(
            schedule_cfg.get("early_trust_regularization_weight", 0.00),
            schedule_cfg.get("late_trust_regularization_weight", 0.05),
            late_progress,
        ),
        "gate_open_bias": _lerp(
            schedule_cfg.get("early_gate_open_bias", 2.0),
            schedule_cfg.get("late_gate_open_bias", 0.0),
            late_progress,
        ),
        "curriculum_fraction": _lerp(
            schedule_cfg.get("early_curriculum_fraction", 0.40),
            schedule_cfg.get("late_curriculum_fraction", 1.00),
            late_progress,
        ),
        "decode_config": {
            "sub_threshold": _lerp(soft_cfg["sub_threshold"], decode_cfg["sub_threshold"], late_progress),
            "del_threshold": _lerp(soft_cfg["del_threshold"], decode_cfg["del_threshold"], late_progress),
            "ins_threshold": _lerp(soft_cfg["ins_threshold"], decode_cfg["ins_threshold"], late_progress),
            "trust_threshold": _lerp(soft_cfg["trust_threshold"], decode_cfg["trust_threshold"], late_progress),
            "max_deletion_length": decode_cfg["max_deletion_length"],
        },
    }
