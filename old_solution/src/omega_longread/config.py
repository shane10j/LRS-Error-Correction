from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfig:
    base_vocab_size: int = 5
    quality_vocab_size: int = 64
    edit_vocab_size: int = 11
    uncertainty_classes: int = 3
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    target_feature_dim: int = 10
    support_feature_dim: int = 17
    max_supports: int = 16
    max_insertions_per_pos: int = 2
    max_deletion_length: int = 4
    run_length_classes: int = 8
    conv_kernel_size: int = 5
    support_mode: str = "full"
    apply_hard_edit_support_filter: bool = False
    hard_edit_min_support_agreement: float = 0.85
    hard_edit_max_support_entropy: float = 0.35
    hard_edit_min_support_depth: float = 2.0
    hard_edit_filter_logit_bias: float = 3.0
    inference_hard_edit_confidence_threshold: float = 0.0
    inference_hard_edit_temperature: float = 1.0
    inference_sub_confidence_threshold: float = 0.8
    inference_del_confidence_threshold: float = 0.92
    inference_ins_confidence_threshold: float = 0.75
    deletion_commit_trust_threshold: float = 0.8
    deletion_candidate_threshold: float = 0.8
    inference_use_deletion_consistency_check: bool = True


@dataclass
class DataConfig:
    train_path: str = "data/train.jsonl"
    val_path: str = "data/val.jsonl"
    test_path: str = "data/test.jsonl"
    manifest_path: str = ""
    max_target_len: int = 1024
    max_supports: int = 16
    max_support_len: int = 1024
    num_workers: int = 4
    region_names: List[str] = field(default_factory=list)
    deletion_oversample_weight: float = 2.5


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 4
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    device: str = "auto"
    mixed_precision: bool = True
    log_every: int = 20
    eval_every: int = 1
    save_dir: str = "checkpoints"
    init_checkpoint: str = ""
    checkpoint_metric: str = "loss"
    checkpoint_metric_mode: str = "min"
    checkpoint_overcorrection_weight: float = 0.5
    checkpoint_hard_edit_fp_weight: float = 0.5
    checkpoint_length_ratio_weight: float = 0.25
    early_stopping_patience: int = 0
    oversample_deletion_windows: bool = False


@dataclass
class LossConfig:
    lambda_edit: float = 1.0
    lambda_sequence: float = 0.0
    lambda_length: float = 0.0
    lambda_insertion_count: float = 0.0
    lambda_trust: float = 0.0
    lambda_hard_edit: float = 0.0
    lambda_hard_edit_precision: float = 0.0
    lambda_selective_hard_edit: float = 0.0
    lambda_deletion_candidate: float = 0.0
    lambda_deletion_length: float = 0.0
    lambda_run_length_aux: float = 0.0
    lambda_deletion_positive_reward: float = 0.0
    lambda_support: float = 0.0
    lambda_preserve: float = 0.15
    lambda_uncertainty: float = 0.1
    homopolymer_weight_scale: float = 0.15
    label_smoothing: float = 0.0
    edit_class_weights: List[float] = field(default_factory=list)
    auto_edit_class_weights: bool = False
    edit_class_weight_power: float = 1.0
    edit_class_weight_count_smoothing: float = 1.0
    edit_class_weight_min: float = 0.25
    edit_class_weight_max: float = 8.0
    substitution_loss_scale: float = 1.0
    deletion_loss_scale: float = 1.0
    insertion_loss_scale: float = 1.0
    hard_edit_uncertainty_power: float = 1.0
    hard_edit_entropy_threshold: float = 0.35
    hard_edit_agreement_threshold: float = 0.85
    hard_edit_entropy_scale: float = 1.0
    hard_edit_low_agreement_scale: float = 1.0
    hard_edit_false_positive_weight: float = 6.0
    hard_edit_false_negative_weight: float = 1.0
    selective_hard_edit_confidence_threshold: float = 0.6
    selective_hard_edit_uncertainty_threshold: float = 0.4
    selective_hard_edit_min_support_agreement: float = 0.85
    deletion_focal_gamma: float = 2.0
    deletion_false_positive_weight: float = 6.0
    deletion_false_negative_weight: float = 1.0
    deletion_positive_reward_scale: float = 1.0


@dataclass
class OmegaConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @staticmethod
    def from_dict(cfg: dict) -> "OmegaConfig":
        return OmegaConfig(
            model=ModelConfig(**cfg.get("model", {})),
            data=DataConfig(**cfg.get("data", {})),
            train=TrainConfig(**cfg.get("train", {})),
            loss=LossConfig(**cfg.get("loss", {})),
        )
