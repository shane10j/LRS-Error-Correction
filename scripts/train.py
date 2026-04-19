#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from omega_longread.config import OmegaConfig
from omega_longread.dataset import LongReadDataset, collate_long_reads
from omega_longread.decode import apply_inference_constraints
from omega_longread.losses import OmegaLoss, resolve_edit_class_weights, summarize_edit_class_weights
from omega_longread.metrics import (
    aggregate_metric_dicts,
    estimate_overcorrection,
    summarize_hard_edit_precision_stratified,
    summarize_edit_predictions,
    summarize_sequence_predictions,
    summarize_support_trust,
)
from omega_longread.model import OmegaModel
from omega_longread.utils import load_config, resolve_torch_device, save_json, set_seed
from omega_longread.vocab import EDIT_TO_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def build_loaders(cfg: OmegaConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = LongReadDataset(cfg.data.train_path)
    val_ds = LongReadDataset(cfg.data.val_path)
    sampler = None
    shuffle = True
    if cfg.train.oversample_deletion_windows:
        weights = []
        for item in train_ds.items:
            edit_labels = item["edit_labels"]
            if edit_labels and isinstance(edit_labels[0], int):
                has_del = any(int(token) == EDIT_TO_ID["DEL"] for token in edit_labels)
            else:
                has_del = any(int(slots[-1]) == EDIT_TO_ID["DEL"] for slots in edit_labels if slots)
            weights.append(float(cfg.data.deletion_oversample_weight) if has_del else 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_long_reads,
        pin_memory=cfg.train.device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_long_reads,
        pin_memory=cfg.train.device == "cuda",
    )
    return train_loader, val_loader


def maybe_empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def compute_checkpoint_score(metrics: Dict[str, float], cfg: OmegaConfig) -> float:
    metric = cfg.train.checkpoint_metric
    if metric == "composite_sequence":
        sequence_identity = metrics.get("sequence_identity", 0.0)
        overcorrection = metrics.get("overcorrection_rate", 1.0)
        hard_edit_fp_rate = metrics.get("hard_edit_false_positive_rate", 1.0)
        length_ratio = metrics.get("predicted_length_ratio", 0.0)
        length_penalty = abs(length_ratio - 1.0)
        return (
            sequence_identity
            - cfg.train.checkpoint_overcorrection_weight * overcorrection
            - cfg.train.checkpoint_hard_edit_fp_weight * hard_edit_fp_rate
            - cfg.train.checkpoint_length_ratio_weight * length_penalty
        )
    if metric not in metrics:
        raise KeyError(f"Checkpoint metric {metric!r} was not found in validation metrics.")
    return metrics[metric]


def get_metric_logits(outputs, batch, cfg: OmegaConfig):
    return apply_inference_constraints(
        outputs["edit_logits"].detach(),
        trust_gate=outputs.get("trust_gate"),
        deletion_candidate_logits=outputs.get("deletion_candidate_logits"),
        deletion_length_logits=outputs.get("deletion_length_logits"),
        local_support_agreement=batch.local_support_agreement,
        deletion_support_fraction=batch.deletion_support_fraction,
        min_sub_confidence=cfg.model.inference_sub_confidence_threshold,
        min_del_confidence=cfg.model.inference_del_confidence_threshold,
        min_ins_confidence=cfg.model.inference_ins_confidence_threshold,
        deletion_candidate_threshold=cfg.model.deletion_candidate_threshold,
        deletion_commit_trust_threshold=cfg.model.deletion_commit_trust_threshold,
        hard_edit_temperature=cfg.model.inference_hard_edit_temperature,
        use_deletion_consistency_check=cfg.model.inference_use_deletion_consistency_check,
    )


def train_epoch(
    model: OmegaModel,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: OmegaLoss,
    device: torch.device,
    scaler: GradScaler,
    cfg: OmegaConfig,
) -> Dict[str, float]:
    model.train()
    metric_rows: List[Dict[str, float]] = []
    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.train.mixed_precision and device.type == "cuda"):
            outputs = model(batch)
            loss, row = criterion(outputs, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        maybe_empty_device_cache(device)

        # Keep training-time bookkeeping lightweight. Full decoded sequence metrics
        # are computed during validation/evaluation, which is what checkpointing uses.
        metric_rows.append(row)

        if step % cfg.train.log_every == 0:
            agg = aggregate_metric_dicts(metric_rows[-cfg.train.log_every :])
            print(f"train step={step}: {json.dumps(agg, indent=None)}")
    return aggregate_metric_dicts(metric_rows)


@torch.no_grad()
def evaluate(
    model: OmegaModel,
    loader: DataLoader,
    criterion: OmegaLoss,
    device: torch.device,
    cfg: OmegaConfig,
) -> Dict[str, float]:
    model.eval()
    metric_rows: List[Dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        with autocast(enabled=cfg.train.mixed_precision and device.type == "cuda"):
            outputs = model(batch)
            _, row = criterion(outputs, batch)
        metric_logits = get_metric_logits(outputs, batch, cfg)
        row.update(summarize_edit_predictions(metric_logits, batch.edit_labels))
        row.update(
            summarize_sequence_predictions(
                metric_logits,
                batch.edit_labels,
                batch.target_bases,
                batch.target_mask,
                batch.target_run_lengths,
                batch.metadata,
                cfg.model.max_insertions_per_pos,
                variant_mask=batch.variant_mask,
                phased_variant_mask=batch.phased_variant_mask,
                region_masks=batch.region_masks,
                support_base_support=batch.support_base_support,
                support_del_mask=batch.support_del_mask,
                support_ins_base_support=batch.support_ins_base_support,
                support_haplotype=batch.support_haplotype,
                support_same_haplotype=batch.support_same_haplotype,
                trust_gate=outputs["trust_gate"],
            )
        )
        row.update(
            summarize_support_trust(
                outputs["trust_gate"],
                batch.support_base_support,
                batch.support_del_mask,
                batch.support_ins_base_support,
                batch.support_haplotype,
                batch.support_same_haplotype,
                batch.target_run_lengths,
                batch.target_mask,
            )
        )
        row.update(
            summarize_hard_edit_precision_stratified(
                metric_logits.argmax(dim=-1),
                batch.edit_labels,
                batch.target_run_lengths,
                batch.support_base_support,
                batch.support_del_mask,
                batch.support_ins_base_support,
                batch.region_masks,
                batch.support_haplotype,
                batch.support_same_haplotype,
            )
        )
        row["overcorrection_rate"] = estimate_overcorrection(metric_logits, batch.preserve_mask, batch.edit_labels)
        metric_rows.append(row)
        maybe_empty_device_cache(device)
    return aggregate_metric_dicts(metric_rows)


def save_checkpoint(model: OmegaModel, optimizer: AdamW, epoch: int, metrics: Dict[str, float], save_dir: str) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": model.get_config(),
    }
    torch.save(ckpt, Path(save_dir) / f"epoch_{epoch:03d}.pt")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    resolved_device, mixed_precision_ok = resolve_torch_device(cfg.train.device)
    cfg.train.device = resolved_device
    cfg.train.mixed_precision = bool(cfg.train.mixed_precision and mixed_precision_ok)
    device = torch.device(cfg.train.device)

    train_loader, val_loader = build_loaders(cfg)
    model = OmegaModel(cfg).to(device)
    if cfg.train.init_checkpoint:
        init_path = Path(cfg.train.init_checkpoint)
        checkpoint = torch.load(init_path, map_location=device)
        init_state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(init_state, strict=False)
        print(f"initialized model from {init_path}")
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.mixed_precision and device.type == "cuda")
    edit_class_weights = resolve_edit_class_weights(
        cfg.loss,
        train_path=cfg.data.train_path,
        edit_vocab_size=cfg.model.edit_vocab_size,
    )
    if edit_class_weights is not None:
        weight_rows = summarize_edit_class_weights(edit_class_weights)
        print("resolved edit class weights:")
        print(json.dumps(weight_rows, indent=2))
        save_json(weight_rows, Path(cfg.train.save_dir) / "edit_class_weights.json")
    criterion = OmegaLoss(cfg.loss, edit_class_weights=edit_class_weights)

    history: List[Dict[str, float]] = []
    best_score = None
    stale_epochs = 0
    for epoch in range(1, cfg.train.epochs + 1):
        print(f"epoch {epoch}/{cfg.train.epochs}")
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg)
        selection_score = compute_checkpoint_score(val_metrics, cfg)
        record = {f"train_{k}": v for k, v in train_metrics.items()} | {f"val_{k}": v for k, v in val_metrics.items()}
        record["epoch"] = epoch
        record["val_selection_score"] = selection_score
        history.append(record)
        print(json.dumps(record, indent=2))
        save_json(history, Path(cfg.train.save_dir) / "history.json")
        save_checkpoint(model, optimizer, epoch, record, cfg.train.save_dir)
        is_better = False
        if best_score is None:
            is_better = True
        elif cfg.train.checkpoint_metric_mode == "max":
            is_better = selection_score > best_score
        elif cfg.train.checkpoint_metric_mode == "min":
            is_better = selection_score < best_score
        else:
            raise ValueError(f"Unsupported checkpoint_metric_mode={cfg.train.checkpoint_metric_mode!r}")
        if is_better:
            best_score = selection_score
            stale_epochs = 0
            torch.save(model.state_dict(), Path(cfg.train.save_dir) / "best_model.pt")
        else:
            stale_epochs += 1
        if cfg.train.early_stopping_patience > 0 and stale_epochs >= cfg.train.early_stopping_patience:
            print(f"early stopping at epoch {epoch} after {stale_epochs} stale epochs")
            break


if __name__ == "__main__":
    main()
