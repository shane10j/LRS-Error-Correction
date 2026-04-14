#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from omega_longread.config import OmegaConfig
from omega_longread.dataset import LongReadDataset, collate_long_reads
from omega_longread.losses import OmegaLoss, resolve_edit_class_weights, summarize_edit_class_weights
from omega_longread.metrics import (
    aggregate_metric_dicts,
    estimate_overcorrection,
    summarize_edit_predictions,
    summarize_sequence_predictions,
    summarize_support_trust,
)
from omega_longread.model import OmegaModel
from omega_longread.utils import load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def build_loaders(cfg: OmegaConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = LongReadDataset(cfg.data.train_path)
    val_ds = LongReadDataset(cfg.data.val_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_long_reads,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_long_reads,
        pin_memory=True,
    )
    return train_loader, val_loader


def compute_checkpoint_score(metrics: Dict[str, float], cfg: OmegaConfig) -> float:
    metric = cfg.train.checkpoint_metric
    if metric == "composite_sequence":
        sequence_identity = metrics.get("sequence_identity", 0.0)
        overcorrection = metrics.get("overcorrection_rate", 1.0)
        length_ratio = metrics.get("predicted_length_ratio", 0.0)
        length_penalty = abs(length_ratio - 1.0)
        return (
            sequence_identity
            - cfg.train.checkpoint_overcorrection_weight * overcorrection
            - cfg.train.checkpoint_length_ratio_weight * length_penalty
        )
    if metric not in metrics:
        raise KeyError(f"Checkpoint metric {metric!r} was not found in validation metrics.")
    return metrics[metric]


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
        with autocast(enabled=cfg.train.mixed_precision):
            outputs = model(batch)
            loss, row = criterion(outputs, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        row.update(summarize_edit_predictions(outputs["edit_logits"].detach(), batch.edit_labels))
        row.update(
            summarize_sequence_predictions(
                outputs["edit_logits"].detach(),
                batch.edit_labels,
                batch.target_bases,
                batch.target_mask,
                batch.target_run_lengths,
                batch.metadata,
                cfg.model.max_insertions_per_pos,
                support_base_support=batch.support_base_support,
                trust_gate=outputs["trust_gate"].detach(),
            )
        )
        row.update(
            summarize_support_trust(
                outputs["trust_gate"].detach(),
                batch.support_base_support,
                batch.target_run_lengths,
                batch.target_mask,
            )
        )
        row["overcorrection_rate"] = estimate_overcorrection(
            outputs["edit_logits"].detach(), batch.preserve_mask, batch.edit_labels
        )
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
        with autocast(enabled=cfg.train.mixed_precision):
            outputs = model(batch)
            _, row = criterion(outputs, batch)
        row.update(summarize_edit_predictions(outputs["edit_logits"], batch.edit_labels))
        row.update(
            summarize_sequence_predictions(
                outputs["edit_logits"],
                batch.edit_labels,
                batch.target_bases,
                batch.target_mask,
                batch.target_run_lengths,
                batch.metadata,
                cfg.model.max_insertions_per_pos,
                support_base_support=batch.support_base_support,
                trust_gate=outputs["trust_gate"],
            )
        )
        row.update(
            summarize_support_trust(
                outputs["trust_gate"],
                batch.support_base_support,
                batch.target_run_lengths,
                batch.target_mask,
            )
        )
        row["overcorrection_rate"] = estimate_overcorrection(outputs["edit_logits"], batch.preserve_mask, batch.edit_labels)
        metric_rows.append(row)
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
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(cfg)
    model = OmegaModel(cfg).to(device)
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
            torch.save(model.state_dict(), Path(cfg.train.save_dir) / "best_model.pt")


if __name__ == "__main__":
    main()
