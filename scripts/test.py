#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from omega_longread.config import OmegaConfig
from omega_longread.dataset import LongReadDataset, collate_long_reads
from omega_longread.decode import apply_edit_ops
from omega_longread.losses import OmegaLoss, resolve_edit_class_weights
from omega_longread.metrics import (
    aggregate_metric_dicts,
    estimate_overcorrection,
    summarize_edit_predictions,
    summarize_sequence_predictions,
    summarize_support_trust,
)
from omega_longread.model import OmegaModel
from omega_longread.tokenizer import DNATokenizer
from omega_longread.utils import load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OMEGA on a held-out JSONL dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt or a saved training checkpoint.")
    parser.add_argument("--data-path", default=None, help="Override cfg.data.test_path.")
    parser.add_argument("--predictions-out", default=None, help="Optional JSONL path for decoded predictions.")
    parser.add_argument("--summary-out", default=None, help="Optional JSON path for summary metrics.")
    return parser.parse_args()


def load_model(checkpoint_path: str, cfg: OmegaConfig, device: torch.device) -> OmegaModel:
    model = OmegaModel(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_path = args.data_path or cfg.data.test_path
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    dataset = LongReadDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_long_reads,
        pin_memory=True,
    )

    model = load_model(args.checkpoint, cfg, device)
    edit_class_weights = resolve_edit_class_weights(
        cfg.loss,
        train_path=cfg.data.train_path,
        edit_vocab_size=cfg.model.edit_vocab_size,
    )
    criterion = OmegaLoss(cfg.loss, edit_class_weights=edit_class_weights)
    tokenizer = DNATokenizer()

    metric_rows: List[Dict[str, float]] = []
    decoded_rows: List[dict] = []

    for batch in loader:
        batch = batch.to(device)
        with autocast(enabled=cfg.train.mixed_precision and device.type == "cuda"):
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

        pred_slots = outputs["edit_logits"].argmax(dim=-1).detach().cpu()
        noisy_batch = batch.target_bases.detach().cpu()
        mask_batch = batch.target_mask.detach().cpu()
        for i, meta in enumerate(batch.metadata):
            valid_len = int(mask_batch[i].sum().item())
            noisy_seq = tokenizer.decode(noisy_batch[i, :valid_len].tolist())
            predicted_seq = apply_edit_ops(
                noisy_seq,
                pred_slots[i, :valid_len].tolist(),
                max_insertions_per_pos=cfg.model.max_insertions_per_pos,
            )
            truth_seq = meta.get("target_sequence", "")
            if args.predictions_out:
                decoded_rows.append(
                    {
                        "read_id": meta.get("read_id"),
                        "source_read_id": meta.get("source_read_id"),
                        "contig": meta.get("contig"),
                        "window_ref_start": meta.get("window_ref_start"),
                        "window_ref_end": meta.get("window_ref_end"),
                        "noisy_sequence": noisy_seq,
                        "truth_sequence": truth_seq,
                        "predicted_sequence": predicted_seq,
                        "predicted_edit_labels": pred_slots[i, :valid_len].tolist(),
                    }
                )
        metric_rows.append(row)

    if not metric_rows:
        raise RuntimeError(f"No evaluable examples found in {data_path}")

    metrics = aggregate_metric_dicts(metric_rows)
    print(json.dumps(metrics, indent=2))

    if args.summary_out:
        save_json(metrics, args.summary_out)
    if args.predictions_out:
        predictions_path = Path(args.predictions_out)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as handle:
            for row in decoded_rows:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    evaluate()
