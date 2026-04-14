#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from omega_longread.config import OmegaConfig
from omega_longread.dataset import LongReadDataset, collate_long_reads
from omega_longread.decode import apply_edit_ops
from omega_longread.metrics import (
    aggregate_metric_dicts,
    summarize_edit_label_predictions,
    summarize_sequence_label_predictions,
)
from omega_longread.support import compute_support_statistics
from omega_longread.tokenizer import DNATokenizer
from omega_longread.utils import load_config, save_json
from omega_longread.vocab import BASE_TO_ID, EDIT_TO_ID, PAD_EDIT_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a conservative support-consensus baseline on OMEGA JSONL windows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", default=None, help="Override cfg.data.test_path.")
    parser.add_argument("--min-support-depth", type=float, default=2.0)
    parser.add_argument("--min-agreement", type=float, default=0.85)
    parser.add_argument("--max-entropy", type=float, default=0.35)
    parser.add_argument("--delete-threshold", type=float, default=0.8)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--predictions-out", default=None)
    return parser.parse_args()


def build_consensus_predictions(batch, cfg: OmegaConfig, args: argparse.Namespace) -> torch.Tensor:
    device = batch.edit_labels.device
    batch_size, seq_len, num_slots = batch.edit_labels.shape
    preds = torch.full((batch_size, seq_len, num_slots), PAD_EDIT_ID, dtype=torch.long, device=device)
    preds[:, :, -1] = EDIT_TO_ID["COPY"]

    support_stats = compute_support_statistics(
        batch.support_base_support,
        batch.support_del_mask,
        batch.support_ins_base_support,
    )
    base_counts = support_stats["counts"]
    evidence_depth = support_stats["depth"]
    delete_frac = support_stats["del_counts"] / evidence_depth.clamp_min(1.0)
    base_probs = base_counts / base_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    base_agreement, best_base_idx = base_probs.max(dim=-1)

    valid_mask = batch.target_mask.bool()
    evidence_mask = evidence_depth >= args.min_support_depth
    low_entropy_mask = support_stats["entropy"] <= args.max_entropy
    confident_base_mask = evidence_mask & low_entropy_mask & (base_agreement >= args.min_agreement)
    confident_delete_mask = evidence_mask & low_entropy_mask & (delete_frac >= args.delete_threshold) & (delete_frac >= base_agreement)

    noisy_base_ids = batch.target_bases
    for base_char in "ACGT":
        base_id = BASE_TO_ID[base_char]
        sub_token = EDIT_TO_ID[f"SUB_{base_char}"]
        sub_mask = confident_base_mask & (best_base_idx == base_id) & (noisy_base_ids != base_id) & valid_mask
        preds[:, :, -1][sub_mask] = sub_token

    del_mask = confident_delete_mask & valid_mask
    preds[:, :, -1][del_mask] = EDIT_TO_ID["DEL"]
    preds[:, :, :-1] = PAD_EDIT_ID
    return preds


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_path = args.data_path or cfg.data.test_path
    dataset = LongReadDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_long_reads,
        pin_memory=True,
    )
    tokenizer = DNATokenizer()

    metric_rows: List[Dict[str, float]] = []
    prediction_rows: List[dict] = []

    for batch in loader:
        preds = build_consensus_predictions(batch, cfg, args)
        row = summarize_edit_label_predictions(preds, batch.edit_labels)
        row.update(
            summarize_sequence_label_predictions(
                preds,
                batch.edit_labels,
                batch.target_bases,
                batch.target_mask,
                batch.target_run_lengths,
                batch.metadata,
                cfg.model.max_insertions_per_pos,
                support_base_support=batch.support_base_support,
                support_del_mask=batch.support_del_mask,
                support_ins_base_support=batch.support_ins_base_support,
            )
        )
        core_preds = preds[:, :, -1]
        core_labels = batch.edit_labels[:, :, -1]
        risky = (batch.preserve_mask > 0) & core_labels.ne(PAD_EDIT_ID)
        if risky.sum() == 0:
            row["overcorrection_rate"] = 0.0
        else:
            overcorrect = (core_preds != EDIT_TO_ID["COPY"]) & risky
            row["overcorrection_rate"] = float((overcorrect.sum().float() / risky.sum().float()).cpu())
        metric_rows.append(row)

        if args.predictions_out:
            preds_cpu = preds.cpu()
            bases_cpu = batch.target_bases.cpu()
            mask_cpu = batch.target_mask.cpu()
            for i, meta in enumerate(batch.metadata):
                valid_len = int(mask_cpu[i].sum().item())
                noisy_seq = tokenizer.decode(bases_cpu[i, :valid_len].tolist())
                predicted_seq = apply_edit_ops(
                    noisy_seq,
                    preds_cpu[i, :valid_len].tolist(),
                    max_insertions_per_pos=cfg.model.max_insertions_per_pos,
                )
                prediction_rows.append(
                    {
                        "read_id": meta.get("read_id"),
                        "source_read_id": meta.get("source_read_id"),
                        "contig": meta.get("contig"),
                        "window_ref_start": meta.get("window_ref_start"),
                        "window_ref_end": meta.get("window_ref_end"),
                        "noisy_sequence": noisy_seq,
                        "truth_sequence": meta.get("target_sequence", ""),
                        "predicted_sequence": predicted_seq,
                        "predicted_edit_labels": preds_cpu[i, :valid_len].tolist(),
                    }
                )

    if not metric_rows:
        raise RuntimeError(f"No evaluable examples found in {data_path}")

    metrics = aggregate_metric_dicts(metric_rows)
    metrics["baseline"] = "conservative_support_consensus"
    metrics["min_support_depth"] = args.min_support_depth
    metrics["min_agreement"] = args.min_agreement
    metrics["max_entropy"] = args.max_entropy
    metrics["delete_threshold"] = args.delete_threshold
    print(json.dumps(metrics, indent=2))

    if args.summary_out:
        save_json(metrics, args.summary_out)
    if args.predictions_out:
        predictions_path = Path(args.predictions_out)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
