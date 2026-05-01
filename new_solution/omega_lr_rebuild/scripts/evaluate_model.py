#!/usr/bin/env python
"""Evaluate a trained checkpoint on the dataset test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from omega_lr.train.checkpointing import load_checkpoint
from omega_lr.train.trainer import build_model, evaluate_loader, hybrid_decode_config, make_loader, neural_only_decode_config, choose_device
from omega_lr.utils import print_config, read_config, save_json, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--decode-variant",
        choices=["config", "full_hybrid", "full_neural_only"],
        default="config",
        help="Evaluate with config decode, forced hybrid rule decode, or neural-only decode.",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    print_config(config)
    checkpoint = load_checkpoint(args.checkpoint)
    _, loader = make_loader(Path(config["dataset"]["output_dir"]) / "test.jsonl", config["train"]["batch_size"], shuffle=False)
    support_input_dim = len(loader.dataset[0]["pileup_features"][0])
    model = build_model(config, checkpoint["run_name"], support_input_dim)
    device = choose_device()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    decode_config = config["decode"]
    output_stem = "reevaluated"
    if args.decode_variant == "full_hybrid":
        decode_config = hybrid_decode_config(config["decode"])
        output_stem = "full_hybrid_reevaluated"
    elif args.decode_variant == "full_neural_only":
        decode_config = neural_only_decode_config(config["decode"])
        output_stem = "full_neural_only_reevaluated"
    rows, summary = evaluate_loader(model, loader, device, decode_config)
    output_dir = Path(args.checkpoint).parent
    save_jsonl(rows, output_dir / f"{output_stem}_predictions.jsonl")
    save_json(summary, output_dir / f"{output_stem}_summary.json")


if __name__ == "__main__":
    main()
