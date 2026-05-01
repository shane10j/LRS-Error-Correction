#!/usr/bin/env python
"""Train target_only or full model."""

from __future__ import annotations

import argparse
from pathlib import Path

from omega_lr.seed import seed_everything
from omega_lr.train.trainer import train_model
from omega_lr.utils import dump_yaml, ensure_dir, load_json, print_config, read_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True, choices=["target_only", "full"])
    args = parser.parse_args()
    config = read_config(args.config)
    print_config(config)
    seed_everything(config.get("seed", 13))
    output_dir = ensure_dir(Path(config["train"]["output_dir"]) / args.run_name)
    dump_yaml(config, output_dir / "config_snapshot.yaml")
    manifest = load_json(Path(config["dataset"]["output_dir"]) / "manifest.json")
    save_json(manifest, output_dir / "manifest.json")
    summary = train_model(config, Path(config["dataset"]["output_dir"]), output_dir, args.run_name)
    save_json(summary, output_dir / "final_summary.json")


if __name__ == "__main__":
    main()
