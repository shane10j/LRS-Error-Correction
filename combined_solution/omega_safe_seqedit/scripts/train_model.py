#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.config import load_config, print_resolved_config
from omega_safe_seqedit.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", choices=["target_only", "full"], default="full")
    args = parser.parse_args()
    config = load_config(args.config)
    print_resolved_config(config)
    train(config, args.run)


if __name__ == "__main__":
    main()
