#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.config import load_config
from omega_safe_seqedit.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    out = Path(config["paths"]["output_dir"])
    summary = {"name": config["name"], "outputs": {}}
    for path in out.rglob("*summary.json"):
        try:
            summary["outputs"][str(path.relative_to(out))] = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    write_json(out / "combined_benchmark_summary.json", summary)
    print(out / "combined_benchmark_summary.json")


if __name__ == "__main__":
    main()
