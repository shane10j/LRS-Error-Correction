#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_safe_seqedit.baselines import conservative_consensus, external_predictions, no_edit, support_rule
from omega_safe_seqedit.config import load_config, print_resolved_config
from omega_safe_seqedit.io_utils import ensure_dir, read_jsonl, write_json
from omega_safe_seqedit.metrics import summarize_predictions


def _score(records: list[dict], fn) -> dict:
    decoded = []
    for record in records:
        decoded.append({**record, **fn(record)})
    return summarize_predictions(decoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    print_resolved_config(config)
    records = read_jsonl(Path(config["paths"]["dataset_dir"]) / f"{args.split}.jsonl")
    out_dir = ensure_dir(config["paths"]["baseline_dir"])
    summaries = {
        "no_edit": _score(records, no_edit),
        "consensus": _score(records, conservative_consensus),
        "support_rule": _score(records, support_rule),
    }
    for ext in config.get("external_baselines", []):
        decoded = external_predictions(ext["path"], records)
        summaries[ext["name"]] = summarize_predictions(decoded)
    write_json(out_dir / f"{args.split}_baseline_summary.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
