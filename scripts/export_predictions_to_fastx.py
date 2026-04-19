#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export decoded prediction JSONL rows to FASTA or FASTQ.")
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["fasta", "fastq"], default="fasta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.predictions_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Predictions JSONL not found: {input_path}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            read_id = row.get("read_id") or row.get("source_read_id") or "read"
            seq = row.get("predicted_sequence", "")
            if args.format == "fasta":
                dst.write(f">{read_id}\n{seq}\n")
            else:
                qual = "I" * len(seq)
                dst.write(f"@{read_id}\n{seq}\n+\n{qual}\n")
    print(str(output_path))


if __name__ == "__main__":
    main()
