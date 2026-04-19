#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from omega_longread.utils import require_existing_path, require_executable, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assembly and compute basic assembly metrics.")
    parser.add_argument("--reads-fastx", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--genome-size", default="3.1g")
    parser.add_argument("--assembler", default="flye")
    parser.add_argument("--assembler-cmd-template", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fasta_lengths(path: Path) -> list[int]:
    lengths: list[int] = []
    current = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if current > 0:
                    lengths.append(current)
                current = 0
            else:
                current += len(line.strip())
    if current > 0:
        lengths.append(current)
    return lengths


def n50(lengths: list[int]) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths, reverse=True)
    total = sum(ordered)
    running = 0
    for length in ordered:
        running += length
        if running >= total / 2:
            return length
    return 0


def main() -> None:
    args = parse_args()
    reads_fastx = require_existing_path(args.reads_fastx, label="reads_fastx")
    reference_fasta = require_existing_path(args.reference_fasta, label="reference_fasta")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = args.assembler_cmd_template.strip() or os.environ.get("OMEGA_ASSEMBLER_CMD_TEMPLATE", "").strip()
    if template:
        command = template.format(
            reads_fastx=reads_fastx,
            reference_fasta=reference_fasta,
            output_dir=str(output_dir),
            genome_size=args.genome_size,
        )
    else:
        assembler = require_executable(args.assembler)
        command = f"{assembler} --nano-hq {reads_fastx} --genome-size {args.genome_size} --out-dir {output_dir}"

    result = {
        "command": command,
        "reads_fastx": reads_fastx,
        "reference_fasta": reference_fasta,
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        completed = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
        assembly_path = output_dir / "assembly.fasta"
        if not assembly_path.exists():
            raise FileNotFoundError(f"Expected assembly at {assembly_path}")
        lengths = fasta_lengths(assembly_path)
        result.update(
            {
                "assembly_fasta": str(assembly_path),
                "assembly_contig_count": len(lengths),
                "assembly_total_bases": sum(lengths),
                "assembly_n50": n50(lengths),
                "assembly_max_contig": max(lengths) if lengths else 0,
            }
        )
    save_json(result, args.summary_out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
