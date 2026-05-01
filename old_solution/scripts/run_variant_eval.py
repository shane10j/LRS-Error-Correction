#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from omega_longread.utils import require_existing_path, require_executable, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run variant-calling impact evaluation for corrected reads.")
    parser.add_argument("--reads-fastx", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--truth-vcf", required=True)
    parser.add_argument("--confident-bed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--aligner", default="minimap2")
    parser.add_argument("--caller-cmd-template", default="")
    parser.add_argument("--happy-cmd-template", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_happy_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Type", "").upper() in {"SNP", "INDEL", "ALL"} or row.get("Filter", "") == "PASS":
                out = {}
                for key in ["METRIC.Precision", "METRIC.Recall", "METRIC.F1_Score"]:
                    if key in row and row[key]:
                        out[f"variant_{key.split('.')[-1].lower()}"] = float(row[key])
                return out
    return {}


def main() -> None:
    args = parse_args()
    reads_fastx = require_existing_path(args.reads_fastx, label="reads_fastx")
    reference_fasta = require_existing_path(args.reference_fasta, label="reference_fasta")
    truth_vcf = require_existing_path(args.truth_vcf, label="truth_vcf")
    confident_bed = require_existing_path(args.confident_bed, label="confident_bed")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bam_path = output_dir / "corrected.bam"
    vcf_path = output_dir / "calls.vcf.gz"
    happy_prefix = output_dir / "happy"
    happy_csv = output_dir / "happy.summary.csv"

    aligner = require_executable(args.aligner)
    align_cmd = f"{aligner} -ax map-ont {reference_fasta} {reads_fastx} > {output_dir / 'corrected.sam'}"
    caller_template = args.caller_cmd_template.strip() or os.environ.get("OMEGA_VARIANT_CALL_CMD_TEMPLATE", "").strip()
    happy_template = args.happy_cmd_template.strip() or os.environ.get("OMEGA_HAPPY_CMD_TEMPLATE", "").strip()
    if not caller_template:
        raise RuntimeError("Missing variant caller command template. Set --caller-cmd-template or $OMEGA_VARIANT_CALL_CMD_TEMPLATE.")
    if not happy_template:
        raise RuntimeError("Missing hap.py command template. Set --happy-cmd-template or $OMEGA_HAPPY_CMD_TEMPLATE.")

    caller_cmd = caller_template.format(
        reads_fastx=reads_fastx,
        reference_fasta=reference_fasta,
        output_dir=str(output_dir),
        bam_path=str(bam_path),
        vcf_path=str(vcf_path),
    )
    happy_cmd = happy_template.format(
        truth_vcf=truth_vcf,
        reference_fasta=reference_fasta,
        query_vcf=str(vcf_path),
        confident_bed=confident_bed,
        output_prefix=str(happy_prefix),
        happy_csv=str(happy_csv),
    )

    result = {
        "align_command": align_cmd,
        "caller_command": caller_cmd,
        "happy_command": happy_cmd,
        "reads_fastx": reads_fastx,
        "reference_fasta": reference_fasta,
        "truth_vcf": truth_vcf,
        "confident_bed": confident_bed,
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        subprocess.run(align_cmd, shell=True, check=True, text=True, capture_output=True)
        subprocess.run(caller_cmd, shell=True, check=True, text=True, capture_output=True)
        subprocess.run(happy_cmd, shell=True, check=True, text=True, capture_output=True)
        result.update(parse_happy_csv(happy_csv))
    save_json(result, args.summary_out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
