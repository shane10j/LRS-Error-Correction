#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from omega_longread.utils import require_existing_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an external baseline command template for HERRO/DeChat-style tools.")
    parser.add_argument("--tool", required=True, choices=["herro", "dechat"])
    parser.add_argument("--reads-bam", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--command-template", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reads_bam = require_existing_path(args.reads_bam, label="reads_bam")
    reference_fasta = require_existing_path(args.reference_fasta, label="reference_fasta")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = args.command_template.strip()
    if not template:
        env_key = f"OMEGA_{args.tool.upper()}_CMD_TEMPLATE"
        template = os.environ.get(env_key, "").strip()
    if not template:
        raise RuntimeError(
            f"No command template provided for {args.tool}. Set --command-template or ${'OMEGA_' + args.tool.upper() + '_CMD_TEMPLATE'}."
        )

    substitutions = {
        "reads_bam": reads_bam,
        "reference_fasta": reference_fasta,
        "output_dir": str(output_dir),
        "summary_out": args.summary_out,
    }
    command = template.format(**substitutions)
    result = {
        "tool": args.tool,
        "command": command,
        "reads_bam": reads_bam,
        "reference_fasta": reference_fasta,
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        completed = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
        result["returncode"] = completed.returncode
    save_json(result, args.summary_out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
