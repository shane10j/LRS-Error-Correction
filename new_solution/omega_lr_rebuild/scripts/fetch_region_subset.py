#!/usr/bin/env python
"""Extract a region subset from BAM/CRAM with lightweight metadata."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pysam

from omega_lr.utils import ensure_dir, print_config, read_config, save_json


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    print_config(config)
    dataset_cfg = config["dataset"]
    input_bam = Path(dataset_cfg["bam_path"])
    output_dir = ensure_dir(Path(args.output_dir or (Path(config["train"]["output_dir"]) / "subset")))
    output_bam = output_dir / input_bam.name
    read_count = 0
    total_bases = 0
    with pysam.AlignmentFile(input_bam, "rb") as src, pysam.AlignmentFile(output_bam, "wb", template=src) as dst:
        contigs = dataset_cfg.get("contigs") or list(src.references)
        for contig in contigs:
            for read in src.fetch(contig):
                if dataset_cfg.get("min_read_length") and (read.query_length or 0) < dataset_cfg["min_read_length"]:
                    continue
                dst.write(read)
                read_count += 1
                total_bases += read.query_length or 0
    pysam.index(str(output_bam))
    metadata = {
        "input_bam": str(input_bam),
        "output_bam": str(output_bam),
        "read_count": read_count,
        "total_bases": total_bases,
        "estimated_coverage": total_bases / max(1, dataset_cfg["max_window_length"] * dataset_cfg.get("max_examples", 1)),
        "file_size_bytes": output_bam.stat().st_size,
        "md5": md5sum(output_bam),
    }
    save_json(metadata, output_dir / "metadata.json")


if __name__ == "__main__":
    main()
