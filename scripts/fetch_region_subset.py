#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pysam

from omega_longread.utils import require_executable, require_existing_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a chromosome-restricted aligned subset from GIAB/HPRC BAM/CRAM.")
    parser.add_argument("--sample", required=True, choices=["HG002", "HG003", "HG004"])
    parser.add_argument("--chromosomes", required=True, help="Comma-separated contigs, e.g. chr20 or chr20,chr21,chr22")
    parser.add_argument("--source", required=True, choices=["giab", "hprc"])
    parser.add_argument("--aligned-url", required=True, help="Local path or remote BAM/CRAM URL.")
    parser.add_argument("--bai-url", default="", help="Optional BAM index path/URL.")
    parser.add_argument("--crai-url", default="", help="Optional CRAM index path/URL.")
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-read-len", type=int, default=10000)
    parser.add_argument("--downsample-frac", type=float, default=None)
    parser.add_argument("--target-coverage", type=float, default=None, help="Optional target coverage after chromosome slicing.")
    parser.add_argument("--write-fastq", action="store_true")
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    require_executable("samtools")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_url = args.aligned_url
    reference_fasta = require_existing_path(args.reference_fasta, label="reference_fasta", base_dir=Path.cwd())
    chroms = [token.strip() for token in args.chromosomes.split(",") if token.strip()]
    if not chroms:
        raise ValueError("At least one chromosome must be provided.")

    if args.downsample_frac is not None and args.target_coverage is not None:
        raise ValueError("Use only one of --downsample-frac or --target-coverage.")

    subset_bam = output_dir / f"{args.sample}.{args.source}.subset.bam"
    subset_fastq = output_dir / f"{args.sample}.{args.source}.subset.fastq.gz"
    metadata_path = output_dir / "metadata.json"

    view_cmd = ["samtools", "view", "-bh", aligned_url, *chroms]
    if args.downsample_frac is not None:
        frac = float(args.downsample_frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError("--downsample-frac must be in (0, 1].")
        view_cmd = ["samtools", "view", "-bh", "-s", f"42.{int(frac * 1000):03d}", aligned_url, *chroms]
    with subset_bam.open("wb") as handle:
        subprocess.run(view_cmd, check=True, stdout=handle)
    run(["samtools", "index", str(subset_bam)])

    total_reads = 0
    kept_reads = 0
    kept_bases = 0
    mapped_bases = 0
    chrom_lengths: dict[str, int] = {}
    with pysam.AlignmentFile(str(subset_bam), "rb", reference_filename=reference_fasta) as bam:
        for sq in bam.header.get("SQ", []):
            if sq.get("SN") in chroms:
                chrom_lengths[str(sq["SN"])] = int(sq["LN"])
        filtered_bam = output_dir / f"{args.sample}.{args.source}.subset.minlen.bam"
        with pysam.AlignmentFile(str(filtered_bam), "wb", header=bam.header) as out_bam:
            for aln in bam.fetch(until_eof=True):
                total_reads += 1
                if aln.is_unmapped or aln.query_length is None or aln.query_length < args.min_read_len:
                    continue
                out_bam.write(aln)
                kept_reads += 1
                kept_bases += int(aln.query_length)
                mapped_bases += int(aln.query_alignment_length or 0)
    subset_bam.unlink()
    filtered_bam = output_dir / f"{args.sample}.{args.source}.subset.minlen.bam"
    filtered_bai = output_dir / f"{args.sample}.{args.source}.subset.minlen.bam.bai"
    filtered_bam.rename(subset_bam)
    if filtered_bai.exists():
        filtered_bai.rename(output_dir / f"{subset_bam.name}.bai")
    else:
        run(["samtools", "index", str(subset_bam)])

    ref_bases = sum(chrom_lengths.values())
    coverage_estimate = (mapped_bases / ref_bases) if ref_bases > 0 else 0.0
    applied_downsample_frac = args.downsample_frac
    if args.target_coverage is not None and coverage_estimate > args.target_coverage and coverage_estimate > 0:
        applied_downsample_frac = max(min(args.target_coverage / coverage_estimate, 1.0), 1e-6)
        downsampled_bam = output_dir / f"{args.sample}.{args.source}.subset.downsampled.bam"
        with downsampled_bam.open("wb") as handle:
            subprocess.run(
                ["samtools", "view", "-bh", "-s", f"42.{int(applied_downsample_frac * 1000):03d}", str(subset_bam)],
                check=True,
                stdout=handle,
            )
        downsampled_bam.replace(subset_bam)
        run(["samtools", "index", str(subset_bam)])
        kept_reads = 0
        kept_bases = 0
        mapped_bases = 0
        with pysam.AlignmentFile(str(subset_bam), "rb", reference_filename=reference_fasta) as bam:
            for aln in bam.fetch(until_eof=True):
                if aln.is_unmapped:
                    continue
                kept_reads += 1
                kept_bases += int(aln.query_length or 0)
                mapped_bases += int(aln.query_alignment_length or 0)
        coverage_estimate = (mapped_bases / ref_bases) if ref_bases > 0 else 0.0

    if args.write_fastq:
        with subset_fastq.open("wb") as handle:
            subprocess.run(["samtools", "fastq", str(subset_bam)], check=True, stdout=handle)
    metadata = {
        "sample": args.sample,
        "source": args.source,
        "chromosomes": chroms,
        "aligned_url": aligned_url,
        "reference_fasta": reference_fasta,
        "subset_bam": str(subset_bam.resolve()),
        "subset_bai": str((output_dir / f"{subset_bam.name}.bai").resolve()),
        "subset_fastq": str(subset_fastq.resolve()) if args.write_fastq else "",
        "min_read_len": args.min_read_len,
        "downsample_frac": applied_downsample_frac,
        "target_coverage": args.target_coverage,
        "total_reads_seen": total_reads,
        "kept_reads": kept_reads,
        "kept_bases": kept_bases,
        "mapped_bases": mapped_bases,
        "reference_bases": ref_bases,
        "coverage_estimate": coverage_estimate,
        "md5": {
            "subset_bam": md5sum(subset_bam),
            "subset_fastq": md5sum(subset_fastq) if args.write_fastq and subset_fastq.exists() else "",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
