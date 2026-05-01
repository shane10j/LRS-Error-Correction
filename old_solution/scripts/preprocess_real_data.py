#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pysam

from omega_longread.preprocessing import (
    IntervalLookup,
    VariantLookup,
    build_read_encoding,
    build_window_example,
    iter_windows,
    split_name_for_contig,
)
from omega_longread.utils import require_existing_path, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test JSONL windows from aligned long reads.")
    parser.add_argument("--reads-bam", required=True, help="Coordinate-sorted, indexed BAM/CRAM of long reads.")
    parser.add_argument("--reference-fasta", required=True, help="Indexed reference FASTA used for alignment.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test JSONL files.")
    parser.add_argument("--truth-vcf", default=None, help="Optional indexed truth VCF for preserve masks.")
    parser.add_argument("--phased-vcf", default=None, help="Optional indexed phased VCF for haplotype-aware masks.")
    parser.add_argument("--confident-bed", default=None, help="Optional BED of high-confidence benchmark regions.")
    parser.add_argument("--haplotagged-bam", default=None, help="Optional haplotagged BAM/CRAM used for support retrieval.")
    parser.add_argument(
        "--region-bed",
        action="append",
        default=[],
        help="Region BED annotation in the form name=/path/to/file.bed. Can be repeated.",
    )
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--window-overlap", type=int, default=256)
    parser.add_argument("--min-window-size", type=int, default=256)
    parser.add_argument("--max-supports", type=int, default=16)
    parser.add_argument("--min-supports-per-window", type=int, default=1)
    parser.add_argument("--max-insertions-per-pos", type=int, default=2)
    parser.add_argument("--max-deletion-length", type=int, default=4)
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--min-mapped-fraction", type=float, default=0.7)
    parser.add_argument("--min-confident-fraction", type=float, default=0.8)
    parser.add_argument("--support-disagreement-threshold", type=float, default=0.7)
    parser.add_argument("--min-support-depth", type=int, default=2)
    parser.add_argument("--min-read-length", type=int, default=512)
    parser.add_argument("--limit-alignments", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--max-examples-per-split", type=int, default=None, help="Optional cap for each split.")
    parser.add_argument("--train-contigs", default="", help="Comma-separated contigs for training.")
    parser.add_argument("--val-contigs", default="", help="Comma-separated contigs for validation.")
    parser.add_argument("--test-contigs", default="", help="Comma-separated contigs for testing.")
    parser.add_argument(
        "--allow-shared-holdout-contigs",
        action="store_true",
        help="Allow validation and test contig sets to overlap. Training contigs must still remain disjoint.",
    )
    return parser.parse_args()


def parse_contig_set(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def parse_region_beds(raw_items: list[str]) -> dict[str, str]:
    region_beds: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Expected --region-bed in name=path form, got {item!r}.")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Expected non-empty region name and path in {item!r}.")
        region_beds[name] = path
    return region_beds


def validate_contig_sets(
    train_contigs: set[str],
    val_contigs: set[str],
    test_contigs: set[str],
    allow_shared_holdout_contigs: bool,
) -> None:
    overlap = train_contigs & (val_contigs | test_contigs)
    if overlap and not allow_shared_holdout_contigs:
        raise ValueError(
            "Train/holdout chromosome overlap is not allowed unless "
            f"--allow-shared-holdout-contigs is set. Overlap: {sorted(overlap)}"
        )
    if not allow_shared_holdout_contigs:
        holdout_overlap = val_contigs & test_contigs
        if holdout_overlap:
            raise ValueError(
                "Validation and test chromosome overlap is not allowed unless "
                f"--allow-shared-holdout-contigs is set. Overlap: {sorted(holdout_overlap)}"
            )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path.cwd()

    reads_bam = require_existing_path(args.reads_bam, label="reads_bam", base_dir=base_dir)
    reference_fasta = require_existing_path(args.reference_fasta, label="reference_fasta", base_dir=base_dir)
    truth_vcf = require_existing_path(args.truth_vcf, label="truth_vcf", base_dir=base_dir, allow_empty=True)
    phased_vcf = require_existing_path(args.phased_vcf, label="phased_vcf", base_dir=base_dir, allow_empty=True)
    confident_bed = require_existing_path(args.confident_bed, label="confident_bed", base_dir=base_dir, allow_empty=True)
    haplotagged_bam = require_existing_path(
        args.haplotagged_bam,
        label="haplotagged_bam",
        base_dir=base_dir,
        allow_empty=True,
    )
    region_bed_paths = {
        name: require_existing_path(path, label=f"region_bed[{name}]", base_dir=base_dir)
        for name, path in parse_region_beds(args.region_bed).items()
    }

    train_contigs = parse_contig_set(args.train_contigs)
    val_contigs = parse_contig_set(args.val_contigs)
    test_contigs = parse_contig_set(args.test_contigs)
    validate_contig_sets(
        train_contigs,
        val_contigs,
        test_contigs,
        allow_shared_holdout_contigs=args.allow_shared_holdout_contigs,
    )

    confidence_lookup = IntervalLookup.from_bed(confident_bed)
    variant_lookup = VariantLookup(truth_vcf)
    phased_variant_lookup = VariantLookup(phased_vcf, phased_only=True) if phased_vcf else None
    region_lookups = {name: IntervalLookup.from_bed(path) for name, path in region_bed_paths.items()}

    counts = defaultdict(int)
    alignments_seen = 0

    output_paths = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }
    writers = {name: open(path, "w", encoding="utf-8") for name, path in output_paths.items()}

    try:
        with pysam.AlignmentFile(
            reads_bam,
            "rb",
            reference_filename=reference_fasta,
        ) as bam, pysam.AlignmentFile(
            haplotagged_bam or reads_bam,
            "rb",
            reference_filename=reference_fasta,
        ) as support_bam, pysam.FastaFile(reference_fasta) as fasta:
            for aln in bam.fetch(until_eof=True):
                if args.limit_alignments is not None and alignments_seen >= args.limit_alignments:
                    break
                alignments_seen += 1

                if (
                    aln.is_unmapped
                    or aln.is_secondary
                    or aln.is_supplementary
                    or aln.mapping_quality < args.min_mapq
                    or aln.reference_name is None
                ):
                    continue

                split_name = split_name_for_contig(
                    aln.reference_name,
                    train_contigs,
                    val_contigs,
                    test_contigs,
                    read_id=aln.query_name,
                )
                if split_name is None:
                    continue
                if args.max_examples_per_split is not None and counts[split_name] >= args.max_examples_per_split:
                    continue

                read_encoding = build_read_encoding(
                    aln=aln,
                    fasta=fasta,
                    max_insertions_per_pos=args.max_insertions_per_pos,
                )
                if read_encoding is None or len(read_encoding.target_bases) < args.min_read_length:
                    continue

                for window_start, window_end in iter_windows(
                    length=len(read_encoding.target_bases),
                    window_size=args.window_size,
                    window_overlap=args.window_overlap,
                    min_window_size=args.min_window_size,
                ):
                    if args.max_examples_per_split is not None and counts[split_name] >= args.max_examples_per_split:
                        break
                    example = build_window_example(
                        target_aln=aln,
                        read_encoding=read_encoding,
                        support_bam=support_bam,
                        variant_lookup=variant_lookup,
                        phased_variant_lookup=phased_variant_lookup,
                        confidence_lookup=confidence_lookup,
                        region_lookups=region_lookups,
                        window_start=window_start,
                        window_end=window_end,
                        max_supports=args.max_supports,
                        min_supports_per_window=args.min_supports_per_window,
                        min_mapq=args.min_mapq,
                        min_confident_fraction=args.min_confident_fraction,
                        min_mapped_fraction=args.min_mapped_fraction,
                        support_disagreement_threshold=args.support_disagreement_threshold,
                        min_support_depth=args.min_support_depth,
                        max_insertions_per_pos=args.max_insertions_per_pos,
                        max_deletion_length=args.max_deletion_length,
                    )
                    if example is None:
                        continue
                    writers[split_name].write(json.dumps(example) + "\n")
                    counts[split_name] += 1
    finally:
        for writer in writers.values():
            writer.close()

    summary = {
        "reads_bam": reads_bam,
        "haplotagged_bam": haplotagged_bam or reads_bam,
        "reference_fasta": reference_fasta,
        "truth_vcf": truth_vcf,
        "phased_vcf": phased_vcf,
        "confident_bed": confident_bed,
        "region_beds": region_bed_paths,
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "max_supports": args.max_supports,
        "min_supports_per_window": args.min_supports_per_window,
        "max_insertions_per_pos": args.max_insertions_per_pos,
        "max_deletion_length": args.max_deletion_length,
        "train_contigs": sorted(train_contigs),
        "val_contigs": sorted(val_contigs),
        "test_contigs": sorted(test_contigs),
        "counts": dict(counts),
        "train_path": str(output_paths["train"]),
        "val_path": str(output_paths["val"]),
        "test_path": str(output_paths["test"]),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
