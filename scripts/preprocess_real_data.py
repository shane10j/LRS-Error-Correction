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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test JSONL windows from aligned long reads.")
    parser.add_argument("--reads-bam", required=True, help="Coordinate-sorted, indexed BAM/CRAM of long reads.")
    parser.add_argument("--reference-fasta", required=True, help="Indexed reference FASTA used for alignment.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test JSONL files.")
    parser.add_argument("--truth-vcf", default=None, help="Optional indexed truth VCF for preserve masks.")
    parser.add_argument("--confident-bed", default=None, help="Optional BED of high-confidence benchmark regions.")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--window-overlap", type=int, default=256)
    parser.add_argument("--min-window-size", type=int, default=256)
    parser.add_argument("--max-supports", type=int, default=16)
    parser.add_argument("--max-insertions-per-pos", type=int, default=2)
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
    return parser.parse_args()


def parse_contig_set(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_contigs = parse_contig_set(args.train_contigs)
    val_contigs = parse_contig_set(args.val_contigs)
    test_contigs = parse_contig_set(args.test_contigs)

    confidence_lookup = IntervalLookup.from_bed(args.confident_bed)
    variant_lookup = VariantLookup(args.truth_vcf)

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
            args.reads_bam,
            "rb",
            reference_filename=args.reference_fasta,
        ) as bam, pysam.FastaFile(args.reference_fasta) as fasta:
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

                split_name = split_name_for_contig(aln.reference_name, train_contigs, val_contigs, test_contigs)
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
                        bam=bam,
                        variant_lookup=variant_lookup,
                        confidence_lookup=confidence_lookup,
                        window_start=window_start,
                        window_end=window_end,
                        max_supports=args.max_supports,
                        min_mapq=args.min_mapq,
                        min_confident_fraction=args.min_confident_fraction,
                        min_mapped_fraction=args.min_mapped_fraction,
                        support_disagreement_threshold=args.support_disagreement_threshold,
                        min_support_depth=args.min_support_depth,
                        max_insertions_per_pos=args.max_insertions_per_pos,
                    )
                    if example is None:
                        continue
                    writers[split_name].write(json.dumps(example) + "\n")
                    counts[split_name] += 1
    finally:
        for writer in writers.values():
            writer.close()

    summary = {
        "reads_bam": args.reads_bam,
        "reference_fasta": args.reference_fasta,
        "truth_vcf": args.truth_vcf,
        "confident_bed": args.confident_bed,
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "max_supports": args.max_supports,
        "max_insertions_per_pos": args.max_insertions_per_pos,
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
