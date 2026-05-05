"""Dataset preprocessing for synthetic and small real BAM presets."""

from __future__ import annotations

import random
from pathlib import Path

from omega_safe_seqedit.io_utils import ensure_dir, write_json, write_jsonl
from omega_safe_seqedit.labels import make_edit_labels
from omega_safe_seqedit.features import pileup_features
from omega_safe_seqedit.synthetic import make_synthetic_split


def _real_records(config: dict) -> dict[str, list[dict]]:
    try:
        import pysam
    except ImportError as exc:
        raise RuntimeError("real_bam preprocessing requires pysam") from exc

    data = config["data"]
    bam_path = Path(data["bam"])
    ref_path = Path(data["reference_fasta"])
    if not bam_path.exists():
        raise FileNotFoundError(f"Missing BAM: {bam_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference FASTA: {ref_path}")

    rng = random.Random(config.get("seed", 47))
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    ref = pysam.FastaFile(str(ref_path))
    contig = data["contig"]
    start = int(data["start"])
    end = int(data["end"])
    window_len = int(data.get("window_length", 512))
    min_read_length = int(data.get("min_read_length", 0))
    max_support = int(data.get("max_support_reads", 8))
    total = int(data["num_train"]) + int(data["num_val"]) + int(data["num_test"])
    reads = [
        read
        for read in bam.fetch(contig, start, end)
        if not read.is_unmapped
        and not read.is_secondary
        and not read.is_supplementary
        and read.query_sequence
        and len(read.query_sequence) >= min_read_length
    ]
    if len(reads) < 2:
        raise RuntimeError("Need at least two usable reads in the selected region")
    rng.shuffle(reads)
    records = []
    for idx, read in enumerate(reads[:total]):
        aligned_start = max(start, read.reference_start or start)
        aligned_end = min(end, aligned_start + window_len)
        truth = ref.fetch(contig, aligned_start, aligned_end).upper()
        target = read.query_sequence[: max(1, min(len(read.query_sequence), len(truth) + 32))].upper()
        target = "".join(base for base in target if base in "ACGT")[: max(16, len(truth))]
        if not target or not truth:
            continue
        support_reads = [r for r in reads if r.query_name != read.query_name]
        rng.shuffle(support_reads)
        support = [
            "".join(base for base in r.query_sequence.upper() if base in "ACGT")[: max(16, len(truth) + 32)]
            for r in support_reads[:max_support]
            if r.query_sequence
        ]
        labels = make_edit_labels(target, truth)
        records.append(
            {
                "example_id": f"real_{idx}_{read.query_name}",
                "sample_id": data.get("sample", "real"),
                "contig": contig,
                "window_start": aligned_start,
                "window_end": aligned_end,
                "target_read_id": read.query_name,
                "target_seq": target,
                "support_read_ids": [r.query_name for r in support_reads[: len(support)]],
                "support_aligned_seqs": support,
                "truth_seq": truth,
                "labels": {
                    "main_type": labels.main_type,
                    "sub_base": labels.sub_base,
                    "insert_before": labels.insert_before,
                    "terminal_insert": labels.terminal_insert,
                },
                "features": pileup_features(target, support),
                "case_type": "real_bam_reference_truth",
            }
        )
    if len(records) < total:
        print(f"Warning: requested {total} records but created {len(records)}")
    train_n = int(data["num_train"])
    val_n = int(data["num_val"])
    return {
        "train": records[:train_n],
        "val": records[train_n : train_n + val_n],
        "test": records[train_n + val_n :],
    }


def build_dataset(config: dict) -> dict[str, list[dict]]:
    data = config["data"]
    if data["kind"] == "synthetic":
        seed = int(config.get("seed", 47))
        return {
            "train": make_synthetic_split(
                "train",
                int(data["num_train"]),
                seed,
                int(data["read_length"]),
                int(data["support_depth"]),
                float(data["support_noise"]),
                bool(data.get("include_neighbor_cases", True)),
                bool(data.get("include_homopolymer_cases", True)),
            ),
            "val": make_synthetic_split(
                "val",
                int(data["num_val"]),
                seed + 1 if not data.get("shared_examples_across_splits", False) else seed,
                int(data["read_length"]),
                int(data["support_depth"]),
                float(data["support_noise"]),
                bool(data.get("include_neighbor_cases", True)),
                bool(data.get("include_homopolymer_cases", True)),
            ),
            "test": make_synthetic_split(
                "test",
                int(data["num_test"]),
                seed + 2 if not data.get("shared_examples_across_splits", False) else seed,
                int(data["read_length"]),
                int(data["support_depth"]),
                float(data["support_noise"]),
                bool(data.get("include_neighbor_cases", True)),
                bool(data.get("include_homopolymer_cases", True)),
            ),
        }
    if data["kind"] == "real_bam":
        return _real_records(config)
    raise ValueError(f"Unsupported data.kind={data['kind']!r}")


def write_dataset(config: dict) -> None:
    dataset_dir = ensure_dir(config["paths"]["dataset_dir"])
    splits = build_dataset(config)
    for split, records in splits.items():
        write_jsonl(dataset_dir / f"{split}.jsonl", records)
    write_json(
        dataset_dir / "manifest.json",
        {
            "name": config["name"],
            "num_examples": {split: len(records) for split, records in splits.items()},
            "data_kind": config["data"]["kind"],
            "config_path": config.get("_config_path"),
        },
    )
