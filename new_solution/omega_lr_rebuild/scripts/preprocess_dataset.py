#!/usr/bin/env python
"""Preprocess synthetic or aligned-read datasets into JSONL examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from omega_lr.data.io_alignments import (
    build_query_ref_maps,
    fetch_support_reads,
    iter_target_windows,
    open_alignment_file,
    parse_alignment_string,
)
from omega_lr.data.io_fastx import read_fasta
from omega_lr.data.labels import generate_labels
from omega_lr.data.manifest import build_manifest, save_manifest
from omega_lr.data.pileup import compute_support_features
from omega_lr.data.schemas import ExampleRecord
from omega_lr.data.support_rules import derive_support_rule_labels
from omega_lr.data.windowing import generate_synthetic_examples
from omega_lr.constants import BASE_TO_ID, ID_TO_EDIT
from omega_lr.logging_utils import get_logger
from omega_lr.utils import dump_yaml, ensure_dir, print_config, read_config, save_jsonl


LOGGER = get_logger(__name__)


def canonicalize_synthetic_insertion_evidence(
    edit_labels: list[int],
    insertion_counts: list[int] | None,
    insertion_base_counts: list[list[int]] | None,
) -> tuple[list[int] | None, list[list[int]] | None]:
    """Move synthetic insertion evidence to the canonical gold label anchor."""
    if insertion_counts is None or insertion_base_counts is None:
        return insertion_counts, insertion_base_counts
    updated_counts = list(insertion_counts)
    updated_base_counts = [list(row) for row in insertion_base_counts]
    for pos, label_id in enumerate(edit_labels):
        label = ID_TO_EDIT[int(label_id)]
        if not label.startswith("INS_"):
            continue
        base_id = BASE_TO_ID[label[-1]]
        if pos < len(updated_base_counts) and updated_base_counts[pos][base_id] > 0:
            continue
        donor_pos = next(
            (
                idx
                for idx, row in enumerate(updated_base_counts)
                if row[base_id] > 0 and (idx >= len(edit_labels) or ID_TO_EDIT[int(edit_labels[idx])] != label)
            ),
            None,
        )
        if donor_pos is None:
            continue
        moved = updated_base_counts[donor_pos][base_id]
        updated_base_counts[pos][base_id] += moved
        updated_counts[pos] += moved
        updated_counts[donor_pos] = max(0, updated_counts[donor_pos] - moved)
        updated_base_counts[donor_pos][base_id] = 0
    return updated_counts, updated_base_counts


def build_synthetic_split(config: dict, split: str) -> list[dict]:
    dataset_cfg = config["dataset"]
    case_names = dataset_cfg.get("synthetic_case_names")
    suite = dataset_cfg.get("synthetic_suite", "basic")
    shared_examples = dataset_cfg.get("shared_examples_across_splits", False)
    split_seed = dataset_cfg["synthetic_seed"] if shared_examples else dataset_cfg["synthetic_seed"]
    rows = []
    split_name = "shared" if shared_examples else split
    for raw in generate_synthetic_examples(split_name, dataset_cfg["splits"][split], split_seed, case_names, suite=suite):
        insertion_counts = raw.pop("_support_insertion_counts", None)
        insertion_base_counts = raw.pop("_support_insertion_base_counts", None)
        raw.pop("_support_insertion_events", None)
        deletion_lengths = raw.pop("_support_deletion_lengths", None)
        labels = generate_labels(raw["target_seq"], raw["truth_seq"], dataset_cfg["max_deletion_length"])
        insertion_counts, insertion_base_counts = canonicalize_synthetic_insertion_evidence(
            labels["edit_labels"],
            insertion_counts,
            insertion_base_counts,
        )
        features = compute_support_features(
            raw["target_seq"],
            raw["support_aligned_seqs"],
            raw["support_strands"],
            insertion_counts,
            deletion_lengths,
            insertion_base_counts,
        )
        support_rule_labels = derive_support_rule_labels(raw["target_seq"], features)
        record = ExampleRecord(
            **raw,
            edit_labels=labels["edit_labels"],
            delete_candidate_labels=labels["delete_candidate_labels"],
            delete_length_labels=labels["delete_length_labels"],
            support_rule_labels=support_rule_labels,
            features=features,
            masks=labels["masks"],
        )
        rows.append(record.to_dict())
    return rows


def build_real_examples(config: dict) -> dict[str, list[dict]]:
    dataset_cfg = config["dataset"]
    reference = read_fasta(dataset_cfg["reference_fasta"])
    truth = read_fasta(dataset_cfg.get("truth_fasta", dataset_cfg["reference_fasta"]))
    bam_path = dataset_cfg["bam_path"]
    max_support = dataset_cfg["max_support_reads"]
    output_rows = []
    with open_alignment_file(bam_path) as bam:
        for window in iter_target_windows(
            bam_path,
            dataset_cfg.get("contigs", []),
            dataset_cfg["max_window_length"],
            dataset_cfg["overlap"],
            dataset_cfg["min_read_length"],
            dataset_cfg["max_examples"],
        ):
            target_read = next(bam.fetch(window["contig"], window["ref_start"], window["ref_end"]))
            if target_read.query_name != window["read_name"]:
                for candidate in bam.fetch(window["contig"], window["ref_start"], window["ref_end"]):
                    if candidate.query_name == window["read_name"]:
                        target_read = candidate
                        break
            query_to_ref, _, target_insertions, _ = build_query_ref_maps(target_read)
            target_seq = target_read.query_sequence[window["query_start"] : window["query_end"]]
            ref_positions = []
            last_ref = window["ref_start"]
            for query_pos in range(window["query_start"], window["query_end"]):
                ref_pos = query_to_ref.get(query_pos, last_ref)
                last_ref = ref_pos
                ref_positions.append(ref_pos)
            support_aligned = []
            support_strands = []
            support_ids = []
            support_offsets = []
            support_deletion_lengths = []
            support_insertion_counts = [0] * len(target_seq)
            for record in fetch_support_reads(bam_path, window["contig"], window["ref_start"], window["ref_end"], window["read_name"], max_support):
                support = parse_alignment_string(bam_path, record)
                _, ref_to_base, insertion_after, deletion_at = build_query_ref_maps(support)
                aligned = []
                deletion_lengths = []
                for idx, ref_pos in enumerate(ref_positions):
                    aligned.append(ref_to_base.get(ref_pos, "-"))
                    deletion_lengths.append(deletion_at.get(ref_pos, 0))
                    support_insertion_counts[idx] += insertion_after.get(ref_pos, 0)
                support_aligned.append("".join(aligned))
                support_ids.append(support.query_name)
                support_offsets.append(window["ref_start"])
                support_strands.append("-" if support.is_reverse else "+")
                support_deletion_lengths.append(deletion_lengths)
            truth_seq = truth[window["contig"]][window["ref_start"] : window["ref_end"]]
            features = compute_support_features(target_seq, support_aligned, support_strands or ["+"], support_insertion_counts, support_deletion_lengths)
            labels = generate_labels(target_seq, truth_seq, dataset_cfg["max_deletion_length"])
            support_rule_labels = derive_support_rule_labels(target_seq, features)
            output_rows.append(
                ExampleRecord(
                    example_id=f"{window['read_name']}_{window['query_start']}_{window['query_end']}",
                    sample_id=dataset_cfg["sample_id"],
                    contig=window["contig"],
                    window_start=window["ref_start"],
                    window_end=window["ref_end"],
                    target_read_id=window["read_name"],
                    target_seq=target_seq,
                    target_qual=[30] * len(target_seq),
                    support_read_ids=support_ids,
                    support_aligned_seqs=support_aligned,
                    support_offsets=support_offsets,
                    support_strands=support_strands,
                    truth_seq=truth_seq,
                    edit_labels=labels["edit_labels"],
                    delete_candidate_labels=labels["delete_candidate_labels"],
                    delete_length_labels=labels["delete_length_labels"],
                    support_rule_labels=support_rule_labels,
                    features=features,
                    masks=labels["masks"],
                ).to_dict()
            )
    total = len(output_rows)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    return {
        "train": output_rows[:train_end],
        "val": output_rows[train_end:val_end],
        "test": output_rows[val_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    print_config(config)
    output_dir = ensure_dir(Path(config["dataset"]["output_dir"]))
    dump_yaml(config, output_dir / "config_snapshot.yaml")
    if config["dataset"]["kind"] == "synthetic":
        splits = {split: build_synthetic_split(config, split) for split in ["train", "val", "test"]}
        source_files = {"generator": "synthetic"}
    else:
        splits = build_real_examples(config)
        source_files = {
            "bam_path": config["dataset"]["bam_path"],
            "reference_fasta": config["dataset"]["reference_fasta"],
            "truth_fasta": config["dataset"].get("truth_fasta", config["dataset"]["reference_fasta"]),
        }
    for split, rows in splits.items():
        save_jsonl(rows, output_dir / f"{split}.jsonl")
        LOGGER.info("saved %s examples to %s", len(rows), output_dir / f"{split}.jsonl")
    manifest = build_manifest(config, {split: len(rows) for split, rows in splits.items()}, source_files)
    save_manifest(manifest, output_dir / "manifest.json")


if __name__ == "__main__":
    main()
