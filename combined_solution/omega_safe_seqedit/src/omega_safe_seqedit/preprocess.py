"""Dataset preprocessing for synthetic and small real BAM presets."""

from __future__ import annotations

import random
from pathlib import Path

from omega_safe_seqedit.io_utils import ensure_dir, write_json, write_jsonl
from omega_safe_seqedit.labels import make_edit_labels
from omega_safe_seqedit.features import pileup_features
from omega_safe_seqedit.synthetic import make_synthetic_split


OLD_PAD_EDIT = 10
OLD_COPY = 0
OLD_SUB_A = 1
OLD_SUB_T = 4
OLD_DEL = 5
OLD_INS_A = 6
OLD_INS_T = 9


def _clean_old_base_seq(seq: str) -> str:
    """Keep length stable while replacing ambiguous bases with a harmless token."""
    return "".join(base if base in "ACGT" else "A" for base in seq.upper())


def _old_labels_to_safe(record: dict, target_seq: str, truth_seq: str) -> dict:
    """Convert old flat edit-slot labels into factorized safe-seqedit labels."""
    main_type = []
    sub_base = []
    insert_before = []
    for pos, slots in enumerate(record["edit_labels"][: len(target_seq)]):
        if not isinstance(slots, list):
            slots = [slots]
        active = [label for label in slots if label != OLD_PAD_EDIT]
        main = 0
        sub = 0
        ins = 0
        for label in active:
            if OLD_SUB_A <= label <= OLD_SUB_T:
                main = 1
                sub = label - OLD_SUB_A
            elif label == OLD_DEL:
                main = 2
            elif OLD_INS_A <= label <= OLD_INS_T:
                ins = (label - OLD_INS_A) + 1
            elif label == OLD_COPY:
                main = main or 0
        main_type.append(main)
        sub_base.append(sub)
        insert_before.append(ins)
    if len(main_type) < len(target_seq):
        fallback = make_edit_labels(target_seq, truth_seq)
        return {
            "main_type": fallback.main_type,
            "sub_base": fallback.sub_base,
            "insert_before": fallback.insert_before,
            "terminal_insert": fallback.terminal_insert,
        }
    return {
        "main_type": main_type,
        "sub_base": sub_base,
        "insert_before": insert_before,
        "terminal_insert": 0,
    }


def _sum_nested_base_counts(value: list, length: int) -> list[list[int]]:
    counts = [[0, 0, 0, 0] for _ in range(length)]
    for support in value or []:
        for pos, row in enumerate(support[:length]):
            for idx in range(min(4, len(row))):
                counts[pos][idx] += int(round(float(row[idx])))
    return counts


def _sum_nested_mask(value: list, length: int) -> list[int]:
    counts = [0] * length
    for support in value or []:
        for pos, item in enumerate(support[:length]):
            counts[pos] += int(round(float(item)))
    return counts


def _mean_nested_mask(value: list, length: int, default: float = 0.0) -> list[float]:
    sums = [0.0] * length
    counts = [0] * length
    for support in value or []:
        for pos, item in enumerate(support[:length]):
            sums[pos] += float(item)
            counts[pos] += 1
    return [sums[pos] / counts[pos] if counts[pos] else default for pos in range(length)]


def _strand_support_stats(value: list, length: int) -> tuple[list[float], list[int], list[int]]:
    """Return forward fraction plus explicit counts for old +/- strand encodings."""
    forward = [0] * length
    reverse = [0] * length
    observed = [0] * length
    for support in value or []:
        for pos, item in enumerate(support[:length]):
            try:
                strand = float(item)
            except (TypeError, ValueError):
                continue
            observed[pos] += 1
            if strand >= 0:
                forward[pos] += 1
            else:
                reverse[pos] += 1
    fraction = [forward[pos] / observed[pos] if observed[pos] else 0.5 for pos in range(length)]
    return fraction, forward, reverse


def _old_vector(record: dict, key: str, length: int, default: int = 0) -> list[int]:
    values = [int(round(float(x))) for x in record.get(key, [])[:length]]
    if len(values) < length:
        values.extend([default] * (length - len(values)))
    return values


def _old_region_mask(record: dict, key: str, length: int) -> list[int]:
    region_masks = record.get("region_masks", {})
    if not isinstance(region_masks, dict):
        return [0] * length
    values = [int(round(float(x))) for x in region_masks.get(key, [])[:length]]
    if len(values) < length:
        values.extend([0] * (length - len(values)))
    return values


def _old_features_to_safe(record: dict, target_seq: str) -> dict:
    length = len(target_seq)
    base_counts = _sum_nested_base_counts(record.get("support_base_support", []), length)
    ins_base_counts = _sum_nested_base_counts(record.get("support_ins_base_support", []), length)
    del_count = [int(round(float(x))) for x in record.get("deletion_support_count", [])[:length]]
    if len(del_count) < length:
        del_count = _sum_nested_mask(record.get("support_del_mask", []), length)
    ins_count = _sum_nested_mask(record.get("support_ins_mask", []), length)
    depth = [int(round(float(x))) for x in record.get("local_support_depth", [])[:length]]
    if len(depth) < length:
        depth = [max(sum(base_counts[pos]) + del_count[pos], 1) for pos in range(length)]
    agreement = [float(x) for x in record.get("local_support_agreement", [])[:length]]
    entropy = [float(x) for x in record.get("local_support_entropy", [])[:length]]
    while len(agreement) < length:
        pos = len(agreement)
        dep = max(depth[pos], 1)
        agreement.append(max(base_counts[pos] + [del_count[pos]]) / dep)
    while len(entropy) < length:
        entropy.append(0.0)
    margin = []
    support_rule_type = []
    support_rule_sub_base = []
    support_rule_ins_base = []
    for pos, base in enumerate(target_seq):
        full_counts = base_counts[pos] + [del_count[pos], ins_count[pos]]
        top = sorted(full_counts, reverse=True)
        margin.append(float(top[0] - top[1]) if len(top) > 1 else float(top[0]))
        dep = max(float(depth[pos]), 1.0)
        best_base = max(range(4), key=lambda idx: base_counts[pos][idx])
        best_ins = max(range(4), key=lambda idx: ins_base_counts[pos][idx]) if sum(ins_base_counts[pos]) else 0
        rule = 0
        sub = 0
        ins = 0
        if "ACGT"[best_base] != base and base_counts[pos][best_base] / dep >= 0.60:
            rule = 1
            sub = best_base
        if ins_count[pos] / dep >= 0.60:
            rule = 3
            ins = best_ins
        if del_count[pos] / dep >= 0.70:
            rule = 2
        support_rule_type.append(rule)
        support_rule_sub_base.append(sub)
        support_rule_ins_base.append(ins)
    homopolymer = [int(x) for x in record.get("target_run_lengths", [])[:length]]
    if len(homopolymer) < length:
        homopolymer.extend([1] * (length - len(homopolymer)))
    tandem = [int(x) for x in record.get("tandem_repeat_flag", [])[:length]]
    if len(tandem) < length:
        tandem.extend([0] * (length - len(tandem)))
    variant_mask = _old_vector(record, "variant_mask", length)
    phased_variant_mask = _old_vector(record, "phased_variant_mask", length)
    preserve_mask = _old_vector(record, "preserve_mask", length)
    uncertainty_label = _old_vector(record, "uncertainty_labels", length)
    region_homopolymer = _old_region_mask(record, "homopolymer", length)
    variant_rich = _old_region_mask(record, "variant_rich", length)
    region_tandem = _old_region_mask(record, "tandem_repeat", length)
    confident_mask = [0 if preserve_mask[pos] or uncertainty_label[pos] else 1 for pos in range(length)]
    support_forward_fraction, support_forward_count, support_reverse_count = _strand_support_stats(record.get("support_strand", []), length)
    support_same_haplotype_fraction = _mean_nested_mask(record.get("support_same_haplotype", []), length)
    support_match_fraction = _mean_nested_mask(record.get("support_match_mask", []), length)
    neighbor = []
    local_rule_density = []
    local_mismatch_density = []
    local_variant_density = []
    nearby_indel_density = []
    left_support_match_fraction = []
    right_support_match_fraction = []
    variant_proximity_flag = []
    repeat_strength = []
    for pos in range(length):
        near = any(support_rule_type[j] != 0 for j in range(max(0, pos - 1), min(length, pos + 2)) if j != pos)
        neighbor.append(1 if near else 0)
        start = max(0, pos - 2)
        end = min(length, pos + 3)
        local_rule_density.append(sum(1 for j in range(start, end) if j != pos and support_rule_type[j] != 0))
        local_mismatch_density.append(sum(1.0 - support_match_fraction[j] for j in range(start, end)) / max(end - start, 1))
        local_variant_density.append(sum(1 for j in range(start, end) if variant_mask[j] or phased_variant_mask[j]) / max(end - start, 1))
        nearby_indel_density.append(sum(1 for j in range(start, end) if j != pos and support_rule_type[j] in {2, 3}) / max(end - start - 1, 1))
        left = support_match_fraction[max(0, pos - 3):pos]
        right = support_match_fraction[pos + 1:min(length, pos + 4)]
        left_support_match_fraction.append(sum(left) / len(left) if left else support_match_fraction[pos])
        right_support_match_fraction.append(sum(right) / len(right) if right else support_match_fraction[pos])
        variant_proximity_flag.append(1 if any(variant_mask[j] or phased_variant_mask[j] for j in range(start, end)) else 0)
        repeat_strength.append(
            min(
                1.0,
                (0.35 if tandem[pos] or region_tandem[pos] else 0.0)
                + (0.35 if region_homopolymer[pos] else 0.0)
                + min(max(homopolymer[pos] - 1, 0) / 8.0, 0.4),
            )
        )
    return {
        "support_base_counts": base_counts,
        "support_del_count": del_count,
        "support_ins_count": ins_count,
        "support_ins_base_counts": ins_base_counts,
        "support_depth": depth,
        "support_agreement": agreement,
        "support_entropy": entropy,
        "support_margin": margin,
        "homopolymer_run_length": homopolymer,
        "tandem_repeat_flag": tandem,
        "region_homopolymer_flag": region_homopolymer,
        "region_tandem_repeat_flag": region_tandem,
        "variant_mask": variant_mask,
        "phased_variant_mask": phased_variant_mask,
        "preserve_mask": preserve_mask,
        "uncertainty_label": uncertainty_label,
        "variant_rich_flag": variant_rich,
        "confident_mask": confident_mask,
        "support_forward_fraction": support_forward_fraction,
        "support_forward_count": support_forward_count,
        "support_reverse_count": support_reverse_count,
        "support_same_haplotype_fraction": support_same_haplotype_fraction,
        "support_match_fraction": support_match_fraction,
        "support_strand_bias": [abs(x - 0.5) * 2.0 for x in support_forward_fraction],
        "local_rule_density": local_rule_density,
        "local_mismatch_density": local_mismatch_density,
        "local_variant_density": local_variant_density,
        "nearby_indel_density": nearby_indel_density,
        "left_support_match_fraction": left_support_match_fraction,
        "right_support_match_fraction": right_support_match_fraction,
        "variant_proximity_flag": variant_proximity_flag,
        "repeat_strength": repeat_strength,
        "mapping_quality_mean": [0.0] * length,
        "mapping_quality_available": [0] * length,
        "reference_kmer_uniqueness": [0.0] * length,
        "reference_kmer_uniqueness_available": [0] * length,
        "window_relative_position": [pos / max(length - 1, 1) for pos in range(length)],
        "boundary_flag": [1 if pos in {0, length - 1} else 0 for pos in range(length)],
        "neighbor_rule_flag": neighbor,
        "support_rule_type": support_rule_type,
        "support_rule_sub_base": support_rule_sub_base,
        "support_rule_ins_base": support_rule_ins_base,
    }


def _old_jsonl_records(config: dict) -> dict[str, list[dict]]:
    from omega_safe_seqedit.io_utils import read_jsonl

    data = config["data"]
    splits = {}
    max_examples = data.get("max_examples", {})
    for split in ["train", "val", "test"]:
        path = Path(data[f"{split}_path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing old JSONL {split} path: {path}")
        records = []
        for idx, old in enumerate(read_jsonl(path)):
            if split in max_examples and idx >= int(max_examples[split]):
                break
            target = _clean_old_base_seq(old["target_bases"])
            truth = _clean_old_base_seq(old.get("target_sequence", target))
            labels = _old_labels_to_safe(old, target, truth)
            records.append(
                {
                    "example_id": old.get("read_id", f"{split}_{idx}"),
                    "sample_id": data.get("sample", "HG002"),
                    "contig": old.get("contig", "chr20"),
                    "window_start": int(old.get("window_ref_start", 0) or 0),
                    "window_end": int(old.get("window_ref_end", len(target)) or len(target)),
                    "target_read_id": old.get("source_read_id", old.get("read_id", f"{split}_{idx}")),
                    "target_seq": target,
                    "support_read_ids": [f"support_{support_idx}" for support_idx in range(len(old.get("support_bases", [])))],
                    "support_aligned_seqs": [_clean_old_base_seq(seq) for seq in old.get("support_bases", [])],
                    "support_strand_tracks": old.get("support_strand", []),
                    "support_cigar_snippets": old.get("support_cigars", old.get("support_cigar", [])),
                    "support_mapping_qualities": old.get("support_mapping_quality", old.get("support_mapq", [])),
                    "truth_seq": truth,
                    "labels": labels,
                    "features": _old_features_to_safe(old, target),
                    "case_type": f"old_hg002_{data.get('coverage', 'unknown')}",
                }
            )
        splits[split] = records
    return splits


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
                str(data.get("synthetic_profile", "standard")),
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
                str(data.get("synthetic_profile", "standard")),
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
                str(data.get("synthetic_profile", "standard")),
            ),
        }
    if data["kind"] == "real_bam":
        return _real_records(config)
    if data["kind"] == "old_jsonl":
        return _old_jsonl_records(config)
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
