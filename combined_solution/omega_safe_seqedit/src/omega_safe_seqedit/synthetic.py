"""Synthetic data for fast edit-learning and safety checks."""

from __future__ import annotations

import random

from omega_safe_seqedit.constants import BASES
from omega_safe_seqedit.features import pileup_features
from omega_safe_seqedit.labels import make_edit_labels


def _rand_seq(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(BASES) for _ in range(length))


def _other_base(rng: random.Random, base: str) -> str:
    choices = [b for b in BASES if b != base]
    return rng.choice(choices)


def _noisy_support(rng: random.Random, truth: str, depth: int, noise: float) -> list[str]:
    reads = []
    for _ in range(depth):
        chars = []
        for base in truth:
            r = rng.random()
            if r < noise * 0.33:
                continue
            if r < noise * 0.66:
                chars.append(_other_base(rng, base))
            else:
                chars.append(base)
            if rng.random() < noise * 0.15:
                chars.append(rng.choice(BASES))
        reads.append("".join(chars))
    return reads


def _mutate_base(seq: str, pos: int, base: str) -> str:
    return seq[:pos] + base + seq[pos + 1 :]


def _delete_base(seq: str, pos: int) -> str:
    return seq[:pos] + seq[pos + 1 :]


def _insert_base(seq: str, pos: int, base: str) -> str:
    return seq[:pos] + base + seq[pos:]


def _false_support(
    rng: random.Random,
    truth: str,
    depth: int,
    noise: float,
    event_type: str,
    pos: int,
) -> list[str]:
    """Create support where a shallow majority suggests a wrong edit."""
    reads = _noisy_support(rng, truth, depth, noise)
    wrong_count = max(2, int(round(depth * 0.60)))
    for idx in range(min(wrong_count, len(reads))):
        if event_type == "SUB":
            if reads[idx]:
                reads[idx] = _mutate_base(
                    reads[idx],
                    min(pos, len(reads[idx]) - 1),
                    _other_base(rng, truth[min(pos, len(truth) - 1)]),
                )
        elif event_type == "DEL" and len(reads[idx]) > pos:
            reads[idx] = _delete_base(reads[idx], pos)
        elif event_type == "INS":
            reads[idx] = _insert_base(reads[idx], min(pos, len(reads[idx])), rng.choice(BASES))
    return reads


def _case_list(profile: str, include_neighbor_cases: bool, include_homopolymer_cases: bool) -> list[str]:
    base_cases = ["copy"] + [f"sub_{base}" for base in BASES] + [f"ins_{base}" for base in BASES] + ["del"]
    if include_homopolymer_cases:
        base_cases += ["homopolymer_ins", "homopolymer_del"]
    if include_neighbor_cases:
        base_cases += ["neighbor_sub_ins", "neighbor_del_sub"]
    if profile == "noisy_curated":
        return [
            "copy",
            "copy_false_sub",
            "copy_false_del",
            "copy_false_ins",
            "sub_A",
            "sub_C",
            "sub_G",
            "sub_T",
            "ins_A",
            "ins_C",
            "ins_G",
            "ins_T",
            "del",
            "homopolymer_del",
            "neighbor_sub_ins",
            "neighbor_del_sub",
            "boundary_ins_A",
            "boundary_ins_G",
            "copy_long",
            "copy_false_del_homopolymer",
            "support_rule_wrong_neighbor",
            "support_rule_wrong_hpoly_del",
            "noisy_true_sub",
            "noisy_true_ins",
        ]
    if profile == "noisy_large":
        return base_cases + [
            "copy_false_sub",
            "copy_false_del",
            "copy_false_ins",
            "copy_false_del_homopolymer",
            "support_rule_wrong_neighbor",
            "support_rule_wrong_hpoly_del",
            "boundary_ins_A",
            "boundary_ins_G",
            "noisy_true_sub",
            "noisy_true_ins",
            "noisy_true_del",
        ]
    if profile == "false_del_regression":
        return [
            "copy_false_del",
            "copy_false_del_homopolymer",
            "support_rule_wrong_neighbor",
            "support_rule_wrong_hpoly_del",
            "neighbor_del_sub",
            "homopolymer_del",
            "del",
            "copy",
        ]
    return base_cases


def _example(example_id: str, sample: str, target: str, truth: str, support: list[str], kind: str) -> dict:
    labels = make_edit_labels(target, truth)
    features = pileup_features(target, support)
    return {
        "example_id": example_id,
        "sample_id": sample,
        "contig": "synthetic",
        "window_start": 0,
        "window_end": len(target),
        "target_read_id": example_id,
        "target_seq": target,
        "support_read_ids": [f"{example_id}_support_{idx}" for idx in range(len(support))],
        "support_aligned_seqs": support,
        "truth_seq": truth,
        "labels": {
            "main_type": labels.main_type,
            "sub_base": labels.sub_base,
            "insert_before": labels.insert_before,
            "terminal_insert": labels.terminal_insert,
        },
        "features": features,
        "case_type": kind,
    }


def make_synthetic_split(
    split: str,
    count: int,
    seed: int,
    read_length: int,
    support_depth: int,
    support_noise: float,
    include_neighbor_cases: bool = True,
    include_homopolymer_cases: bool = True,
    profile: str = "standard",
) -> list[dict]:
    rng = random.Random(seed)
    cases = _case_list(profile, include_neighbor_cases, include_homopolymer_cases)
    records = []
    for idx in range(count):
        case = cases[idx % len(cases)]
        truth = _rand_seq(rng, read_length)
        if "long" in case:
            truth = _rand_seq(rng, read_length * 2)
        pos = rng.randint(5, len(truth) - 6)
        target = truth
        if case.startswith("sub_"):
            wanted = case[-1]
            truth = truth[:pos] + wanted + truth[pos + 1 :]
            target = truth[:pos] + _other_base(rng, wanted) + truth[pos + 1 :]
        elif case.startswith("ins_"):
            wanted = case[-1]
            truth = truth[:pos] + wanted + truth[pos:]
            target = truth[:pos] + truth[pos + 1 :]
        elif case == "del":
            extra = rng.choice(BASES)
            target = truth[:pos] + extra + truth[pos:]
        elif case == "homopolymer_ins":
            truth = truth[:pos] + "AAAAA" + truth[pos + 5 :]
            target = truth[: pos + 2] + truth[pos + 3 :]
        elif case == "homopolymer_del":
            truth = truth[:pos] + "CCCC" + truth[pos + 4 :]
            target = truth[: pos + 2] + "G" + truth[pos + 2 :]
        elif case == "neighbor_sub_ins":
            truth = truth[:pos] + "T" + truth[pos + 1 :]
            target = truth[:pos] + _other_base(rng, "T") + truth[pos + 1 :]
            truth = truth[: pos + 1] + "A" + truth[pos + 1 :]
        elif case == "neighbor_del_sub":
            truth = truth[:pos] + "G" + truth[pos + 1 :]
            target = truth[:pos] + "T" + _other_base(rng, "G") + truth[pos + 1 :]
        elif case == "boundary_ins_A":
            truth = "A" + truth
            target = truth[1:]
            pos = 0
        elif case == "boundary_ins_G":
            truth = "G" + truth
            target = truth[1:]
            pos = 0
        elif case == "noisy_true_sub":
            wanted = rng.choice(BASES)
            truth = truth[:pos] + wanted + truth[pos + 1 :]
            target = truth[:pos] + _other_base(rng, wanted) + truth[pos + 1 :]
        elif case == "noisy_true_ins":
            wanted = rng.choice(BASES)
            truth = truth[:pos] + wanted + truth[pos:]
            target = truth[:pos] + truth[pos + 1 :]
        elif case == "noisy_true_del":
            target = truth[:pos] + rng.choice(BASES) + truth[pos:]
        elif case == "copy_false_del_homopolymer":
            truth = truth[:pos] + "AAAAAA" + truth[pos + 6 :]
            target = truth
            pos = pos + 2
        elif case == "support_rule_wrong_neighbor":
            # Truth contains one real neighboring event; support also suggests an extra false nearby edit.
            truth = truth[:pos] + "T" + truth[pos + 1 :]
            target = truth[:pos] + _other_base(rng, "T") + truth[pos + 1 :]
        elif case == "support_rule_wrong_hpoly_del":
            truth = truth[:pos] + "CCCCC" + truth[pos + 5 :]
            target = truth
            pos = pos + 2
        if case == "copy_false_sub":
            support = _false_support(rng, truth, support_depth, support_noise, "SUB", pos)
        elif case == "copy_false_del":
            support = _false_support(rng, truth, support_depth, support_noise, "DEL", pos)
        elif case == "copy_false_ins":
            support = _false_support(rng, truth, support_depth, support_noise, "INS", pos)
        elif case == "copy_false_del_homopolymer":
            support = _false_support(rng, truth, support_depth, support_noise, "DEL", pos)
        elif case == "support_rule_wrong_neighbor":
            support = _false_support(rng, truth, support_depth, support_noise, "DEL", min(pos + 1, len(truth) - 1))
        elif case == "support_rule_wrong_hpoly_del":
            support = _false_support(rng, truth, support_depth, support_noise, "DEL", pos)
        else:
            support = _noisy_support(rng, truth, support_depth, support_noise)
        records.append(_example(f"{split}_{case}_{idx}", split, target, truth, support, case))
    return records
