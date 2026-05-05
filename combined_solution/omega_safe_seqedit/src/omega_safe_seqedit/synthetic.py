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
) -> list[dict]:
    rng = random.Random(seed)
    cases = ["copy"]
    cases += [f"sub_{base}" for base in BASES]
    cases += [f"ins_{base}" for base in BASES]
    cases += ["del"]
    if include_homopolymer_cases:
        cases += ["homopolymer_ins", "homopolymer_del"]
    if include_neighbor_cases:
        cases += ["neighbor_sub_ins", "neighbor_del_sub"]
    records = []
    for idx in range(count):
        case = cases[idx % len(cases)]
        truth = _rand_seq(rng, read_length)
        pos = rng.randint(5, read_length - 6)
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
        support = _noisy_support(rng, truth, support_depth, support_noise)
        records.append(_example(f"{split}_{case}_{idx}", split, target, truth, support, case))
    return records
