#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

from omega_longread.vocab import EDIT_TO_ID

BASES = ["A", "C", "G", "T"]
MAX_INSERTIONS = 2


def mutate(seq: str) -> tuple[str, list[list[int]]]:
    noisy = []
    labels = []
    for ch in seq:
        insertions = []
        for _ in range(MAX_INSERTIONS):
            if random.random() < 0.03:
                inserted = random.choice(BASES)
                noisy.append(inserted)
                insertions.append(EDIT_TO_ID[f"INS_{inserted}"])
            else:
                insertions.append(EDIT_TO_ID["PAD"])

        r = random.random()
        if r < 0.08:
            alt = random.choice([b for b in BASES if b != ch])
            noisy.append(alt)
            labels.append(insertions + [EDIT_TO_ID[f"SUB_{ch}"]])
        elif r < 0.12:
            labels.append(insertions + [EDIT_TO_ID["DEL"]])
        else:
            noisy.append(ch)
            labels.append(insertions + [EDIT_TO_ID["COPY"]])
    return "".join(noisy), labels[: len(seq)]


def make_example(i: int) -> dict:
    truth = "".join(random.choice(BASES) for _ in range(256))
    noisy, labels = mutate(truth)
    noisy = noisy[:256]
    labels = labels[: len(noisy)]
    target_run_lengths = []
    prev = None
    run = 0
    for ch in noisy:
        if ch == prev:
            run += 1
        else:
            run = 1
        prev = ch
        target_run_lengths.append(run)
    support = []
    for _ in range(4):
        s, _ = mutate(truth)
        support.append(s[: len(noisy)].ljust(len(noisy), "A"))
    support_qualities = [[30] * len(noisy) for _ in support]
    masks = [[1] * len(noisy) for _ in support]
    base_support = [[[0.25, 0.25, 0.25, 0.25] for _ in range(len(noisy))] for _ in support]
    return {
        "read_id": f"read_{i}",
        "target_bases": noisy,
        "target_qualities": [30] * len(noisy),
        "target_run_lengths": target_run_lengths,
        "support_bases": support,
        "support_match_mask": masks,
        "support_ins_mask": [[0] * len(noisy) for _ in support],
        "support_del_mask": [[0] * len(noisy) for _ in support],
        "support_qualities": support_qualities,
        "support_base_support": base_support,
        "target_sequence": truth,
        "edit_labels": labels,
        "preserve_mask": [0] * len(noisy),
        "uncertainty_labels": [0] * len(noisy),
    }


def write_jsonl(path: str, n: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps(make_example(i)) + "\n")


if __name__ == "__main__":
    random.seed(7)
    write_jsonl("data/train.jsonl", 64)
    write_jsonl("data/val.jsonl", 16)
