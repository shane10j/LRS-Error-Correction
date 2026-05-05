"""Small DNA helpers."""

from __future__ import annotations

from omega_safe_seqedit.constants import BASES


def clean_dna(seq: str) -> str:
    return "".join(base for base in seq.upper() if base in BASES)


def homopolymer_lengths(seq: str) -> list[int]:
    if not seq:
        return []
    out = [1] * len(seq)
    start = 0
    for idx in range(1, len(seq) + 1):
        if idx == len(seq) or seq[idx] != seq[start]:
            run = idx - start
            for pos in range(start, idx):
                out[pos] = run
            start = idx
    return out


def tandem_repeat_flags(seq: str, max_motif: int = 4) -> list[int]:
    flags = [0] * len(seq)
    for motif_len in range(2, max_motif + 1):
        for start in range(0, max(0, len(seq) - motif_len * 2 + 1)):
            motif = seq[start : start + motif_len]
            if seq[start + motif_len : start + motif_len * 2] == motif:
                for pos in range(start, min(len(seq), start + motif_len * 2)):
                    flags[pos] = 1
    return flags


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def identity(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return 1.0 - edit_distance(a, b) / denom
