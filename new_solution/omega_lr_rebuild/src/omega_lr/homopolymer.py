"""Homopolymer utilities."""


def run_lengths(seq: str) -> list[int]:
    if not seq:
        return []
    lengths = [1] * len(seq)
    start = 0
    while start < len(seq):
        end = start + 1
        while end < len(seq) and seq[end] == seq[start]:
            end += 1
        value = end - start
        for idx in range(start, end):
            lengths[idx] = value
        start = end
    return lengths


def homopolymer_mask(seq: str, threshold: int = 4) -> list[int]:
    return [int(length >= threshold) for length in run_lengths(seq)]

