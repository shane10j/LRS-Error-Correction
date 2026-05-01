"""Tiny tandem-repeat heuristics."""


def tandem_repeat_mask(seq: str, max_unit: int = 3, min_repeats: int = 2) -> list[int]:
    mask = [0] * len(seq)
    for unit_size in range(1, max_unit + 1):
        for start in range(len(seq) - unit_size * min_repeats + 1):
            unit = seq[start : start + unit_size]
            span = unit * min_repeats
            if seq.startswith(span, start):
                end = start + len(span)
                while end + unit_size <= len(seq) and seq[end : end + unit_size] == unit:
                    end += unit_size
                for idx in range(start, end):
                    mask[idx] = 1
    return mask
