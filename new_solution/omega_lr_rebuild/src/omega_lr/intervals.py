"""Interval helpers."""


def overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def clipped_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> tuple[int, int] | None:
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    return (start, end) if start < end else None

