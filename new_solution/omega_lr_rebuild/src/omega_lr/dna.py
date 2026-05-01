"""DNA helpers."""

from omega_lr.constants import BASE_TO_ID, PAD_BASE_ID


def normalize_base(base: str) -> str:
    base = base.upper()
    return base if base in BASE_TO_ID else "A"


def encode_sequence(seq: str) -> list[int]:
    return [BASE_TO_ID.get(normalize_base(base), PAD_BASE_ID) for base in seq]


def decode_sequence(ids: list[int]) -> str:
    inverse = {idx: base for base, idx in BASE_TO_ID.items()}
    return "".join(inverse.get(idx, "N") for idx in ids)


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]

