"""Shared constants."""

from __future__ import annotations


BASES = ["A", "C", "G", "T"]
BASE_TO_ID = {base: idx for idx, base in enumerate(BASES)}
ID_TO_BASE = {idx: base for base, idx in BASE_TO_ID.items()}
PAD_BASE_ID = len(BASES)
UNKNOWN_BASE = "N"

EDIT_TYPE_LABELS = ["COPY", "SUB", "DEL", "INS"]
EDIT_TYPE_TO_ID = {label: idx for idx, label in enumerate(EDIT_TYPE_LABELS)}
ID_TO_EDIT_TYPE = {idx: label for label, idx in EDIT_TYPE_TO_ID.items()}

EDIT_LABELS = [
    "COPY",
    "SUB_A",
    "SUB_C",
    "SUB_G",
    "SUB_T",
    "DEL",
    "INS_A",
    "INS_C",
    "INS_G",
    "INS_T",
]
EDIT_TO_ID = {label: idx for idx, label in enumerate(EDIT_LABELS)}
ID_TO_EDIT = {idx: label for label, idx in EDIT_TO_ID.items()}

HARD_EDIT_LABELS = {"DEL", "SUB_A", "SUB_C", "SUB_G", "SUB_T", "INS_A", "INS_C", "INS_G", "INS_T"}
DEFAULT_GAP_BUCKETS = [1, 2, 3, 4, 5]


def edit_label_to_type_and_payload(label: str) -> tuple[str, int | None]:
    if label == "COPY":
        return "COPY", None
    if label == "DEL":
        return "DEL", None
    if label.startswith("SUB_"):
        return "SUB", BASE_TO_ID[label[-1]]
    if label.startswith("INS_"):
        return "INS", BASE_TO_ID[label[-1]]
    raise KeyError(f"Unknown edit label: {label}")


def compose_edit_label(edit_type: str, payload_base_id: int | None = None) -> str:
    if edit_type == "COPY":
        return "COPY"
    if edit_type == "DEL":
        return "DEL"
    if edit_type == "SUB":
        if payload_base_id is None:
            raise ValueError("SUB requires a payload base id.")
        return f"SUB_{ID_TO_BASE[int(payload_base_id)]}"
    if edit_type == "INS":
        if payload_base_id is None:
            raise ValueError("INS requires a payload base id.")
        return f"INS_{ID_TO_BASE[int(payload_base_id)]}"
    raise KeyError(f"Unknown edit type: {edit_type}")
