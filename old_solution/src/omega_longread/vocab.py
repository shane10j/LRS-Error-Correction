from __future__ import annotations

BASES = ["A", "C", "G", "T", "N"]
BASE_TO_ID = {b: i for i, b in enumerate(BASES)}
ID_TO_BASE = {i: b for b, i in BASE_TO_ID.items()}
PAD_BASE_ID = BASE_TO_ID["N"]

EDIT_TOKENS = [
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
    "PAD",
]
EDIT_TO_ID = {tok: i for i, tok in enumerate(EDIT_TOKENS)}
ID_TO_EDIT = {i: tok for tok, i in EDIT_TO_ID.items()}
PAD_EDIT_ID = EDIT_TO_ID["PAD"]

UNCERTAINTY_CLASSES = ["error", "ambiguous", "variant_supported"]
UNCERTAINTY_TO_ID = {name: i for i, name in enumerate(UNCERTAINTY_CLASSES)}
