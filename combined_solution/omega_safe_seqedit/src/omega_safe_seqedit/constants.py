"""Shared constants for aligned edit-script correction."""

BASES = "ACGT"
BASE_TO_ID = {base: idx for idx, base in enumerate(BASES)}
ID_TO_BASE = {idx: base for base, idx in BASE_TO_ID.items()}
PAD_BASE_ID = 4
EOS_BASE_ID = 5

MAIN_TYPES = ["COPY", "SUB", "DEL"]
MAIN_TO_ID = {label: idx for idx, label in enumerate(MAIN_TYPES)}
ID_TO_MAIN = {idx: label for label, idx in MAIN_TO_ID.items()}

INS_LABELS = ["NONE", "A", "C", "G", "T"]
INS_TO_ID = {label: idx for idx, label in enumerate(INS_LABELS)}
ID_TO_INS = {idx: label for label, idx in INS_TO_ID.items()}

SUPPORT_RULE_TYPES = ["COPY", "SUB", "DEL", "INS"]
RULE_TO_ID = {label: idx for idx, label in enumerate(SUPPORT_RULE_TYPES)}
ID_TO_RULE = {idx: label for label, idx in RULE_TO_ID.items()}
