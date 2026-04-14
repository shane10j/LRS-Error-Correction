from __future__ import annotations

from typing import Iterable, List

import torch

from .vocab import ID_TO_EDIT
from .vocab import EDIT_TO_ID

_COPY_ID = EDIT_TO_ID["COPY"]
_CORE_HARD_EDIT_IDS = {
    EDIT_TO_ID["SUB_A"],
    EDIT_TO_ID["SUB_C"],
    EDIT_TO_ID["SUB_G"],
    EDIT_TO_ID["SUB_T"],
    EDIT_TO_ID["DEL"],
}


def apply_edit_ops(noisy_bases: str, edit_ids: Iterable[int] | Iterable[Iterable[int]], max_insertions_per_pos: int = 2) -> str:
    out: List[str] = []
    noisy = list(noisy_bases)
    edit_ids = list(edit_ids)
    if edit_ids and isinstance(edit_ids[0], int):
        edit_ids = [[int(token)] for token in edit_ids]

    for i, base in enumerate(noisy):
        if i >= len(edit_ids):
            out.append(base)
            continue
        slots = list(edit_ids[i])
        if not slots:
            out.append(base)
            continue
        insertion_slots = slots[:-1]
        core_slot = slots[-1]

        for token in insertion_slots[:max_insertions_per_pos]:
            op = ID_TO_EDIT.get(int(token), "PAD")
            if op.startswith("INS_"):
                out.append(op[-1])

        core_op = ID_TO_EDIT.get(int(core_slot), "COPY")
        if core_op == "COPY":
            out.append(noisy[i])
        elif core_op.startswith("SUB_"):
            out.append(core_op[-1])
        elif core_op == "DEL":
            continue
        else:
            out.append(noisy[i])
    return "".join(out)


def filter_low_confidence_hard_edits(
    edit_logits: torch.Tensor,
    min_hard_edit_confidence: float = 0.0,
    hard_edit_temperature: float = 1.0,
) -> torch.Tensor:
    if min_hard_edit_confidence <= 0.0 and abs(hard_edit_temperature - 1.0) < 1e-8:
        return edit_logits
    core_logits = edit_logits[:, :, -1, :]
    temperature = max(float(hard_edit_temperature), 1e-4)
    core_probs = torch.softmax(core_logits / temperature, dim=-1)
    pred_confidence, pred_ids = core_probs.max(dim=-1)
    hard_edit_mask = torch.zeros_like(pred_ids, dtype=torch.bool)
    for token_id in _CORE_HARD_EDIT_IDS:
        hard_edit_mask |= pred_ids == token_id
    low_conf_hard_edit_mask = hard_edit_mask & (pred_confidence < min_hard_edit_confidence)
    if not low_conf_hard_edit_mask.any():
        return edit_logits
    forced_copy = torch.full_like(core_logits, -1e4)
    forced_copy[..., _COPY_ID] = 1e4
    filtered_core_logits = torch.where(low_conf_hard_edit_mask.unsqueeze(-1), forced_copy, core_logits)
    filtered_edit_logits = edit_logits.clone()
    filtered_edit_logits[:, :, -1, :] = filtered_core_logits
    return filtered_edit_logits
