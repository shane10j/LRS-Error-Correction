from __future__ import annotations

from typing import Iterable, List

import torch

from .vocab import ID_TO_EDIT
from .vocab import EDIT_TO_ID
from .vocab import PAD_EDIT_ID

_COPY_ID = EDIT_TO_ID["COPY"]
_CORE_HARD_EDIT_IDS = {
    EDIT_TO_ID["SUB_A"],
    EDIT_TO_ID["SUB_C"],
    EDIT_TO_ID["SUB_G"],
    EDIT_TO_ID["SUB_T"],
    EDIT_TO_ID["DEL"],
}
_SUB_IDS = {
    EDIT_TO_ID["SUB_A"],
    EDIT_TO_ID["SUB_C"],
    EDIT_TO_ID["SUB_G"],
    EDIT_TO_ID["SUB_T"],
}
_INS_IDS = {
    EDIT_TO_ID["INS_A"],
    EDIT_TO_ID["INS_C"],
    EDIT_TO_ID["INS_G"],
    EDIT_TO_ID["INS_T"],
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


def apply_inference_constraints(
    edit_logits: torch.Tensor,
    *,
    trust_gate: torch.Tensor | None = None,
    deletion_candidate_logits: torch.Tensor | None = None,
    deletion_length_logits: torch.Tensor | None = None,
    local_support_agreement: torch.Tensor | None = None,
    deletion_support_fraction: torch.Tensor | None = None,
    min_sub_confidence: float = 0.8,
    min_del_confidence: float = 0.92,
    min_ins_confidence: float = 0.75,
    deletion_candidate_threshold: float = 0.8,
    deletion_commit_trust_threshold: float = 0.8,
    hard_edit_temperature: float = 1.0,
    use_deletion_consistency_check: bool = True,
) -> torch.Tensor:
    filtered = edit_logits.clone()
    temperature = max(float(hard_edit_temperature), 1e-4)

    if filtered.shape[2] > 1:
        ins_logits = filtered[:, :, :-1, :]
        ins_probs = torch.softmax(ins_logits / temperature, dim=-1)
        ins_confidence, ins_pred = ins_probs.max(dim=-1)
        low_conf_insertions = torch.zeros_like(ins_pred, dtype=torch.bool)
        for token_id in _INS_IDS:
            low_conf_insertions |= (ins_pred == token_id) & (ins_confidence < min_ins_confidence)
        if low_conf_insertions.any():
            forced_pad = torch.full_like(ins_logits, -1e4)
            forced_pad[..., PAD_EDIT_ID] = 1e4
            filtered[:, :, :-1, :] = torch.where(low_conf_insertions.unsqueeze(-1), forced_pad, ins_logits)

    core_logits = filtered[:, :, -1, :]
    core_probs = torch.softmax(core_logits / temperature, dim=-1)
    core_confidence, core_pred = core_probs.max(dim=-1)
    force_copy_mask = torch.zeros_like(core_pred, dtype=torch.bool)
    sub_mask = torch.zeros_like(core_pred, dtype=torch.bool)
    for token_id in _SUB_IDS:
        sub_mask |= core_pred == token_id
    del_mask = core_pred == EDIT_TO_ID["DEL"]
    force_copy_mask |= sub_mask & (core_confidence < min_sub_confidence)
    force_copy_mask |= del_mask & (core_confidence < min_del_confidence)

    if del_mask.any():
        if deletion_candidate_logits is not None:
            del_candidate_prob = torch.sigmoid(deletion_candidate_logits)
            force_copy_mask |= del_mask & (del_candidate_prob < deletion_candidate_threshold)
        if deletion_length_logits is not None:
            del_length_pred = deletion_length_logits.argmax(dim=-1)
            force_copy_mask |= del_mask & (del_length_pred <= 0)
        if trust_gate is not None:
            force_copy_mask |= del_mask & (trust_gate < deletion_commit_trust_threshold)
        if use_deletion_consistency_check:
            if local_support_agreement is not None and deletion_support_fraction is not None:
                force_copy_mask |= del_mask & (deletion_support_fraction + 1e-6 < local_support_agreement)
            elif deletion_support_fraction is not None:
                force_copy_mask |= del_mask & (deletion_support_fraction < 0.5)

    if force_copy_mask.any():
        forced_copy = torch.full_like(core_logits, -1e4)
        forced_copy[..., _COPY_ID] = 1e4
        filtered[:, :, -1, :] = torch.where(force_copy_mask.unsqueeze(-1), forced_copy, core_logits)
    return filtered
