from __future__ import annotations

from typing import Iterable, List

from .vocab import ID_TO_EDIT


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
