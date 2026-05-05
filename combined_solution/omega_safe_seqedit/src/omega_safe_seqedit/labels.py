"""Monotonic sequence-to-edit-sequence labels."""

from __future__ import annotations

from dataclasses import dataclass

from omega_safe_seqedit.constants import BASE_TO_ID, INS_TO_ID, MAIN_TO_ID


@dataclass
class EditLabels:
    main_type: list[int]
    sub_base: list[int]
    insert_before: list[int]
    terminal_insert: int


def global_alignment(source: str, target: str) -> list[tuple[str | None, str | None]]:
    """Return source/target aligned pairs using unit-cost edit distance."""
    n, m = len(source), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    move = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        move[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        move[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            subst = dp[i - 1][j - 1] + (source[i - 1] != target[j - 1])
            dele = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = min(subst, dele, ins)
            dp[i][j] = best
            move[i][j] = "M" if best == subst else "D" if best == dele else "I"
    pairs: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "M":
            pairs.append((source[i - 1], target[j - 1]))
            i -= 1
            j -= 1
        elif step == "D":
            pairs.append((source[i - 1], None))
            i -= 1
        else:
            pairs.append((None, target[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def make_edit_labels(target_seq: str, truth_seq: str) -> EditLabels:
    main = [MAIN_TO_ID["COPY"]] * len(target_seq)
    sub = [0] * len(target_seq)
    insert_before = [INS_TO_ID["NONE"]] * len(target_seq)
    terminal_insert = INS_TO_ID["NONE"]
    target_pos = 0
    for source_base, truth_base in global_alignment(target_seq, truth_seq):
        if source_base is None and truth_base is not None:
            ins_id = INS_TO_ID.get(truth_base, INS_TO_ID["NONE"])
            if target_pos < len(target_seq):
                insert_before[target_pos] = ins_id
            else:
                terminal_insert = ins_id
            continue
        if source_base is not None and truth_base is None:
            main[target_pos] = MAIN_TO_ID["DEL"]
            target_pos += 1
            continue
        if source_base is not None and truth_base is not None:
            if source_base == truth_base:
                main[target_pos] = MAIN_TO_ID["COPY"]
            else:
                main[target_pos] = MAIN_TO_ID["SUB"]
                sub[target_pos] = BASE_TO_ID.get(truth_base, 0)
            target_pos += 1
    return EditLabels(main, sub, insert_before, terminal_insert)


def apply_edit_labels(target_seq: str, labels: EditLabels | dict) -> str:
    if isinstance(labels, dict):
        labels = EditLabels(
            labels["main_type"],
            labels["sub_base"],
            labels["insert_before"],
            labels.get("terminal_insert", 0),
        )
    out: list[str] = []
    bases = "ACGT"
    for idx, base in enumerate(target_seq):
        ins_id = labels.insert_before[idx]
        if ins_id > 0:
            out.append(bases[ins_id - 1])
        action = labels.main_type[idx]
        if action == MAIN_TO_ID["COPY"]:
            out.append(base)
        elif action == MAIN_TO_ID["SUB"]:
            out.append(bases[labels.sub_base[idx]])
        elif action == MAIN_TO_ID["DEL"]:
            continue
    if labels.terminal_insert > 0:
        out.append(bases[labels.terminal_insert - 1])
    return "".join(out)


def label_events(labels: dict, target_seq: str) -> list[dict]:
    events = []
    for pos, ins_id in enumerate(labels["insert_before"]):
        if ins_id > 0:
            events.append({"pos": pos, "type": "INS", "base": "ACGT"[ins_id - 1]})
    for pos, action in enumerate(labels["main_type"]):
        if action == MAIN_TO_ID["SUB"]:
            events.append({"pos": pos, "type": "SUB", "base": "ACGT"[labels["sub_base"][pos]]})
        elif action == MAIN_TO_ID["DEL"]:
            events.append({"pos": pos, "type": "DEL", "base": target_seq[pos]})
    if labels.get("terminal_insert", 0) > 0:
        events.append({"pos": len(target_seq), "type": "INS", "base": "ACGT"[labels["terminal_insert"] - 1]})
    return events
