"""Support pileup features and rule labels."""

from __future__ import annotations

import math

from omega_safe_seqedit.constants import BASES, BASE_TO_ID, INS_TO_ID, RULE_TO_ID
from omega_safe_seqedit.dna import homopolymer_lengths, tandem_repeat_flags
from omega_safe_seqedit.labels import global_alignment


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counts:
        if count:
            p = count / total
            ent -= p * math.log(p)
    return ent / math.log(max(2, len(counts)))


def pileup_features(target_seq: str, support_seqs: list[str]) -> dict:
    length = len(target_seq)
    base_counts = [[0, 0, 0, 0] for _ in range(length)]
    del_count = [0] * length
    ins_count = [0] * length
    ins_base_counts = [[0, 0, 0, 0] for _ in range(length)]
    depth = [0] * length

    for support in support_seqs:
        target_pos = 0
        for target_base, support_base in global_alignment(target_seq, support):
            if target_base is None and support_base in BASE_TO_ID:
                anchor = min(target_pos, max(0, length - 1))
                if length:
                    ins_count[anchor] += 1
                    ins_base_counts[anchor][BASE_TO_ID[support_base]] += 1
                continue
            if target_base is not None and target_pos < length:
                depth[target_pos] += 1
                if support_base is None:
                    del_count[target_pos] += 1
                elif support_base in BASE_TO_ID:
                    base_counts[target_pos][BASE_TO_ID[support_base]] += 1
                target_pos += 1

    agreement = []
    entropy = []
    margin = []
    for counts, dels, dep in zip(base_counts, del_count, depth):
        full = counts + [dels]
        sorted_counts = sorted(full, reverse=True)
        agreement.append((sorted_counts[0] / max(dep, 1)) if dep else 0.0)
        margin.append((sorted_counts[0] - sorted_counts[1]) if len(sorted_counts) > 1 else sorted_counts[0])
        entropy.append(_entropy(full))

    support_rule_type = []
    support_rule_sub_base = []
    support_rule_ins_base = []
    for pos, target_base in enumerate(target_seq):
        rule = RULE_TO_ID["COPY"]
        sub_base = 0
        ins_base = 0
        dep = max(depth[pos], 1)
        best_base_id = max(range(4), key=lambda idx: base_counts[pos][idx])
        best_base = BASES[best_base_id]
        best_base_fraction = base_counts[pos][best_base_id] / dep
        best_ins_id = max(range(4), key=lambda idx: ins_base_counts[pos][idx])
        ins_fraction = ins_count[pos] / dep
        del_fraction = del_count[pos] / dep
        if best_base != target_base and best_base_fraction >= 0.60:
            rule = RULE_TO_ID["SUB"]
            sub_base = best_base_id
        if ins_fraction >= 0.60:
            rule = RULE_TO_ID["INS"]
            ins_base = best_ins_id
        if del_fraction >= 0.70:
            rule = RULE_TO_ID["DEL"]
        support_rule_type.append(rule)
        support_rule_sub_base.append(sub_base)
        support_rule_ins_base.append(ins_base)

    homopolymer = homopolymer_lengths(target_seq)
    tandem = tandem_repeat_flags(target_seq)
    boundary = [1 if pos in {0, length - 1} else 0 for pos in range(length)]
    rule_hard = [1 if x != RULE_TO_ID["COPY"] else 0 for x in support_rule_type]
    neighbor = []
    for pos in range(length):
        near = any(rule_hard[j] for j in range(max(0, pos - 1), min(length, pos + 2)) if j != pos)
        neighbor.append(1 if near else 0)

    return {
        "support_base_counts": base_counts,
        "support_del_count": del_count,
        "support_ins_count": ins_count,
        "support_ins_base_counts": ins_base_counts,
        "support_depth": depth,
        "support_agreement": agreement,
        "support_entropy": entropy,
        "support_margin": margin,
        "homopolymer_run_length": homopolymer,
        "tandem_repeat_flag": tandem,
        "boundary_flag": boundary,
        "neighbor_rule_flag": neighbor,
        "support_rule_type": support_rule_type,
        "support_rule_sub_base": support_rule_sub_base,
        "support_rule_ins_base": support_rule_ins_base,
    }


def feature_matrix(example: dict) -> list[list[float]]:
    f = example["features"]
    rows = []
    for pos in range(len(example["target_seq"])):
        depth = max(float(f["support_depth"][pos]), 1.0)
        base_fracs = [x / depth for x in f["support_base_counts"][pos]]
        ins_fracs = [x / depth for x in f["support_ins_base_counts"][pos]]
        rows.append(
            base_fracs
            + [
                f["support_del_count"][pos] / depth,
                f["support_ins_count"][pos] / depth,
                f["support_agreement"][pos],
                f["support_entropy"][pos],
                f["support_margin"][pos] / depth,
                min(f["homopolymer_run_length"][pos], 10) / 10.0,
                float(f["tandem_repeat_flag"][pos]),
                float(f["boundary_flag"][pos]),
                float(f["neighbor_rule_flag"][pos]),
            ]
            + ins_fracs
        )
    return rows


def rule_feature_matrix(example: dict) -> list[list[float]]:
    f = example["features"]
    rows = []
    for pos, base in enumerate(example["target_seq"]):
        depth = max(float(f["support_depth"][pos]), 1.0)
        best_base_id = max(range(4), key=lambda idx: f["support_base_counts"][pos][idx])
        best_ins_id = max(range(4), key=lambda idx: f["support_ins_base_counts"][pos][idx])
        rows.append(
            [
                1.0 if BASES[best_base_id] != base else 0.0,
                f["support_base_counts"][pos][best_base_id] / depth,
                f["support_ins_count"][pos] / depth,
                f["support_del_count"][pos] / depth,
                f["support_entropy"][pos],
                f["support_margin"][pos] / depth,
                float(best_base_id) / 3.0,
                float(best_ins_id) / 3.0,
                float(f["neighbor_rule_flag"][pos]),
                float(f["boundary_flag"][pos]),
                1.0 if f["homopolymer_run_length"][pos] >= 4 else 0.0,
            ]
        )
    return rows
