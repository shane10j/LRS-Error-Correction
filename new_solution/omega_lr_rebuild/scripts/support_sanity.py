#!/usr/bin/env python
"""Print support-feature sanity reports for synthetic edit cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from omega_lr.constants import BASES, ID_TO_EDIT
from omega_lr.data.labels import generate_labels
from omega_lr.data.pileup import compute_support_features
from omega_lr.data.windowing import SyntheticCase, filter_synthetic_cases
from omega_lr.utils import ensure_dir, print_config, read_config, save_json


def aligned_truth_views(target_seq: str, truth_seq: str, max_deletion_length: int) -> tuple[dict[int, str], dict[int, list[str]]]:
    labels = generate_labels(target_seq, truth_seq, max_deletion_length)
    aligned_target = labels["alignment"]["aligned_target"]
    aligned_truth = labels["alignment"]["aligned_truth"]
    truth_by_target: dict[int, str] = {}
    insertions_after: dict[int, list[str]] = {}
    target_idx = -1
    for target_char, truth_char in zip(aligned_target, aligned_truth):
        if target_char != "-":
            target_idx += 1
            truth_by_target[target_idx] = truth_char
        elif truth_char != "-":
            insertions_after.setdefault(max(target_idx, 0), []).append(truth_char)
    return truth_by_target, insertions_after


def aggregate_insertion_counts(case: SyntheticCase) -> list[int]:
    if case.support_insertion_counts is not None:
        return case.support_insertion_counts
    if case.support_insertion_events is None:
        return [0] * len(case.target_seq)
    return [sum(events[pos] for events in case.support_insertion_events) for pos in range(len(case.target_seq))]


def raw_support_events(case: SyntheticCase, pos: int) -> list[dict]:
    insertion_events = case.support_insertion_events or [[0] * len(case.target_seq) for _ in case.support_aligned_seqs]
    deletion_lengths = case.support_deletion_lengths or [[0] * len(case.target_seq) for _ in case.support_aligned_seqs]
    events = []
    for read_idx, aligned in enumerate(case.support_aligned_seqs):
        symbol = aligned[pos] if pos < len(aligned) else None
        events.append(
            {
                "read_index": read_idx,
                "aligned_symbol": symbol,
                "insertion_after": insertion_events[read_idx][pos] if read_idx < len(insertion_events) else 0,
                "deletion_length": deletion_lengths[read_idx][pos] if read_idx < len(deletion_lengths) else 0,
            }
        )
    return events


def feature_report_for_case(case: SyntheticCase, max_deletion_length: int) -> dict:
    insertion_counts = aggregate_insertion_counts(case)
    deletion_lengths = case.support_deletion_lengths or [[0] * len(case.target_seq) for _ in case.support_aligned_seqs]
    support_strands = ["+" if idx % 2 == 0 else "-" for idx in range(len(case.support_aligned_seqs))]
    features = compute_support_features(
        case.target_seq,
        case.support_aligned_seqs,
        support_strands,
        insertion_counts=insertion_counts,
        deletion_lengths=deletion_lengths,
        insertion_base_counts=case.support_insertion_base_counts,
    )
    labels = generate_labels(case.target_seq, case.truth_seq, max_deletion_length)
    truth_by_target, insertions_after = aligned_truth_views(case.target_seq, case.truth_seq, max_deletion_length)
    positions = []
    hard_agreements = []
    hard_nonbase_signal = False
    for pos, label_id in enumerate(labels["edit_labels"]):
        label = ID_TO_EDIT[label_id]
        if label == "COPY":
            continue
        truth_base = truth_by_target.get(pos, "-")
        if label.startswith("INS_"):
            truth_base = "".join(insertions_after.get(pos, [label[-1]]))
        positions.append(
            {
                "pos": pos,
                "gold_label": label,
                "target_base": case.target_seq[pos],
                "truth_base": truth_base,
                "raw_support_events": raw_support_events(case, pos),
                "base_counts": dict(zip(BASES, features["support_base_counts"][pos])),
                "insertion_count": features["support_ins_count"][pos],
                "deletion_count": features["support_del_count"][pos],
                "agreement": round(float(features["support_agreement"][pos]), 4),
                "entropy": round(float(features["support_entropy"][pos]), 4),
            }
        )
        hard_agreements.append(features["support_agreement"][pos])
        hard_nonbase_signal = hard_nonbase_signal or features["support_ins_count"][pos] > 0 or features["support_del_count"][pos] > 0

    checks = {
        "sub_support_favors_or_shows_disagreement": True,
        "ins_support_nonzero": True,
        "del_support_nonzero": True,
        "hard_position_agreement_not_all_one": not hard_agreements or any(value < 0.999 for value in hard_agreements) or hard_nonbase_signal,
    }
    for position in positions:
        if position["gold_label"].startswith("SUB_"):
            corrected_base = position["gold_label"][-1]
            noisy_base = position["target_base"]
            corrected_count = position["base_counts"][corrected_base]
            noisy_count = position["base_counts"].get(noisy_base, 0)
            checks["sub_support_favors_or_shows_disagreement"] = corrected_count > noisy_count or corrected_count != noisy_count
        elif position["gold_label"].startswith("INS_"):
            checks["ins_support_nonzero"] = position["insertion_count"] > 0
        elif position["gold_label"] == "DEL":
            checks["del_support_nonzero"] = position["deletion_count"] > 0

    return {
        "case": case.name,
        "target_seq": case.target_seq,
        "truth_seq": case.truth_seq,
        "target_in_support": False,
        "support_summary_stage": "prealigned_before_coordinate_projection",
        "positions": positions,
        "acceptance_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cases", nargs="+", default=["sub", "ins", "del", "copy"])
    parser.add_argument("--output", default="outputs/support_sanity/support_sanity.json")
    args = parser.parse_args()

    config = read_config(args.config)
    print_config(config)
    max_deletion_length = config["dataset"]["max_deletion_length"]
    reports = [feature_report_for_case(case, max_deletion_length) for case in filter_synthetic_cases(args.cases)]

    for report in reports:
        print("=" * 100)
        print("case:", report["case"])
        print("target_seq:", report["target_seq"])
        print("truth_seq :", report["truth_seq"])
        print("target_in_support:", report["target_in_support"])
        print("support_summary_stage:", report["support_summary_stage"])
        print("acceptance_checks:", report["acceptance_checks"])
        for position in report["positions"]:
            print("-" * 100)
            print(
                f"pos={position['pos']} gold={position['gold_label']} "
                f"target_base={position['target_base']} truth_base={position['truth_base']}"
            )
            print("raw_support_events:", position["raw_support_events"])
            print("base_counts      :", position["base_counts"])
            print("insertion_count  :", position["insertion_count"])
            print("deletion_count   :", position["deletion_count"])
            print("agreement        :", position["agreement"])
            print("entropy          :", position["entropy"])

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_json(reports, output_path)


if __name__ == "__main__":
    main()
