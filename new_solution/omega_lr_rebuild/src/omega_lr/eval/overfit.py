"""Strict correction-quality scoring for tiny overfit gates."""

from __future__ import annotations

from omega_lr.constants import ID_TO_EDIT


def _gold_labels(row: dict) -> list[str]:
    return [ID_TO_EDIT[int(label)] for label in row["gold_edit_labels"]]


def correction_quality(rows: list[dict]) -> dict:
    """Score decoded rows by exact correction quality, not loss.

    This intentionally favors checkpoints with exact decoded sequence matches
    and zero false hard edits. It is used only for debug/overfit selection.
    """
    exact_matches = 0
    false_hard_edits = 0
    label_mismatches = 0
    total_positions = 0
    hard_gold_positions = 0
    hard_gold_correct = 0
    for row in rows:
        exact_matches += int(row["prediction"] == row["truth_seq"])
        gold_labels = _gold_labels(row)
        predicted_labels = row["predicted_labels"]
        for predicted, gold in zip(predicted_labels, gold_labels):
            total_positions += 1
            label_mismatches += int(predicted != gold)
            false_hard_edits += int(gold == "COPY" and predicted != "COPY")
            if gold != "COPY":
                hard_gold_positions += 1
                hard_gold_correct += int(predicted == gold)
    num_examples = len(rows)
    exact_match_rate = exact_matches / max(num_examples, 1)
    false_hard_edit_rate = false_hard_edits / max(total_positions, 1)
    label_mismatch_rate = label_mismatches / max(total_positions, 1)
    hard_gold_accuracy = hard_gold_correct / max(hard_gold_positions, 1)
    selection_score = (
        10.0 * exact_match_rate
        + hard_gold_accuracy
        - 5.0 * false_hard_edit_rate
        - label_mismatch_rate
    )
    return {
        "num_examples": num_examples,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_match_rate,
        "false_hard_edit_count": false_hard_edits,
        "false_hard_edit_rate": false_hard_edit_rate,
        "label_mismatch_count": label_mismatches,
        "label_mismatch_rate": label_mismatch_rate,
        "hard_gold_positions": hard_gold_positions,
        "hard_gold_correct": hard_gold_correct,
        "hard_gold_accuracy": hard_gold_accuracy,
        "exact_zero_false_edits": exact_matches == num_examples and false_hard_edits == 0,
        "selection_score": selection_score,
    }


def should_use_overfit_selection(config: dict) -> bool:
    decode_cfg = config.get("decode", {})
    return bool(
        decode_cfg.get("mode") == "debug"
        and config.get("dataset", {}).get("shared_examples_across_splits", False)
    )
