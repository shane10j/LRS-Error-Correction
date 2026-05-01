"""Composite usable score."""


def usable_score(summary: dict, identity_weight: float = 1.0, overcorrection_weight: float = 0.5, fp_weight: float = 0.5) -> float:
    identity = summary["sequence"]["identity"]
    overcorrection = summary["safety"]["overcorrection_rate"]
    hard_fp = summary["safety"]["hard_edit_false_positive_rate"]
    return identity_weight * identity - overcorrection_weight * overcorrection - fp_weight * hard_fp

