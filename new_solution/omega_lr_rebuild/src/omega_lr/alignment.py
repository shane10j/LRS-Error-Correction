"""Small dynamic-programming alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentResult:
    aligned_target: str
    aligned_truth: str
    score: int


def global_align(target: str, truth: str, match: int = 2, mismatch: int = -1, gap: int = -2) -> AlignmentResult:
    rows = len(target) + 1
    cols = len(truth) + 1
    dp = [[0] * cols for _ in range(rows)]
    trace = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i * gap
        trace[i][0] = "U"
    for j in range(1, cols):
        dp[0][j] = j * gap
        trace[0][j] = "L"
    for i in range(1, rows):
        for j in range(1, cols):
            diag = dp[i - 1][j - 1] + (match if target[i - 1] == truth[j - 1] else mismatch)
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            best = max(diag, up, left)
            dp[i][j] = best
            trace[i][j] = "D" if best == diag else ("U" if best == up else "L")
    i, j = len(target), len(truth)
    aligned_target = []
    aligned_truth = []
    while i > 0 or j > 0:
        move = trace[i][j]
        if move == "D":
            aligned_target.append(target[i - 1])
            aligned_truth.append(truth[j - 1])
            i -= 1
            j -= 1
        elif move == "U":
            aligned_target.append(target[i - 1])
            aligned_truth.append("-")
            i -= 1
        else:
            aligned_target.append("-")
            aligned_truth.append(truth[j - 1])
            j -= 1
    return AlignmentResult("".join(reversed(aligned_target)), "".join(reversed(aligned_truth)), dp[-1][-1])


def left_normalize_indels(aligned_target: str, aligned_truth: str) -> tuple[str, str]:
    target = list(aligned_target)
    truth = list(aligned_truth)
    moved = True
    while moved:
        moved = False
        for idx in range(1, len(target)):
            if target[idx] == "-" and target[idx - 1] == truth[idx] and truth[idx - 1] != "-":
                target[idx], target[idx - 1] = target[idx - 1], target[idx]
                truth[idx], truth[idx - 1] = truth[idx - 1], truth[idx]
                moved = True
            if truth[idx] == "-" and truth[idx - 1] == target[idx] and target[idx - 1] != "-":
                target[idx], target[idx - 1] = target[idx - 1], target[idx]
                truth[idx], truth[idx - 1] = truth[idx - 1], truth[idx]
                moved = True
    return "".join(target), "".join(truth)

