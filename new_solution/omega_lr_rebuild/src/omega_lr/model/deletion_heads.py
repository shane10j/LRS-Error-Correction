"""Deletion heads."""

from __future__ import annotations

from torch import nn


class DeleteCandidateHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, fused):
        return self.linear(fused).squeeze(-1)


class DeleteLengthHead(nn.Module):
    def __init__(self, d_model: int, max_deletion_length: int):
        super().__init__()
        self.linear = nn.Linear(d_model, max_deletion_length + 1)

    def forward(self, fused):
        return self.linear(fused)

