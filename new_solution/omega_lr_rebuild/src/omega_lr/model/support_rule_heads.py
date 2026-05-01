"""Auxiliary heads that force support-rule representations to stay interpretable."""

from __future__ import annotations

from torch import nn

from omega_lr.constants import BASES


class SupportMajorityBaseHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, len(BASES))

    def forward(self, support_repr):
        return self.linear(support_repr)


class SupportFlagHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, support_repr):
        return self.linear(support_repr).squeeze(-1)
