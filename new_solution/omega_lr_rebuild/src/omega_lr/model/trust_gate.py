"""Trust gate."""

from __future__ import annotations

import torch
from torch import nn


class TrustGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward_logits(self, target_repr: torch.Tensor, support_repr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([target_repr, support_repr], dim=-1))

    def forward(self, target_repr: torch.Tensor, support_repr: torch.Tensor, gate_open_bias: float = 0.0) -> torch.Tensor:
        logits = self.forward_logits(target_repr, support_repr) + gate_open_bias
        return torch.sigmoid(logits)
