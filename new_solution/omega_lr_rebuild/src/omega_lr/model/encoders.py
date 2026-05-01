"""Sequence encoders."""

from __future__ import annotations

import torch
from torch import nn


class TargetEncoder(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
            nn.GELU(),
        )

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        encoded = self.net(embedded.transpose(1, 2)).transpose(1, 2)
        return encoded

