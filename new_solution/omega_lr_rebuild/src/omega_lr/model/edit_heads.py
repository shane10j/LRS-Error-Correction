"""Structured edit heads plus a derived flat label view."""

from __future__ import annotations

import torch
from torch import nn

from omega_lr.constants import BASES, EDIT_TYPE_LABELS


class TypeHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, len(EDIT_TYPE_LABELS))

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.linear(fused)


class SubBaseHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, len(BASES))

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.linear(fused)


class InsBaseHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, len(BASES))

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.linear(fused)


def compose_flat_edit_logits(
    type_logits: torch.Tensor,
    sub_base_logits: torch.Tensor,
    ins_base_logits: torch.Tensor,
) -> torch.Tensor:
    copy_logits = type_logits[..., 0:1]
    sub_logits = type_logits[..., 1:2] + sub_base_logits
    del_logits = type_logits[..., 2:3]
    ins_logits = type_logits[..., 3:4] + ins_base_logits
    return torch.cat([copy_logits, sub_logits, del_logits, ins_logits], dim=-1)
