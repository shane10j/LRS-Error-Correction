"""Embeddings."""

from __future__ import annotations

import torch
from torch import nn


class TargetEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_length: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        return self.embedding(tokens) + self.position_embedding(positions)
