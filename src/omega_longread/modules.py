from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16384) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ConvStem(nn.Module):
    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return x + y


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.ln1(x)
        attn_out, _ = self.attn(z, z, z, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, ff_mult: int, dropout: float, kernel_size: int) -> None:
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.stem = ConvStem(d_model, kernel_size)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, ff_mult, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        original_mask = mask
        safe_mask = mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
        x = self.pos(x)
        x = self.stem(x)
        key_padding_mask = ~safe_mask
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.ln(x)
        return x * original_mask.unsqueeze(-1)


class OverlapAggregator(nn.Module):
    def __init__(self, d_model: int, dropout: float, support_feature_dim: int) -> None:
        super().__init__()
        self.target_ln = nn.LayerNorm(d_model)
        self.support_ln = nn.LayerNorm(d_model)
        self.out_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model + support_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        target_hidden: torch.Tensor,
        support_hidden: torch.Tensor,
        support_token_mask: torch.Tensor,
        support_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_hidden = self.target_ln(target_hidden)
        support_hidden = self.support_ln(support_hidden)
        target_expanded = target_hidden.unsqueeze(1).expand(-1, support_hidden.shape[1], -1, -1)
        score_input = torch.cat([target_expanded, support_hidden], dim=-1)
        support_scores = self.score(score_input).squeeze(-1)
        support_scores = support_scores.masked_fill(~support_token_mask, -1e4)
        support_weights = torch.softmax(support_scores, dim=1)
        support_weights = support_weights * support_token_mask.float()
        support_weight_norm = support_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        support_weights = support_weights / support_weight_norm
        support_summary = (support_weights.unsqueeze(-1) * support_hidden).sum(dim=1)
        no_support = ~support_token_mask.any(dim=1)
        support_summary = support_summary * (~no_support).unsqueeze(-1)
        gate = self.gate(torch.cat([target_hidden, support_summary, support_features], dim=-1))
        fused = gate * support_summary + (1.0 - gate) * target_hidden
        return self.out_ln(target_hidden + self.dropout(fused)), gate.squeeze(-1)
