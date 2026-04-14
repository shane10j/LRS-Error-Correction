from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch
from torch import nn

from .config import OmegaConfig
from .modules import OverlapAggregator, SequenceEncoder
from .support import compute_support_statistics


class OmegaModel(nn.Module):
    def __init__(self, cfg: OmegaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model
        d_model = mcfg.d_model
        self.num_edit_slots = 1 + mcfg.max_insertions_per_pos

        self.base_emb = nn.Embedding(mcfg.base_vocab_size, d_model)
        self.qual_emb = nn.Embedding(mcfg.quality_vocab_size, d_model)
        self.run_len_proj = nn.Linear(1, d_model)

        self.support_base_emb = nn.Embedding(mcfg.base_vocab_size, d_model)
        self.support_qual_emb = nn.Embedding(mcfg.quality_vocab_size, d_model)
        self.support_feat_proj = nn.Linear(mcfg.support_feature_dim, d_model)

        self.target_encoder = SequenceEncoder(
            d_model=d_model,
            num_heads=mcfg.num_heads,
            num_layers=mcfg.num_layers,
            ff_mult=mcfg.ff_mult,
            dropout=mcfg.dropout,
            kernel_size=mcfg.conv_kernel_size,
        )
        self.support_encoder = SequenceEncoder(
            d_model=d_model,
            num_heads=mcfg.num_heads,
            num_layers=max(2, mcfg.num_layers // 2),
            ff_mult=mcfg.ff_mult,
            dropout=mcfg.dropout,
            kernel_size=mcfg.conv_kernel_size,
        )
        self.aggregator = OverlapAggregator(d_model, mcfg.num_heads, mcfg.dropout)

        self.edit_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(mcfg.dropout),
            nn.Linear(d_model, self.num_edit_slots * mcfg.edit_vocab_size),
        )
        self.support_dist_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(mcfg.dropout),
            nn.Linear(d_model, mcfg.uncertainty_classes),
        )
        self.preserve_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        if mcfg.support_mode not in {"full", "target_only", "support_only", "masked_target"}:
            raise ValueError(f"Unsupported support_mode={mcfg.support_mode!r}.")

    def encode_target(self, target_bases: torch.Tensor, target_quals: torch.Tensor, target_run_lengths: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        x = self.base_emb(target_bases) + self.qual_emb(target_quals.clamp_min(0))
        x = x + self.run_len_proj(target_run_lengths.unsqueeze(-1).float())
        return self.target_encoder(x, target_mask)

    def encode_masked_target(self, target_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = target_mask.shape
        device = target_mask.device
        d_model = self.base_emb.embedding_dim
        x = torch.zeros(batch_size, seq_len, d_model, device=device, dtype=self.base_emb.weight.dtype)
        return self.target_encoder(x, target_mask)

    def encode_support(
        self,
        support_bases: torch.Tensor,
        support_quals: torch.Tensor,
        support_match_mask: torch.Tensor,
        support_ins_mask: torch.Tensor,
        support_del_mask: torch.Tensor,
        support_valid_mask: torch.Tensor,
        support_base_support: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_support, support_len = support_bases.shape
        feat = torch.stack(
            [
                support_match_mask.float(),
                support_ins_mask.float(),
                support_del_mask.float(),
                support_valid_mask.float(),
            ],
            dim=-1,
        )
        feat = torch.cat([feat, support_base_support], dim=-1)
        x = self.support_base_emb(support_bases) + self.support_qual_emb(support_quals.clamp_min(0))
        x = x + self.support_feat_proj(feat)
        x = x.view(batch * num_support, support_len, -1)
        mask = support_valid_mask.view(batch * num_support, support_len)
        h = self.support_encoder(x, mask)
        return h.view(batch, num_support, support_len, -1)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        support_mode = self.cfg.model.support_mode
        if support_mode == "masked_target":
            target_hidden = self.encode_masked_target(batch.target_mask)
        else:
            target_hidden = self.encode_target(
                batch.target_bases,
                batch.target_quals,
                batch.target_run_lengths,
                batch.target_mask,
            )
        support_hidden = self.encode_support(
            batch.support_bases,
            batch.support_quals,
            batch.support_match_mask,
            batch.support_ins_mask,
            batch.support_del_mask,
            batch.support_valid_mask,
            batch.support_base_support,
        )
        support_stats = compute_support_statistics(batch.support_base_support)
        support_features = torch.stack(
            [
                support_stats["agreement"],
                1.0 - support_stats["entropy"],
                support_stats["depth_norm"],
            ],
            dim=-1,
        )
        support_summary, trust_gate = self.aggregator(
            target_hidden,
            support_hidden,
            batch.support_valid_mask,
            support_features,
        )
        if support_mode == "target_only":
            fused_target = target_hidden
            fused_support = torch.zeros_like(support_summary)
            trust_gate = torch.zeros_like(trust_gate)
        elif support_mode == "support_only":
            fused_target = torch.zeros_like(target_hidden)
            fused_support = support_summary
            trust_gate = torch.ones_like(trust_gate)
        elif support_mode == "masked_target":
            fused_target = torch.zeros_like(target_hidden)
            fused_support = support_summary
        else:
            fused_target = target_hidden
            fused_support = support_summary
        fused = torch.cat([fused_target, fused_support], dim=-1)
        edit_logits = self.edit_head(fused).view(
            fused.shape[0],
            fused.shape[1],
            self.num_edit_slots,
            self.cfg.model.edit_vocab_size,
        )
        return {
            "edit_logits": edit_logits,
            "support_logits": self.support_dist_head(fused),
            "uncertainty_logits": self.uncertainty_head(fused),
            "preserve_logits": self.preserve_head(fused).squeeze(-1),
            "trust_gate": trust_gate,
            "support_confidence": support_stats["confidence"],
            "support_uncertainty": support_stats["uncertainty"],
            "support_entropy": support_stats["entropy"],
            "support_agreement": support_stats["agreement"],
            "support_depth": support_stats["depth"],
        }

    def get_config(self) -> dict:
        return asdict(self.cfg)
