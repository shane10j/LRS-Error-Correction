from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch
from torch import nn

from .config import OmegaConfig
from .modules import OverlapAggregator, SequenceEncoder
from .support import compute_support_statistics
from .vocab import EDIT_TO_ID

_CORE_HARD_EDIT_IDS = [
    EDIT_TO_ID["SUB_A"],
    EDIT_TO_ID["SUB_C"],
    EDIT_TO_ID["SUB_G"],
    EDIT_TO_ID["SUB_T"],
    EDIT_TO_ID["DEL"],
]


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
        self.target_feat_proj = nn.Linear(mcfg.target_feature_dim, d_model)

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
        self.aggregator = OverlapAggregator(d_model, mcfg.dropout, support_feature_dim=9)

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
        self.deletion_candidate_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(mcfg.dropout),
            nn.Linear(d_model, 1),
        )
        self.deletion_length_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(mcfg.dropout),
            nn.Linear(d_model, mcfg.max_deletion_length + 1),
        )
        self.run_length_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(mcfg.dropout),
            nn.Linear(d_model, mcfg.run_length_classes + 1),
        )
        if mcfg.support_mode not in {"full", "target_only", "support_only", "masked_target"}:
            raise ValueError(f"Unsupported support_mode={mcfg.support_mode!r}.")

    def apply_support_confidence_filter(
        self,
        edit_logits: torch.Tensor,
        support_stats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.cfg.model.apply_hard_edit_support_filter:
            return edit_logits
        core_logits = edit_logits[:, :, -1, :]
        agreement = support_stats["agreement"]
        entropy = support_stats["entropy"]
        depth = support_stats["depth"]
        allow_hard_edit = (
            (agreement >= self.cfg.model.hard_edit_min_support_agreement)
            & (entropy <= self.cfg.model.hard_edit_max_support_entropy)
            & (depth >= self.cfg.model.hard_edit_min_support_depth)
        )
        filtered_core_logits = core_logits.clone()
        hard_edit_index = torch.tensor(_CORE_HARD_EDIT_IDS, device=core_logits.device)
        penalty = (~allow_hard_edit).float().unsqueeze(-1) * self.cfg.model.hard_edit_filter_logit_bias
        filtered_core_logits[..., hard_edit_index] = filtered_core_logits[..., hard_edit_index] - penalty
        edit_logits = edit_logits.clone()
        edit_logits[:, :, -1, :] = filtered_core_logits
        return edit_logits

    def encode_target(
        self,
        target_bases: torch.Tensor,
        target_quals: torch.Tensor,
        target_run_lengths: torch.Tensor,
        target_mask: torch.Tensor,
        tandem_repeat_flag: torch.Tensor,
        deletion_support_count: torch.Tensor,
        deletion_support_fraction: torch.Tensor,
        local_support_entropy: torch.Tensor,
        local_support_agreement: torch.Tensor,
        local_support_depth: torch.Tensor,
        gap_length_histogram: torch.Tensor,
    ) -> torch.Tensor:
        x = self.base_emb(target_bases) + self.qual_emb(target_quals.clamp_min(0))
        x = x + self.run_len_proj(target_run_lengths.unsqueeze(-1).float())
        target_features = torch.cat(
            [
                tandem_repeat_flag.unsqueeze(-1).float(),
                deletion_support_count.unsqueeze(-1).float(),
                deletion_support_fraction.unsqueeze(-1).float(),
                local_support_entropy.unsqueeze(-1).float(),
                local_support_agreement.unsqueeze(-1).float(),
                local_support_depth.unsqueeze(-1).float(),
                gap_length_histogram.float(),
            ],
            dim=-1,
        )
        x = x + self.target_feat_proj(target_features)
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
        support_ins_base_support: torch.Tensor,
        support_strand: torch.Tensor,
        support_haplotype: torch.Tensor,
        support_same_haplotype: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_support, support_len = support_bases.shape
        hap_one_hot = torch.stack(
            [
                support_haplotype.eq(0).float(),
                support_haplotype.eq(1).float(),
                support_haplotype.eq(2).float(),
            ],
            dim=-1,
        )
        feat = torch.stack(
            [
                support_match_mask.float(),
                support_ins_mask.float(),
                support_del_mask.float(),
                support_valid_mask.float(),
                support_strand.float(),
                support_same_haplotype.float(),
            ],
            dim=-1,
        )
        feat = torch.cat([feat, support_base_support, support_ins_base_support, hap_one_hot], dim=-1)
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
                batch.tandem_repeat_flag,
                batch.deletion_support_count,
                batch.deletion_support_fraction,
                batch.local_support_entropy,
                batch.local_support_agreement,
                batch.local_support_depth,
                batch.gap_length_histogram,
            )
        support_hidden = self.encode_support(
            batch.support_bases,
            batch.support_quals,
            batch.support_match_mask,
            batch.support_ins_mask,
            batch.support_del_mask,
            batch.support_valid_mask,
            batch.support_base_support,
            batch.support_ins_base_support,
            batch.support_strand,
            batch.support_haplotype,
            batch.support_same_haplotype,
        )
        support_token_mask = batch.support_valid_mask & (
            batch.support_del_mask.bool()
            | batch.support_ins_mask.bool()
            | batch.support_base_support.sum(dim=-1).gt(0)
        )
        support_stats = compute_support_statistics(
            batch.support_base_support,
            batch.support_del_mask,
            batch.support_ins_base_support,
            batch.support_haplotype,
            batch.support_same_haplotype,
        )
        support_features = torch.stack(
            [
                support_stats["agreement"],
                1.0 - support_stats["entropy"],
                support_stats["depth_norm"],
                support_stats["ins_agreement"],
                1.0 - support_stats["ins_entropy"],
                support_stats["ins_depth_norm"],
                support_stats["hap_agreement"],
                support_stats["hap_consistency"],
                support_stats["phasing_depth_norm"],
            ],
            dim=-1,
        )
        support_summary, trust_gate = self.aggregator(
            target_hidden,
            support_hidden,
            support_token_mask,
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
        edit_logits = self.apply_support_confidence_filter(edit_logits, support_stats)
        return {
            "edit_logits": edit_logits,
            "support_logits": self.support_dist_head(fused),
            "uncertainty_logits": self.uncertainty_head(fused),
            "preserve_logits": self.preserve_head(fused).squeeze(-1),
            "deletion_candidate_logits": self.deletion_candidate_head(fused).squeeze(-1),
            "deletion_length_logits": self.deletion_length_head(fused),
            "run_length_logits": self.run_length_head(fused),
            "trust_gate": trust_gate,
            "support_confidence": support_stats["confidence"],
            "support_uncertainty": support_stats["uncertainty"],
            "support_entropy": support_stats["entropy"],
            "support_agreement": support_stats["agreement"],
            "support_depth": support_stats["depth"],
            "support_del_count": support_stats["del_counts"],
            "support_del_fraction": support_stats["del_fraction"],
            "support_ins_depth": support_stats["ins_depth"],
            "support_ins_agreement": support_stats["ins_agreement"],
        }

    def get_config(self) -> dict:
        return asdict(self.cfg)
