"""Small sequence-to-edit-sequence model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from omega_safe_seqedit.constants import INS_LABELS, MAIN_TYPES, PAD_BASE_ID, SUPPORT_RULE_TYPES


@dataclass
class SeqEditModelConfig:
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.05
    max_len: int = 1024
    feature_dim: int = 17
    rule_feature_dim: int = 11
    use_support: bool = True
    use_rule_features: bool = True


class SafeSeqEditModel(nn.Module):
    def __init__(self, config: SeqEditModelConfig):
        super().__init__()
        self.config = config
        self.base_embedding = nn.Embedding(PAD_BASE_ID + 1, config.d_model, padding_idx=PAD_BASE_ID)
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)
        self.support_proj = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.rule_proj = nn.Linear(config.rule_feature_dim, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        self.main_head = nn.Linear(config.d_model, len(MAIN_TYPES))
        self.sub_head = nn.Linear(config.d_model, 4)
        self.insert_head = nn.Linear(config.d_model, len(INS_LABELS))
        self.allow_head = nn.Linear(config.d_model, 1)
        self.support_rule_head = nn.Linear(config.d_model, len(SUPPORT_RULE_TYPES))

    def forward(self, target_ids: torch.Tensor, features: torch.Tensor, rule_features: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        batch, length = target_ids.shape
        positions = torch.arange(length, device=target_ids.device).unsqueeze(0).expand(batch, length)
        target = self.base_embedding(target_ids) + self.pos_embedding(positions.clamp(max=self.config.max_len - 1))
        encoded = self.encoder(target, src_key_padding_mask=attention_mask.eq(0))
        if self.config.use_support:
            support = self.support_proj(features)
        else:
            support = torch.zeros_like(encoded)
        if self.config.use_rule_features:
            rules = self.rule_proj(rule_features)
        else:
            rules = torch.zeros_like(encoded)
        fused = self.fusion(torch.cat([encoded, support, rules], dim=-1))
        return {
            "main_logits": self.main_head(fused),
            "sub_logits": self.sub_head(fused),
            "insert_logits": self.insert_head(fused),
            "allow_logits": self.allow_head(fused).squeeze(-1),
            "support_rule_logits": self.support_rule_head(fused),
        }


def model_from_config(config: dict, feature_dim: int = 17, rule_feature_dim: int = 11, use_support: bool | None = None) -> SafeSeqEditModel:
    model_cfg = config["model"]
    return SafeSeqEditModel(
        SeqEditModelConfig(
            d_model=int(model_cfg["d_model"]),
            num_layers=int(model_cfg["num_layers"]),
            num_heads=int(model_cfg["num_heads"]),
            dropout=float(model_cfg.get("dropout", 0.05)),
            max_len=int(model_cfg.get("max_len", 1024)),
            feature_dim=feature_dim,
            rule_feature_dim=rule_feature_dim,
            use_support=bool(model_cfg.get("use_support", True)) if use_support is None else use_support,
            use_rule_features=bool(model_cfg.get("use_rule_features", True)),
        )
    )
