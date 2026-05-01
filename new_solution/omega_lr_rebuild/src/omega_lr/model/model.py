"""Model assembly."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from omega_lr.constants import PAD_BASE_ID
from omega_lr.model.deletion_heads import DeleteCandidateHead, DeleteLengthHead
from omega_lr.model.edit_heads import InsBaseHead, SubBaseHead, TypeHead, compose_flat_edit_logits
from omega_lr.model.embeddings import TargetEmbedding
from omega_lr.model.encoders import TargetEncoder
from omega_lr.model.support_encoder import SupportEncoder
from omega_lr.model.support_rule_heads import SupportFlagHead, SupportMajorityBaseHead
from omega_lr.model.trust_gate import TrustGate


@dataclass
class ModelConfig:
    d_model: int
    conv_kernel_size: int
    support_hidden_dim: int
    max_window_length: int
    max_deletion_length: int
    use_support: bool
    support_input_dim: int
    rule_feature_dim: int
    use_trust_gate: bool
    use_delete_length_head: bool
    payload_type_coupling_boost: float = 0.0


class OmegaEditModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.target_embedding = TargetEmbedding(PAD_BASE_ID + 1, config.d_model, config.max_window_length)
        self.target_encoder = TargetEncoder(config.d_model, config.conv_kernel_size)
        self.support_encoder = SupportEncoder(config.support_input_dim, config.d_model)
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
        )
        self.trust_gate = TrustGate(config.d_model)
        self.type_head = TypeHead(config.d_model + config.rule_feature_dim + 5)
        self.sub_base_head = SubBaseHead(config.d_model)
        self.ins_base_head = InsBaseHead(config.d_model)
        self.delete_candidate_head = DeleteCandidateHead(config.d_model)
        self.delete_length_head = DeleteLengthHead(config.d_model, config.max_deletion_length)
        self.trust_head = nn.Linear(config.d_model, 1)
        self.support_majority_base_head = SupportMajorityBaseHead(config.d_model)
        self.support_suggests_sub_head = SupportFlagHead(config.d_model)
        self.support_suggests_ins_head = SupportFlagHead(config.d_model)
        self.support_suggests_del_head = SupportFlagHead(config.d_model)

    def forward(
        self,
        target_tokens: torch.Tensor,
        pileup_features: torch.Tensor,
        rule_features: torch.Tensor | None = None,
        gate_open_bias: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        target_repr = self.target_encoder(self.target_embedding(target_tokens))
        support_repr = self.support_encoder(pileup_features)
        if rule_features is None:
            rule_features = torch.zeros(
                target_tokens.shape[0],
                target_tokens.shape[1],
                self.config.rule_feature_dim,
                device=target_tokens.device,
                dtype=support_repr.dtype,
            )
        if self.config.use_support:
            fused = self.fusion(torch.cat([target_repr, support_repr], dim=-1))
            if self.config.use_trust_gate:
                trust = self.trust_gate(target_repr, support_repr, gate_open_bias=gate_open_bias)
            else:
                trust = torch.ones_like(target_repr[..., :1])
        else:
            fused = target_repr
            trust = torch.ones_like(target_repr[..., :1])
            rule_features = torch.zeros_like(rule_features)
        sub_base_logits = self.sub_base_head(fused)
        ins_base_logits = self.ins_base_head(fused)
        sub_probs = torch.softmax(sub_base_logits, dim=-1)
        ins_probs = torch.softmax(ins_base_logits, dim=-1)
        max_sub_prob, sub_argmax = sub_probs.max(dim=-1, keepdim=True)
        max_ins_prob, _ = ins_probs.max(dim=-1, keepdim=True)
        target_base_ids = target_tokens.clamp(min=0, max=3).unsqueeze(-1)
        sub_argmax_differs = (sub_argmax != target_base_ids).float()
        majority_differs = rule_features[..., 0:1]
        insertion_fraction = rule_features[..., 2:3]
        sub_payload_support = max_sub_prob * sub_argmax_differs * majority_differs
        ins_payload_support = max_ins_prob * insertion_fraction
        type_inputs = torch.cat(
            [
                fused,
                rule_features,
                max_sub_prob,
                sub_argmax_differs,
                max_ins_prob,
                sub_payload_support,
                ins_payload_support,
            ],
            dim=-1,
        )
        type_logits = self.type_head(type_inputs)
        if self.config.use_support and self.config.payload_type_coupling_boost > 0.0:
            type_logits = type_logits.clone()
            type_logits[..., 1] = type_logits[..., 1] + self.config.payload_type_coupling_boost * sub_payload_support.squeeze(-1)
            type_logits[..., 3] = type_logits[..., 3] + self.config.payload_type_coupling_boost * ins_payload_support.squeeze(-1)
        if self.config.use_delete_length_head:
            delete_length_logits = self.delete_length_head(fused)
        else:
            delete_length_logits = torch.zeros(
                fused.shape[0],
                fused.shape[1],
                self.config.max_deletion_length + 1,
                device=fused.device,
                dtype=fused.dtype,
            )
        edit_logits = compose_flat_edit_logits(type_logits, sub_base_logits, ins_base_logits)
        return {
            "type_logits": type_logits,
            "sub_base_logits": sub_base_logits,
            "ins_base_logits": ins_base_logits,
            "edit_logits": edit_logits,
            "delete_candidate_logits": self.delete_candidate_head(fused),
            "delete_length_logits": delete_length_logits,
            "trust": trust.squeeze(-1),
            "trust_logits": self.trust_head(fused).squeeze(-1),
            "support_majority_base_logits": self.support_majority_base_head(support_repr),
            "support_suggests_sub_logits": self.support_suggests_sub_head(support_repr),
            "support_suggests_ins_logits": self.support_suggests_ins_head(support_repr),
            "support_suggests_del_logits": self.support_suggests_del_head(support_repr),
        }
