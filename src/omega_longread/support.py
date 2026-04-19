from __future__ import annotations

import math
from typing import Dict

import torch


def _normalized_entropy(probs: torch.Tensor, num_classes: int) -> torch.Tensor:
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy / math.log(float(max(num_classes, 2)))


def compute_support_statistics(
    support_base_support: torch.Tensor,
    support_del_mask: torch.Tensor | None = None,
    support_ins_base_support: torch.Tensor | None = None,
    support_haplotype: torch.Tensor | None = None,
    support_same_haplotype: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    base_counts = support_base_support.sum(dim=1)
    if support_del_mask is None:
        del_counts = torch.zeros(
            *base_counts.shape[:2],
            device=support_base_support.device,
            dtype=support_base_support.dtype,
        )
    else:
        del_counts = support_del_mask.float().sum(dim=1)
    if support_ins_base_support is None:
        ins_counts = torch.zeros_like(base_counts)
    else:
        ins_counts = support_ins_base_support.sum(dim=1)

    core_counts = torch.cat([base_counts, del_counts.unsqueeze(-1)], dim=-1)
    depth = core_counts.sum(dim=-1)
    support_mask = depth.gt(0)
    probs = core_counts / depth.unsqueeze(-1).clamp_min(1.0)
    agreement = probs.max(dim=-1).values * support_mask.float()
    entropy = _normalized_entropy(probs, num_classes=core_counts.shape[-1]) * support_mask.float()
    max_depth = max(int(support_base_support.shape[1]), 1)
    depth_norm = (depth / float(max_depth)).clamp(0.0, 1.0)
    confidence = (agreement * depth_norm).clamp(0.0, 1.0)
    uncertainty = (0.5 * entropy + 0.5 * (1.0 - confidence)).clamp(0.0, 1.0)
    del_fraction = del_counts / depth.clamp_min(1.0)

    ins_depth = ins_counts.sum(dim=-1)
    ins_mask = ins_depth.gt(0)
    ins_probs = ins_counts / ins_depth.unsqueeze(-1).clamp_min(1.0)
    ins_agreement = ins_probs.max(dim=-1).values * ins_mask.float()
    ins_entropy = _normalized_entropy(ins_probs, num_classes=ins_counts.shape[-1]) * ins_mask.float()
    ins_depth_norm = (ins_depth / float(max_depth)).clamp(0.0, 1.0)
    ins_confidence = (ins_agreement * ins_depth_norm).clamp(0.0, 1.0)

    if support_haplotype is None:
        hap1_counts = torch.zeros_like(depth)
        hap2_counts = torch.zeros_like(depth)
    else:
        hap1_counts = support_haplotype.eq(1).float().sum(dim=1)
        hap2_counts = support_haplotype.eq(2).float().sum(dim=1)
    phased_depth = hap1_counts + hap2_counts
    hap_counts = torch.stack([hap1_counts, hap2_counts], dim=-1)
    hap_probs = hap_counts / phased_depth.unsqueeze(-1).clamp_min(1.0)
    hap_agreement = hap_probs.max(dim=-1).values * phased_depth.gt(0).float()
    phasing_depth_norm = (phased_depth / float(max_depth)).clamp(0.0, 1.0)
    if support_same_haplotype is None:
        hap_consistency = torch.zeros_like(depth_norm)
    else:
        support_valid_count = max(int(support_same_haplotype.shape[1]), 1)
        hap_consistency = (support_same_haplotype.float().sum(dim=1) / float(support_valid_count)).clamp(0.0, 1.0)

    return {
        "counts": base_counts,
        "base_counts": base_counts,
        "del_counts": del_counts,
        "del_fraction": del_fraction,
        "ins_counts": ins_counts,
        "core_counts": core_counts,
        "depth": depth,
        "depth_norm": depth_norm,
        "probs": probs,
        "agreement": agreement,
        "entropy": entropy,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "mask": support_mask,
        "ins_depth": ins_depth,
        "ins_depth_norm": ins_depth_norm,
        "ins_probs": ins_probs,
        "ins_agreement": ins_agreement,
        "ins_entropy": ins_entropy,
        "ins_confidence": ins_confidence,
        "ins_mask": ins_mask,
        "hap1_counts": hap1_counts,
        "hap2_counts": hap2_counts,
        "phased_depth": phased_depth,
        "phasing_depth_norm": phasing_depth_norm,
        "hap_probs": hap_probs,
        "hap_agreement": hap_agreement,
        "hap_consistency": hap_consistency,
    }
