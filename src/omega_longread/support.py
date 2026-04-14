from __future__ import annotations

import math
from typing import Dict

import torch


def compute_support_statistics(support_base_support: torch.Tensor) -> Dict[str, torch.Tensor]:
    counts = support_base_support.sum(dim=1)
    depth = counts.sum(dim=-1)
    support_mask = depth.gt(0)
    probs = counts / depth.unsqueeze(-1).clamp_min(1.0)
    agreement = probs.max(dim=-1).values * support_mask.float()
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    entropy = (entropy / math.log(4.0)) * support_mask.float()
    max_depth = max(int(support_base_support.shape[1]), 1)
    depth_norm = (depth / float(max_depth)).clamp(0.0, 1.0)
    confidence = (agreement * depth_norm).clamp(0.0, 1.0)
    uncertainty = (0.5 * entropy + 0.5 * (1.0 - confidence)).clamp(0.0, 1.0)
    return {
        "counts": counts,
        "depth": depth,
        "depth_norm": depth_norm,
        "probs": probs,
        "agreement": agreement,
        "entropy": entropy,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "mask": support_mask,
    }
