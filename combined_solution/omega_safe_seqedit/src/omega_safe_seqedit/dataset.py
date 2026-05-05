"""Torch dataset and collate for seq-edit examples."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from omega_safe_seqedit.constants import BASE_TO_ID, PAD_BASE_ID
from omega_safe_seqedit.features import feature_matrix, rule_feature_matrix
from omega_safe_seqedit.io_utils import read_jsonl


class SeqEditDataset(Dataset):
    def __init__(self, path: str):
        self.records = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        target_ids = [BASE_TO_ID.get(base, PAD_BASE_ID) for base in record["target_seq"]]
        return {
            "record": record,
            "target_ids": target_ids,
            "features": feature_matrix(record),
            "rule_features": rule_feature_matrix(record),
            "main_type": record["labels"]["main_type"],
            "sub_base": record["labels"]["sub_base"],
            "insert_before": record["labels"]["insert_before"],
            "support_rule_type": record["features"]["support_rule_type"],
            "support_rule_sub_base": record["features"]["support_rule_sub_base"],
            "support_rule_ins_base": record["features"]["support_rule_ins_base"],
        }


def _pad_1d(items: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x in items)
    out = torch.full((len(items), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(items), max_len), dtype=torch.float32)
    for idx, seq in enumerate(items):
        out[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[idx, : len(seq)] = 1.0
    return out, mask


def _pad_features(items: list[list[list[float]]]) -> torch.Tensor:
    max_len = max(len(x) for x in items)
    width = len(items[0][0]) if items and items[0] else 1
    out = torch.zeros((len(items), max_len, width), dtype=torch.float32)
    for idx, rows in enumerate(items):
        if rows:
            out[idx, : len(rows), :] = torch.tensor(rows, dtype=torch.float32)
    return out


def collate_seqedit(samples: list[dict]) -> dict:
    target_ids, mask = _pad_1d([s["target_ids"] for s in samples], PAD_BASE_ID)
    main_type, _ = _pad_1d([s["main_type"] for s in samples], 0)
    sub_base, _ = _pad_1d([s["sub_base"] for s in samples], 0)
    insert_before, _ = _pad_1d([s["insert_before"] for s in samples], 0)
    support_rule_type, _ = _pad_1d([s["support_rule_type"] for s in samples], 0)
    support_rule_sub_base, _ = _pad_1d([s["support_rule_sub_base"] for s in samples], 0)
    support_rule_ins_base, _ = _pad_1d([s["support_rule_ins_base"] for s in samples], 0)
    return {
        "records": [s["record"] for s in samples],
        "target_ids": target_ids,
        "attention_mask": mask,
        "features": _pad_features([s["features"] for s in samples]),
        "rule_features": _pad_features([s["rule_features"] for s in samples]),
        "main_type": main_type,
        "sub_base": sub_base,
        "insert_before": insert_before,
        "support_rule_type": support_rule_type,
        "support_rule_sub_base": support_rule_sub_base,
        "support_rule_ins_base": support_rule_ins_base,
    }
