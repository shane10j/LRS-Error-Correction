from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .tokenizer import DNATokenizer
from .utils import read_jsonl
from .vocab import PAD_BASE_ID, PAD_EDIT_ID


@dataclass
class Batch:
    target_bases: torch.Tensor
    target_quals: torch.Tensor
    target_run_lengths: torch.Tensor
    target_mask: torch.Tensor
    support_bases: torch.Tensor
    support_quals: torch.Tensor
    support_match_mask: torch.Tensor
    support_ins_mask: torch.Tensor
    support_del_mask: torch.Tensor
    support_valid_mask: torch.Tensor
    support_base_support: torch.Tensor
    support_ins_base_support: torch.Tensor
    support_strand: torch.Tensor
    support_haplotype: torch.Tensor
    support_same_haplotype: torch.Tensor
    tandem_repeat_flag: torch.Tensor
    deletion_support_count: torch.Tensor
    deletion_support_fraction: torch.Tensor
    local_support_entropy: torch.Tensor
    local_support_agreement: torch.Tensor
    local_support_depth: torch.Tensor
    gap_length_histogram: torch.Tensor
    target_sequence_ids: torch.Tensor
    target_sequence_mask: torch.Tensor
    edit_labels: torch.Tensor
    deletion_candidate_labels: torch.Tensor
    deletion_length_labels: torch.Tensor
    variant_mask: torch.Tensor
    phased_variant_mask: torch.Tensor
    preserve_mask: torch.Tensor
    uncertainty_labels: torch.Tensor
    region_masks: Dict[str, torch.Tensor]
    metadata: List[Dict[str, Any]]

    def to(self, device: torch.device | str) -> "Batch":
        kwargs = {}
        for field_name, value in self.__dict__.items():
            if torch.is_tensor(value):
                kwargs[field_name] = value.to(device)
            elif isinstance(value, dict):
                kwargs[field_name] = {
                    key: tensor.to(device) if torch.is_tensor(tensor) else tensor
                    for key, tensor in value.items()
                }
            else:
                kwargs[field_name] = value
        return Batch(**kwargs)


class LongReadDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.items = list(read_jsonl(path))
        self.tokenizer = DNATokenizer()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        edit_labels = item["edit_labels"]
        if edit_labels and isinstance(edit_labels[0], int):
            edit_labels = [[label] for label in edit_labels]
        return {
            "read_id": item.get("read_id", str(idx)),
            "source_read_id": item.get("source_read_id", item.get("read_id", str(idx))),
            "contig": item.get("contig"),
            "window_ref_start": item.get("window_ref_start"),
            "window_ref_end": item.get("window_ref_end"),
            "target_bases": self.tokenizer.encode(item["target_bases"]),
            "target_quals": item.get("target_qualities", [0] * len(item["target_bases"])),
            "target_run_lengths": item.get("target_run_lengths", [1] * len(item["target_bases"])),
            "support_bases": self.tokenizer.batch_encode(item.get("support_bases", [])),
            "support_quals": item.get("support_qualities", []),
            "support_match_mask": item.get("support_match_mask", []),
            "support_ins_mask": item.get("support_ins_mask", []),
            "support_del_mask": item.get("support_del_mask", []),
            "support_base_support": item.get("support_base_support", []),
            "support_ins_base_support": item.get("support_ins_base_support"),
            "support_strand": item.get("support_strand", []),
            "support_haplotype": item.get("support_haplotype", []),
            "support_same_haplotype": item.get("support_same_haplotype", []),
            "tandem_repeat_flag": item.get("tandem_repeat_flag", [0] * len(item["edit_labels"])),
            "deletion_support_count": item.get("deletion_support_count", [0.0] * len(item["edit_labels"])),
            "deletion_support_fraction": item.get("deletion_support_fraction", [0.0] * len(item["edit_labels"])),
            "local_support_entropy": item.get("local_support_entropy", [0.0] * len(item["edit_labels"])),
            "local_support_agreement": item.get("local_support_agreement", [0.0] * len(item["edit_labels"])),
            "local_support_depth": item.get("local_support_depth", [0.0] * len(item["edit_labels"])),
            "gap_length_histogram": item.get("gap_length_histogram", [[0.0, 0.0, 0.0, 0.0] for _ in item["edit_labels"]]),
            "target_sequence_ids": self.tokenizer.encode(item.get("target_sequence", "")),
            "edit_labels": edit_labels,
            "deletion_candidate_labels": item.get("deletion_candidate_labels", [0] * len(item["edit_labels"])),
            "deletion_length_labels": item.get("deletion_length_labels", [0] * len(item["edit_labels"])),
            "variant_mask": item.get("variant_mask", [0] * len(item["edit_labels"])),
            "phased_variant_mask": item.get("phased_variant_mask", [0] * len(item["edit_labels"])),
            "preserve_mask": item.get("preserve_mask", [0] * len(item["edit_labels"])),
            "uncertainty_labels": item.get("uncertainty_labels", [0] * len(item["edit_labels"])),
            "region_masks": item.get("region_masks", {}),
            "target_haplotype": item.get("target_haplotype", 0),
            "target_sequence": item.get("target_sequence", ""),
        }


def _pad_1d(seqs: List[List[int]], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return out, mask


def _pad_1d_float(seqs: List[List[float]], pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), float(pad_value), dtype=torch.float32)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.float32)
        mask[i, : len(seq)] = True
    return out, mask


def _pad_2d_nested(
    nested: List[List[List[int]]], pad_value: int, dtype: torch.dtype = torch.long
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(nested)
    max_items = max(len(x) for x in nested) if nested else 0
    max_len = max((len(y) for x in nested for y in x), default=1)
    out = torch.full((batch_size, max_items, max_len), pad_value, dtype=dtype)
    valid = torch.zeros((batch_size, max_items, max_len), dtype=torch.bool)
    for i, group in enumerate(nested):
        for j, seq in enumerate(group):
            if not seq:
                continue
            t = torch.tensor(seq, dtype=dtype)
            out[i, j, : len(seq)] = t
            valid[i, j, : len(seq)] = True
    return out, valid


def _pad_edit_labels(
    nested: List[List[List[int]]], pad_value: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(nested)
    max_len = max(len(seq) for seq in nested)
    max_slots = max((len(slots) for seq in nested for slots in seq), default=1)
    out = torch.full((batch_size, max_len, max_slots), pad_value, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len, max_slots), dtype=torch.bool)
    for i, seq in enumerate(nested):
        for j, slots in enumerate(seq):
            if not slots:
                continue
            out[i, j, : len(slots)] = torch.tensor(slots, dtype=torch.long)
            mask[i, j, : len(slots)] = True
    return out, mask


def _pad_base_support(nested: List[List[List[List[float]]]]) -> torch.Tensor:
    batch_size = len(nested)
    max_items = max(len(x) for x in nested) if nested else 0
    max_len = max((len(y) for x in nested for y in x), default=1)
    out = torch.zeros((batch_size, max_items, max_len, 4), dtype=torch.float32)
    for i, group in enumerate(nested):
        for j, seq in enumerate(group):
            if len(seq) == 0:
                continue
            t = torch.tensor(seq, dtype=torch.float32)
            out[i, j, : t.shape[0], :] = t
    return out


def _pad_2d_feature(nested: List[List[List[float]]], width: int) -> torch.Tensor:
    batch_size = len(nested)
    max_len = max((len(seq) for seq in nested), default=1)
    out = torch.zeros((batch_size, max_len, width), dtype=torch.float32)
    for i, seq in enumerate(nested):
        if not seq:
            continue
        t = torch.tensor(seq, dtype=torch.float32)
        out[i, : t.shape[0], : min(width, t.shape[1])] = t[:, :width]
    return out


def _zero_nested_scalar_support(support_bases: List[List[int]], value: float = 0.0) -> List[List[float]]:
    return [[value for _ in seq] for seq in support_bases]


def _zero_nested_base_support(support_bases: List[List[int]]) -> List[List[List[float]]]:
    return [[[0.0, 0.0, 0.0, 0.0] for _ in seq] for seq in support_bases]


def collate_long_reads(samples: List[Dict[str, Any]]) -> Batch:
    target_bases, target_mask = _pad_1d([s["target_bases"] for s in samples], PAD_BASE_ID)
    target_quals, _ = _pad_1d([s["target_quals"] for s in samples], 0)
    target_run_lengths, _ = _pad_1d([s["target_run_lengths"] for s in samples], 1)
    target_sequence_ids, target_sequence_mask = _pad_1d([s["target_sequence_ids"] for s in samples], PAD_BASE_ID)
    edit_labels, _ = _pad_edit_labels([s["edit_labels"] for s in samples], PAD_EDIT_ID)
    deletion_candidate_labels, _ = _pad_1d([s["deletion_candidate_labels"] for s in samples], 0)
    deletion_length_labels, _ = _pad_1d([s["deletion_length_labels"] for s in samples], 0)
    variant_mask, _ = _pad_1d([s["variant_mask"] for s in samples], 0)
    phased_variant_mask, _ = _pad_1d([s["phased_variant_mask"] for s in samples], 0)
    preserve_mask, _ = _pad_1d([s["preserve_mask"] for s in samples], 0)
    uncertainty_labels, _ = _pad_1d([s["uncertainty_labels"] for s in samples], 0)

    support_bases, support_valid_mask = _pad_2d_nested([s["support_bases"] for s in samples], PAD_BASE_ID)
    support_quals, _ = _pad_2d_nested([s["support_quals"] for s in samples], 0)
    support_match_mask, _ = _pad_2d_nested([s["support_match_mask"] for s in samples], 0)
    support_ins_mask, _ = _pad_2d_nested([s["support_ins_mask"] for s in samples], 0)
    support_del_mask, _ = _pad_2d_nested([s["support_del_mask"] for s in samples], 0)
    support_strand, _ = _pad_2d_nested(
        [
            s["support_strand"] if s.get("support_strand") else _zero_nested_scalar_support(s["support_bases"])
            for s in samples
        ],
        0,
        dtype=torch.float32,
    )
    support_haplotype, _ = _pad_2d_nested(
        [
            s["support_haplotype"] if s.get("support_haplotype") else _zero_nested_scalar_support(s["support_bases"])
            for s in samples
        ],
        0,
        dtype=torch.float32,
    )
    support_same_haplotype, _ = _pad_2d_nested(
        [
            s["support_same_haplotype"] if s.get("support_same_haplotype") else _zero_nested_scalar_support(s["support_bases"])
            for s in samples
        ],
        0,
        dtype=torch.float32,
    )
    support_base_support = _pad_base_support([s["support_base_support"] for s in samples])
    support_ins_base_support = _pad_base_support(
        [
            s["support_ins_base_support"]
            if s.get("support_ins_base_support") is not None
            else _zero_nested_base_support(s["support_bases"])
            for s in samples
        ]
    )
    tandem_repeat_flag, _ = _pad_1d([s["tandem_repeat_flag"] for s in samples], 0)
    deletion_support_count, _ = _pad_1d_float([s["deletion_support_count"] for s in samples], 0.0)
    deletion_support_fraction, _ = _pad_1d_float([s["deletion_support_fraction"] for s in samples], 0.0)
    local_support_entropy, _ = _pad_1d_float([s["local_support_entropy"] for s in samples], 0.0)
    local_support_agreement, _ = _pad_1d_float([s["local_support_agreement"] for s in samples], 0.0)
    local_support_depth, _ = _pad_1d_float([s["local_support_depth"] for s in samples], 0.0)
    gap_width = max((len(row[0]) for row in [s["gap_length_histogram"] for s in samples] if row), default=4)
    gap_length_histogram = _pad_2d_feature([s["gap_length_histogram"] for s in samples], gap_width)
    region_names = sorted({name for sample in samples for name in sample.get("region_masks", {}).keys()})
    region_masks = {}
    for name in region_names:
        region_masks[name], _ = _pad_1d(
            [sample.get("region_masks", {}).get(name, [0] * len(sample["edit_labels"])) for sample in samples],
            0,
        )

    metadata = [
        {
            "read_id": s["read_id"],
            "source_read_id": s.get("source_read_id"),
            "contig": s.get("contig"),
            "window_ref_start": s.get("window_ref_start"),
            "window_ref_end": s.get("window_ref_end"),
            "target_haplotype": s.get("target_haplotype", 0),
            "target_sequence": s.get("target_sequence", ""),
        }
        for s in samples
    ]
    return Batch(
        target_bases=target_bases,
        target_quals=target_quals,
        target_run_lengths=target_run_lengths,
        target_mask=target_mask,
        support_bases=support_bases,
        support_quals=support_quals,
        support_match_mask=support_match_mask,
        support_ins_mask=support_ins_mask,
        support_del_mask=support_del_mask,
        support_valid_mask=support_valid_mask,
        support_base_support=support_base_support,
        support_ins_base_support=support_ins_base_support,
        support_strand=support_strand,
        support_haplotype=support_haplotype,
        support_same_haplotype=support_same_haplotype,
        tandem_repeat_flag=tandem_repeat_flag.float(),
        deletion_support_count=deletion_support_count.float(),
        deletion_support_fraction=deletion_support_fraction.float(),
        local_support_entropy=local_support_entropy.float(),
        local_support_agreement=local_support_agreement.float(),
        local_support_depth=local_support_depth.float(),
        gap_length_histogram=gap_length_histogram.float(),
        target_sequence_ids=target_sequence_ids,
        target_sequence_mask=target_sequence_mask,
        edit_labels=edit_labels,
        deletion_candidate_labels=deletion_candidate_labels.float(),
        deletion_length_labels=deletion_length_labels.long(),
        variant_mask=variant_mask.float(),
        phased_variant_mask=phased_variant_mask.float(),
        preserve_mask=preserve_mask.float(),
        uncertainty_labels=uncertainty_labels,
        region_masks={name: mask.float() for name, mask in region_masks.items()},
        metadata=metadata,
    )
