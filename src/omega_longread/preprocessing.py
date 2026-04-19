from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import pysam

from .decode import apply_edit_ops
from .vocab import EDIT_TO_ID, PAD_EDIT_ID, UNCERTAINTY_TO_ID

_MATCH_OPS = {0, 7, 8}
_REF_GAP_OPS = {2, 3}
_QUERY_GAP_OPS = {1}
_QUERY_CLIP_OPS = {4}
_REF_SKIP_OPS = {5, 6}
_DNA_BASES = {"A", "C", "G", "T"}


@dataclass
class ReadEncoding:
    read_id: str
    contig: str
    target_bases: str
    target_qualities: List[int]
    target_run_lengths: List[int]
    edit_labels: List[List[int]]
    ref_positions: List[int | None]


@dataclass
class SupportProjection:
    base_by_ref: Dict[int, str]
    qual_by_ref: Dict[int, int]
    deleted_ref_positions: set[int]
    insertion_bases_by_ref: Dict[int, List[str]]
    haplotype: int
    strand: int


class IntervalLookup:
    def __init__(self, intervals: Dict[str, List[tuple[int, int]]]) -> None:
        self.intervals = {contig: sorted(ranges) for contig, ranges in intervals.items()}
        self.starts = {contig: [start for start, _ in ranges] for contig, ranges in self.intervals.items()}

    @classmethod
    def from_bed(cls, path: str | None) -> "IntervalLookup | None":
        if not path:
            return None
        intervals: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) < 3:
                    continue
                contig, start, end = fields[:3]
                intervals[contig].append((int(start), int(end)))
        return cls(intervals)

    def contains(self, contig: str, pos: int) -> bool:
        ranges = self.intervals.get(contig)
        if not ranges:
            return False
        starts = self.starts[contig]
        idx = bisect_right(starts, pos) - 1
        if idx < 0:
            return False
        start, end = ranges[idx]
        return start <= pos < end

    def covered_fraction(self, contig: str, positions: Sequence[int]) -> float:
        if not positions:
            return 0.0
        covered = sum(1 for pos in positions if self.contains(contig, pos))
        return covered / len(positions)


class VariantLookup:
    def __init__(self, path: str | None, phased_only: bool = False) -> None:
        self.path = path
        self.phased_only = phased_only
        self.vcf = pysam.VariantFile(path) if path else None

    def positions(self, contig: str, start: int, end: int) -> set[int]:
        if self.vcf is None:
            return set()
        hits: set[int] = set()
        try:
            records = self.vcf.fetch(contig, start, end)
        except (KeyError, ValueError):
            return hits
        for rec in records:
            if self.phased_only:
                samples = list(rec.samples.values())
                if not samples or not any(getattr(sample, "phased", False) for sample in samples):
                    continue
            rec_start = max(start, rec.start)
            rec_end = min(end, max(rec.stop, rec.start + 1))
            for pos in range(rec_start, rec_end):
                hits.add(pos)
        return hits


def normalize_base(base: str | None) -> str:
    if not base:
        return "N"
    base = base.upper()
    return base if base in _DNA_BASES else "N"


def cap_quality(qual: int | None, upper: int = 63) -> int:
    if qual is None:
        return 0
    return max(0, min(int(qual), upper))


def compute_run_lengths(seq: str) -> List[int]:
    out: List[int] = []
    prev = None
    run = 0
    for base in seq:
        if base == prev:
            run += 1
        else:
            run = 1
            prev = base
        out.append(run)
    return out


def detect_tandem_repeat_flags(seq: str, max_motif_len: int = 3, min_repeats: int = 2) -> List[int]:
    flags = [0] * len(seq)
    for motif_len in range(1, max_motif_len + 1):
        span = motif_len * min_repeats
        for start in range(0, max(len(seq) - span + 1, 0)):
            motif = seq[start : start + motif_len]
            if "N" in motif or len(motif) < motif_len:
                continue
            matched = True
            offset = start
            repeat_count = 0
            while offset + motif_len <= len(seq) and seq[offset : offset + motif_len] == motif:
                repeat_count += 1
                offset += motif_len
            if matched and repeat_count >= min_repeats:
                for idx in range(start, min(offset, len(seq))):
                    flags[idx] = 1
    return flags


def _banded_align_to_edit_labels(
    noisy_seq: str,
    target_seq: str,
    max_insertions_per_pos: int,
    band_width: int = 12,
) -> List[List[int]]:
    n = len(noisy_seq)
    m = len(target_seq)
    inf = 10 ** 9
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: List[List[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        if i <= band_width:
            dp[i][0] = i
            back[i][0] = (i - 1, 0, "D")
    for j in range(1, m + 1):
        if j <= band_width:
            dp[0][j] = j
            back[0][j] = (0, j - 1, "I")

    for i in range(1, n + 1):
        j_lo = max(1, i - band_width)
        j_hi = min(m, i + band_width)
        for j in range(j_lo, j_hi + 1):
            candidates: List[tuple[int, tuple[int, int, str]]] = []
            sub_cost = 0 if noisy_seq[i - 1] == target_seq[j - 1] else 1
            if dp[i - 1][j - 1] < inf:
                candidates.append((dp[i - 1][j - 1] + sub_cost, (i - 1, j - 1, "M")))
            if dp[i - 1][j] < inf:
                candidates.append((dp[i - 1][j] + 1, (i - 1, j, "D")))
            if dp[i][j - 1] < inf:
                candidates.append((dp[i][j - 1] + 1, (i, j - 1, "I")))
            # Prefer deletions, then insertions, then substitutions on ties to left-normalize.
            score, prev = min(
                candidates,
                key=lambda item: (
                    item[0],
                    {"D": 0, "I": 1, "M": 2}[item[1][2]],
                ),
            )
            dp[i][j] = score
            back[i][j] = prev

    i, j = n, m
    ops: List[str] = []
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            if i > 0:
                prev = (i - 1, j, "D")
            else:
                prev = (i, j - 1, "I")
        pi, pj, op = prev
        ops.append(op)
        i, j = pi, pj
    ops.reverse()

    edit_labels: List[List[int]] = []
    pending_insertions: List[int] = []
    noisy_idx = 0
    target_idx = 0
    for op in ops:
        if op == "I":
            if target_idx < len(target_seq):
                target_base = normalize_base(target_seq[target_idx])
                if target_base in _DNA_BASES:
                    pending_insertions.append(EDIT_TO_ID[f"INS_{target_base}"])
            target_idx += 1
            continue
        if noisy_idx >= len(noisy_seq):
            break
        noisy_base = normalize_base(noisy_seq[noisy_idx])
        if op == "D":
            core_label = EDIT_TO_ID["DEL"]
        else:
            target_base = normalize_base(target_seq[target_idx]) if target_idx < len(target_seq) else noisy_base
            core_label = EDIT_TO_ID["COPY"] if noisy_base == target_base or target_base == "N" else EDIT_TO_ID[f"SUB_{target_base}"]
            target_idx += 1
        edit_labels.append(_pad_insertion_labels(pending_insertions, max_insertions_per_pos) + [core_label])
        pending_insertions = []
        noisy_idx += 1

    while len(edit_labels) < len(noisy_seq):
        edit_labels.append([PAD_EDIT_ID] * max_insertions_per_pos + [EDIT_TO_ID["COPY"]])
    return edit_labels


def canonicalize_indel_windows(
    noisy_seq: str,
    edit_labels: List[List[int]],
    max_insertions_per_pos: int,
    flank: int = 6,
    band_width: int = 12,
) -> List[List[int]]:
    hard_positions = [
        idx
        for idx, slots in enumerate(edit_labels)
        if (slots and slots[-1] == EDIT_TO_ID["DEL"]) or any(token != PAD_EDIT_ID for token in slots[:-1])
    ]
    if not hard_positions:
        return edit_labels

    corrected_seq = apply_edit_ops(noisy_seq, edit_labels, max_insertions_per_pos=max_insertions_per_pos)
    canonical_full = _banded_align_to_edit_labels(
        noisy_seq,
        corrected_seq,
        max_insertions_per_pos=max_insertions_per_pos,
        band_width=band_width,
    )
    normalized = [list(slots) for slots in edit_labels]
    for pos in hard_positions:
        start = max(0, pos - flank)
        end = min(len(edit_labels), pos + flank + 1)
        for idx in range(start, end):
            normalized[idx] = canonical_full[idx]
    return normalized


def one_hot_base(base: str) -> List[float]:
    if base == "A":
        return [1.0, 0.0, 0.0, 0.0]
    if base == "C":
        return [0.0, 1.0, 0.0, 0.0]
    if base == "G":
        return [0.0, 0.0, 1.0, 0.0]
    if base == "T":
        return [0.0, 0.0, 0.0, 1.0]
    return [0.0, 0.0, 0.0, 0.0]


def base_count_vector(bases: Sequence[str]) -> List[float]:
    counts = [0.0, 0.0, 0.0, 0.0]
    for base in bases:
        base = normalize_base(base)
        if base == "A":
            counts[0] += 1.0
        elif base == "C":
            counts[1] += 1.0
        elif base == "G":
            counts[2] += 1.0
        elif base == "T":
            counts[3] += 1.0
    return counts


def extract_haplotype_tag(aln: pysam.AlignedSegment) -> int:
    try:
        hp = int(aln.get_tag("HP"))
    except (KeyError, TypeError, ValueError):
        return 0
    return hp if hp in {1, 2} else 0


def _pad_insertion_labels(tokens: List[int], max_insertions_per_pos: int) -> List[int]:
    clipped = tokens[:max_insertions_per_pos]
    return clipped + [PAD_EDIT_ID] * (max_insertions_per_pos - len(clipped))


def build_read_encoding(
    aln: pysam.AlignedSegment,
    fasta: pysam.FastaFile,
    max_insertions_per_pos: int,
) -> ReadEncoding | None:
    if (
        aln.is_unmapped
        or aln.is_secondary
        or aln.is_supplementary
        or not aln.cigartuples
        or aln.query_sequence is None
        or aln.reference_name is None
        or aln.reference_start is None
        or aln.reference_end is None
    ):
        return None

    query = aln.query_sequence
    quals = list(aln.query_qualities or [0] * len(query))
    ref_seq = fasta.fetch(aln.reference_name, aln.reference_start, aln.reference_end).upper()

    q_idx = 0
    r_idx = aln.reference_start
    pending_ref_insertions: List[int] = []

    target_bases: List[str] = []
    target_qualities: List[int] = []
    edit_labels: List[List[int]] = []
    ref_positions: List[int | None] = []

    for op, length in aln.cigartuples:
        if op in _QUERY_CLIP_OPS:
            q_idx += length
            continue
        if op in _MATCH_OPS:
            for _ in range(length):
                q_base = normalize_base(query[q_idx])
                ref_base = normalize_base(ref_seq[r_idx - aln.reference_start])
                core_label = EDIT_TO_ID["COPY"] if q_base == ref_base or ref_base == "N" else EDIT_TO_ID[f"SUB_{ref_base}"]
                target_bases.append(q_base)
                target_qualities.append(cap_quality(quals[q_idx]))
                edit_labels.append(_pad_insertion_labels(pending_ref_insertions, max_insertions_per_pos) + [core_label])
                ref_positions.append(r_idx)
                pending_ref_insertions = []
                q_idx += 1
                r_idx += 1
            continue
        if op in _QUERY_GAP_OPS:
            for _ in range(length):
                target_bases.append(normalize_base(query[q_idx]))
                target_qualities.append(cap_quality(quals[q_idx]))
                edit_labels.append(_pad_insertion_labels(pending_ref_insertions, max_insertions_per_pos) + [EDIT_TO_ID["DEL"]])
                ref_positions.append(None)
                pending_ref_insertions = []
                q_idx += 1
            continue
        if op in _REF_GAP_OPS:
            for _ in range(length):
                ref_base = normalize_base(ref_seq[r_idx - aln.reference_start])
                if ref_base in _DNA_BASES:
                    pending_ref_insertions.append(EDIT_TO_ID[f"INS_{ref_base}"])
                r_idx += 1
            continue
        if op in _REF_SKIP_OPS:
            continue
        if op in {5, 6}:
            continue
        raise ValueError(f"Unsupported CIGAR op {op} for read {aln.query_name}")

    if not target_bases:
        return None
    target_seq = "".join(target_bases)
    canonical_edit_labels = canonicalize_indel_windows(
        target_seq,
        edit_labels,
        max_insertions_per_pos=max_insertions_per_pos,
    )
    return ReadEncoding(
        read_id=aln.query_name,
        contig=aln.reference_name,
        target_bases=target_seq,
        target_qualities=target_qualities,
        target_run_lengths=compute_run_lengths(target_seq),
        edit_labels=canonical_edit_labels,
        ref_positions=ref_positions,
    )


def build_support_projection(aln: pysam.AlignedSegment) -> SupportProjection | None:
    if (
        aln.is_unmapped
        or aln.is_secondary
        or aln.is_supplementary
        or not aln.cigartuples
        or aln.query_sequence is None
        or aln.reference_name is None
        or aln.reference_start is None
    ):
        return None

    query = aln.query_sequence
    quals = list(aln.query_qualities or [0] * len(query))
    q_idx = 0
    r_idx = aln.reference_start
    pending_insertion_bases: List[str] = []

    base_by_ref: Dict[int, str] = {}
    qual_by_ref: Dict[int, int] = {}
    deleted_ref_positions: set[int] = set()
    insertion_bases_by_ref: Dict[int, List[str]] = defaultdict(list)

    def flush_pending_insertion(anchor_ref_pos: int) -> None:
        nonlocal pending_insertion_bases
        if not pending_insertion_bases:
            return
        insertion_bases_by_ref[anchor_ref_pos].extend(pending_insertion_bases)
        pending_insertion_bases = []

    for op, length in aln.cigartuples:
        if op in _QUERY_CLIP_OPS:
            q_idx += length
            continue
        if op in _MATCH_OPS:
            for _ in range(length):
                flush_pending_insertion(r_idx)
                base_by_ref[r_idx] = normalize_base(query[q_idx])
                qual_by_ref[r_idx] = cap_quality(quals[q_idx])
                q_idx += 1
                r_idx += 1
            continue
        if op in _QUERY_GAP_OPS:
            for _ in range(length):
                pending_insertion_bases.append(normalize_base(query[q_idx]))
                q_idx += 1
            continue
        if op in _REF_GAP_OPS:
            for _ in range(length):
                flush_pending_insertion(r_idx)
                deleted_ref_positions.add(r_idx)
                r_idx += 1
            continue
        if op in _REF_SKIP_OPS:
            continue
        if op in {5, 6}:
            continue
        raise ValueError(f"Unsupported CIGAR op {op} for support read {aln.query_name}")

    return SupportProjection(
        base_by_ref=base_by_ref,
        qual_by_ref=qual_by_ref,
        deleted_ref_positions=deleted_ref_positions,
        insertion_bases_by_ref=dict(insertion_bases_by_ref),
        haplotype=extract_haplotype_tag(aln),
        strand=-1 if aln.is_reverse else 1,
    )


def fetch_support_alignments(
    bam: pysam.AlignmentFile,
    target_aln: pysam.AlignedSegment,
    window_ref_start: int,
    window_ref_end: int,
    max_supports: int,
    min_mapq: int,
) -> List[pysam.AlignedSegment]:
    candidates: List[tuple[int, int, pysam.AlignedSegment]] = []
    for aln in bam.fetch(target_aln.reference_name, window_ref_start, window_ref_end, multiple_iterators=True):
        if (
            aln.is_unmapped
            or aln.is_secondary
            or aln.is_supplementary
            or aln.mapping_quality < min_mapq
            or aln.query_name == target_aln.query_name
            or aln.reference_start is None
            or aln.reference_end is None
        ):
            continue
        overlap = min(window_ref_end, aln.reference_end) - max(window_ref_start, aln.reference_start)
        if overlap <= 0:
            continue
        candidates.append((overlap, aln.mapping_quality, aln))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [aln for _, _, aln in candidates[:max_supports]]


def project_support_onto_window(
    projection: SupportProjection,
    window_ref_positions: Sequence[int | None],
    target_window_bases: str,
    target_haplotype: int,
) -> dict:
    support_chars: List[str] = []
    support_quals: List[int] = []
    support_match_mask: List[int] = []
    support_ins_mask: List[int] = []
    support_del_mask: List[int] = []
    support_base_support: List[List[float]] = []
    support_ins_base_support: List[List[float]] = []
    support_strand: List[float] = []
    support_haplotype: List[float] = []
    support_same_haplotype: List[float] = []

    for ref_pos, target_base in zip(window_ref_positions, target_window_bases):
        support_strand.append(float(projection.strand))
        support_haplotype.append(float(projection.haplotype))
        support_same_haplotype.append(float(target_haplotype > 0 and projection.haplotype == target_haplotype))
        if ref_pos is None:
            support_chars.append("N")
            support_quals.append(0)
            support_match_mask.append(0)
            support_ins_mask.append(0)
            support_del_mask.append(0)
            support_base_support.append([0.0, 0.0, 0.0, 0.0])
            support_ins_base_support.append([0.0, 0.0, 0.0, 0.0])
            continue

        inserted_bases = projection.insertion_bases_by_ref.get(ref_pos, [])
        support_ins_mask.append(int(bool(inserted_bases)))
        support_ins_base_support.append(base_count_vector(inserted_bases))
        if ref_pos in projection.base_by_ref:
            base = projection.base_by_ref[ref_pos]
            support_chars.append(base)
            support_quals.append(projection.qual_by_ref[ref_pos])
            support_match_mask.append(int(base == target_base))
            support_del_mask.append(0)
            support_base_support.append(one_hot_base(base))
            continue
        if ref_pos in projection.deleted_ref_positions:
            support_chars.append("N")
            support_quals.append(0)
            support_match_mask.append(0)
            support_del_mask.append(1)
            support_base_support.append([0.0, 0.0, 0.0, 0.0])
            continue
        support_chars.append("N")
        support_quals.append(0)
        support_match_mask.append(0)
        support_del_mask.append(0)
        support_base_support.append([0.0, 0.0, 0.0, 0.0])

    return {
        "support_bases": "".join(support_chars),
        "support_qualities": support_quals,
        "support_match_mask": support_match_mask,
        "support_ins_mask": support_ins_mask,
        "support_del_mask": support_del_mask,
        "support_base_support": support_base_support,
        "support_ins_base_support": support_ins_base_support,
        "support_strand": support_strand,
        "support_haplotype": support_haplotype,
        "support_same_haplotype": support_same_haplotype,
    }


def infer_uncertainty_labels(
    contig: str,
    window_ref_positions: Sequence[int | None],
    support_base_support: Sequence[Sequence[Sequence[float]]],
    support_del_mask: Sequence[Sequence[int]],
    variant_positions: set[int],
    disagreement_threshold: float,
    min_support_depth: int,
) -> tuple[List[int], List[int]]:
    preserve_mask: List[int] = []
    uncertainty_labels: List[int] = []
    num_positions = len(window_ref_positions)

    for i in range(num_positions):
        ref_pos = window_ref_positions[i]
        counts = [0.0, 0.0, 0.0, 0.0]
        del_count = 0.0
        for support_dist in support_base_support:
            pos_dist = support_dist[i]
            for j in range(4):
                counts[j] += pos_dist[j]
        for del_row in support_del_mask:
            del_count += float(del_row[i])
        core_counts = counts + [del_count]
        depth = int(sum(core_counts))
        top_fraction = max(core_counts) / depth if depth else 0.0
        is_variant = ref_pos is not None and ref_pos in variant_positions

        if is_variant:
            preserve_mask.append(1)
            uncertainty_labels.append(UNCERTAINTY_TO_ID["variant_supported"])
        elif ref_pos is None or depth < min_support_depth or top_fraction < disagreement_threshold:
            preserve_mask.append(1 if depth >= min_support_depth and ref_pos is not None else 0)
            uncertainty_labels.append(UNCERTAINTY_TO_ID["ambiguous"])
        else:
            preserve_mask.append(0)
            uncertainty_labels.append(UNCERTAINTY_TO_ID["error"])

    return preserve_mask, uncertainty_labels


def build_deletion_targets(
    window_edit_labels: Sequence[Sequence[int]],
    max_deletion_length: int,
) -> tuple[List[int], List[int]]:
    candidate_labels = [0] * len(window_edit_labels)
    length_labels = [0] * len(window_edit_labels)
    i = 0
    while i < len(window_edit_labels):
        is_del = bool(window_edit_labels[i]) and window_edit_labels[i][-1] == EDIT_TO_ID["DEL"]
        if not is_del:
            i += 1
            continue
        start = i
        while i < len(window_edit_labels) and window_edit_labels[i][-1] == EDIT_TO_ID["DEL"]:
            i += 1
        run_len = i - start
        candidate_labels[start] = 1
        length_labels[start] = min(run_len, max_deletion_length)
    return candidate_labels, length_labels


def compute_gap_length_histogram(
    support_del_mask: Sequence[Sequence[int]],
    max_deletion_length: int,
) -> List[List[float]]:
    if not support_del_mask:
        return []
    seq_len = len(support_del_mask[0])
    hist = [[0.0 for _ in range(max_deletion_length)] for _ in range(seq_len)]
    for row in support_del_mask:
        idx = 0
        while idx < seq_len:
            if not row[idx]:
                idx += 1
                continue
            start = idx
            while idx < seq_len and row[idx]:
                idx += 1
            run_len = idx - start
            bucket = min(run_len, max_deletion_length) - 1
            for pos in range(start, idx):
                hist[pos][bucket] += 1.0
    return hist


def compute_local_support_features(
    support_base_support: Sequence[Sequence[Sequence[float]]],
    support_del_mask: Sequence[Sequence[int]],
    max_deletion_length: int,
) -> dict:
    if not support_base_support:
        return {
            "deletion_support_count": [],
            "deletion_support_fraction": [],
            "local_support_entropy": [],
            "local_support_agreement": [],
            "local_support_depth": [],
            "gap_length_histogram": [],
        }
    seq_len = len(support_base_support[0])
    gap_hist = compute_gap_length_histogram(support_del_mask, max_deletion_length)
    deletion_support_count: List[float] = []
    deletion_support_fraction: List[float] = []
    local_support_entropy: List[float] = []
    local_support_agreement: List[float] = []
    local_support_depth: List[float] = []
    for pos in range(seq_len):
        base_counts = [0.0, 0.0, 0.0, 0.0]
        del_count = 0.0
        for row in support_base_support:
            for base_idx in range(4):
                base_counts[base_idx] += float(row[pos][base_idx])
        for del_row in support_del_mask:
            del_count += float(del_row[pos])
        core_counts = base_counts + [del_count]
        depth = sum(core_counts)
        if depth <= 0:
            deletion_support_count.append(0.0)
            deletion_support_fraction.append(0.0)
            local_support_entropy.append(0.0)
            local_support_agreement.append(0.0)
            local_support_depth.append(0.0)
            continue
        probs = [count / depth for count in core_counts]
        entropy = 0.0
        for prob in probs:
            if prob > 0.0:
                entropy -= prob * math.log(prob)
        entropy /= math.log(len(core_counts))
        deletion_support_count.append(del_count)
        deletion_support_fraction.append(del_count / depth)
        local_support_entropy.append(entropy)
        local_support_agreement.append(max(probs))
        local_support_depth.append(depth)
    return {
        "deletion_support_count": deletion_support_count,
        "deletion_support_fraction": deletion_support_fraction,
        "local_support_entropy": local_support_entropy,
        "local_support_agreement": local_support_agreement,
        "local_support_depth": local_support_depth,
        "gap_length_histogram": gap_hist,
    }


def build_region_masks(
    contig: str,
    window_ref_positions: Sequence[int | None],
    region_lookups: Mapping[str, IntervalLookup | None],
    target_run_lengths: Sequence[int],
) -> Dict[str, List[int]]:
    masks: Dict[str, List[int]] = {}
    for name, lookup in region_lookups.items():
        mask: List[int] = []
        for ref_pos in window_ref_positions:
            mask.append(int(ref_pos is not None and lookup is not None and lookup.contains(contig, ref_pos)))
        masks[name] = mask
    if "homopolymer" not in masks:
        masks["homopolymer"] = [int(run_len >= 3) for run_len in target_run_lengths]
    return masks


def iter_windows(length: int, window_size: int, window_overlap: int, min_window_size: int) -> Iterator[tuple[int, int]]:
    step = max(1, window_size - window_overlap)
    start = 0
    while start < length:
        end = min(length, start + window_size)
        if end - start >= min_window_size:
            yield start, end
        if end == length:
            break
        start += step


def split_name_for_contig(
    contig: str,
    train_contigs: set[str],
    val_contigs: set[str],
    test_contigs: set[str],
    read_id: str | None = None,
) -> str | None:
    memberships = {
        split_name
        for split_name, split_contigs in (
            ("train", train_contigs),
            ("val", val_contigs),
            ("test", test_contigs),
        )
        if contig in split_contigs
    }
    if not memberships:
        if not train_contigs:
            return "train"
        return None

    if memberships == {"train", "val", "test"}:
        if not read_id:
            return "train"
        digest = sum(ord(ch) for ch in read_id) % 10
        if digest == 0:
            return "test"
        if digest == 1:
            return "val"
        return "train"
    if memberships == {"val", "test"}:
        if not read_id:
            return "val"
        digest = sum(ord(ch) for ch in read_id) % 2
        return "val" if digest == 0 else "test"
    if memberships == {"train", "val"}:
        if not read_id:
            return "train"
        digest = sum(ord(ch) for ch in read_id) % 10
        return "val" if digest == 0 else "train"
    if memberships == {"train", "test"}:
        if not read_id:
            return "train"
        digest = sum(ord(ch) for ch in read_id) % 10
        return "test" if digest == 0 else "train"
    if "test" in memberships:
        return "test"
    if "val" in memberships:
        return "val"
    if "train" in memberships or not train_contigs:
        return "train"
    return None


def build_window_example(
    target_aln: pysam.AlignedSegment,
    read_encoding: ReadEncoding,
    support_bam: pysam.AlignmentFile,
    variant_lookup: VariantLookup,
    phased_variant_lookup: VariantLookup | None,
    confidence_lookup: IntervalLookup | None,
    region_lookups: Mapping[str, IntervalLookup | None],
    window_start: int,
    window_end: int,
    max_supports: int,
    min_supports_per_window: int,
    min_mapq: int,
    min_confident_fraction: float,
    min_mapped_fraction: float,
    support_disagreement_threshold: float,
    min_support_depth: int,
    max_insertions_per_pos: int,
    max_deletion_length: int = 4,
) -> dict | None:
    window_ref_positions = read_encoding.ref_positions[window_start:window_end]
    mapped_positions = [pos for pos in window_ref_positions if pos is not None]
    if not mapped_positions:
        return None
    if len(mapped_positions) / len(window_ref_positions) < min_mapped_fraction:
        return None

    window_ref_start = min(mapped_positions)
    window_ref_end = max(mapped_positions) + 1
    if confidence_lookup is not None:
        if confidence_lookup.covered_fraction(read_encoding.contig, mapped_positions) < min_confident_fraction:
            return None

    support_alignments = fetch_support_alignments(
        bam=support_bam,
        target_aln=target_aln,
        window_ref_start=window_ref_start,
        window_ref_end=window_ref_end,
        max_supports=max_supports,
        min_mapq=min_mapq,
    )
    if not support_alignments:
        return None

    target_window_bases = read_encoding.target_bases[window_start:window_end]
    target_haplotype = extract_haplotype_tag(target_aln)
    support_rows = []
    support_base_support = []
    support_del_masks = []
    for aln in support_alignments:
        projection = build_support_projection(aln)
        if projection is None:
            continue
        row = project_support_onto_window(
            projection,
            window_ref_positions,
            target_window_bases,
            target_haplotype=target_haplotype,
        )
        support_rows.append(row)
        support_base_support.append(row["support_base_support"])
        support_del_masks.append(row["support_del_mask"])
    if len(support_rows) < min_supports_per_window:
        return None

    variant_positions = variant_lookup.positions(read_encoding.contig, window_ref_start, window_ref_end)
    phased_variant_positions = (
        phased_variant_lookup.positions(read_encoding.contig, window_ref_start, window_ref_end)
        if phased_variant_lookup is not None
        else set()
    )
    preserve_mask, uncertainty_labels = infer_uncertainty_labels(
        contig=read_encoding.contig,
        window_ref_positions=window_ref_positions,
        support_base_support=support_base_support,
        support_del_mask=support_del_masks,
        variant_positions=variant_positions,
        disagreement_threshold=support_disagreement_threshold,
        min_support_depth=min_support_depth,
    )
    variant_mask = [int(ref_pos is not None and ref_pos in variant_positions) for ref_pos in window_ref_positions]
    phased_variant_mask = [int(ref_pos is not None and ref_pos in phased_variant_positions) for ref_pos in window_ref_positions]
    region_masks = build_region_masks(
        contig=read_encoding.contig,
        window_ref_positions=window_ref_positions,
        region_lookups=region_lookups,
        target_run_lengths=read_encoding.target_run_lengths[window_start:window_end],
    )
    if "variant_rich" not in region_masks:
        region_masks["variant_rich"] = variant_mask
    if "tandem_repeat" not in region_masks:
        region_masks["tandem_repeat"] = detect_tandem_repeat_flags(target_window_bases)

    window_edit_labels = read_encoding.edit_labels[window_start:window_end]
    if any(len(labels) != max_insertions_per_pos + 1 for labels in window_edit_labels):
        return None
    deletion_candidate_labels, deletion_length_labels = build_deletion_targets(
        window_edit_labels,
        max_deletion_length=max_deletion_length,
    )
    local_support_features = compute_local_support_features(
        support_base_support,
        support_del_masks,
        max_deletion_length=max_deletion_length,
    )

    return {
        "read_id": f"{read_encoding.read_id}:{window_start}-{window_end}",
        "source_read_id": read_encoding.read_id,
        "contig": read_encoding.contig,
        "window_ref_start": window_ref_start,
        "window_ref_end": window_ref_end,
        "target_bases": target_window_bases,
        "target_qualities": read_encoding.target_qualities[window_start:window_end],
        "target_run_lengths": read_encoding.target_run_lengths[window_start:window_end],
        "target_haplotype": target_haplotype,
        "support_bases": [row["support_bases"] for row in support_rows],
        "support_match_mask": [row["support_match_mask"] for row in support_rows],
        "support_ins_mask": [row["support_ins_mask"] for row in support_rows],
        "support_del_mask": [row["support_del_mask"] for row in support_rows],
        "support_qualities": [row["support_qualities"] for row in support_rows],
        "support_base_support": [row["support_base_support"] for row in support_rows],
        "support_ins_base_support": [row["support_ins_base_support"] for row in support_rows],
        "support_strand": [row["support_strand"] for row in support_rows],
        "support_haplotype": [row["support_haplotype"] for row in support_rows],
        "support_same_haplotype": [row["support_same_haplotype"] for row in support_rows],
        "target_sequence": apply_edit_ops(target_window_bases, window_edit_labels, max_insertions_per_pos=max_insertions_per_pos),
        "edit_labels": window_edit_labels,
        "deletion_candidate_labels": deletion_candidate_labels,
        "deletion_length_labels": deletion_length_labels,
        "variant_mask": variant_mask,
        "phased_variant_mask": phased_variant_mask,
        "region_masks": region_masks,
        "preserve_mask": preserve_mask,
        "uncertainty_labels": uncertainty_labels,
        "tandem_repeat_flag": region_masks["tandem_repeat"],
        "deletion_support_count": local_support_features["deletion_support_count"],
        "deletion_support_fraction": local_support_features["deletion_support_fraction"],
        "local_support_entropy": local_support_features["local_support_entropy"],
        "local_support_agreement": local_support_features["local_support_agreement"],
        "local_support_depth": local_support_features["local_support_depth"],
        "gap_length_histogram": local_support_features["gap_length_histogram"],
    }


def write_examples_jsonl(examples: Iterable[dict], path: str | Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")
            count += 1
    return count
