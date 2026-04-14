# OMEGA Long-Read Correction (PyTorch)

This repository contains a research-oriented PyTorch implementation of an overlap-conditioned monotonic edit model for long-read error correction.

## What is implemented

- Target-read encoder with local convolution + Transformer blocks
- Support-read encoder with overlap-aware features
- Coverage-aware cross-attention from target positions to retrieved overlap pileups
- Monotonic edit decoder in edit space with insertion slots per noisy base
- Preservation / overcorrection-aware losses
- Support consistency regularization
- Uncertainty head
- Dataset and batching utilities for windowed training
- Real-data preprocessing from aligned long reads
- Training and held-out test scripts

## Installation

```bash
pip install -r requirements.txt
```

## Real-data preprocessing

The preprocessing entrypoint is:

```bash
PYTHONPATH=src python scripts/preprocess_real_data.py \
  --reads-bam /path/to/ont_reads.bam \
  --reference-fasta /path/to/reference.fa \
  --truth-vcf /path/to/giab_truth.vcf.gz \
  --confident-bed /path/to/high_confidence.bed \
  --output-dir data/real \
  --window-size 1024 \
  --window-overlap 256 \
  --max-supports 16 \
  --max-insertions-per-pos 2 \
  --min-mapq 20 \
  --train-contigs chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16 \
  --val-contigs chr17,chr18,chr19 \
  --test-contigs chr20,chr21,chr22,chrX
```

Required inputs:

- coordinate-sorted, indexed long-read `BAM` or `CRAM`
- indexed reference `FASTA`

Optional truth resources:

- indexed truth `VCF` to mark preservation-sensitive loci
- confidence `BED` to keep only benchmarkable windows

The script writes `train.jsonl`, `val.jsonl`, `test.jsonl`, and `manifest.json` under the chosen output directory.

## Training

```bash
PYTHONPATH=src python scripts/train.py --config configs/real_data.yaml
```

## Testing

```bash
PYTHONPATH=src python scripts/test.py \
  --config configs/real_data.yaml \
  --checkpoint checkpoints/real_data/best_model.pt \
  --predictions-out outputs/test_predictions.jsonl \
  --summary-out outputs/test_summary.json
```

## Data format

Each training item should serialize to a JSON line with fields similar to:

```json
{
  "read_id": "read_001",
  "target_bases": "ACGTACGT...",
  "target_qualities": [30, 31, 29, ...],
  "target_run_lengths": [1,1,1,2,...],
  "contig": "chr20",
  "window_ref_start": 1000123,
  "window_ref_end": 1001147,
  "support_bases": ["ACGT...", "AC-T...", ...],
  "support_match_mask": [[1,1,0,...], [1,0,1,...]],
  "support_ins_mask": [[0,0,1,...], [0,0,0,...]],
  "support_del_mask": [[0,1,0,...], [0,0,1,...]],
  "support_qualities": [[20,25,...], [18,19,...]],
  "support_base_support": [[0.1,0.7,0.1,0.1], ...],
  "target_sequence": "ACGTACGT...",
  "edit_labels": [[10,10,0], [10,10,4], [6,10,0], ...],
  "preserve_mask": [0,0,1,1,...],
  "uncertainty_labels": [0,0,2,1,...]
}
```

`edit_labels` are aligned to noisy-read positions and use `max_insertions_per_pos + 1` slots per position:

- insertion slots before the noisy base
- one core slot for `COPY`, `SUB_*`, or `DEL`

The preprocessing script fills these labels directly from the read-vs-reference CIGAR path.

## Suggested ablations

- plain seq2seq baseline
- edit decoder only
- edit decoder + support encoder
- support encoder + preservation loss
- full model
