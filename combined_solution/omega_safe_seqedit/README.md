# Omega Safe SeqEdit

Omega Safe SeqEdit is a combined long-read error-correction scaffold that blends the two earlier project directions:

- the old solution's sequence-aware target/support encoding;
- the new solution's factorized edit heads, support-rule diagnostics, conservative hybrid decoding, and overcorrection-first evaluation.

The model is not a free-form seq2seq translator. It is a monotonic sequence-to-edit-sequence corrector: for each target-read position it predicts whether to copy, substitute, delete, or insert before that position, then a conservative decoder decides whether the edit is safe enough to apply.

## Install

Use the existing conda environment if available:

```bash
conda activate lrs_err_correct_env
pip install -e combined_solution/omega_safe_seqedit
```

Or install from scratch:

```bash
pip install -r combined_solution/omega_safe_seqedit/requirements.txt
pip install -e combined_solution/omega_safe_seqedit
```

## Quick Mac Smoke Run

```bash
cd combined_solution/omega_safe_seqedit
python scripts/preprocess_dataset.py --config configs/mac_debug.yaml
python scripts/run_baselines.py --config configs/mac_debug.yaml
python scripts/train_model.py --config configs/mac_debug.yaml --run full
python scripts/evaluate_predictions.py --config configs/mac_debug.yaml --run full
```

The default debug preset is synthetic and intentionally tiny.

## Real Dataset Path

Edit `configs/local_real_mac.yaml` and set:

- `data.bam`
- `data.reference_fasta`
- `data.contig`
- `data.start`
- `data.end`
- optional `external_baselines`

Then run the notebook:

```bash
jupyter notebook notebooks/safe_seqedit_real_world.ipynb
```

The notebook can run:

- `no_edit`
- conservative support consensus
- support rule baseline
- target-only SeqEdit
- full support-conditioned SeqEdit
- external corrected FASTA/FASTQ/JSONL predictions

## Main Metrics

Every evaluation reports:

- identity
- edit distance
- normalized edit distance
- predicted length ratio
- overcorrection rate
- hard-edit false-positive rate
- substitution/deletion/insertion precision, recall, and F1
- per-base substitution/insertion recall
- stratified metrics for homopolymer, tandem-repeat, high-entropy, low-agreement, boundary, and neighboring-edit contexts
- false-edit tables for safety debugging

## Design Intent

This codebase is meant to answer one narrow question:

Can a sequence-aware but conservative edit-script model beat support consensus on real long-read correction while keeping overcorrection close to zero?

It is intentionally Mac-runnable first. Larger SOTA comparisons should be imported as external corrected FASTA/FASTQ outputs rather than reproduced inside this lightweight repo.
