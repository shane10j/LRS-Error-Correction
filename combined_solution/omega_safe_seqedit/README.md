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

## Requested Validation Ladder

The notebook exposes these switches. The benchmark-plan section defaults to the recommended HG002 old 12x run; turn that switch off if you only want to inspect existing outputs.

1. `synthetic_noisy_large_multiseed`: seed 47/48 large noisy validation, intended to check whether false `DEL/SUB/INS` errors drop at scale.
2. `synthetic_noisy_small_curated`: fast safety gate with clean copy, false-support, neighboring-edit, homopolymer, boundary insertion, and true hard-edit cases.
3. `false_del_regression`: fixed noisy-neighbor/homopolymer false-deletion regression suite.
4. `local_real_mac`: first real BAM benchmark.
5. `hg002_chr20_mac`: first project-report HG002 chr20 benchmark.

For the most useful local HG002/SOTA-style comparisons, use the pre-windowed old HG002 chr20 presets:

- `configs/hg002_old_8x_mac.yaml`: low-coverage stress test.
- `configs/hg002_old_12x_mac.yaml`: recommended first substantial benchmark.
- `configs/hg002_old_20x_mac.yaml`: high-support validation.

These import the old solution's already-materialized HG002 chr20 JSONL windows and point to the existing old BAM/reference/truth assets. They cap examples by default for Mac runtime; remove `data.max_examples` in the YAML files to use the full old splits.

Run the multiseed synthetic validation directly with:

```bash
python scripts/run_multiseed_benchmark.py --config configs/synthetic_noisy_large_multiseed.yaml
```

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
- support-rule gap table: support rule was correct but hybrid missed
- vetoed true-edit table: neural would have been correct but hybrid vetoed
- neural-vs-rule confusion counts
- hybrid contribution counts: forced by rule, neural agreed, rescued by neural
- false `SUB/INS/DEL` counts, neighbor-induced false edits, and homopolymer false deletions
- HG002 false-edit context counts for variant masks, phased variants, preserve/low-confidence regions, repeat-rich regions, support fraction/margin, entropy, and consensus agreement

## HG002 Safety Audit And Calibration

The notebook now treats HG002 false-substitution reduction as the main optimization target, not synthetic recall. After running `hg002_old_12x_mac`, use the HG002 audit section to inspect every full-hybrid false edit by mechanism:

- possible true variant or haplotype-preservation site
- repeat or homopolymer ambiguity
- neighboring-edit/alignment ambiguity
- support-rule false positive
- neural hallucination or unsafe rescue

The calibration cells export candidate edits with real-data safety labels:

```bash
python scripts/build_allow_gate_dataset.py \
  --predictions outputs/hg002_old_12x_mac/runs/full/test_hybrid_predictions.jsonl \
  --output outputs/hg002_old_12x_mac/calibration/allow_gate_candidates.jsonl \
  --summary-output outputs/hg002_old_12x_mac/calibration/allow_gate_candidate_summary.json

python scripts/calibrate_allow_gate.py \
  --candidates outputs/hg002_old_12x_mac/calibration/allow_gate_candidates.jsonl \
  --output outputs/hg002_old_12x_mac/calibration/allow_gate_thresholds.json \
  --max-false-positives 0

python scripts/train_allow_gate.py \
  --candidates outputs/hg002_old_12x_mac/calibration/allow_gate_candidates.jsonl \
  --output outputs/hg002_old_12x_mac/calibration/learned_allow_gate.json \
  --max-false-positives 0 \
  --max-false-positive-rate 0.001
```

HG002 configs intentionally stop forcing support-rule substitutions by default. A support-rule `SUB` is treated as a candidate and must pass stricter repeat/neighbor/variant safety checks plus local-window support reranking and neural agreement.

HG002 indels now have hard safety vetoes:

- support-rule `INS` requires normalized top inserted-base fraction >= 0.90 and inserted-base margin >= 2; all real-data insertions require a learned allow gate, and repeat/tandem insertions are hard-vetoed unless the gate explicitly approves.
- support-rule `DEL` is vetoed in repeat/tandem, neighbor, zero-base-support, or homopolymer-run >= 2 contexts and requires high deletion fraction in non-repeat contexts.
- ambiguous `INS/DEL` candidates run a narrow local reranker before they can be applied.
- insertion payload fractions are normalized against insertion support, so audit values are bounded by 1.
- the notebook exports a `would_have_corrected_table` for true support-rule edits that were vetoed, which is the recovery list to inspect after false edits are near zero.
- SUB recovery is disabled by default because the hand-tuned rule was unsafe on HG002.
- `sub_recovery` is now an explicit A/B notebook branch gated by a learned SUB-only allow classifier; without that classifier it should abstain.
- The notebook exports `support_rule_sub_candidates.jsonl`, a false-SUB vs true-SUB contrast table with support evidence, neural probabilities, repeat/variant/confidence context, local mismatch density, window coordinate, strand balance, and haplotype-consistency features.

External SOTA comparisons should wait until `full_hybrid` beats `no_edit`, conservative consensus, and `support_rule` on HG002 usable score with near-zero false edits.

## Design Intent

This codebase is meant to answer one narrow question:

Can a sequence-aware but conservative edit-script model beat support consensus on real long-read correction while keeping overcorrection close to zero?

It is intentionally Mac-runnable first. Larger SOTA comparisons should be imported as external corrected FASTA/FASTQ outputs rather than reproduced inside this lightweight repo.
