# omega_lr_rebuild

Small, standalone long-read error-correction research codebase rebuilt for fast iteration, conservative edits, and explicit deletion handling.

## Environment

This repo mirrors the old `lrs_err_correct_env` footprint:

```bash
conda env create -f environment.yml
conda activate lrs_err_correct_env
pip install -e .
```

If the environment already exists on this machine, the notebook kernel is configured to use:

```bash
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python
```

## Quick Start

Run the tiny end-to-end debug path:

```bash
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/preprocess_dataset.py --config configs/debug_tiny.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/run_baselines.py --config configs/debug_tiny.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/train_model.py --config configs/debug_tiny.yaml --run-name target_only
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/train_model.py --config configs/debug_tiny.yaml --run-name full
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/export_summary.py --config configs/debug_tiny.yaml
```

Run the next synthetic generalization benchmark with non-shared splits and all edit payloads:

```bash
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/preprocess_dataset.py --config configs/debug_synthetic_generalization.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/run_baselines.py --config configs/debug_synthetic_generalization.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/train_model.py --config configs/debug_synthetic_generalization.yaml --run-name target_only
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/train_model.py --config configs/debug_synthetic_generalization.yaml --run-name full
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/export_summary.py --config configs/debug_synthetic_generalization.yaml
```

Run the staged synthetic curriculum before moving to real data:

```bash
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/run_synthetic_curriculum.py --base-config configs/debug_synthetic_generalization.yaml
```

Run the larger multi-seed synthetic scale check:

```bash
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/run_synthetic_multiseed.py --base-config configs/debug_synthetic_generalization_large.yaml --seeds 47,48,49,50,51
```

To only write the per-seed configs before committing to a full run:

```bash
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/run_synthetic_multiseed.py --skip-training
```

Run the local real-data smoke path:

```bash
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/preprocess_dataset.py --config configs/local_small.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/run_baselines.py --config configs/local_small.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/train_model.py --config configs/local_small.yaml --run-name full
```

Run the chr20 benchmark path after filling in the reference/truth paths:

```bash
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/fetch_region_subset.py --config configs/hg002_chr20_small.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/preprocess_dataset.py --config configs/hg002_chr20_small.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/run_baselines.py --config configs/hg002_chr20_small.yaml
/Users/shanejayasundera/anaconda3/envs/lrs_err_correct_env/bin/python scripts/train_model.py --config configs/hg002_chr20_small.yaml --run-name full
```

The workspace currently includes the HG002 chr20 BAM subset plus benchmark VCF/BED, but not the referenced chr20 truth/reference FASTA files. The config is wired for Phase C, and those two fasta paths need to be provided before preprocessing can run.

## Experiment Ladder

1. `debug_tiny`: keep the shared-split exact-learning guardrail green.
2. `debug_synthetic_generalization`: measure unseen synthetic generalization with `shared_examples_across_splits: false`.
3. `debug_synthetic_curriculum`: prove COPY+SUB, COPY+INS, COPY+DEL, mixed hard edits, then full harder synthetic.
4. `debug_synthetic_generalization_large` and `run_synthetic_multiseed.py`: test whether zero-FP hybrid behavior survives scale and random seeds.
5. `local_small`: prove utility on a tiny real dataset and approach consensus.
6. `hg002_chr20_small`: compare conservative learned correction against consensus on a small human benchmark.

## Outputs

Each run directory saves:

- `config_snapshot.yaml`
- `manifest.json` 
- `history.json`
- `test_summary.json`
- `benchmark_summary.json`

## Notebook

Use [notebooks/train_and_benchmark.ipynb](/Users/shanejayasundera/LRS-Error-Correction/new_solution/omega_lr_rebuild/notebooks/train_and_benchmark.ipynb) for thin orchestration only.
