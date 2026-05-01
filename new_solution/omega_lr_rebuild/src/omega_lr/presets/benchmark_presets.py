"""Benchmark preset metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkPreset:
    name: str
    runs: list[str]
    description: str
    max_examples: int
    epochs: int


PRESETS = {
    "debug_tiny": BenchmarkPreset(
        name="debug_tiny",
        runs=["no_edit", "consensus", "target_only", "full"],
        description="CPU-safe synthetic correctness loop",
        max_examples=100,
        epochs=6,
    ),
    "debug_synthetic_generalization": BenchmarkPreset(
        name="debug_synthetic_generalization",
        runs=["no_edit", "support_rule", "consensus", "target_only", "full_hybrid", "full_neural_only"],
        description="Non-shared synthetic payload/context generalization loop",
        max_examples=102,
        epochs=12,
    ),
    "debug_synthetic_generalization_large": BenchmarkPreset(
        name="debug_synthetic_generalization_large",
        runs=["no_edit", "support_rule", "consensus", "target_only", "full_hybrid", "full_neural_only"],
        description="Larger multi-seed synthetic generalization loop with boundary insertions",
        max_examples=1120,
        epochs=18,
    ),
    "local_small": BenchmarkPreset(
        name="local_small",
        runs=["no_edit", "consensus", "target_only", "full"],
        description="Small real-data local smoke path",
        max_examples=1500,
        epochs=5,
    ),
    "hg002_chr20_small": BenchmarkPreset(
        name="hg002_chr20_small",
        runs=["no_edit", "consensus", "target_only", "full"],
        description="Small chr20 benchmark with conservative comparison to consensus",
        max_examples=3000,
        epochs=6,
    ),
}
