#!/usr/bin/env python
"""Forwarder for the rebuilt omega_lr synthetic multi-seed runner.

This keeps root-level commands working while the standalone codebase lives
under new_solution/omega_lr_rebuild.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "new_solution" / "omega_lr_rebuild"
INNER_SCRIPT = PROJECT_ROOT / "scripts" / "run_synthetic_multiseed.py"


def main() -> None:
    if not INNER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing rebuilt runner: {INNER_SCRIPT}")
    os.chdir(PROJECT_ROOT)
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    sys.argv = [str(INNER_SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(INNER_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
