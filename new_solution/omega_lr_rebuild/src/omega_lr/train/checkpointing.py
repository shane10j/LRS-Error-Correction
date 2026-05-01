"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state: dict, path: str | Path) -> None:
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict:
    return torch.load(path, map_location=map_location or "cpu")

