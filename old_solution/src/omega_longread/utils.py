from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from .config import OmegaConfig


def load_config(path: str | os.PathLike[str]) -> OmegaConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return OmegaConfig.from_dict(raw)


def save_json(data: Any, path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | os.PathLike[str]) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_path(
    value: str | os.PathLike[str] | None,
    *,
    env_var: str | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> str:
    raw = ""
    if env_var:
        raw = os.environ.get(env_var, "").strip()
    if not raw and value is not None:
        raw = str(value).strip()
    if not raw:
        return ""
    raw = os.path.expandvars(os.path.expanduser(raw))
    path = Path(raw)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return str(path.resolve())


def require_existing_path(
    value: str | os.PathLike[str] | None,
    *,
    label: str,
    env_var: str | None = None,
    base_dir: str | os.PathLike[str] | None = None,
    allow_empty: bool = False,
) -> str:
    path = resolve_path(value, env_var=env_var, base_dir=base_dir)
    if not path:
        if allow_empty:
            return ""
        source = f" or ${env_var}" if env_var else ""
        raise FileNotFoundError(f"Missing required path for {label}{source}.")
    if not Path(path).exists():
        raise FileNotFoundError(f"Expected {label} at {path}, but it does not exist.")
    return path


def resolve_torch_device(preferred: str | None = None) -> tuple[str, bool]:
    preferred = (preferred or "").lower().strip()
    if preferred in {"", "auto"}:
        preferred = "cuda"
    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda", True
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", False
        return "cpu", False
    if preferred == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", False
        if torch.cuda.is_available():
            return "cuda", True
        return "cpu", False
    if preferred == "cpu":
        return "cpu", False
    if preferred.startswith("cuda:"):
        if torch.cuda.is_available():
            return preferred, True
        return "cpu", False
    return "cpu", False


def require_executable(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise FileNotFoundError(f"Required executable {name!r} was not found on PATH.")
    return resolved
