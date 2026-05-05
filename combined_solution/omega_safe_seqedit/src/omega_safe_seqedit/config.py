"""Config loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(config_path.parents[1])
    out = Path(config["paths"]["output_dir"])
    if not out.is_absolute():
        out = config_path.parents[1] / out
    config["paths"]["output_dir"] = str(out)
    config["paths"]["dataset_dir"] = str(out / "dataset")
    config["paths"]["baseline_dir"] = str(out / "baselines")
    config["paths"]["runs_dir"] = str(out / "runs")
    return config


def print_resolved_config(config: dict[str, Any]) -> None:
    print(yaml.safe_dump(config, sort_keys=False))
