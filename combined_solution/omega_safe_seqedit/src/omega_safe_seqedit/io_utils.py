"""Inspectable JSONL and FASTX IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_fastx(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            return records
        handle.seek(0)
        if first.startswith(">"):
            name = None
            chunks: list[str] = []
            for line in handle:
                line = line.strip()
                if line.startswith(">"):
                    if name is not None:
                        records[name] = "".join(chunks).upper()
                    name = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
            if name is not None:
                records[name] = "".join(chunks).upper()
        elif first.startswith("@"):
            while True:
                name = handle.readline().strip()
                if not name:
                    break
                seq = handle.readline().strip()
                handle.readline()
                handle.readline()
                records[name[1:].split()[0]] = seq.upper()
        else:
            raise ValueError(f"Unsupported FASTX file: {path}")
    return records
