"""Training and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from omega_safe_seqedit.dataset import SeqEditDataset, collate_seqedit
from omega_safe_seqedit.decode import decode_record
from omega_safe_seqedit.io_utils import ensure_dir, write_json, write_jsonl
from omega_safe_seqedit.losses import compute_loss
from omega_safe_seqedit.metrics import summarize_predictions
from omega_safe_seqedit.model import model_from_config


def choose_device(config: dict) -> torch.device:
    requested = config["train"].get("device", "auto")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train(config: dict, run_name: str = "full") -> dict:
    dataset_dir = Path(config["paths"]["dataset_dir"])
    run_dir = ensure_dir(Path(config["paths"]["runs_dir"]) / run_name)
    train_ds = SeqEditDataset(str(dataset_dir / "train.jsonl"))
    val_ds = SeqEditDataset(str(dataset_dir / "val.jsonl"))
    sample = train_ds[0]
    use_support = run_name != "target_only"
    model = model_from_config(config, len(sample["features"][0]), len(sample["rule_features"][0]), use_support=use_support)
    device = choose_device(config)
    model.to(device)
    loader = DataLoader(train_ds, batch_size=int(config["train"]["batch_size"]), shuffle=True, collate_fn=collate_seqedit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]), weight_decay=float(config["train"].get("weight_decay", 0.0)))
    history = []
    best_score = -1e9
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["target_ids"], batch["features"], batch["rule_features"], batch["attention_mask"])
            loss, parts = compute_loss(batch, outputs, config, device)
            loss.backward()
            optimizer.step()
            losses.append(parts)
        eval_mode = "neural" if run_name == "target_only" else config["decode"].get("mode", "hybrid")
        val_summary = evaluate_model(config, model, val_ds, device, mode=eval_mode)
        train_loss = sum(x["loss"] for x in losses) / max(len(losses), 1)
        row = {"epoch": epoch, "train_loss": train_loss, "val_usable_score": val_summary["usable_score"], "val_identity": val_summary["identity"], "val_false_edits": val_summary["false_edits"]}
        history.append(row)
        torch.save({"model": model.state_dict(), "config": config, "run_name": run_name}, run_dir / "last.ckpt")
        if val_summary["usable_score"] > best_score:
            best_score = val_summary["usable_score"]
            torch.save({"model": model.state_dict(), "config": config, "run_name": run_name}, run_dir / "best.ckpt")
            write_json(run_dir / "best_val_summary.json", val_summary)
        print(json.dumps(row, sort_keys=True))
    write_json(run_dir / "history.json", {"history": history})
    return {"run_dir": str(run_dir), "history": history}


def evaluate_model(config: dict, model: torch.nn.Module, dataset: SeqEditDataset, device: torch.device, mode: str = "hybrid") -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=int(config["train"]["batch_size"]), shuffle=False, collate_fn=collate_seqedit)
    decoded = []
    with torch.no_grad():
        for batch in loader:
            raw_records = batch["records"]
            moved = _move(batch, device)
            outputs = model(moved["target_ids"], moved["features"], moved["rule_features"], moved["attention_mask"])
            for idx, record in enumerate(raw_records):
                pred = decode_record(outputs, idx, record, config, mode=mode)
                decoded.append({**record, **pred})
    return summarize_predictions(decoded)


def evaluate_checkpoint(config: dict, run_name: str, split: str = "test", mode: str = "hybrid") -> dict:
    dataset_dir = Path(config["paths"]["dataset_dir"])
    run_dir = Path(config["paths"]["runs_dir"]) / run_name
    dataset = SeqEditDataset(str(dataset_dir / f"{split}.jsonl"))
    sample = dataset[0]
    checkpoint = torch.load(run_dir / "best.ckpt", map_location="cpu")
    model = model_from_config(config, len(sample["features"][0]), len(sample["rule_features"][0]), use_support=run_name != "target_only")
    model.load_state_dict(checkpoint["model"])
    device = choose_device(config)
    model.to(device)
    eval_mode = "neural" if run_name == "target_only" else mode
    summary = evaluate_model(config, model, dataset, device, mode=eval_mode)
    write_json(run_dir / f"{split}_{eval_mode}_summary.json", summary)
    # Save decoded rows for qualitative notebook inspection.
    loader = DataLoader(dataset, batch_size=int(config["train"]["batch_size"]), shuffle=False, collate_fn=collate_seqedit)
    decoded = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            raw_records = batch["records"]
            moved = _move(batch, device)
            outputs = model(moved["target_ids"], moved["features"], moved["rule_features"], moved["attention_mask"])
            for idx, record in enumerate(raw_records):
                pred = decode_record(outputs, idx, record, config, mode=eval_mode)
                decoded.append({**record, **pred})
    write_jsonl(run_dir / f"{split}_{eval_mode}_predictions.jsonl", decoded)
    return summary
