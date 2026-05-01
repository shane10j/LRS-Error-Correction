"""Training and evaluation loop."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from omega_lr.data.collate import collate_examples
from omega_lr.data.datasets import InMemoryDataset, JsonlDataset
from omega_lr.data.samplers import edit_richness_score, oversample_hard_edit_rows
from omega_lr.eval.overfit import correction_quality, should_use_overfit_selection
from omega_lr.eval.summaries import benchmark_record, build_summary
from omega_lr.constants import ID_TO_EDIT
from omega_lr.model.decode import decode_batch, decode_batch_argmax
from omega_lr.model.model import ModelConfig, OmegaEditModel
from omega_lr.train.checkpointing import save_checkpoint
from omega_lr.train.early_stopping import EarlyStopping
from omega_lr.train.losses import compute_losses
from omega_lr.train.metrics import batch_label_metrics, finalize_train_metrics
from omega_lr.train.schedule import epoch_schedule
from omega_lr.utils import save_json, save_jsonl


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: dict, run_name: str, support_input_dim: int) -> OmegaEditModel:
    model_flags = config.get("model_debug", {})
    model_cfg = ModelConfig(
        d_model=config["model"]["d_model"],
        conv_kernel_size=config["model"]["conv_kernel_size"],
        support_hidden_dim=config["model"]["support_hidden_dim"],
        max_window_length=config["dataset"]["max_window_length"],
        max_deletion_length=config["dataset"]["max_deletion_length"],
        use_support=config["model"][run_name]["use_support"],
        support_input_dim=support_input_dim,
        rule_feature_dim=6,
        use_trust_gate=model_flags.get("use_trust_gate", True),
        use_delete_length_head=model_flags.get("use_delete_length_head", True),
        payload_type_coupling_boost=float(config["model"].get("payload_type_coupling_boost", 0.0)),
    )
    return OmegaEditModel(model_cfg)


def make_loader(path: Path, batch_size: int, shuffle: bool) -> tuple[JsonlDataset, DataLoader]:
    dataset = JsonlDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_examples)
    return dataset, loader


def make_in_memory_loader(rows: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = InMemoryDataset(rows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_examples)


def curriculum_rows(rows: list[dict], fraction: float) -> list[dict]:
    ranked = sorted(rows, key=edit_richness_score, reverse=True)
    keep = max(1, int(len(ranked) * fraction))
    return ranked[:keep]


def prediction_rows(batch: dict, decoded: list[dict]) -> list[dict]:
    rows = []
    for example, prediction in zip(batch["raw_examples"], decoded):
        rows.append(
            {
                "example_id": example["example_id"],
                "target_seq": example["target_seq"],
                "truth_seq": example["truth_seq"],
                "prediction": prediction["prediction"],
                "predicted_labels": prediction["predicted_labels"],
                "gold_edit_labels": example["edit_labels"],
                "features": example["features"],
                "masks": example["masks"],
                "trust": prediction["trust"],
            }
        )
    return rows


def evaluate_loader(model: OmegaEditModel, loader: DataLoader, device: torch.device, decode_config: dict, gate_open_bias: float = 0.0) -> tuple[list[dict], dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["target_tokens"].to(device),
                batch["pileup_features"].to(device),
                batch["rule_features"].to(device),
                gate_open_bias=gate_open_bias,
            )
            decoded = decode_batch(batch, outputs, decode_config)
            rows.extend(prediction_rows(batch, decoded))
    summary = build_summary(rows)
    return rows, summary


def evaluate_loader_argmax(model: OmegaEditModel, loader: DataLoader, device: torch.device, gate_open_bias: float = 0.0) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["target_tokens"].to(device),
                batch["pileup_features"].to(device),
                batch["rule_features"].to(device),
                gate_open_bias=gate_open_bias,
            )
            decoded = decode_batch_argmax(batch, outputs)
            rows.extend(prediction_rows(batch, decoded))
    return rows


def neural_only_decode_config(decode_config: dict) -> dict:
    updated = deepcopy(decode_config)
    updated["hybrid_rule_decode"] = False
    return updated


def hybrid_decode_config(decode_config: dict) -> dict:
    updated = deepcopy(decode_config)
    updated["hybrid_rule_decode"] = True
    return updated


def hybrid_gap_diagnostics(model: OmegaEditModel, loader: DataLoader, device: torch.device, decode_config: dict) -> dict:
    model.eval()
    hybrid_config = hybrid_decode_config(decode_config)
    neural_config = neural_only_decode_config(decode_config)
    positions = []
    summary = {
        "hard_edit_positions": 0,
        "neural_argmax_correct": 0,
        "neural_only_correct": 0,
        "hybrid_correct": 0,
        "rule_forced_count": 0,
        "rule_forced_correct": 0,
        "correct_requires_rule_forcing": 0,
        "rule_forced_wrong": 0,
    }
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["target_tokens"].to(device),
                batch["pileup_features"].to(device),
                batch["rule_features"].to(device),
            )
            hybrid_decoded = decode_batch(batch, outputs, hybrid_config)
            neural_decoded = decode_batch(batch, outputs, neural_config)
            argmax_decoded = decode_batch_argmax(batch, outputs)
            for example, hybrid, neural, argmax in zip(batch["raw_examples"], hybrid_decoded, neural_decoded, argmax_decoded):
                gold_labels = [ID_TO_EDIT[int(label)] for label in example["edit_labels"]]
                for pos, gold in enumerate(gold_labels):
                    if gold == "COPY":
                        continue
                    neural_argmax = argmax["predicted_labels"][pos] if pos < len(argmax["predicted_labels"]) else None
                    neural_final = neural["predicted_labels"][pos] if pos < len(neural["predicted_labels"]) else None
                    hybrid_final = hybrid["predicted_labels"][pos] if pos < len(hybrid["predicted_labels"]) else None
                    trace = next((item for item in hybrid.get("trace", []) if item["pos"] == pos), {})
                    forced_by_rule = bool(trace.get("forced_by_rule", False))
                    neural_argmax_correct = neural_argmax == gold
                    neural_only_correct = neural_final == gold
                    hybrid_correct = hybrid_final == gold
                    summary["hard_edit_positions"] += 1
                    summary["neural_argmax_correct"] += int(neural_argmax_correct)
                    summary["neural_only_correct"] += int(neural_only_correct)
                    summary["hybrid_correct"] += int(hybrid_correct)
                    summary["rule_forced_count"] += int(forced_by_rule)
                    summary["rule_forced_correct"] += int(forced_by_rule and hybrid_correct)
                    summary["correct_requires_rule_forcing"] += int(hybrid_correct and not neural_only_correct and forced_by_rule)
                    summary["rule_forced_wrong"] += int(forced_by_rule and not hybrid_correct)
                    positions.append(
                        {
                            "example_id": example["example_id"],
                            "pos": pos,
                            "gold": gold,
                            "neural_argmax": neural_argmax,
                            "neural_only_final": neural_final,
                            "hybrid_final": hybrid_final,
                            "forced_by_rule": forced_by_rule,
                            "candidate_label": trace.get("candidate_label"),
                            "type_probs": trace.get("type_probs"),
                            "sub_base_probs": trace.get("sub_base_probs"),
                            "ins_base_probs": trace.get("ins_base_probs"),
                            "support_ins_count": example["features"]["support_ins_count"][pos],
                            "support_del_count": example["features"]["support_del_count"][pos],
                            "support_base_counts": example["features"]["support_base_counts"][pos],
                            "support_agreement": example["features"]["support_agreement"][pos],
                            "support_entropy": example["features"]["support_entropy"][pos],
                        }
                    )
    denom = max(summary["hard_edit_positions"], 1)
    summary["neural_argmax_hard_edit_accuracy"] = round(summary["neural_argmax_correct"] / denom, 4)
    summary["neural_only_hard_edit_accuracy"] = round(summary["neural_only_correct"] / denom, 4)
    summary["hybrid_hard_edit_accuracy"] = round(summary["hybrid_correct"] / denom, 4)
    summary["rule_forced_precision"] = round(summary["rule_forced_correct"] / max(summary["rule_forced_count"], 1), 4)
    return {"summary": summary, "positions": positions}


def train_model(config: dict, dataset_dir: Path, run_dir: Path, run_name: str) -> dict:
    config["_active_run_name"] = run_name
    device = choose_device()
    train_dataset, _ = make_loader(dataset_dir / "train.jsonl", config["train"]["batch_size"], shuffle=True)
    val_dataset, val_loader = make_loader(dataset_dir / "val.jsonl", config["train"]["batch_size"], shuffle=False)
    _, test_loader = make_loader(dataset_dir / "test.jsonl", config["train"]["batch_size"], shuffle=False)
    support_input_dim = len(val_dataset[0]["pileup_features"][0])
    model = build_model(config, run_name, support_input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"])
    early_stopping = EarlyStopping(config["train"]["patience"])
    history = []
    best_summary = None
    best_selection = None
    use_overfit_selection = should_use_overfit_selection(config)
    for epoch in range(config["train"]["epochs"]):
        schedule = epoch_schedule(config, epoch)
        config["_runtime_schedule"] = schedule
        active_rows = curriculum_rows(train_dataset.rows, schedule["curriculum_fraction"])
        active_rows = oversample_hard_edit_rows(active_rows, int(config["train"].get("hard_edit_oversample_factor", 1)))
        train_loader = make_in_memory_loader(active_rows, config["train"]["batch_size"], shuffle=True)
        model.train()
        running = {"loss": 0.0, "positive_reward": 0.0}
        metric_totals: dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                batch["target_tokens"].to(device),
                batch["pileup_features"].to(device),
                batch["rule_features"].to(device),
                gate_open_bias=schedule["gate_open_bias"],
            )
            losses = compute_losses(batch, outputs, config, device)
            losses["total"].backward()
            optimizer.step()
            metrics = batch_label_metrics(batch, outputs)
            running["loss"] += float(losses["total"].item())
            for loss_name, loss_value in losses.items():
                if loss_name == "total":
                    continue
                running[loss_name] = running.get(loss_name, 0.0) + float(loss_value.item())
            for key, value in metrics.items():
                if key == "hard_margin_min" and key in metric_totals:
                    metric_totals[key] = min(metric_totals[key], float(value))
                else:
                    metric_totals[key] = metric_totals.get(key, 0.0) + float(value)
            batches += 1
        train_metrics = finalize_train_metrics(metric_totals)
        train_metrics["loss"] = round(running["loss"] / max(batches, 1), 4)
        train_metrics["loss_components"] = {
            key: round(value / max(batches, 1), 4)
            for key, value in running.items()
            if key != "loss"
        }
        val_rows, val_summary = evaluate_loader(
            model,
            val_loader,
            device,
            schedule["decode_config"],
            gate_open_bias=schedule["gate_open_bias"],
        )
        val_selection_rows = (
            evaluate_loader_argmax(model, val_loader, device, gate_open_bias=schedule["gate_open_bias"])
            if use_overfit_selection
            else val_rows
        )
        if use_overfit_selection:
            save_jsonl(val_selection_rows, run_dir / "val_argmax_predictions.jsonl")
        val_quality = correction_quality(val_selection_rows)
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_summary,
                "val_correction_quality": val_quality,
                "schedule": schedule,
            }
        )
        save_json(history, run_dir / "history.json")
        save_jsonl(val_rows, run_dir / "val_predictions.jsonl")
        last_state = {
            "model_state": model.state_dict(),
            "config": config,
            "run_name": run_name,
            "epoch": epoch + 1,
            "val_summary": val_summary,
            "val_correction_quality": val_quality,
        }
        save_checkpoint(last_state, run_dir / "last.ckpt")
        selection_value = (
            val_quality["selection_score"]
            if use_overfit_selection
            else val_summary["usable_score"]
        )
        best_value = None
        if best_selection is not None:
            best_value = best_selection["selection_score" if use_overfit_selection else "usable_score"]
        if best_summary is None or selection_value >= best_value:
            best_summary = val_summary
            best_selection = {
                "epoch": epoch + 1,
                "selection_mode": "exact_correction_quality" if use_overfit_selection else "usable_score",
                "selection_score": selection_value,
                "usable_score": val_summary["usable_score"],
                "correction_quality": val_quality,
            }
            save_json(best_summary, run_dir / "best_val_summary.json")
            save_json(best_selection, run_dir / "best_selection_summary.json")
            save_checkpoint({**last_state, "best_selection": best_selection}, run_dir / "best.ckpt")
        early_stop_value = selection_value if use_overfit_selection else val_summary["usable_score"]
        if early_stopping.update(early_stop_value):
            break
    checkpoint = torch.load(run_dir / "best.ckpt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    config["_runtime_schedule"] = epoch_schedule(config, config["train"]["epochs"] - 1)
    test_decode_config = hybrid_decode_config(config["decode"]) if run_name == "full" else config["decode"]
    test_rows, test_summary = evaluate_loader(model, test_loader, device, test_decode_config, gate_open_bias=0.0)
    save_jsonl(test_rows, run_dir / "test_predictions.jsonl")
    save_json(test_summary, run_dir / "test_summary.json")
    benchmark_name = "full_hybrid" if run_name == "full" else run_name
    save_json(benchmark_record(benchmark_name, test_summary), run_dir / "benchmark_summary.json")
    if run_name == "full":
        neural_rows, neural_summary = evaluate_loader(model, test_loader, device, neural_only_decode_config(config["decode"]), gate_open_bias=0.0)
        save_jsonl(neural_rows, run_dir / "full_neural_only_test_predictions.jsonl")
        save_json(neural_summary, run_dir / "full_neural_only_test_summary.json")
        save_json(benchmark_record("full_neural_only", neural_summary), run_dir / "full_neural_only_benchmark_summary.json")
        gap = hybrid_gap_diagnostics(model, test_loader, device, config["decode"])
        save_json(gap, run_dir / "hybrid_gap_diagnostics.json")
        save_json(gap["summary"], run_dir / "hybrid_gap_summary.json")
    return test_summary
