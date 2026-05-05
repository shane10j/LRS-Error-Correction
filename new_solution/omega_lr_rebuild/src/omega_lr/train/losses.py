"""Training losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from omega_lr.constants import BASES, EDIT_TO_ID, EDIT_TYPE_TO_ID


def build_class_weights(config: dict) -> torch.Tensor:
    weights = torch.ones(len(EDIT_TO_ID))
    class_cfg = config["train"]["class_weights"]
    for label, idx in EDIT_TO_ID.items():
        if label == "COPY":
            weights[idx] = class_cfg["COPY"]
        elif label == "DEL":
            weights[idx] = class_cfg["DEL"]
        elif label.startswith("SUB_"):
            weights[idx] = class_cfg["SUB"]
        elif label.startswith("INS_"):
            weights[idx] = class_cfg["INS"]
    return weights


def build_type_weights(config: dict) -> torch.Tensor:
    weights = torch.ones(len(EDIT_TYPE_TO_ID))
    class_cfg = config["train"]["class_weights"]
    weights[EDIT_TYPE_TO_ID["COPY"]] = class_cfg["COPY"]
    weights[EDIT_TYPE_TO_ID["SUB"]] = class_cfg["SUB"]
    weights[EDIT_TYPE_TO_ID["DEL"]] = class_cfg["DEL"]
    weights[EDIT_TYPE_TO_ID["INS"]] = class_cfg["INS"]
    return weights


def focal_binary_loss(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(labels > 0.5, torch.full_like(labels, alpha), torch.full_like(labels, 1.0 - alpha))
    loss = -alpha_t * ((1.0 - pt) ** gamma) * torch.log(pt.clamp(min=1e-6))
    return loss.mean()


def structured_targets(edit_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    type_labels = torch.full_like(edit_labels, EDIT_TYPE_TO_ID["COPY"])
    sub_base_labels = torch.zeros_like(edit_labels)
    ins_base_labels = torch.zeros_like(edit_labels)

    sub_start = EDIT_TO_ID["SUB_A"]
    ins_start = EDIT_TO_ID["INS_A"]
    del_id = EDIT_TO_ID["DEL"]

    sub_mask = (edit_labels >= sub_start) & (edit_labels < sub_start + 4)
    ins_mask = (edit_labels >= ins_start) & (edit_labels < ins_start + 4)
    del_mask = edit_labels == del_id

    type_labels[sub_mask] = EDIT_TYPE_TO_ID["SUB"]
    type_labels[del_mask] = EDIT_TYPE_TO_ID["DEL"]
    type_labels[ins_mask] = EDIT_TYPE_TO_ID["INS"]
    sub_base_labels[sub_mask] = edit_labels[sub_mask] - sub_start
    ins_base_labels[ins_mask] = edit_labels[ins_mask] - ins_start
    return type_labels, sub_base_labels, ins_base_labels


def base_position_weights(base_labels: torch.Tensor, config_weights: dict, device: torch.device) -> torch.Tensor:
    """Return per-position weights for A/C/G/T-specific rescue pressure."""
    weights = torch.ones_like(base_labels, dtype=torch.float32, device=device)
    if not config_weights:
        return weights
    for base, weight in config_weights.items():
        weights[base_labels == BASES.index(base)] = float(weight)
    return weights


def weighted_masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


def compute_losses(batch: dict, outputs: dict, config: dict, device: torch.device) -> dict[str, torch.Tensor]:
    schedule = config.get("_runtime_schedule", {})
    active_run = config.get("_active_run_name", "full")
    support_loss_scale = 1.0 if config.get("model", {}).get(active_run, {}).get("use_support", True) else 0.0
    mask = batch["attention_mask"].to(device)
    edit_labels = batch["edit_labels"].to(device)
    delete_candidate_labels = batch["delete_candidate_labels"].to(device)
    delete_length_labels = batch["delete_length_labels"].to(device)
    type_labels, sub_base_labels, ins_base_labels = structured_targets(edit_labels)

    type_weights = build_type_weights(config).to(device)
    type_loss = F.cross_entropy(outputs["type_logits"].transpose(1, 2), type_labels, weight=type_weights, reduction="none")
    hard_type_loss_weight = schedule.get("hard_type_loss_weight", config["train"].get("hard_type_loss_weight", 1.0))
    type_position_weights = torch.ones_like(type_loss)
    type_position_weights[(type_labels != EDIT_TYPE_TO_ID["COPY"]) & (mask > 0.5)] = hard_type_loss_weight
    type_loss = (type_loss * mask * type_position_weights).sum() / (mask * type_position_weights).sum().clamp(min=1.0)

    sub_positions = (type_labels == EDIT_TYPE_TO_ID["SUB"]) & (mask > 0.5)
    if sub_positions.any():
        sub_payload_losses = F.cross_entropy(
            outputs["sub_base_logits"][sub_positions],
            sub_base_labels[sub_positions],
            reduction="none",
        )
        sub_base_weight_cfg = config["train"].get("sub_payload_base_weights", {})
        if sub_base_weight_cfg:
            sub_base_weights = base_position_weights(sub_base_labels[sub_positions], sub_base_weight_cfg, device)
            sub_payload_losses = sub_payload_losses * sub_base_weights
        sub_payload_loss = sub_payload_losses.mean()
    else:
        sub_payload_loss = torch.tensor(0.0, device=device)

    ins_positions = (type_labels == EDIT_TYPE_TO_ID["INS"]) & (mask > 0.5)
    if ins_positions.any():
        ins_payload_losses = F.cross_entropy(
            outputs["ins_base_logits"][ins_positions],
            ins_base_labels[ins_positions],
            reduction="none",
        )
        base_weight_cfg = config["train"].get("ins_payload_base_weights", {})
        if base_weight_cfg:
            base_weights = torch.ones(4, device=device)
            for base, weight in base_weight_cfg.items():
                base_weights[BASES.index(base)] = float(weight)
            ins_payload_losses = ins_payload_losses * base_weights[ins_base_labels[ins_positions]]
        ins_payload_loss = ins_payload_losses.mean()
    else:
        ins_payload_loss = torch.tensor(0.0, device=device)

    delete_candidate_loss = focal_binary_loss(outputs["delete_candidate_logits"] * mask, delete_candidate_labels * mask)
    delete_positions = (type_labels == EDIT_TYPE_TO_ID["DEL"]) & (mask > 0.5)
    use_delete_length_head = config.get("model_debug", {}).get("use_delete_length_head", True)
    use_trust_gate = config.get("model_debug", {}).get("use_trust_gate", True)
    if use_delete_length_head and delete_positions.any():
        delete_length_loss = F.cross_entropy(outputs["delete_length_logits"][delete_positions], delete_length_labels[delete_positions])
    else:
        delete_length_loss = torch.tensor(0.0, device=device)

    support_majority_labels = batch.get("support_majority_base_labels", torch.zeros_like(edit_labels)).to(device)
    support_inserted_base_labels = batch.get("support_inserted_base_labels", torch.zeros_like(edit_labels)).to(device).clamp(min=0, max=3)
    support_sub_labels = batch.get("support_suggests_sub_labels", torch.zeros_like(mask)).to(device)
    support_ins_labels = batch.get("support_suggests_ins_labels", torch.zeros_like(mask)).to(device)
    support_del_labels = batch.get("support_suggests_del_labels", torch.zeros_like(mask)).to(device)
    support_rule_type_labels = batch.get("support_rule_type_labels", torch.zeros_like(edit_labels)).to(device)
    if "support_majority_base_logits" in outputs:
        support_majority_loss = F.cross_entropy(
            outputs["support_majority_base_logits"].transpose(1, 2),
            support_majority_labels,
            reduction="none",
        )
        support_majority_loss = (support_majority_loss * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        support_majority_loss = torch.tensor(0.0, device=device)

    def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    support_sub_loss = (
        masked_bce(outputs["support_suggests_sub_logits"], support_sub_labels)
        if "support_suggests_sub_logits" in outputs
        else torch.tensor(0.0, device=device)
    )
    support_ins_loss = (
        masked_bce(outputs["support_suggests_ins_logits"], support_ins_labels)
        if "support_suggests_ins_logits" in outputs
        else torch.tensor(0.0, device=device)
    )
    support_ins_base_positions = (support_ins_labels > 0.5) & (mask > 0.5)
    if support_ins_base_positions.any():
        support_ins_base_losses = F.cross_entropy(
            outputs["ins_base_logits"][support_ins_base_positions],
            support_inserted_base_labels[support_ins_base_positions],
            reduction="none",
        )
        support_ins_base_weights = base_position_weights(
            support_inserted_base_labels[support_ins_base_positions],
            config["train"].get("support_ins_base_weights", config["train"].get("ins_payload_base_weights", {})),
            device,
        )
        support_ins_base_loss = weighted_masked_mean(support_ins_base_losses, support_ins_base_weights)
    else:
        support_ins_base_loss = torch.tensor(0.0, device=device)
    support_sub_base_positions = (support_sub_labels > 0.5) & (mask > 0.5)
    if support_sub_base_positions.any():
        support_sub_base_losses = F.cross_entropy(
            outputs["sub_base_logits"][support_sub_base_positions],
            support_majority_labels[support_sub_base_positions].clamp(min=0, max=3),
            reduction="none",
        )
        support_sub_base_weights = base_position_weights(
            support_majority_labels[support_sub_base_positions],
            config["train"].get("support_sub_base_weights", config["train"].get("sub_payload_base_weights", {})),
            device,
        )
        support_sub_base_loss = weighted_masked_mean(support_sub_base_losses, support_sub_base_weights)
    else:
        support_sub_base_loss = torch.tensor(0.0, device=device)
    support_del_loss = (
        masked_bce(outputs["support_suggests_del_logits"], support_del_labels)
        if "support_suggests_del_logits" in outputs
        else torch.tensor(0.0, device=device)
    )
    support_rule_type_loss = F.cross_entropy(
        outputs["type_logits"].transpose(1, 2),
        support_rule_type_labels,
        reduction="none",
    )
    support_rule_type_loss = (support_rule_type_loss * mask).sum() / mask.sum().clamp(min=1.0)

    type_probs = torch.softmax(outputs["type_logits"], dim=-1)
    gold_type_probs = type_probs.gather(-1, type_labels.unsqueeze(-1)).squeeze(-1)
    hard_edit_mask = (type_labels != EDIT_TYPE_TO_ID["COPY"]) & (mask > 0.5)
    copy_mask = (type_labels == EDIT_TYPE_TO_ID["COPY"]) & (mask > 0.5)
    positive_reward = -(gold_type_probs[hard_edit_mask].mean() if hard_edit_mask.any() else torch.tensor(0.0, device=device))

    type_margin = schedule.get("type_margin", config["train"].get("type_margin", config["train"].get("copy_margin", 0.35)))
    gold_type_logits = outputs["type_logits"].gather(-1, type_labels.unsqueeze(-1)).squeeze(-1)
    gold_type_mask = F.one_hot(type_labels, num_classes=len(EDIT_TYPE_TO_ID)).bool()
    wrong_type_logits = outputs["type_logits"].masked_fill(gold_type_mask, float("-inf"))
    strongest_wrong_logits = wrong_type_logits.max(dim=-1).values
    margin_loss = (
        torch.relu(type_margin - (gold_type_logits[mask > 0.5] - strongest_wrong_logits[mask > 0.5])).mean()
        if (mask > 0.5).any()
        else torch.tensor(0.0, device=device)
    )

    del_logits = outputs["type_logits"][..., EDIT_TYPE_TO_ID["DEL"]]
    sub_logits = outputs["type_logits"][..., EDIT_TYPE_TO_ID["SUB"]]
    ins_logits = outputs["type_logits"][..., EDIT_TYPE_TO_ID["INS"]]
    copy_logits = outputs["type_logits"][..., EDIT_TYPE_TO_ID["COPY"]]
    rule_edit_mask = (support_rule_type_labels != EDIT_TYPE_TO_ID["COPY"]) & (mask > 0.5)
    rule_type_logits = outputs["type_logits"].gather(-1, support_rule_type_labels.unsqueeze(-1)).squeeze(-1)
    rule_copy_margin = schedule.get("rule_copy_margin", config["train"].get("rule_copy_margin", 1.25))
    rule_positive_copy_margin_loss = (
        torch.relu(rule_copy_margin - (rule_type_logits[rule_edit_mask] - copy_logits[rule_edit_mask])).mean()
        if rule_edit_mask.any()
        else torch.tensor(0.0, device=device)
    )
    rule_copy_mask = (support_rule_type_labels == EDIT_TYPE_TO_ID["COPY"]) & (mask > 0.5)
    hard_type_probs = type_probs[..., [EDIT_TYPE_TO_ID["SUB"], EDIT_TYPE_TO_ID["DEL"], EDIT_TYPE_TO_ID["INS"]]]
    rule_copy_noncopy_penalty = (
        hard_type_probs[rule_copy_mask].sum(dim=-1).mean()
        if rule_copy_mask.any()
        else torch.tensor(0.0, device=device)
    )
    strongest_hard_logits = torch.stack([sub_logits, del_logits, ins_logits], dim=-1).max(dim=-1).values
    rule_copy_safety_margin = schedule.get("rule_copy_safety_margin", config["train"].get("rule_copy_safety_margin", 1.50))
    rule_copy_safety_margin_loss = (
        torch.relu(rule_copy_safety_margin - (copy_logits[rule_copy_mask] - strongest_hard_logits[rule_copy_mask])).mean()
        if rule_copy_mask.any()
        else torch.tensor(0.0, device=device)
    )
    non_del_mask = (type_labels != EDIT_TYPE_TO_ID["DEL"]) & (mask > 0.5)
    non_del_margin = schedule.get("non_del_margin", config["train"].get("non_del_margin", 0.50))
    del_margin_loss = (
        torch.relu(non_del_margin - (gold_type_logits[non_del_mask] - del_logits[non_del_mask])).mean()
        if non_del_mask.any()
        else torch.tensor(0.0, device=device)
    )
    del_fallback_penalty = (
        torch.relu(del_logits[non_del_mask] - gold_type_logits[non_del_mask]).mean()
        if non_del_mask.any()
        else torch.tensor(0.0, device=device)
    )
    hard_copy_penalty = (
        torch.relu(copy_logits[hard_edit_mask] - gold_type_logits[hard_edit_mask]).mean()
        if hard_edit_mask.any()
        else torch.tensor(0.0, device=device)
    )
    sub_copy_margin = schedule.get("sub_copy_margin", config["train"].get("sub_copy_margin", 1.00))
    sub_del_margin = schedule.get("sub_del_margin", config["train"].get("sub_del_margin", 1.00))
    ins_del_margin = schedule.get("ins_del_margin", config["train"].get("ins_del_margin", 0.75))
    sub_copy_margin_loss = (
        torch.relu(sub_copy_margin - (sub_logits[sub_positions] - copy_logits[sub_positions])).mean()
        if sub_positions.any()
        else torch.tensor(0.0, device=device)
    )
    sub_del_margin_loss = (
        torch.relu(sub_del_margin - (sub_logits[sub_positions] - del_logits[sub_positions])).mean()
        if sub_positions.any()
        else torch.tensor(0.0, device=device)
    )
    ins_del_margin_loss = (
        torch.relu(ins_del_margin - (ins_logits[ins_positions] - del_logits[ins_positions])).mean()
        if ins_positions.any()
        else torch.tensor(0.0, device=device)
    )
    sub_probs = torch.softmax(outputs["sub_base_logits"], dim=-1)
    ins_probs = torch.softmax(outputs["ins_base_logits"], dim=-1)
    max_sub_probs, sub_argmax = sub_probs.max(dim=-1)
    max_ins_probs, ins_argmax = ins_probs.max(dim=-1)
    target_base_ids = batch.get("target_tokens", torch.zeros_like(edit_labels)).to(device).clamp(min=0, max=3)
    sub_payload_support = (sub_argmax != target_base_ids) & (support_sub_labels > 0.5) & (mask > 0.5)
    ins_payload_support = (support_ins_labels > 0.5) & (mask > 0.5)
    payload_type_margin = schedule.get("payload_type_margin", config["train"].get("payload_type_margin", 0.50))
    sub_payload_type_consistency_loss = (
        weighted_masked_mean(
            torch.relu(payload_type_margin - (sub_logits[sub_payload_support] - copy_logits[sub_payload_support]))
            * max_sub_probs[sub_payload_support],
            base_position_weights(
                support_majority_labels[sub_payload_support],
                config["train"].get("sub_type_activation_base_weights", {}),
                device,
            ),
        )
        if sub_payload_support.any()
        else torch.tensor(0.0, device=device)
    )
    ins_payload_type_consistency_loss = (
        weighted_masked_mean(
            torch.relu(payload_type_margin - (ins_logits[ins_payload_support] - copy_logits[ins_payload_support]))
            * max_ins_probs[ins_payload_support],
            base_position_weights(
                support_inserted_base_labels[ins_payload_support],
                config["train"].get("ins_type_activation_base_weights", {}),
                device,
            ),
        )
        if ins_payload_support.any()
        else torch.tensor(0.0, device=device)
    )
    hybrid_payload_threshold = config.get("decode", {}).get("hybrid_payload_threshold", 0.50)
    hybrid_sub_payload_threshold = config.get("decode", {}).get("hybrid_sub_payload_threshold", hybrid_payload_threshold)
    hybrid_ins_payload_threshold = config.get("decode", {}).get("hybrid_ins_payload_threshold", hybrid_payload_threshold)
    support_majority_labels = support_majority_labels.clamp(min=0, max=3)
    sub_threshold_by_base = config.get("decode", {}).get("hybrid_sub_payload_threshold_by_base", {})
    ins_threshold_by_base = config.get("decode", {}).get("hybrid_ins_payload_threshold_by_base", {})
    sub_payload_thresholds = torch.full_like(max_sub_probs, float(hybrid_sub_payload_threshold))
    ins_payload_thresholds = torch.full_like(max_ins_probs, float(hybrid_ins_payload_threshold))
    for base, threshold in sub_threshold_by_base.items():
        sub_payload_thresholds[support_majority_labels == BASES.index(base)] = float(threshold)
    for base, threshold in ins_threshold_by_base.items():
        ins_payload_thresholds[ins_base_labels == BASES.index(base)] = float(threshold)
    hybrid_sub_mask = (
        (support_rule_type_labels == EDIT_TYPE_TO_ID["SUB"])
        & (sub_argmax == support_majority_labels)
        & (max_sub_probs >= sub_payload_thresholds)
        & (mask > 0.5)
    )
    hybrid_ins_mask = (
        (support_rule_type_labels == EDIT_TYPE_TO_ID["INS"])
        & (type_labels == EDIT_TYPE_TO_ID["INS"])
        & (ins_argmax == ins_base_labels)
        & (max_ins_probs >= ins_payload_thresholds)
        & (mask > 0.5)
    )
    hybrid_del_mask = (
        (support_rule_type_labels == EDIT_TYPE_TO_ID["DEL"])
        & (mask > 0.5)
    )
    hybrid_forced_mask = hybrid_sub_mask | hybrid_ins_mask | hybrid_del_mask
    if hybrid_forced_mask.any():
        hybrid_forced_losses = F.cross_entropy(
            outputs["type_logits"][hybrid_forced_mask],
            support_rule_type_labels[hybrid_forced_mask],
            reduction="none",
        )
        hybrid_forced_base_labels = torch.where(
            support_rule_type_labels == EDIT_TYPE_TO_ID["SUB"],
            support_majority_labels,
            ins_base_labels,
        )
        forced_sub_weights = base_position_weights(
            hybrid_forced_base_labels[hybrid_forced_mask],
            config["train"].get("sub_type_activation_base_weights", {}),
            device,
        )
        forced_ins_weights = base_position_weights(
            hybrid_forced_base_labels[hybrid_forced_mask],
            config["train"].get("ins_type_activation_base_weights", {}),
            device,
        )
        forced_type_labels = support_rule_type_labels[hybrid_forced_mask]
        forced_weights = torch.ones_like(hybrid_forced_losses)
        forced_weights = torch.where(forced_type_labels == EDIT_TYPE_TO_ID["SUB"], forced_sub_weights, forced_weights)
        forced_weights = torch.where(forced_type_labels == EDIT_TYPE_TO_ID["INS"], forced_ins_weights, forced_weights)
        hybrid_forced_type_loss = weighted_masked_mean(hybrid_forced_losses, forced_weights)
    else:
        hybrid_forced_type_loss = torch.tensor(0.0, device=device)
    hybrid_forced_margin = schedule.get("hybrid_forced_margin", config["train"].get("hybrid_forced_margin", 1.75))
    hybrid_forced_type_logits = outputs["type_logits"].gather(-1, support_rule_type_labels.unsqueeze(-1)).squeeze(-1)
    hybrid_forced_copy_margin_loss = (
        weighted_masked_mean(
            torch.relu(hybrid_forced_margin - (hybrid_forced_type_logits[hybrid_forced_mask] - copy_logits[hybrid_forced_mask])),
            forced_weights,
        )
        if hybrid_forced_mask.any()
        else torch.tensor(0.0, device=device)
    )

    non_copy_type_mass = 1.0 - type_probs[..., EDIT_TYPE_TO_ID["COPY"]]
    false_positive_penalty = non_copy_type_mass[copy_mask].mean() if copy_mask.any() else torch.tensor(0.0, device=device)
    copy_to_sub_penalty = type_probs[..., EDIT_TYPE_TO_ID["SUB"]][copy_mask].mean() if copy_mask.any() else torch.tensor(0.0, device=device)
    copy_to_del_penalty = type_probs[..., EDIT_TYPE_TO_ID["DEL"]][copy_mask].mean() if copy_mask.any() else torch.tensor(0.0, device=device)
    copy_to_ins_penalty = type_probs[..., EDIT_TYPE_TO_ID["INS"]][copy_mask].mean() if copy_mask.any() else torch.tensor(0.0, device=device)
    no_ins_evidence_copy_mask = copy_mask & (support_ins_labels < 0.5)
    no_ins_evidence_copy_to_ins_penalty = (
        type_probs[..., EDIT_TYPE_TO_ID["INS"]][no_ins_evidence_copy_mask].mean()
        if no_ins_evidence_copy_mask.any()
        else torch.tensor(0.0, device=device)
    )
    no_ins_evidence_ins_margin = schedule.get("no_ins_evidence_ins_margin", config["train"].get("no_ins_evidence_ins_margin", 1.00))
    no_ins_evidence_ins_margin_loss = (
        torch.relu(no_ins_evidence_ins_margin - (copy_logits[no_ins_evidence_copy_mask] - ins_logits[no_ins_evidence_copy_mask])).mean()
        if no_ins_evidence_copy_mask.any()
        else torch.tensor(0.0, device=device)
    )
    trust_regularization = (
        (outputs["trust"].mean() - 0.5).abs() * schedule.get("trust_regularization_weight", 0.05)
        if use_trust_gate
        else torch.tensor(0.0, device=device)
    )
    total = (
        schedule.get("type_loss_weight", config["train"].get("type_loss_weight", 1.0)) * type_loss
        + schedule.get("sub_payload_loss_weight", config["train"].get("sub_payload_loss_weight", 1.0)) * sub_payload_loss
        + schedule.get("ins_payload_loss_weight", config["train"].get("ins_payload_loss_weight", 1.0)) * ins_payload_loss
        + schedule.get("delete_candidate_aux_weight", config["train"].get("delete_candidate_aux_weight", 0.10)) * delete_candidate_loss
        + delete_length_loss
        + support_loss_scale * schedule.get("support_majority_loss_weight", config["train"].get("support_majority_loss_weight", 0.0)) * support_majority_loss
        + support_loss_scale * schedule.get("support_sub_loss_weight", config["train"].get("support_sub_loss_weight", 0.0)) * support_sub_loss
        + support_loss_scale * schedule.get("support_sub_base_loss_weight", config["train"].get("support_sub_base_loss_weight", 0.0)) * support_sub_base_loss
        + support_loss_scale * schedule.get("support_ins_loss_weight", config["train"].get("support_ins_loss_weight", 0.0)) * support_ins_loss
        + support_loss_scale * schedule.get("support_ins_base_loss_weight", config["train"].get("support_ins_base_loss_weight", 0.0)) * support_ins_base_loss
        + support_loss_scale * schedule.get("support_del_loss_weight", config["train"].get("support_del_loss_weight", 0.0)) * support_del_loss
        + support_loss_scale * schedule.get("support_rule_type_loss_weight", config["train"].get("support_rule_type_loss_weight", 0.0)) * support_rule_type_loss
        + support_loss_scale * schedule.get("rule_positive_copy_margin_weight", config["train"].get("rule_positive_copy_margin_weight", 0.0)) * rule_positive_copy_margin_loss
        + support_loss_scale * schedule.get("rule_copy_noncopy_penalty_weight", config["train"].get("rule_copy_noncopy_penalty_weight", 0.0)) * rule_copy_noncopy_penalty
        + support_loss_scale * schedule.get("rule_copy_safety_margin_weight", config["train"].get("rule_copy_safety_margin_weight", 0.0)) * rule_copy_safety_margin_loss
        + schedule.get("type_margin_weight", config["train"].get("type_margin_weight", config["train"].get("copy_margin_weight", 0.50))) * margin_loss
        + schedule.get("non_del_margin_weight", config["train"].get("non_del_margin_weight", 0.50)) * del_margin_loss
        + schedule.get("del_fallback_penalty_weight", config["train"].get("del_fallback_penalty_weight", 0.50)) * del_fallback_penalty
        + schedule.get("hard_copy_penalty_weight", config["train"].get("hard_copy_penalty_weight", 0.50)) * hard_copy_penalty
        + schedule.get("sub_copy_margin_weight", config["train"].get("sub_copy_margin_weight", 0.0)) * sub_copy_margin_loss
        + schedule.get("sub_del_margin_weight", config["train"].get("sub_del_margin_weight", 0.0)) * sub_del_margin_loss
        + schedule.get("ins_del_margin_weight", config["train"].get("ins_del_margin_weight", 0.0)) * ins_del_margin_loss
        + support_loss_scale * schedule.get("sub_payload_type_consistency_weight", config["train"].get("sub_payload_type_consistency_weight", 0.0)) * sub_payload_type_consistency_loss
        + support_loss_scale * schedule.get("ins_payload_type_consistency_weight", config["train"].get("ins_payload_type_consistency_weight", 0.0)) * ins_payload_type_consistency_loss
        + support_loss_scale * schedule.get("hybrid_forced_type_loss_weight", config["train"].get("hybrid_forced_type_loss_weight", 0.0)) * hybrid_forced_type_loss
        + support_loss_scale * schedule.get("hybrid_forced_copy_margin_weight", config["train"].get("hybrid_forced_copy_margin_weight", 0.0)) * hybrid_forced_copy_margin_loss
        + schedule.get("positive_hard_edit_reward_weight", 0.35) * positive_reward
        + schedule.get("false_positive_penalty_weight", config["train"].get("false_positive_penalty_weight", 0.20)) * false_positive_penalty
        + schedule.get("copy_to_sub_penalty_weight", config["train"].get("copy_to_sub_penalty_weight", 0.0)) * copy_to_sub_penalty
        + schedule.get("copy_to_del_penalty_weight", config["train"].get("copy_to_del_penalty_weight", 0.0)) * copy_to_del_penalty
        + schedule.get("copy_to_ins_penalty_weight", config["train"].get("copy_to_ins_penalty_weight", 0.0)) * copy_to_ins_penalty
        + support_loss_scale * schedule.get("no_ins_evidence_copy_to_ins_penalty_weight", config["train"].get("no_ins_evidence_copy_to_ins_penalty_weight", 0.0)) * no_ins_evidence_copy_to_ins_penalty
        + support_loss_scale * schedule.get("no_ins_evidence_ins_margin_weight", config["train"].get("no_ins_evidence_ins_margin_weight", 0.0)) * no_ins_evidence_ins_margin_loss
        + trust_regularization
    )
    return {
        "total": total,
        "type_loss": type_loss.detach(),
        "sub_payload_loss": sub_payload_loss.detach(),
        "ins_payload_loss": ins_payload_loss.detach(),
        "delete_candidate_loss": delete_candidate_loss.detach(),
        "delete_length_loss": delete_length_loss.detach(),
        "support_majority_loss": support_majority_loss.detach(),
        "support_sub_loss": support_sub_loss.detach(),
        "support_sub_base_loss": support_sub_base_loss.detach(),
        "support_ins_loss": support_ins_loss.detach(),
        "support_ins_base_loss": support_ins_base_loss.detach(),
        "support_del_loss": support_del_loss.detach(),
        "support_rule_type_loss": support_rule_type_loss.detach(),
        "rule_positive_copy_margin_loss": rule_positive_copy_margin_loss.detach(),
        "rule_copy_noncopy_penalty": rule_copy_noncopy_penalty.detach(),
        "rule_copy_safety_margin_loss": rule_copy_safety_margin_loss.detach(),
        "margin_loss": margin_loss.detach(),
        "del_margin_loss": del_margin_loss.detach(),
        "del_fallback_penalty": del_fallback_penalty.detach(),
        "hard_copy_penalty": hard_copy_penalty.detach(),
        "sub_copy_margin_loss": sub_copy_margin_loss.detach(),
        "sub_del_margin_loss": sub_del_margin_loss.detach(),
        "ins_del_margin_loss": ins_del_margin_loss.detach(),
        "sub_payload_type_consistency_loss": sub_payload_type_consistency_loss.detach(),
        "ins_payload_type_consistency_loss": ins_payload_type_consistency_loss.detach(),
        "hybrid_forced_type_loss": hybrid_forced_type_loss.detach(),
        "hybrid_forced_copy_margin_loss": hybrid_forced_copy_margin_loss.detach(),
        "false_positive_penalty": false_positive_penalty.detach(),
        "copy_to_sub_penalty": copy_to_sub_penalty.detach(),
        "copy_to_del_penalty": copy_to_del_penalty.detach(),
        "copy_to_ins_penalty": copy_to_ins_penalty.detach(),
        "no_ins_evidence_copy_to_ins_penalty": no_ins_evidence_copy_to_ins_penalty.detach(),
        "no_ins_evidence_ins_margin_loss": no_ins_evidence_ins_margin_loss.detach(),
        "positive_reward": (-positive_reward).detach(),
    }
