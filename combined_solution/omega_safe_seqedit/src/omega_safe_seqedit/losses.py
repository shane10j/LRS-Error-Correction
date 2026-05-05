"""Losses for conservative edit-script learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from omega_safe_seqedit.constants import INS_TO_ID, MAIN_TO_ID, RULE_TO_ID


def compute_loss(batch: dict, outputs: dict, config: dict, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["attention_mask"].to(device)
    main = batch["main_type"].to(device)
    sub = batch["sub_base"].to(device)
    insert = batch["insert_before"].to(device)
    support_rule = batch["support_rule_type"].to(device)
    support_rule_sub = batch["support_rule_sub_base"].to(device)
    support_rule_ins = batch["support_rule_ins_base"].to(device)
    train = config["train"]

    weights = torch.tensor(
        [0.35, float(train.get("hard_edit_weight", 3.0)), float(train.get("hard_edit_weight", 3.0))],
        device=device,
    )
    main_loss = F.cross_entropy(outputs["main_logits"].transpose(1, 2), main, weight=weights, reduction="none")
    main_loss = (main_loss * mask).sum() / mask.sum().clamp(min=1.0)

    sub_mask = (main == MAIN_TO_ID["SUB"]) & mask.bool()
    if sub_mask.any():
        sub_loss = F.cross_entropy(outputs["sub_logits"][sub_mask], sub[sub_mask])
    else:
        sub_loss = torch.tensor(0.0, device=device)

    insert_loss = F.cross_entropy(outputs["insert_logits"].transpose(1, 2), insert, reduction="none")
    insert_weights = torch.ones_like(insert_loss)
    insert_weights[(insert != INS_TO_ID["NONE"]) & mask.bool()] = float(train.get("hard_edit_weight", 3.0))
    insert_loss = (insert_loss * insert_weights * mask).sum() / (insert_weights * mask).sum().clamp(min=1.0)

    rule_distill = F.cross_entropy(outputs["support_rule_logits"].transpose(1, 2), support_rule, reduction="none")
    rule_distill = (rule_distill * mask).sum() / mask.sum().clamp(min=1.0)

    main_probs = torch.softmax(outputs["main_logits"], dim=-1)
    insert_probs = torch.softmax(outputs["insert_logits"], dim=-1)
    copy_positions = (main == MAIN_TO_ID["COPY"]) & (insert == INS_TO_ID["NONE"]) & mask.bool()
    false_main_pressure = main_probs[..., [MAIN_TO_ID["SUB"], MAIN_TO_ID["DEL"]]].sum(dim=-1)
    false_insert_pressure = 1.0 - insert_probs[..., INS_TO_ID["NONE"]]
    fp_penalty = (
        (false_main_pressure[copy_positions] + false_insert_pressure[copy_positions]).mean()
        if copy_positions.any()
        else torch.tensor(0.0, device=device)
    )

    gold_main = outputs["main_logits"].gather(-1, main.unsqueeze(-1)).squeeze(-1)
    wrong_main = outputs["main_logits"].masked_fill(F.one_hot(main, 3).bool(), float("-inf")).max(dim=-1).values
    hard_positions = ((main != MAIN_TO_ID["COPY"]) | (insert != INS_TO_ID["NONE"])) & mask.bool()
    margin = float(train.get("margin", 0.75))
    margin_loss = (
        torch.relu(margin - (gold_main[hard_positions] - wrong_main[hard_positions])).mean()
        if hard_positions.any()
        else torch.tensor(0.0, device=device)
    )

    # Payload imitation from support rules helps the neural heads learn reusable support logic.
    rule_sub_mask = (support_rule == RULE_TO_ID["SUB"]) & mask.bool()
    if rule_sub_mask.any():
        support_sub_loss = F.cross_entropy(outputs["sub_logits"][rule_sub_mask], support_rule_sub[rule_sub_mask])
    else:
        support_sub_loss = torch.tensor(0.0, device=device)
    rule_ins_mask = (support_rule == RULE_TO_ID["INS"]) & mask.bool()
    if rule_ins_mask.any():
        support_ins_targets = support_rule_ins[rule_ins_mask] + 1
        support_ins_loss = F.cross_entropy(outputs["insert_logits"][rule_ins_mask], support_ins_targets)
    else:
        support_ins_loss = torch.tensor(0.0, device=device)

    total = (
        main_loss
        + float(train.get("sub_payload_weight", 1.5)) * sub_loss
        + float(train.get("ins_payload_weight", 1.5)) * insert_loss
        + float(train.get("support_rule_distill_weight", 0.8)) * rule_distill
        + float(train.get("false_edit_penalty", 1.0)) * fp_penalty
        + float(train.get("margin_weight", 0.5)) * margin_loss
        + 0.4 * support_sub_loss
        + 0.4 * support_ins_loss
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "main_loss": float(main_loss.detach().cpu()),
        "sub_loss": float(sub_loss.detach().cpu()),
        "insert_loss": float(insert_loss.detach().cpu()),
        "rule_distill": float(rule_distill.detach().cpu()),
        "fp_penalty": float(fp_penalty.detach().cpu()),
        "margin_loss": float(margin_loss.detach().cpu()),
    }
    return total, parts
