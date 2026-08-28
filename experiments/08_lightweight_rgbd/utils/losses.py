"""Base and hard-mask distillation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    true_positive = (probability * target).sum(dims)
    false_positive = (probability * (1.0 - target)).sum(dims)
    false_negative = ((1.0 - probability) * target).sum(dims)
    score = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    return 1.0 - score.mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return tversky_loss(logits, target, alpha=0.5, beta=0.5)


def boundary_loss(
    logits: torch.Tensor,
    binary_target: torch.Tensor,
    distance_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positive = binary_target.sum()
    negative = binary_target.numel() - positive
    positive_weight = torch.clamp(
        negative / torch.clamp(positive, min=1.0), 1.0, 20.0
    ).detach()
    binary = F.binary_cross_entropy_with_logits(
        logits, binary_target, pos_weight=positive_weight
    )
    distance = F.smooth_l1_loss(torch.sigmoid(logits), distance_target)
    combined = 0.5 * binary + 0.5 * distance
    return combined, {"boundary_bce": binary, "boundary_distance": distance}


def base_loss(
    mask_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    mask: torch.Tensor,
    boundary: torch.Tensor,
    boundary_distance: torch.Tensor,
    *,
    bce_weight: float,
    tversky_weight: float,
    boundary_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(mask_logits, mask)
    tversky = tversky_loss(
        mask_logits, mask, alpha=tversky_alpha, beta=tversky_beta
    )
    edge, edge_parts = boundary_loss(boundary_logits, boundary, boundary_distance)
    total = bce_weight * bce + tversky_weight * tversky + boundary_weight * edge
    parts = {"bce": bce, "tversky": tversky, "boundary": edge, **edge_parts}
    return total, {key: float(value.detach()) for key, value in parts.items()}


def distilled_loss(
    mask_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    mask: torch.Tensor,
    boundary: torch.Tensor,
    boundary_distance: torch.Tensor,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    *,
    hard_weight: float,
    teacher_a_weight: float,
    teacher_b_weight: float,
    boundary_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce_gt = F.binary_cross_entropy_with_logits(mask_logits, mask)
    dice_gt = dice_loss(mask_logits, mask)
    # BCEWithLogits is mathematically BCE(sigmoid(logits), target) while being
    # numerically stable. Teacher targets remain strictly binary, not logits.
    teacher_a_bce = F.binary_cross_entropy_with_logits(mask_logits, teacher_a)
    teacher_b_bce = F.binary_cross_entropy_with_logits(mask_logits, teacher_b)
    edge, edge_parts = boundary_loss(boundary_logits, boundary, boundary_distance)
    total = (
        hard_weight * (dice_gt + bce_gt)
        + teacher_a_weight * teacher_a_bce
        + teacher_b_weight * teacher_b_bce
        + boundary_weight * edge
    )
    parts = {
        "dice_gt": dice_gt,
        "bce_gt": bce_gt,
        "teacher_a_bce": teacher_a_bce,
        "teacher_b_bce": teacher_b_bce,
        "boundary": edge,
        **edge_parts,
    }
    return total, {key: float(value.detach()) for key, value in parts.items()}
