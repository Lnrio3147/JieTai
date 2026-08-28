"""Foreground and boundary losses for the RGB-only segmenter."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (probability * target).sum(dims)
    fp = (probability * (1.0 - target)).sum(dims)
    fn = ((1.0 - probability) * target).sum(dims)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return 1.0 - score.mean()


def segmentation_loss(
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
    positive = boundary.sum()
    negative = boundary.numel() - positive
    pos_weight = torch.clamp(
        negative / torch.clamp(positive, min=1.0), 1.0, 20.0
    ).detach()
    boundary_bce = F.binary_cross_entropy_with_logits(
        boundary_logits, boundary, pos_weight=pos_weight
    )
    boundary_distance_loss = F.smooth_l1_loss(
        torch.sigmoid(boundary_logits), boundary_distance
    )
    edge = 0.5 * boundary_bce + 0.5 * boundary_distance_loss
    total = bce_weight * bce + tversky_weight * tversky + boundary_weight * edge
    return total, {
        "bce": float(bce.detach()),
        "tversky": float(tversky.detach()),
        "boundary": float(edge.detach()),
    }
