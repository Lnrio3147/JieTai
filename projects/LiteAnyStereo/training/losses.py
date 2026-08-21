"""Loss functions shared by LAS1 and LAS2 fine-tuning."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _as_prediction_list(predictions):
    if torch.is_tensor(predictions):
        return [predictions]
    if isinstance(predictions, Sequence) and predictions:
        if all(torch.is_tensor(prediction) for prediction in predictions):
            return list(predictions)
    raise TypeError("Model predictions must be a tensor or a non-empty sequence of tensors.")


def multi_prediction_smooth_l1(
    predictions,
    target,
    valid,
    aux_weight=0.5,
    beta=1.0,
):
    """Compute weighted Smooth L1 loss for a model's full-resolution predictions.

    The first prediction is treated as the final output and receives weight 1.0.
    Every following auxiliary prediction receives ``aux_weight ** index``.
    """

    prediction_list = _as_prediction_list(predictions)
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"Expected target shape [B,1,H,W], got {tuple(target.shape)}")
    if valid.shape != target.shape:
        raise ValueError(f"Valid mask shape {tuple(valid.shape)} does not match target {tuple(target.shape)}")
    if not 0.0 <= aux_weight <= 1.0:
        raise ValueError("aux_weight must be in [0, 1]")

    valid = valid.bool() & torch.isfinite(target)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError("The batch contains no valid disparity pixels.")

    total_loss = target.new_zeros(())
    components = []
    for index, prediction in enumerate(prediction_list):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction {index} shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}"
            )
        weight = aux_weight ** index
        component = F.smooth_l1_loss(prediction[valid], target[valid], beta=beta, reduction="mean")
        total_loss = total_loss + weight * component
        components.append(component.detach())

    return total_loss, components, valid_count
