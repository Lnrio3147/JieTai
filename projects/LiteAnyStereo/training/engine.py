"""Single-GPU training and validation loops."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from core.utils.utils import InputPadder
from .losses import multi_prediction_smooth_l1
from .metrics import DisparityMetrics, aggregate_tradition_metrics, disparity_metric_row
from .visualization import save_validation_vis


def _final_prediction(predictions):
    return predictions if torch.is_tensor(predictions) else predictions[0]


def _move_batch(batch, device, max_disp):
    left = batch["left"].to(device, non_blocking=True)
    right = batch["right"].to(device, non_blocking=True)
    target = batch["disparity"].to(device, non_blocking=True)
    # Clone because in-place mask refinement must not mutate a CPU batch tensor.
    valid = batch["valid"].to(device, non_blocking=True).bool().clone()
    valid &= torch.isfinite(target) & (target > 0.0)
    if max_disp is not None:
        valid &= target < max_disp
    return left, right, target, valid


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    global_step,
    max_disp,
    aux_weight,
    grad_clip,
    amp,
    logger,
    jsonl_writer=None,
    log_interval=10,
    max_steps=0,
):
    model.train()
    metrics = DisparityMetrics()
    loss_sum = 0.0
    batches = 0
    started = time.perf_counter()
    stopped_early = False

    for batch_index, batch in enumerate(loader):
        if max_steps and global_step >= max_steps:
            stopped_early = True
            break

        left, right, target, valid = _move_batch(batch, device, max_disp)
        if not valid.any():
            logger.warning("Skipping batch %d because it contains no valid disparity pixels", batch_index)
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            predictions = model(left, right, max_disp=max_disp, test_mode=False)
            loss, components, valid_count = multi_prediction_smooth_l1(
                predictions,
                target,
                valid,
                aux_weight=aux_weight,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_skipped = amp and scaler.get_scale() < scale_before_step

        batches += 1
        loss_value = float(loss.detach().item())
        loss_sum += loss_value
        metrics.update(_final_prediction(predictions).detach(), target, valid)

        if optimizer_step_skipped:
            logger.warning(
                "AMP overflow at epoch=%d batch=%d; optimizer and scheduler steps were skipped",
                epoch,
                batch_index,
            )
            if jsonl_writer is not None:
                jsonl_writer(
                    {
                        "event": "amp_overflow",
                        "epoch": epoch,
                        "step": global_step,
                        "batch": batch_index,
                        "loss": loss_value,
                        "scale_before": scale_before_step,
                        "scale_after": scaler.get_scale(),
                    }
                )
            continue

        if scheduler is not None:
            scheduler.step()
        global_step += 1

        if global_step == 1 or global_step % log_interval == 0:
            current = metrics.compute()
            learning_rate = optimizer.param_groups[0]["lr"]
            record = {
                "event": "train_step",
                "epoch": epoch,
                "step": global_step,
                "batch": batch_index,
                "loss": loss_value,
                "prediction_losses": [float(value.item()) for value in components],
                "epe": current["epe"],
                "d1": current["d1"],
                "bad1": current["bad1"],
                "bad2": current["bad2"],
                "bad3": current["bad3"],
                "valid_pixels": valid_count,
                "lr": learning_rate,
            }
            logger.info(
                "epoch=%d step=%d batch=%d loss=%.5f epe=%.4f d1=%.3f%% lr=%.3e",
                epoch,
                global_step,
                batch_index,
                loss_value,
                current["epe"],
                current["d1"],
                learning_rate,
            )
            if jsonl_writer is not None:
                jsonl_writer(record)

    result = metrics.compute()
    result.update(
        {
            "loss": loss_sum / max(batches, 1),
            "batches": batches,
            "seconds": time.perf_counter() - started,
        }
    )
    return result, global_step, stopped_early


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    max_disp,
    amp,
    logger,
    evaluation_protocol="standard",
    excluded_scenes=(),
    epe_threshold=None,
    return_scene_metrics=False,
    save_vis_dir=None,
    vis_error_max=20.0,
):
    if evaluation_protocol not in {"standard", "tradition"}:
        raise ValueError(f"Unsupported evaluation protocol: {evaluation_protocol}")
    model.eval()
    metrics = DisparityMetrics()
    scene_rows = []
    started = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        # tradition_stereo evaluates every finite GT pixel above zero, including
        # references beyond the network's configured search range.
        validation_max_disp = None if evaluation_protocol == "tradition" else max_disp
        left, right, target, valid = _move_batch(batch, device, validation_max_disp)
        if left.shape[0] != 1:
            raise ValueError("Validation requires batch_size=1 because source image sizes may vary.")

        padder = InputPadder(left.shape, divis_by=32)
        padded_left, padded_right = padder.pad(left, right)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            prediction = model(padded_left, padded_right, max_disp=max_disp, test_mode=True)
        prediction = padder.unpad(prediction.float())
        metrics.update(prediction, target, valid)
        if save_vis_dir is not None:
            scene_name = Path(str(batch["name"][0])).name
            traditional_label = batch.get("traditional_label", "Previous algorithm")
            if isinstance(traditional_label, (list, tuple)):
                traditional_label = traditional_label[0]
            save_validation_vis(
                Path(save_vis_dir) / scene_name / "vis.png",
                left=left[0],
                prediction=prediction[0],
                target=target[0],
                valid=valid[0],
                evaluation_protocol=evaluation_protocol,
                traditional=batch.get("traditional_disparity"),
                traditional_label=str(traditional_label),
                disparity_max=max_disp,
                error_max=vis_error_max,
            )
        total_pixels_value = batch.get("evaluation_pixels", target[0].numel())
        if torch.is_tensor(total_pixels_value):
            total_pixels_value = int(total_pixels_value.reshape(-1)[0].item())
        elif isinstance(total_pixels_value, (list, tuple)):
            total_pixels_value = int(total_pixels_value[0])
        scene_rows.append(
            disparity_metric_row(
                prediction,
                target,
                valid,
                name=batch["name"][0],
                total_pixels=total_pixels_value,
            )
        )

        if batch_index == 0:
            logger.info("validation first_sample=%s shape=%s", batch["name"][0], tuple(left.shape))

    if evaluation_protocol == "tradition":
        result = aggregate_tradition_metrics(
            scene_rows,
            excluded_scenes=excluded_scenes,
            epe_threshold=epe_threshold,
        )
    else:
        result = metrics.compute()
        result["scene_count"] = len(scene_rows)
        result["aggregation"] = "pixel_micro"
    if return_scene_metrics:
        result["scene_metrics"] = scene_rows
    if save_vis_dir is not None:
        result["vis_output_dir"] = str(Path(save_vis_dir).expanduser().resolve())
    result["seconds"] = time.perf_counter() - started
    return result
