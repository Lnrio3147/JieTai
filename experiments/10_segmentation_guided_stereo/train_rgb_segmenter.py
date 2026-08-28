#!/usr/bin/env python3
"""Train the RGB-only foreground segmenter used before stereo matching."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config_experiment10 as config
from models.rgb_segmenter import create_rgb_segmenter
from utils.data import RGBMaskDataset
from utils.losses import segmentation_loss
from utils.metrics import aggregate_metrics, select_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--output", type=Path, default=config.SEGMENTER_RUN_DIR)
    parser.add_argument("--rgbd-initialization", type=Path, default=config.RGBD_INITIALIZATION)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--encoder-learning-rate", type=float, default=config.ENCODER_LEARNING_RATE
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-rgbd-initialization", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.inference_mode()
def predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    output = {}
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(rgb)
        probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for name, probability in zip(batch["name"], probabilities):
            output[name] = np.asarray(probability, dtype=np.float32)
    return output


def resized_predictions(
    probabilities: dict[str, np.ndarray], threshold: float
) -> dict[str, np.ndarray]:
    return {
        name: cv2.resize(
            (probability >= threshold).astype(np.uint8),
            (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        for name, probability in probabilities.items()
    }


def write_threshold_sweep(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_everything(config.SEED)
    dataset_path = args.dataset.resolve()
    output = args.output.resolve()
    initialization = args.rgbd_initialization.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    use_rgbd_initialization = (
        not args.no_rgbd_initialization and initialization.is_file()
    )
    use_imagenet = not args.no_pretrained and not use_rgbd_initialization
    model = create_rgb_segmenter(pretrained=use_imagenet)
    initialization_summary = None
    if use_rgbd_initialization:
        initialization_summary = model.initialize_from_rgbd(initialization)
    model.to(device)

    train_set = RGBMaskDataset(
        dataset_path,
        "train",
        config.IMAGE_WIDTH,
        config.IMAGE_HEIGHT,
        augment=True,
        seed=config.SEED,
    )
    val_set = RGBMaskDataset(
        dataset_path,
        "val",
        config.IMAGE_WIDTH,
        config.IMAGE_HEIGHT,
        augment=False,
        seed=config.SEED,
    )
    test_set = RGBMaskDataset(
        dataset_path,
        "test",
        config.IMAGE_WIDTH,
        config.IMAGE_HEIGHT,
        augment=False,
        seed=config.SEED,
    )
    counts = Counter(record["category"] for record in train_set.records)
    weights = [
        counts[record["category"]] ** (-config.CATEGORY_BALANCE_POWER)
        for record in train_set.records
    ]
    samples_per_epoch = int(math.ceil(len(train_set) * 1.2))
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    encoder_parameters = list(model.rgb_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": decoder_parameters, "lr": args.learning_rate},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "RGB-only MobileNetV4-Conv-S + EMCAD",
        "dataset": str(dataset_path),
        "image_size_width_height": [config.IMAGE_WIDTH, config.IMAGE_HEIGHT],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "amp": amp,
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "rgbd_initialization": str(initialization) if use_rgbd_initialization else None,
        "initialization_summary": initialization_summary,
        "imagenet_pretrained": use_imagenet,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    history = []
    best_iou = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        running = defaultdict(float)
        batches = 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            boundary = batch["boundary"].to(device, non_blocking=True)
            boundary_distance = batch["boundary_distance"].to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                mask_logits, boundary_logits = model(rgb)
                loss, parts = segmentation_loss(
                    mask_logits,
                    boundary_logits,
                    mask,
                    boundary,
                    boundary_distance,
                    bce_weight=config.BCE_WEIGHT,
                    tversky_weight=config.TVERSKY_WEIGHT,
                    boundary_weight=config.BOUNDARY_WEIGHT,
                    tversky_alpha=config.TVERSKY_ALPHA,
                    tversky_beta=config.TVERSKY_BETA,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running["loss"] += float(loss.detach())
            for key, value in parts.items():
                running[key] += value
            batches += 1
        scheduler.step()

        probabilities = predict_probabilities(model, val_loader, device, amp)
        validation = aggregate_metrics(
            dataset_path,
            val_set.records,
            resized_predictions(probabilities, 0.5),
            config.BOUNDARY_TOLERANCE,
        )
        metrics = validation["overall"]
        record = {
            "epoch": epoch,
            **{key: value / max(batches, 1) for key, value in running.items()},
            "learning_rate": optimizer.param_groups[1]["lr"],
            "val_iou_at_0_5": metrics["foreground_iou"],
            "val_precision_at_0_5": metrics["precision"],
            "val_recall_at_0_5": metrics["recall"],
            "val_boundary_f1_at_0_5": metrics["boundary_f1"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        score = float(metrics["foreground_iou"])
        if score > best_iou:
            best_iou = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_at_0_5": validation,
                    "model_name": config.MODEL_NAME,
                },
                output / "best.pt",
            )
        else:
            stale_epochs += 1
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in history), encoding="utf-8"
        )
        if stale_epochs >= args.patience:
            break

    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    validation_probabilities = predict_probabilities(model, val_loader, device, amp)
    resized_validation_probabilities = {
        name: cv2.resize(
            probability,
            (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        for name, probability in validation_probabilities.items()
    }
    threshold, sweep = select_threshold(
        dataset_path,
        val_set.records,
        resized_validation_probabilities,
        config.THRESHOLD_CANDIDATES,
        config.THRESHOLD_RECALL_FLOOR,
        config.BOUNDARY_TOLERANCE,
    )
    validation = aggregate_metrics(
        dataset_path,
        val_set.records,
        {
            name: probability >= threshold
            for name, probability in resized_validation_probabilities.items()
        },
        config.BOUNDARY_TOLERANCE,
    )
    test_probabilities = predict_probabilities(model, test_loader, device, amp)
    test = aggregate_metrics(
        dataset_path,
        test_set.records,
        resized_predictions(test_probabilities, threshold),
        config.BOUNDARY_TOLERANCE,
    )
    checkpoint["selected_threshold"] = threshold
    checkpoint["validation"] = validation
    torch.save(checkpoint, output / "best.pt")
    write_threshold_sweep(output / "threshold_sweep.csv", sweep)
    summary = {
        "completed": True,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "selected_threshold": threshold,
        "validation": validation,
        "frozen_test": test,
        "checkpoint": str(output / "best.pt"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
