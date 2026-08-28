#!/usr/bin/env python3
"""Train a one-step x-prediction clean-mask projector."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config_experiment9 as config
from data import CleanMaskDataset
from models.clean_mask_projector import create_projector

sys.path.insert(0, str(config.EXP8_DIR))
from utils.losses import base_loss  # noqa: E402
from utils.metrics import aggregate_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--coarse-dir", type=Path, default=config.COARSE_DIR)
    parser.add_argument("--output", type=Path, default=config.RUN_DIR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: Path,
    records: list[dict[str, str]],
    device: torch.device,
    amp: bool,
) -> dict:
    model.eval()
    masks: dict[str, np.ndarray] = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(features)
        probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for name, probability in zip(batch["name"], probabilities):
            masks[name] = probability >= 0.5
    return aggregate_metrics(
        dataset, records, masks, config.BOUNDARY_TOLERANCE
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    validation: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "validation": validation,
            "method": "JiT-inspired direct clean-mask x-prediction",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(config.SEED)
    dataset_path = args.dataset.resolve()
    coarse_dir = args.coarse_dir.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_options = {"dataset": dataset_path, "coarse_dir": coarse_dir}
    train_set = CleanMaskDataset(
        "train", augment=True, seed=config.SEED, **dataset_options
    )
    val_set = CleanMaskDataset(
        "val", augment=False, seed=config.SEED, **dataset_options
    )
    counts = Counter(record["category"] for record in train_set.records)
    sample_weights = [
        counts[record["category"]] ** (-config.CATEGORY_BALANCE_POWER)
        for record in train_set.records
    ]
    samples_per_epoch = int(math.ceil(len(train_set) * 1.2))
    sampler = WeightedRandomSampler(sample_weights, samples_per_epoch, replacement=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = create_projector(
        in_channels=config.PROJECTOR_INPUT_CHANNELS,
        channels=config.PROJECTOR_CHANNELS,
        bottleneck_channels=config.BOTTLENECK_CHANNELS,
        patch_size=config.PATCH_SIZE,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > config.MAX_EXTRA_PARAMS:
        raise RuntimeError(
            f"Projector has {parameter_count:,} parameters; limit is "
            f"{config.MAX_EXTRA_PARAMS:,}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "single-pass residual Clean-Mask Projector",
        "paper_idea": "JiT x-prediction + low-dimensional manifold bottleneck",
        "not_used": ["iterative diffusion sampling", "large Vision Transformer"],
        "input_channels": [
            "RGB x3",
            "robust disparity",
            "Sobel disparity",
            "disparity validity",
            "Exp8 Base probability",
            "Exp8 Base boundary",
            "corruption severity",
        ],
        "projector_size_width_height": [
            config.PROJECTOR_WIDTH,
            config.PROJECTOR_HEIGHT,
        ],
        "parameter_count": parameter_count,
        "dataset": str(dataset_path),
        "coarse_dir": str(coarse_dir),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": config.WEIGHT_DECAY,
        "amp": amp,
        "device": str(device),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "corruption_probability": config.CORRUPTION_PROBABILITY,
        "loss": "0.35 BCE + 0.55 Tversky + 0.10 Boundary(BCE+DT)",
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    # With a zero-initialized correction head, epoch 0 is exactly Exp8 Base.
    initial_validation = validate(
        model, val_loader, dataset_path, val_set.records, device, amp
    )
    best_iou = float(initial_validation["overall"]["foreground_iou"])
    best_epoch = 0
    save_checkpoint(output / "best.pt", model, 0, initial_validation)
    history: list[dict] = [
        {
            "epoch": 0,
            "val_iou": best_iou,
            "val_precision": initial_validation["overall"]["precision"],
            "val_recall": initial_validation["overall"]["recall"],
            "val_boundary_f1": initial_validation["overall"]["boundary_f1"],
        }
    ]
    print(json.dumps(history[0]), flush=True)
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        running = defaultdict(float)
        batches = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            boundary = batch["boundary"].to(device, non_blocking=True)
            boundary_distance = batch["boundary_distance"].to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits, boundary_logits = model(features)
                loss, parts = base_loss(
                    logits,
                    boundary_logits,
                    target,
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

        validation = validate(
            model, val_loader, dataset_path, val_set.records, device, amp
        )
        metrics = validation["overall"]
        record = {
            "epoch": epoch,
            **{key: value / max(batches, 1) for key, value in running.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_iou": metrics["foreground_iou"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_boundary_f1": metrics["boundary_f1"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        score = float(metrics["foreground_iou"])
        if score > best_iou:
            best_iou = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(output / "best.pt", model, epoch, validation)
        else:
            stale_epochs += 1
        if epoch % 10 == 0:
            save_checkpoint(output / f"epoch_{epoch:03d}.pt", model, epoch, validation)
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in history), encoding="utf-8"
        )
        if stale_epochs >= args.patience:
            break

    summary = {
        "completed": True,
        "best_epoch": best_epoch,
        "best_validation_iou": best_iou,
        "epochs_run": len(history) - 1,
        "checkpoint": str(output / "best.pt"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
