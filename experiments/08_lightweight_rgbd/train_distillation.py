#!/usr/bin/env python3
"""Train the Experiment 8 base or hard-mask-distilled student."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config_experiment8 as config
from models.student_network import create_student
from utils.data import WorkpieceStudentDataset
from utils.losses import base_loss, distilled_loss
from utils.metrics import aggregate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "distilled"), required=True)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument(
        "--teacher-root", type=Path, default=config.TEACHER_TARGET_DIR
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--encoder-learning-rate", type=float, default=config.ENCODER_LEARNING_RATE
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--prepare-teachers", action="store_true")
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
        geometry = batch["geometry"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(rgb, geometry)
        probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for name, probability in zip(batch["name"], probabilities):
            output[name] = np.asarray(probability, dtype=np.float32)
    return output


def main() -> None:
    args = parse_args()
    seed_everything(config.SEED)
    dataset_path = args.dataset.resolve()
    teacher_root = args.teacher_root.resolve()
    default_output = (
        config.BASE_RUN_DIR if args.mode == "base" else config.DISTILLED_RUN_DIR
    )
    output = (args.output or default_output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    require_teachers = args.mode == "distilled"
    if require_teachers and args.prepare_teachers:
        from prepare_teacher_targets import generate_teacher_targets

        generate_teacher_targets(
            dataset=dataset_path,
            output=teacher_root,
            workers=max(1, args.workers // 2),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    model = create_student(pretrained=not args.no_pretrained).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= config.MAX_SEGMENTER_PARAMS:
        raise RuntimeError(
            f"Student has {parameter_count:,} parameters; limit is "
            f"{config.MAX_SEGMENTER_PARAMS:,}"
        )

    dataset_options = {
        "root": config.ROOT,
        "dataset": dataset_path,
        "width": config.IMAGE_WIDTH,
        "height": config.IMAGE_HEIGHT,
        "seed": config.SEED,
        "teacher_root": teacher_root,
        "require_teachers": require_teachers,
        "teacher_a_erosion_kernel": config.TEACHER_A_EROSION_KERNEL,
        "teacher_a_erosion_iterations": config.TEACHER_A_EROSION_ITERATIONS,
    }
    train_set = WorkpieceStudentDataset(split="train", augment=True, **dataset_options)
    val_set = WorkpieceStudentDataset(
        split="val", augment=False, **{**dataset_options, "require_teachers": False}
    )
    counts = Counter(record["category"] for record in train_set.records)
    sample_weights = [
        counts[record["category"]] ** (-config.CATEGORY_BALANCE_POWER)
        for record in train_set.records
    ]
    samples_per_epoch = int(math.ceil(len(train_set) * 1.2))
    sampler = WeightedRandomSampler(sample_weights, samples_per_epoch, replacement=True)
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
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    encoder_parameters = list(model.rgb_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    student_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": student_parameters, "lr": args.learning_rate},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "architecture": "MobileNetV4-Conv-S + three-layer geometry + four spatial gates + EMCAD",
        "image_size_width_height": [config.IMAGE_WIDTH, config.IMAGE_HEIGHT],
        "tensor_shape": [args.batch_size, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH],
        "dataset": str(dataset_path),
        "teacher_checkpoint": str(config.TEACHER_CHECKPOINT) if require_teachers else None,
        "teacher_target_dir": str(teacher_root) if require_teachers else None,
        "parameter_count": parameter_count,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "weight_decay": config.WEIGHT_DECAY,
        "pretrained": not args.no_pretrained,
        "amp": amp,
        "device": str(device),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "samples_per_epoch": samples_per_epoch,
        "loss": (
            "0.35 BCE + 0.55 Tversky + 0.10 Boundary(BCE+DT)"
            if args.mode == "base"
            else "0.5*(Dice_GT+BCE_GT)+0.3*BCE_TeacherA+0.2*BCE_TeacherB+0.1*Boundary(BCE+DT)"
        ),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    val_records = val_set.records
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
            geometry = batch["geometry"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            boundary = batch["boundary"].to(device, non_blocking=True)
            boundary_distance = batch["boundary_distance"].to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                mask_logits, boundary_logits = model(rgb, geometry)
                if args.mode == "base":
                    loss, parts = base_loss(
                        mask_logits,
                        boundary_logits,
                        mask,
                        boundary,
                        boundary_distance,
                        bce_weight=config.BASE_BCE_WEIGHT,
                        tversky_weight=config.BASE_TVERSKY_WEIGHT,
                        boundary_weight=config.BASE_BOUNDARY_WEIGHT,
                        tversky_alpha=config.TVERSKY_ALPHA,
                        tversky_beta=config.TVERSKY_BETA,
                    )
                else:
                    loss, parts = distilled_loss(
                        mask_logits,
                        boundary_logits,
                        mask,
                        boundary,
                        boundary_distance,
                        batch["teacher_a"].to(device, non_blocking=True),
                        batch["teacher_b"].to(device, non_blocking=True),
                        hard_weight=config.DISTILL_HARD_WEIGHT,
                        teacher_a_weight=config.DISTILL_TEACHER_A_WEIGHT,
                        teacher_b_weight=config.DISTILL_TEACHER_B_WEIGHT,
                        boundary_weight=config.DISTILL_BOUNDARY_WEIGHT,
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
        masks = {
            name: cv2.resize(
                (probability >= 0.5).astype(np.uint8),
                (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for name, probability in probabilities.items()
        }
        validation = aggregate_metrics(
            dataset_path, val_records, masks, config.BOUNDARY_TOLERANCE
        )
        metrics = validation["overall"]
        record = {
            "epoch": epoch,
            **{key: value / max(batches, 1) for key, value in running.items()},
            "learning_rate": optimizer.param_groups[1]["lr"],
            "val_iou": metrics["foreground_iou"],
            "val_macro_category_iou": metrics["macro_category_iou"],
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "model_name": config.MODEL_NAME,
                    "mode": args.mode,
                },
                output / "best.pt",
            )
        else:
            stale_epochs += 1
        if epoch % config.SNAPSHOT_EVERY == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "validation": validation},
                output / f"epoch_{epoch:03d}.pt",
            )
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in history), encoding="utf-8"
        )
        if stale_epochs >= args.patience:
            break

    summary = {
        "completed": True,
        "best_epoch": best_epoch,
        "best_validation_iou": best_iou,
        "epochs_run": len(history),
        "checkpoint": str(output / "best.pt"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
