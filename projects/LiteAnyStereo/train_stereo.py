#!/usr/bin/env python3
"""Single-GPU fine-tuning entry point for LiteAnyStereo."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.models import (
    build_model,
    load_model_weights,
    model_label,
    normalize_model_size,
    normalize_version,
    require_checkpoint,
    resolve_checkpoint,
)
from training.checkpoint import (
    environment_summary,
    safe_torch_load,
    save_training_checkpoint,
    write_json,
)
from training.data import build_datasets
from training.engine import train_one_epoch, validate
from training.metrics import TRADITION_EXCLUDED_SCENES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LiteAnyStereo with JMP-LF6020, ETH3D, KITTI 2015, or a CSV manifest."
    )
    parser.add_argument("--version", default="las1", help="model version: las1/v1 or las2/v2")
    parser.add_argument("--model_size", "--model-size", default=None, help="LAS2 size: s, m, l, or h")
    parser.add_argument(
        "--restore_ckpt",
        default=None,
        help="initial weights; defaults to the official checkpoint, or use 'none' for random initialization",
    )
    parser.add_argument("--resume", default=None, help="resume a training checkpoint produced by this script")
    parser.add_argument("--fnet_pretrained", action="store_true", help="request ImageNet backbone weights")

    parser.add_argument(
        "--dataset", choices=["synthetic", "jmp", "eth3d", "kitti15", "manifest"], default="synthetic"
    )
    parser.add_argument("--data_root", default=None, help="dataset root for --dataset jmp/eth3d/kitti15")
    parser.add_argument("--manifest", default=None, help="CSV manifest for --dataset manifest")
    parser.add_argument(
        "--evaluation_protocol",
        choices=["auto", "standard", "tradition"],
        default="auto",
        help="validation metric protocol; auto selects tradition for JMP and standard otherwise",
    )
    parser.add_argument(
        "--tradition_eval_root",
        default="../tradition_stereo/datasets/FDJYP-3",
        help="FDJYP-3 root containing full images and 818x512 disp_cropped.npy references",
    )
    parser.add_argument("--eval_epe_threshold", type=float, default=20.0)
    parser.add_argument(
        "--eval_epe_filter", action=argparse.BooleanOptionalAction, default=True,
        help="filter scenes above --eval_epe_threshold in tradition evaluation",
    )
    parser.add_argument(
        "--eval_exclude_scenes", action=argparse.BooleanOptionalAction, default=True,
        help="exclude tradition_stereo's four configured scenes",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2, help="KITTI validation fraction")
    parser.add_argument("--split_seed", type=int, default=42, help="seed for deterministic KITTI split")
    parser.add_argument("--synthetic_train_samples", type=int, default=16)
    parser.add_argument("--synthetic_val_samples", type=int, default=4)

    parser.add_argument("--output_dir", default="./runs/training/las1_finetune")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing run's files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crop_height", type=int, default=256)
    parser.add_argument("--crop_width", type=int, default=512)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate_before_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="stop after this many optimizer steps; 0 means no limit (useful for smoke tests)",
    )
    return parser.parse_args()


def validate_args(args):
    if args.resume and args.restore_ckpt is not None:
        raise ValueError("--resume and --restore_ckpt are mutually exclusive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch_size must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.crop_height <= 0 or args.crop_width <= 0:
        raise ValueError("crop dimensions must be positive")
    if args.crop_height % 32 or args.crop_width % 32:
        raise ValueError("crop dimensions must both be divisible by 32")
    if args.max_disp <= 0 or args.max_disp % 4:
        raise ValueError("--max_disp must be positive and divisible by 4")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("--lr must be positive and --weight_decay cannot be negative")
    if not 0.0 <= args.aux_weight <= 1.0:
        raise ValueError("--aux_weight must be between 0 and 1")
    if args.log_interval <= 0 or args.save_every <= 0 or args.max_steps < 0:
        raise ValueError("log/save intervals must be positive and --max_steps cannot be negative")
    if args.eval_epe_threshold <= 0:
        raise ValueError("--eval_epe_threshold must be positive")
    if args.dataset == "jmp" and args.evaluation_protocol == "tradition":
        if not args.tradition_eval_root:
            raise ValueError("--tradition_eval_root is required for tradition evaluation")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false; use --device cpu explicitly")


def evaluation_signature(config):
    """Return the validation settings that determine whether best EPE is comparable."""
    dataset = config.get("dataset")
    protocol = config.get("evaluation_protocol") or "standard"
    if protocol == "auto":
        protocol = "tradition" if dataset == "jmp" else "standard"
    signature = {"protocol": protocol}
    if protocol == "tradition":
        signature.update(
            {
                "epe_filter": bool(config.get("eval_epe_filter", True)),
                "epe_threshold": float(config.get("eval_epe_threshold", 20.0)),
                "exclude_scenes": bool(config.get("eval_exclude_scenes", True)),
            }
        )
    return signature


def configure_logging(output_dir, append):
    logger = logging.getLogger("liteanystereo.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    log_file = logging.FileHandler(output_dir / "train.log", mode="a" if append else "w", encoding="utf-8")
    log_file.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(log_file)
    return logger


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, args, training):
    generator = torch.Generator()
    generator.manual_seed(args.seed + (0 if training else 1))
    return DataLoader(
        dataset,
        batch_size=args.batch_size if training else 1,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def prepare_output(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "latest.pth"
    if latest.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"{latest} already exists. Use --resume {latest}, choose another --output_dir, "
            "or explicitly pass --overwrite."
        )
    return output_dir


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.evaluation_protocol == "auto":
        args.evaluation_protocol = "tradition" if args.dataset == "jmp" else "standard"
    validate_args(args)
    args.version = normalize_version(args.version)
    args.model_size = normalize_model_size(args.version, args.model_size)
    output_dir = prepare_output(args)
    logger = configure_logging(output_dir, append=bool(args.resume))
    seed_everything(args.seed, args.deterministic)

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    label = model_label(args.version, args.model_size)
    logger.info("Starting %s fine-tuning on %s (AMP=%s)", label, device, use_amp)

    resolved_resume = Path(args.resume).expanduser().resolve() if args.resume else None
    resolved_restore = None
    if not args.resume:
        restore_value = resolve_checkpoint(args.version, args.restore_ckpt, model_size=args.model_size)
        resolved_restore = Path(restore_value).expanduser().resolve() if restore_value is not None else None

    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["command"] = " ".join(shlex.quote(value) for value in sys.argv)
    config["resolved_resume"] = str(resolved_resume) if resolved_resume else None
    config["resolved_restore_ckpt"] = str(resolved_restore) if resolved_restore else None
    if resolved_restore and resolved_restore.is_file():
        config["restore_ckpt_sha256"] = file_sha256(resolved_restore)
    if args.resume:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write_json(output_dir / f"config_resume_{stamp}.json", config)
        write_json(output_dir / f"environment_resume_{stamp}.json", environment_summary())
    else:
        write_json(output_dir / "config.json", config)
        write_json(output_dir / "environment.json", environment_summary())

    train_dataset, val_dataset = build_datasets(args)
    train_loader = make_loader(train_dataset, args, training=True)
    val_loader = make_loader(val_dataset, args, training=False)
    if not train_loader:
        raise ValueError("Training data loader is empty")
    logger.info(
        "Dataset ready: train=%d val=%d evaluation_protocol=%s",
        len(train_dataset),
        len(val_dataset),
        args.evaluation_protocol,
    )
    validation_kwargs = {
        "evaluation_protocol": args.evaluation_protocol,
        "excluded_scenes": TRADITION_EXCLUDED_SCENES if args.eval_exclude_scenes else (),
        "epe_threshold": (
            args.eval_epe_threshold
            if args.evaluation_protocol == "tradition" and args.eval_epe_filter
            else None
        ),
    }

    model = build_model(
        args.version,
        fnet_pretrained=args.fnet_pretrained,
        model_size=args.model_size,
        max_disp=args.max_disp,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # LRScheduler performs an initial step during construction. Adding one keeps
    # the final optimizer update at OneCycle's minimum instead of one step past it.
    total_steps = args.epochs * len(train_loader) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        cycle_momentum=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    global_step = 0
    best_epe = float("inf")
    if args.resume:
        resume_path = resolved_resume
        require_checkpoint(resume_path)
        checkpoint = safe_torch_load(resume_path, map_location=device)
        load_model_weights(model, checkpoint, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        checkpoint_signature = evaluation_signature(checkpoint.get("config") or {})
        current_signature = evaluation_signature(vars(args))
        if checkpoint_signature == current_signature:
            best_epe = float(checkpoint.get("best_epe", best_epe))
        else:
            logger.warning(
                "Validation settings changed from %s to %s; resetting best EPE because the "
                "saved value is not comparable.",
                checkpoint_signature,
                current_signature,
            )
        logger.info("Resumed %s at epoch=%d step=%d", resume_path, start_epoch, global_step)
    else:
        restore_path = resolved_restore
        if restore_path is not None:
            require_checkpoint(restore_path)
            logger.info("Loading initial weights from %s", restore_path)
            load_model_weights(model, safe_torch_load(restore_path, map_location=device), strict=True)
        else:
            logger.warning("Training from random initialization")

    metrics_path = output_dir / "metrics.jsonl"
    metrics_mode = "a" if args.resume else "w"
    final_record = None
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_handle:
        def write_metric(record):
            metrics_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_handle.flush()

        if args.validate_before_training and not args.resume:
            baseline = validate(
                model, val_loader, device, args.max_disp, use_amp, logger, **validation_kwargs
            )
            baseline_record = {"event": "baseline_validation", "epoch": 0, "step": 0, **baseline}
            write_metric(baseline_record)
            logger.info(
                "Baseline validation: epe=%.4f d1=%.3f%% bad1=%.3f%% bad2=%.3f%% "
                "bad3=%.3f%% scenes=%d time=%.1fs",
                baseline["epe"],
                baseline["d1"],
                baseline["bad1"],
                baseline["bad2"],
                baseline["bad3"],
                baseline["scene_count"],
                baseline["seconds"],
            )

        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics, global_step, stopped = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
                global_step=global_step,
                max_disp=args.max_disp,
                aux_weight=args.aux_weight,
                grad_clip=args.grad_clip,
                amp=use_amp,
                logger=logger,
                jsonl_writer=write_metric,
                log_interval=args.log_interval,
                max_steps=args.max_steps,
            )
            val_metrics = validate(
                model, val_loader, device, args.max_disp, use_amp, logger, **validation_kwargs
            )
            final_record = {
                "event": "epoch_end",
                "epoch": epoch,
                "step": global_step,
                "train": train_metrics,
                "validation": val_metrics,
            }
            write_metric(final_record)
            logger.info(
                "Epoch %d complete: train_loss=%.5f train_epe=%.4f val_epe=%.4f "
                "val_d1=%.3f%% bad1=%.3f%% bad2=%.3f%% bad3=%.3f%%",
                epoch,
                train_metrics["loss"],
                train_metrics["epe"],
                val_metrics["epe"],
                val_metrics["d1"],
                val_metrics["bad1"],
                val_metrics["bad2"],
                val_metrics["bad3"],
            )

            improved = val_metrics["epe"] < best_epe
            if improved:
                best_epe = val_metrics["epe"]
            checkpoint_args = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_epe=best_epe,
                config=config,
            )
            save_training_checkpoint(output_dir / "latest.pth", **checkpoint_args)
            if improved:
                save_training_checkpoint(output_dir / "best.pth", **checkpoint_args)
                logger.info("New best checkpoint: epe=%.4f", best_epe)
            if epoch % args.save_every == 0:
                save_training_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", **checkpoint_args)

            if stopped or (args.max_steps and global_step >= args.max_steps):
                logger.info("Reached --max_steps=%d", args.max_steps)
                break

    summary = {
        "status": "completed",
        "model": label,
        "dataset": args.dataset,
        "evaluation_protocol": args.evaluation_protocol,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_requested": args.epochs,
        "last_epoch": final_record["epoch"] if final_record else start_epoch - 1,
        "global_step": global_step,
        "best_epe": best_epe,
        "last_validation": final_record["validation"] if final_record else None,
        "latest_checkpoint": str(output_dir / "latest.pth"),
        "best_checkpoint": str(output_dir / "best.pth"),
    }
    write_json(output_dir / "summary.json", summary)
    logger.info("Training completed. Summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
