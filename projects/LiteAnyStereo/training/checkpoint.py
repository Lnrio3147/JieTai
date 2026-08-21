"""Checkpoint and run-record helpers."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch


def safe_torch_load(path, map_location="cpu"):
    """Load tensor/state-dict checkpoints without allowing arbitrary pickle globals."""

    return torch.load(path, map_location=map_location, weights_only=True)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_epe,
    config,
):
    target_model = model.module if hasattr(model, "module") else model
    payload = {
        "model": target_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_epe": float(best_epe),
        "config": dict(config),
    }
    atomic_torch_save(payload, path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def environment_summary():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "numpy": _package_version("numpy"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }


def _package_version(name):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None
