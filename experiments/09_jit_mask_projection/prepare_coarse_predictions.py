#!/usr/bin/env python3
"""Cache frozen Experiment 8 Base probabilities and boundary predictions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import config_experiment9 as config

sys.path.insert(0, str(config.EXP8_DIR))
from models.student_network import create_student  # noqa: E402
from utils.data import WorkpieceStudentDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=config.EXP8_BASE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=config.COARSE_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def generate(
    dataset: Path = config.DATASET_DIR,
    checkpoint: Path = config.EXP8_BASE_CHECKPOINT,
    output: Path = config.COARSE_DIR,
    batch_size: int = 2,
    workers: int = 2,
    force: bool = False,
) -> dict:
    dataset_path = dataset.resolve()
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_student(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    manifest = []
    for split in ("train", "val", "test"):
        destination = output / split
        destination.mkdir(parents=True, exist_ok=True)
        split_dataset = WorkpieceStudentDataset(
            root=config.ROOT,
            dataset=dataset_path,
            split=split,
            width=config.BASE_INPUT_WIDTH,
            height=config.BASE_INPUT_HEIGHT,
            augment=False,
            seed=config.SEED,
        )
        missing = [
            record for record in split_dataset.records
            if force or not (destination / f"{record['name']}.npz").is_file()
        ]
        if not missing:
            manifest.append(
                {"split": split, "count": len(split_dataset), "skipped": True}
            )
            continue
        loader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        written = 0
        for batch in loader:
            names = batch["name"]
            if not force and all((destination / f"{name}.npz").is_file() for name in names):
                continue
            rgb = batch["rgb"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                mask_logits, boundary_logits = model(rgb, geometry)
            probabilities = torch.sigmoid(mask_logits).float().cpu().numpy()[:, 0]
            boundaries = torch.sigmoid(boundary_logits).float().cpu().numpy()[:, 0]
            for name, probability, boundary in zip(names, probabilities, boundaries):
                path = destination / f"{name}.npz"
                if path.is_file() and not force:
                    continue
                size = (config.PROJECTOR_WIDTH, config.PROJECTOR_HEIGHT)
                probability = cv2.resize(probability, size, interpolation=cv2.INTER_LINEAR)
                boundary = cv2.resize(boundary, size, interpolation=cv2.INTER_LINEAR)
                np.savez_compressed(
                    path,
                    probability=probability.astype(np.float16),
                    boundary=boundary.astype(np.float16),
                )
                written += 1
        manifest.append(
            {
                "split": split,
                "count": len(split_dataset),
                "written": written,
                "skipped": False,
            }
        )
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "source_checkpoint": str(checkpoint),
        "ground_truth_used": False,
        "splits": manifest,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            generate(
                dataset=args.dataset,
                checkpoint=args.checkpoint,
                output=args.output,
                batch_size=args.batch_size,
                workers=args.workers,
                force=args.force,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
