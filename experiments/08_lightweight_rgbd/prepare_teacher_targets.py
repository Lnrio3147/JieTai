#!/usr/bin/env python3
"""Precompute binary Experiment 7.2/7.1 teacher targets for all splits.

Teacher A is Experiment 7.2 (adaptive overflow rescue). Teacher B is Experiment
7.1 (continuous solid). No ground-truth mask is read while producing targets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config_experiment8 as config
from utils.data import disparity_path, read_disparity, read_records
from utils.postprocess import experiment4_refine, find_overflow_rescue, geometric_refine


def load_exp7_training_module():
    path = config.EXP7_DIR / "scripts/train_rgbd_fusion.py"
    spec = importlib.util.spec_from_file_location("experiment7_training", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=config.TEACHER_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=config.TEACHER_TARGET_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_existing_targets(
    dataset: Path, split: str, output: Path, force: bool
) -> bool:
    if dataset.resolve() != config.DATASET_DIR.resolve():
        return False
    if split not in ("val", "test"):
        return False
    sources = {
        "teacher_a": config.EXP7_OVERFLOW_RUN / split / "masks",
        "teacher_b": config.EXP7_GEOMETRY_RUN / split / "masks",
    }
    records = read_records(dataset, split)
    if not all(
        (directory / f"{record['name']}.png").is_file()
        for directory in sources.values()
        for record in records
    ):
        return False
    for teacher, source in sources.items():
        destination = output / teacher / split
        destination.mkdir(parents=True, exist_ok=True)
        for record in records:
            target = destination / f"{record['name']}.png"
            if force or not target.is_file():
                shutil.copy2(source / target.name, target)
    return True


class TeacherInferenceDataset(Dataset):
    """RGB-D inputs for the frozen Experiment 7 model without reading GT."""

    def __init__(self, exp7, dataset: Path, split: str) -> None:
        self.exp7 = exp7
        self.dataset = dataset
        self.records = read_records(dataset, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = cv2.imread(
            str(self.dataset / record["image"]), cv2.IMREAD_COLOR
        )
        if image is None:
            raise FileNotFoundError(self.dataset / record["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if record.get("disparity"):
            selected_disparity = self.dataset / record["disparity"]
        else:
            selected_disparity = disparity_path(config.ROOT, record)
        raw_disparity = read_disparity(selected_disparity)
        disparity, valid = self.exp7.normalize_disparity(raw_disparity)
        size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        disparity = cv2.resize(disparity, size, interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        image = (image - self.exp7.RGB_MEAN) / self.exp7.RGB_STD
        depth = self.exp7.depth_channels(disparity, valid)
        return {
            "rgb": torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)),
            "depth": torch.from_numpy(depth.transpose(2, 0, 1)),
            "name": record["name"],
        }


@torch.inference_mode()
def predict_split(
    exp7,
    model: torch.nn.Module,
    dataset: Path,
    split: str,
    output: Path,
    threshold: float,
    device: torch.device,
    amp: bool,
    batch_size: int,
    workers: int,
    force: bool,
) -> dict:
    records = read_records(dataset, split)
    teacher_a_dir = output / "teacher_a" / split
    teacher_b_dir = output / "teacher_b" / split
    for directory in (teacher_a_dir, teacher_b_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not force and all(
        (teacher_a_dir / f"{record['name']}.png").is_file()
        and (teacher_b_dir / f"{record['name']}.png").is_file()
        for record in records
    ):
        return {"split": split, "count": len(records), "skipped": True}

    inference_dataset = TeacherInferenceDataset(exp7, dataset, split)
    loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    probabilities: dict[str, np.ndarray] = {}
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(rgb, depth)
        values = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for name, probability in zip(batch["name"], values):
            probabilities[name] = cv2.resize(
                probability,
                (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

    triggered = []
    for record in records:
        name = record["name"]
        probability = probabilities[name]
        if record.get("disparity"):
            selected_disparity = dataset / record["disparity"]
        else:
            selected_disparity = disparity_path(config.ROOT, record)
        raw_disparity = read_disparity(selected_disparity)
        disparity = cv2.resize(
            raw_disparity,
            (probability.shape[1], probability.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        exp4, _ = experiment4_refine(probability, disparity, threshold)
        teacher_b = geometric_refine(
            exp4,
            gaussian_sigma=config.GEOMETRY_GAUSSIAN_SIGMA,
            binary_threshold=config.GEOMETRY_BINARY_THRESHOLD,
            closing_radius=config.GEOMETRY_CLOSING_RADIUS,
            preserve_hole_area=config.GEOMETRY_PRESERVE_HOLE_AREA,
        )
        rescue, decision = find_overflow_rescue(
            probability,
            low_threshold=threshold,
            reference_threshold=config.OVERFLOW_REFERENCE_THRESHOLD,
            search_start=config.OVERFLOW_SEARCH_START,
            search_stop=config.OVERFLOW_SEARCH_STOP,
            search_step=config.OVERFLOW_SEARCH_STEP,
            max_reference_area_ratio=config.OVERFLOW_MAX_REFERENCE_AREA_RATIO,
            min_reference_bbox_fraction=config.OVERFLOW_MIN_REFERENCE_BBOX_FRACTION,
            max_step_area_ratio=config.OVERFLOW_MAX_STEP_AREA_RATIO,
            max_bbox_contraction_ratio=config.OVERFLOW_MAX_BBOX_CONTRACTION_RATIO,
            min_candidate_area_fraction=config.OVERFLOW_MIN_CANDIDATE_AREA_FRACTION,
            closing_radius=config.OVERFLOW_CLOSING_RADIUS,
        )
        teacher_a = teacher_b if rescue is None else rescue
        if decision["triggered"]:
            triggered.append(name)
        cv2.imwrite(str(teacher_a_dir / f"{name}.png"), teacher_a.astype(np.uint8) * 255)
        cv2.imwrite(str(teacher_b_dir / f"{name}.png"), teacher_b.astype(np.uint8) * 255)
    return {"split": split, "count": len(records), "triggered": triggered, "skipped": False}


def generate_teacher_targets(
    dataset: Path = config.DATASET_DIR,
    checkpoint: Path = config.TEACHER_CHECKPOINT,
    output: Path = config.TEACHER_TARGET_DIR,
    batch_size: int = 1,
    workers: int = 2,
    force: bool = False,
) -> dict:
    dataset = dataset.resolve()
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.mkdir(parents=True, exist_ok=True)

    copied = {
        split: copy_existing_targets(dataset, split, output, force)
        for split in ("val", "test")
    }
    split_records = {
        split: read_records(dataset, split) for split in ("train", "val", "test")
    }
    splits_to_predict = [
        split
        for split in ("train", "val", "test")
        if not copied.get(split, False)
        and (
            force
            or not all(
                (output / teacher / split / f"{record['name']}.png").is_file()
                for teacher in ("teacher_a", "teacher_b")
                for record in split_records[split]
            )
        )
    ]
    results: list[dict] = []
    if splits_to_predict:
        exp7 = load_exp7_training_module()
        run_config = json.loads(
            (config.EXP7_SOURCE_RUN / "run_config.json").read_text(encoding="utf-8")
        )
        pretrained_path = Path(run_config["pretrained"])
        pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model = exp7.RGBDFusionNet(pretrained)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        amp = device.type == "cuda"
        threshold = float(
            json.loads(
                (config.EXP7_SOURCE_RUN / "summary.json").read_text(encoding="utf-8")
            )["selected_probability_threshold"]
        )
        for split in splits_to_predict:
            results.append(
                predict_split(
                    exp7,
                    model,
                    dataset,
                    split,
                    output,
                    threshold,
                    device,
                    amp,
                    batch_size,
                    workers,
                    force,
                )
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        results.extend(
            {
                "split": split,
                "count": len(split_records[split]),
                "skipped": True,
                "reason": "all frozen teacher masks already exist",
            }
            for split in ("train", "val", "test")
        )
    for split in ("val", "test"):
        if copied[split]:
            results.append(
                {
                    "split": split,
                    "copied_from_experiment7": True,
                    "count": len(split_records[split]),
                }
            )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "teacher_checkpoint": str(checkpoint),
        "teacher_a": "Experiment 7.2 final binary mask; erosion is applied after training resize",
        "teacher_b": "Experiment 7.1 continuous-solid binary mask",
        "ground_truth_used_to_generate_targets": False,
        "splits": results,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = generate_teacher_targets(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output=args.output,
        batch_size=args.batch_size,
        workers=args.workers,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
