"""Inputs and structured mask corruptions for clean-mask projection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config_experiment9 as config

sys.path.append(str(config.EXP8_DIR))
from utils.data import (  # noqa: E402
    RGB_MEAN,
    RGB_STD,
    binary_boundary,
    distance_boundary_target,
    geometry_channels,
    read_image_and_geometry,
    read_records,
)


def _ellipse_on_boundary(
    mask: np.ndarray, rng: np.random.Generator, add: bool, severity: float
) -> np.ndarray:
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    points = np.concatenate(contours, axis=0)[:, 0] if contours else np.empty((0, 2))
    if not len(points):
        return mask
    x, y = points[int(rng.integers(0, len(points)))]
    long_axis = max(3, int(round((8 + 24 * severity) * rng.uniform(0.7, 1.2))))
    short_axis = max(2, int(round((4 + 12 * severity) * rng.uniform(0.7, 1.2))))
    angle = float(rng.uniform(0.0, 180.0))
    value = 1.0 if add else 0.0
    output = mask.copy()
    cv2.ellipse(output, (int(x), int(y)), (long_axis, short_axis), angle, 0, 360, value, -1)
    return output


def _interior_hole(
    mask: np.ndarray, gt: np.ndarray, rng: np.random.Generator, severity: float
) -> np.ndarray:
    locations = np.argwhere(gt > 0.5)
    if not len(locations):
        return mask
    y, x = locations[int(rng.integers(0, len(locations)))]
    axis_x = max(2, int(round(3 + 13 * severity * rng.uniform(0.5, 1.1))))
    axis_y = max(2, int(round(3 + 13 * severity * rng.uniform(0.5, 1.1))))
    output = mask.copy()
    cv2.ellipse(output, (int(x), int(y)), (axis_x, axis_y), 0, 0, 360, 0.0, -1)
    return output


def structured_corruption(
    probability: np.ndarray,
    gt: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Simulate the connected overflows, missing edges and holes seen in practice."""
    if rng.random() >= config.CORRUPTION_PROBABILITY:
        return probability.astype(np.float32), 0.0
    severity = float(rng.uniform(0.15, 1.0))
    source = probability if rng.random() < 0.55 else gt.astype(np.float32)
    corrupted = source.copy()
    operation_count = int(rng.integers(1, config.MAX_STRUCTURED_OPERATIONS + 1))
    for _ in range(operation_count):
        operation = int(rng.integers(0, 5))
        if operation in (0, 1):
            radius = max(1, int(round(1 + severity * (config.MAX_MORPH_RADIUS - 1))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            mode = cv2.MORPH_DILATE if operation == 0 else cv2.MORPH_ERODE
            corrupted = cv2.morphologyEx(corrupted, mode, kernel)
        elif operation == 2:
            corrupted = _ellipse_on_boundary(corrupted, rng, add=True, severity=severity)
        elif operation == 3:
            corrupted = _ellipse_on_boundary(corrupted, rng, add=False, severity=severity)
        else:
            corrupted = _interior_hole(corrupted, gt, rng, severity)
    sigma = 0.3 + 1.2 * severity
    corrupted = cv2.GaussianBlur(corrupted.astype(np.float32), (0, 0), sigma)
    corrupted += rng.normal(0.0, 0.02 * severity, corrupted.shape).astype(np.float32)
    # Keep part of the real Base prediction so synthetic samples remain on the
    # same input distribution as deployment.
    corrupted = 0.80 * corrupted + 0.20 * probability
    return np.clip(corrupted, 1e-4, 1.0 - 1e-4), severity


class CleanMaskDataset(Dataset):
    def __init__(
        self,
        split: str,
        augment: bool,
        seed: int,
        dataset: Path = config.DATASET_DIR,
        coarse_dir: Path = config.COARSE_DIR,
    ) -> None:
        self.split = split
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.dataset = dataset.resolve()
        self.coarse_dir = coarse_dir.resolve()
        self.records = read_records(self.dataset, split)
        missing = [
            record["name"]
            for record in self.records
            if not (self.coarse_dir / split / f"{record['name']}.npz").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} coarse predictions; run "
                "prepare_coarse_predictions.py first"
            )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        width, height = config.PROJECTOR_WIDTH, config.PROJECTOR_HEIGHT
        image, disparity, valid = read_image_and_geometry(
            config.ROOT, self.dataset, record, width, height
        )
        mask = cv2.imread(
            str(self.dataset / record["mask"]), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise FileNotFoundError(self.dataset / record["mask"])
        gt = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 127
        with np.load(
            self.coarse_dir / self.split / f"{record['name']}.npz",
            allow_pickle=False,
        ) as values:
            coarse = values["probability"].astype(np.float32)
            coarse_boundary = values["boundary"].astype(np.float32)

        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index * 997)
        severity = 0.0
        if self.augment:
            coarse, severity = structured_corruption(coarse, gt, rng)
            if severity > 0.0:
                synthetic_boundary = binary_boundary(coarse >= 0.5)
                synthetic_boundary = cv2.GaussianBlur(
                    synthetic_boundary, (0, 0), 0.8
                )
                coarse_boundary = np.clip(
                    0.8 * synthetic_boundary + 0.2 * coarse_boundary, 0.0, 1.0
                )
            if rng.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                disparity = np.ascontiguousarray(disparity[:, ::-1])
                valid = np.ascontiguousarray(valid[:, ::-1])
                gt = np.ascontiguousarray(gt[:, ::-1])
                coarse = np.ascontiguousarray(coarse[:, ::-1])
                coarse_boundary = np.ascontiguousarray(coarse_boundary[:, ::-1])

        image = image.astype(np.float32) / 255.0
        rgb = ((image - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
        geometry = geometry_channels(disparity, valid).transpose(2, 0, 1)
        severity_map = np.full((1, height, width), severity, dtype=np.float32)
        features = np.concatenate(
            (
                rgb.astype(np.float32),
                geometry.astype(np.float32),
                coarse[None].astype(np.float32),
                coarse_boundary[None].astype(np.float32),
                severity_map,
            ),
            axis=0,
        )
        gt_float = gt.astype(np.float32)
        boundary = binary_boundary(gt_float)
        boundary_distance = distance_boundary_target(boundary)
        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(gt_float[None]),
            "boundary": torch.from_numpy(boundary[None]),
            "boundary_distance": torch.from_numpy(boundary_distance[None]),
            "name": record["name"],
            "category": record["category"],
        }
