"""RGB-only dataset and preprocessing for Experiment 10."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_records(dataset: Path, split: str) -> list[dict[str, str]]:
    path = dataset / "index" / f"{split}.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"Empty dataset split: {path}")
    return records


def binary_boundary(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return (
        cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    ).astype(np.float32)


def distance_boundary_target(boundary: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    inverse = (boundary <= 0.5).astype(np.uint8)
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    return np.exp(-distance / sigma).astype(np.float32)


def preprocess_rgb(image_bgr: np.ndarray, width: int, height: int) -> torch.Tensor:
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise ValueError("Expected a BGR image with three channels")
    image = cv2.cvtColor(image_bgr[..., :3], cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    rgb = ((image - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))


class RGBMaskDataset(Dataset):
    def __init__(
        self,
        dataset: Path,
        split: str,
        width: int,
        height: int,
        augment: bool,
        seed: int,
    ) -> None:
        self.dataset = Path(dataset)
        self.split = split
        self.records = read_records(self.dataset, split)
        self.width = int(width)
        self.height = int(height)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image_path = self.dataset / record["image"]
        mask_path = self.dataset / record["mask"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        if mask is None:
            raise FileNotFoundError(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )

        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index * 997)
        if self.augment and rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = image.astype(np.float32) / 255.0
        if self.augment:
            contrast = float(rng.uniform(0.80, 1.20))
            brightness = float(rng.uniform(-0.15, 0.15))
            gamma = float(rng.uniform(0.85, 1.15))
            image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
            image = np.power(image, gamma, dtype=np.float32)

        rgb = ((image - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
        binary = (mask > 127).astype(np.float32)
        boundary = binary_boundary(binary)
        distance = distance_boundary_target(boundary)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32)),
            "mask": torch.from_numpy(binary[None]),
            "boundary": torch.from_numpy(boundary[None]),
            "boundary_distance": torch.from_numpy(distance[None]),
            "name": record["name"],
            "category": record["category"],
        }
