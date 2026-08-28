"""Dataset and preprocessing for Experiment 8."""

from __future__ import annotations

import csv
import re
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


def scene_name(record: dict[str, str]) -> str:
    category = record["category"]
    prefix = f"{category}_"
    if not record["name"].startswith(prefix):
        raise ValueError(f"Unexpected sample name: {record['name']}")
    suffix = record["name"][len(prefix) :]
    if category in ("general", "scale", "jop1"):
        return suffix.replace("_", "-")
    stem, frame = suffix.rsplit("_", 1)
    return f"{stem}-{frame}"


def disparity_path(root: Path, record: dict[str, str]) -> Path:
    category = record["category"]
    scene = scene_name(record)
    if category == "jop1":
        return (
            root
            / "experiments/01_stereo_comparison/jop1/results/final_9/liteanystereo"
            / scene
            / "disp.npy"
        )
    group = {"general": "general_1221", "scale": "scale_1221"}.get(
        category, category
    )
    return (
        root
        / "experiments/01_stereo_comparison/rec_img_set/results/final_203/outputs"
        / group
        / scene
        / "liteanystereo/disp_full.npy"
    )


def read_disparity(path: Path) -> np.ndarray:
    """Read the NPY/PFM disparity formats used by the combined V3 dataset."""
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".pfm":
        with path.open("rb") as stream:
            header = stream.readline().decode("ascii").strip()
            if header not in ("PF", "Pf"):
                raise ValueError(f"Invalid PFM header in {path}: {header}")
            color = header == "PF"
            dimensions = stream.readline().decode("ascii").strip()
            while dimensions.startswith("#"):
                dimensions = stream.readline().decode("ascii").strip()
            match = re.fullmatch(r"(\d+)\s+(\d+)", dimensions)
            if match is None:
                raise ValueError(f"Invalid PFM dimensions in {path}: {dimensions}")
            width, height = (int(value) for value in match.groups())
            scale = float(stream.readline().decode("ascii").strip())
            dtype = "<f4" if scale < 0 else ">f4"
            channels = 3 if color else 1
            value = np.fromfile(
                stream, dtype=dtype, count=width * height * channels
            )
        expected = width * height * channels
        if value.size != expected:
            raise ValueError(f"Truncated PFM {path}: {value.size}/{expected}")
        shape = (height, width, channels) if color else (height, width)
        value = np.flipud(value.reshape(shape))
    else:
        raise ValueError(f"Unsupported disparity format: {path}")
    if value.ndim == 3:
        value = value[..., 0]
    return np.asarray(value, dtype=np.float32)


def robust_normalize_disparity(disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median/MAD normalization robust to glare and disparity outliers."""
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    values = disparity[valid]
    if values.size < 32:
        return np.zeros_like(disparity, dtype=np.float32), valid.astype(np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_scale = max(1.4826 * mad, 1e-3)
    z_score = (np.nan_to_num(disparity, nan=median) - median) / robust_scale
    normalized = (np.clip(z_score, -3.0, 3.0) + 3.0) / 6.0
    normalized[~valid] = 0.0
    return normalized.astype(np.float32), valid.astype(np.float32)


def geometry_channels(disparity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    values = magnitude[valid > 0.5]
    scale = float(np.percentile(values, 95.0)) if values.size else 1.0
    magnitude = np.clip(magnitude / max(scale, 1e-4), 0.0, 1.0)
    return np.stack((disparity, magnitude, valid), axis=2).astype(np.float32)


def binary_boundary(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return (
        cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    ).astype(np.float32)


def distance_boundary_target(boundary: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    inverse = (boundary <= 0.5).astype(np.uint8)
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    return np.exp(-distance / sigma).astype(np.float32)


def read_image_and_geometry(
    root: Path,
    dataset: Path,
    record: dict[str, str],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(dataset / record["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if record.get("disparity"):
        selected_disparity = dataset / record["disparity"]
    else:
        selected_disparity = disparity_path(root, record)
    raw_disparity = read_disparity(selected_disparity)
    disparity, valid = robust_normalize_disparity(raw_disparity)
    size = (width, height)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    disparity = cv2.resize(disparity, size, interpolation=cv2.INTER_LINEAR)
    valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    return image, disparity, valid


class WorkpieceStudentDataset(Dataset):
    def __init__(
        self,
        root: Path,
        dataset: Path,
        split: str,
        width: int,
        height: int,
        augment: bool,
        seed: int,
        teacher_root: Path | None = None,
        require_teachers: bool = False,
        teacher_a_erosion_kernel: int = 3,
        teacher_a_erosion_iterations: int = 1,
    ) -> None:
        self.root = root
        self.dataset = dataset
        self.split = split
        self.records = read_records(dataset, split)
        self.width = width
        self.height = height
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.teacher_root = teacher_root
        self.require_teachers = require_teachers
        self.teacher_a_erosion_kernel = teacher_a_erosion_kernel
        self.teacher_a_erosion_iterations = teacher_a_erosion_iterations
        if require_teachers:
            if teacher_root is None:
                raise ValueError("teacher_root is required for distilled training")
            for teacher in ("teacher_a", "teacher_b"):
                directory = teacher_root / teacher / split
                missing = [
                    record["name"]
                    for record in self.records
                    if not (directory / f"{record['name']}.png").is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Missing {len(missing)} {teacher} targets in {directory}; "
                        "run prepare_teacher_targets.py first"
                    )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _read_teacher(self, teacher: str, name: str) -> np.ndarray:
        assert self.teacher_root is not None
        path = self.teacher_root / teacher / self.split / f"{name}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        return cv2.resize(
            mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image, disparity, valid = read_image_and_geometry(
            self.root, self.dataset, record, self.width, self.height
        )
        mask = cv2.imread(str(self.dataset / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.dataset / record["mask"])
        mask = cv2.resize(
            mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        teacher_a = teacher_b = None
        if self.require_teachers:
            teacher_a = self._read_teacher("teacher_a", record["name"])
            teacher_b = self._read_teacher("teacher_b", record["name"])

        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index * 997)
        if self.augment and rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            disparity = np.ascontiguousarray(disparity[:, ::-1])
            valid = np.ascontiguousarray(valid[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            if teacher_a is not None and teacher_b is not None:
                teacher_a = np.ascontiguousarray(teacher_a[:, ::-1])
                teacher_b = np.ascontiguousarray(teacher_b[:, ::-1])

        image = image.astype(np.float32) / 255.0
        if self.augment:
            contrast = float(rng.uniform(0.85, 1.15))
            brightness = float(rng.uniform(-0.15, 0.15))
            image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

        rgb = (image - RGB_MEAN) / RGB_STD
        binary = (mask > 127).astype(np.float32)
        boundary = binary_boundary(binary)
        distance_target = distance_boundary_target(boundary)
        geometry = geometry_channels(disparity, valid)
        output: dict[str, torch.Tensor | str] = {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32)),
            "geometry": torch.from_numpy(geometry.transpose(2, 0, 1)),
            "mask": torch.from_numpy(binary[None]),
            "boundary": torch.from_numpy(boundary[None]),
            "boundary_distance": torch.from_numpy(distance_target[None]),
            "name": record["name"],
            "category": record["category"],
        }
        if teacher_a is not None and teacher_b is not None:
            erosion_kernel = np.ones(
                (self.teacher_a_erosion_kernel, self.teacher_a_erosion_kernel),
                dtype=np.uint8,
            )
            teacher_a = cv2.erode(
                (teacher_a > 127).astype(np.uint8),
                erosion_kernel,
                iterations=self.teacher_a_erosion_iterations,
            )
            output["teacher_a"] = torch.from_numpy(teacher_a[None].astype(np.float32))
            output["teacher_b"] = torch.from_numpy(
                (teacher_b[None] > 127).astype(np.float32)
            )
        return output
