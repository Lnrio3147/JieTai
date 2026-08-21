#!/usr/bin/env python3
"""Train and evaluate a frozen-disparity RGB-D workpiece segmenter.

LiteAnyStereo predictions are read from the existing Experiment 1 outputs and
never updated.  A pretrained RGB encoder and a separately initialized disparity
encoder are fused at four scales.  The decoder predicts the outer workpiece mask
and an auxiliary boundary map.  The held-out test split is evaluated only after
checkpoint and probability-threshold selection on the validation split.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet18


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/07_rgbd_fusion"
BASELINE_EXPERIMENT = ROOT / "experiments/06_multidomain_segmentation"
DATASET_DEFAULT = ROOT / "datasets/training/workpiece-seg-isat-v2"
OUTPUT_DEFAULT = EXPERIMENT / "results/rgbd_fusion_v1"
PRETRAINED_DEFAULT = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
EVALUATOR = BASELINE_EXPERIMENT / "scripts/evaluate_exp4_exp5.py"
BASELINE_DEFAULT = (
    BASELINE_EXPERIMENT
    / "results/tune01_jop_reflective_rescue_test_v2/masks/jop_reflective_rescue"
)
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_evaluator():
    spec = importlib.util.spec_from_file_location("exp6_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise ImportError(EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = load_evaluator()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED_DEFAULT)
    parser.add_argument("--baseline-masks", type=Path, default=BASELINE_DEFAULT)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--category-balance-power", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_records(dataset: Path, split: str) -> list[dict]:
    with (dataset / "index" / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"Empty split: {split}")
    return records


def rasterize_outer_mask(record: dict, shape: tuple[int, int]) -> np.ndarray:
    annotation = json.loads(Path(record["source_annotation"]).read_text(encoding="utf-8"))
    mask = np.zeros(shape, dtype=np.uint8)
    ordered = sorted(
        enumerate(annotation["objects"]),
        key=lambda pair: (float(pair[1].get("layer", pair[0] + 1)), pair[0]),
    )
    for _, item in ordered:
        points = np.rint(np.asarray(item["segmentation"], dtype=np.float32)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
        value = 0 if item["category"] == "__background__" else 255
        cv2.fillPoly(mask, [points], value)
    return mask


def normalize_disparity(disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    values = disparity[valid]
    if values.size < 32:
        return np.zeros_like(disparity, dtype=np.float32), valid.astype(np.float32)
    low, high = np.percentile(values, [2.0, 98.0])
    scale = max(float(high - low), 1e-3)
    normalized = np.clip((np.nan_to_num(disparity, nan=low) - low) / scale, 0.0, 1.0)
    normalized[~valid] = 0.0
    return normalized.astype(np.float32), valid.astype(np.float32)


def apply_geometry(
    image: np.ndarray,
    disparity: np.ndarray,
    valid: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        disparity = np.ascontiguousarray(disparity[:, ::-1])
        valid = np.ascontiguousarray(valid[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if rng.random() < 0.45:
        height, width = mask.shape
        angle = float(rng.uniform(-4.0, 4.0))
        scale = float(rng.uniform(0.97, 1.03))
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        disparity = cv2.warpAffine(
            disparity, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        valid = cv2.warpAffine(
            valid, matrix, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = cv2.warpAffine(
            mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return image, disparity, valid, mask


def apply_reflective_augmentation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    gamma = float(rng.uniform(0.72, 1.35))
    image = np.power(np.clip(image, 0.0, 1.0), gamma)
    image = image * float(rng.uniform(0.82, 1.18)) + float(rng.uniform(-0.08, 0.08))
    if rng.random() < 0.45:
        height, width = image.shape[:2]
        glare = np.zeros((height, width), dtype=np.float32)
        center = (int(rng.uniform(0.15, 0.85) * width), int(rng.uniform(0.1, 0.9) * height))
        axes = (int(rng.uniform(0.04, 0.18) * width), int(rng.uniform(0.08, 0.30) * height))
        cv2.ellipse(glare, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        sigma = max(axes) * float(rng.uniform(0.25, 0.7))
        glare = cv2.GaussianBlur(glare, (0, 0), sigmaX=max(sigma, 1.0))
        strength = float(rng.uniform(0.25, 0.75))
        image = image * (1.0 - strength * glare[..., None]) + strength * glare[..., None]
    if rng.random() < 0.18:
        kernel = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    return np.clip(image, 0.0, 1.0)


def depth_channels(disparity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    values = gradient[valid > 0.5]
    scale = float(np.percentile(values, 95.0)) if values.size else 1.0
    gradient = np.clip(gradient / max(scale, 1e-4), 0.0, 1.0)
    return np.stack([disparity, gradient, valid], axis=2).astype(np.float32)


def boundary_target(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.uint8)
    return (cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0).astype(np.float32)


class WorkpieceRGBD(Dataset):
    def __init__(self, dataset: Path, split: str, size: tuple[int, int], augment: bool, seed: int):
        self.dataset = dataset
        self.records = read_records(dataset, split)
        self.width, self.height = size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_bgr = cv2.imread(record["source_image"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(record["source_image"])
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = rasterize_outer_mask(record, image.shape[:2])
        raw_disparity = np.load(EVAL.disparity_path(record))
        disparity, valid = normalize_disparity(raw_disparity)
        size = (self.width, self.height)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        disparity = cv2.resize(disparity, size, interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)

        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index * 997)
        if self.augment:
            image, disparity, valid, mask = apply_geometry(image, disparity, valid, mask, rng)
            image = apply_reflective_augmentation(image, rng)
        else:
            image = image.astype(np.float32) / 255.0

        image = (image.astype(np.float32) - RGB_MEAN) / RGB_STD
        binary = (mask > 127).astype(np.float32)
        depth = depth_channels(disparity, valid)
        boundary = boundary_target(binary)
        return {
            "rgb": torch.from_numpy(image.transpose(2, 0, 1)),
            "depth": torch.from_numpy(depth.transpose(2, 0, 1)),
            "mask": torch.from_numpy(binary[None]),
            "boundary": torch.from_numpy(boundary[None]),
            "name": record["name"],
            "category": record["category"],
        }


class Encoder(nn.Module):
    def __init__(self, pretrained: dict[str, torch.Tensor], depth_input: bool = False):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.load_state_dict(pretrained)
        if depth_input:
            with torch.no_grad():
                averaged = backbone.conv1.weight.mean(dim=1, keepdim=True)
                backbone.conv1.weight.copy_(averaged.repeat(1, 3, 1, 1))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


class GatedFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([rgb, depth], dim=1))
        return self.refine(torch.cat([rgb, gate * depth], dim=1)) + rgb


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class RGBDFusionNet(nn.Module):
    def __init__(self, pretrained: dict[str, torch.Tensor]):
        super().__init__()
        self.rgb_encoder = Encoder(pretrained, depth_input=False)
        self.depth_encoder = Encoder(pretrained, depth_input=True)
        channels = [64, 128, 256, 512]
        self.fusions = nn.ModuleList([GatedFusion(value) for value in channels])
        self.decode3 = DecoderBlock(512, 256, 256)
        self.decode2 = DecoderBlock(256, 128, 128)
        self.decode1 = DecoderBlock(128, 64, 64)
        self.full = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(32, 1, 1)
        self.boundary_head = nn.Conv2d(32, 1, 1)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        fused = [layer(r, d) for layer, r, d in zip(self.fusions, rgb_features, depth_features)]
        x = self.decode3(fused[3], fused[2])
        x = self.decode2(x, fused[1])
        x = self.decode1(x, fused[0])
        x = F.interpolate(x, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        x = self.full(x)
        return self.mask_head(x), self.boundary_head(x)


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    true_positive = (probability * target).sum(dims)
    false_positive = (probability * (1.0 - target)).sum(dims)
    false_negative = ((1.0 - probability) * target).sum(dims)
    score = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    return 1.0 - score.mean()


def segmentation_loss(
    mask_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    mask: torch.Tensor,
    boundary: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    binary = F.binary_cross_entropy_with_logits(mask_logits, mask)
    tversky = tversky_loss(mask_logits, mask)
    positive = boundary.sum()
    negative = boundary.numel() - positive
    weight = torch.clamp(negative / torch.clamp(positive, min=1.0), 1.0, 20.0)
    boundary_loss = F.binary_cross_entropy_with_logits(
        boundary_logits, boundary, pos_weight=weight.detach()
    )
    total = 0.35 * binary + 0.55 * tversky + 0.10 * boundary_loss
    return total, {
        "bce": float(binary.detach()),
        "tversky": float(tversky.detach()),
        "boundary": float(boundary_loss.detach()),
    }


def confusion(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return EVAL.confusion_counts(gt.astype(bool), prediction.astype(bool))


def aggregate(records: list[dict], predictions: dict[str, np.ndarray], tolerance: int) -> dict:
    total_confusion = np.zeros((2, 2), dtype=np.int64)
    total_boundary = np.zeros(4, dtype=np.int64)
    category_confusion = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))
    category_boundary = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    dataset = Path(records[0]["dataset"])
    for record in records:
        gt = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        pred = predictions[record["name"]]
        counts = confusion(gt, pred)
        edges = EVAL.boundary_counts(gt, pred, tolerance)
        total_confusion += counts
        total_boundary += edges
        category_confusion[record["category"]] += counts
        category_boundary[record["category"]] += edges
    overall = EVAL.metrics_from_confusion(total_confusion)
    EVAL.add_boundary_metrics(overall, total_boundary)
    per_category = {}
    for category in sorted(category_confusion):
        metrics = EVAL.metrics_from_confusion(category_confusion[category])
        EVAL.add_boundary_metrics(metrics, category_boundary[category])
        per_category[category] = metrics
    overall["macro_category_iou"] = float(
        np.mean([metrics["foreground_iou"] for metrics in per_category.values()])
    )
    overall["macro_category_f2"] = float(
        np.mean(
            [
                5.0 * item["precision"] * item["recall"]
                / max(4.0 * item["precision"] + item["recall"], 1e-12)
                for item in per_category.values()
            ]
        )
    )
    return {"overall": overall, "per_category": per_category}


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device, target_size: tuple[int, int], amp: bool
) -> dict[str, np.ndarray]:
    model.eval()
    output = {}
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits, _ = model(rgb, depth)
        probability = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for name, array in zip(batch["name"], probability):
            output[name] = cv2.resize(array, target_size, interpolation=cv2.INTER_LINEAR)
    return output


def calibration(
    records: list[dict], probabilities: dict[str, np.ndarray], tolerance: int
) -> tuple[float, list[dict]]:
    table = []
    for threshold in np.arange(0.025, 0.751, 0.025):
        masks = {name: value >= threshold for name, value in probabilities.items()}
        metrics = aggregate(records, masks, tolerance)["overall"]
        table.append({"threshold": float(round(threshold, 3)), **metrics})
    # Recall is the primary objective, but an unconstrained F2 optimum can accept
    # an almost full-frame mask on this foreground-heavy dataset.  The precision
    # floor keeps the selected operating point useful for a clean point cloud.
    eligible = [item for item in table if item["precision"] >= 0.94]
    if not eligible:
        eligible = table
    selected = max(
        eligible,
        key=lambda item: (
            item["macro_category_f2"], item["macro_category_iou"], item["foreground_iou"]
        ),
    )
    return float(selected["threshold"]), table


def refine_with_experiment4(
    records: list[dict], probabilities: dict[str, np.ndarray], threshold: float
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        disparity = np.load(EVAL.disparity_path(record))
        probability = probabilities[record["name"]]
        disparity = cv2.resize(
            disparity, (probability.shape[1], probability.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        output[record["name"]] = EVAL.experiment4_refine(probability, disparity, threshold)[0]
    return output


def load_baseline(records: list[dict], directory: Path) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        path = directory / f"{record['name']}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        output[record["name"]] = mask > 127
    return output


def sample_metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return EVAL.metrics_from_confusion(confusion(gt, prediction))


def save_outputs(
    output: Path,
    records: list[dict],
    probabilities: dict[str, np.ndarray],
    methods: dict[str, dict[str, np.ndarray]],
    threshold: float,
) -> None:
    mask_root = output / "test_masks"
    for method, values in methods.items():
        directory = mask_root / method
        directory.mkdir(parents=True, exist_ok=True)
        for name, mask in values.items():
            cv2.imwrite(str(directory / f"{name}.png"), mask.astype(np.uint8) * 255)
    probability_root = output / "test_probabilities"
    probability_root.mkdir(parents=True, exist_ok=True)
    for name, probability in probabilities.items():
        np.save(probability_root / f"{name}.npy", probability.astype(np.float16))

    rows = []
    panels = []
    dataset = Path(records[0]["dataset"])
    for record in records:
        name = record["name"]
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        row = {"name": name, "category": record["category"], "threshold": threshold}
        for method, values in methods.items():
            for key, value in sample_metrics(gt, values[name]).items():
                if key in ("foreground_iou", "precision", "recall", "dice"):
                    row[f"{method}_{key}"] = value
        rows.append(row)
        panel = np.hstack(
            [
                EVAL.label_panel(image, "image"),
                EVAL.mask_panel(gt, "human outer contour"),
                EVAL.mask_panel(methods["v4_1"][name], "V4.1"),
                EVAL.mask_panel(methods["rgbd_raw"][name], "RGB-D raw"),
                EVAL.mask_panel(methods["rgbd_exp4"][name], "RGB-D + Exp4"),
            ]
        )
        cv2.putText(panel, name, (5, panel.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        panels.append(cv2.resize(panel, (900, 320), interpolation=cv2.INTER_AREA))
    with (output / "per_scene.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(output / "test_contact_sheet.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 91])


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    pretrained_path = args.pretrained.resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; use a new experiment version: {output}")
    if not pretrained_path.is_file():
        raise FileNotFoundError(pretrained_path)
    output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp

    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    model = RGBDFusionNet(pretrained).to(device)
    train_set = WorkpieceRGBD(dataset, "train", (args.width, args.height), True, args.seed)
    val_set = WorkpieceRGBD(dataset, "val", (args.width, args.height), False, args.seed)
    test_set = WorkpieceRGBD(dataset, "test", (args.width, args.height), False, args.seed)
    counts = Counter(record["category"] for record in train_set.records)
    weights = [counts[record["category"]] ** (-args.category_balance_power) for record in train_set.records]
    samples_per_epoch = int(math.ceil(sum(counts.values()) * 1.2))
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        # Recreate workers every epoch so the deterministic epoch-specific
        # augmentation seed reaches each worker copy of the dataset.
        persistent_workers=False,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_options)

    encoder_parameters = list(model.rgb_encoder.parameters()) + list(model.depth_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    new_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": new_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    target_size = (288, 512)
    val_records = [{**record, "dataset": str(dataset)} for record in val_set.records]
    test_records = [{**record, "dataset": str(dataset)} for record in test_set.records]
    history = []
    best_score = -1.0
    best_epoch = 0
    stale = 0

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "dual_resnet18_gated_rgb_disparity_unet",
        "rgb_input": "ImageNet-normalized left RGB",
        "depth_input": "per-scene robust-normalized LiteAnyStereo disparity + Sobel magnitude + validity",
        "stereo_training": "LiteAnyStereo frozen; existing predictions loaded from disk",
        "supervision": "outer workpiece mask plus derived boundary; no supervised internal-hole labels",
        "loss": "0.35 BCE + 0.55 Tversky(alpha=0.3,beta=0.7) + 0.10 boundary BCE",
        "checkpoint_selection": "validation macro-category IoU at threshold 0.5",
        "threshold_selection": "validation macro-category F2 with overall precision >= 0.94",
        "dataset": str(dataset),
        "output": str(output),
        "pretrained": str(pretrained_path),
        "device": str(device),
        "torch": torch.__version__,
        "image_size": {"width": args.width, "height": args.height},
        "evaluation_size": {"width": target_size[0], "height": target_size[1]},
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "samples_per_epoch": samples_per_epoch,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "category_balance_power": args.category_balance_power,
        "seed": args.seed,
        "amp": amp,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        running = defaultdict(float)
        batch_count = 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            boundary = batch["boundary"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                mask_logits, boundary_logits = model(rgb, depth)
                loss, parts = segmentation_loss(mask_logits, boundary_logits, mask, boundary)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running["loss"] += float(loss.detach())
            for key, value in parts.items():
                running[key] += value
            batch_count += 1
        scheduler.step()

        val_probabilities = predict_probabilities(model, val_loader, device, target_size, amp)
        val_masks = {name: probability >= 0.5 for name, probability in val_probabilities.items()}
        val_metrics = aggregate(val_records, val_masks, args.boundary_tolerance)["overall"]
        record = {
            "epoch": epoch,
            **{key: value / max(batch_count, 1) for key, value in running.items()},
            "learning_rate": optimizer.param_groups[1]["lr"],
            "val_foreground_iou": val_metrics["foreground_iou"],
            "val_recall": val_metrics["recall"],
            "val_boundary_f1": val_metrics["boundary_f1"],
            "val_macro_category_iou": val_metrics["macro_category_iou"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        score = val_metrics["macro_category_iou"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "validation": val_metrics},
                output / "best.pt",
            )
        else:
            stale += 1
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in history), encoding="utf-8"
        )
        if stale >= args.patience:
            break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_probabilities = predict_probabilities(model, val_loader, device, target_size, amp)
    selected_threshold, threshold_table = calibration(
        val_records, val_probabilities, args.boundary_tolerance
    )
    test_probabilities = predict_probabilities(model, test_loader, device, target_size, amp)
    raw_masks = {
        name: probability >= selected_threshold for name, probability in test_probabilities.items()
    }
    refined_masks = refine_with_experiment4(test_records, test_probabilities, selected_threshold)
    baseline_masks = load_baseline(test_records, args.baseline_masks.resolve())
    methods = {"v4_1": baseline_masks, "rgbd_raw": raw_masks, "rgbd_exp4": refined_masks}
    evaluation = {
        method: aggregate(test_records, masks, args.boundary_tolerance)
        for method, masks in methods.items()
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "development_comparison_test",
        "warning": "This 21-image split was used by prior engineering comparisons and is not a pristine final test set.",
        "best_epoch": best_epoch,
        "checkpoint_selection_score": best_score,
        "selected_probability_threshold": selected_threshold,
        "threshold_selection": "validation macro-category F2 with overall precision >= 0.94",
        "validation_threshold_table": threshold_table,
        "test": evaluation,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_outputs(output, test_records, test_probabilities, methods, selected_threshold)
    print(json.dumps({"completed": True, "output": str(output), "summary": summary["test"]}, indent=2))


if __name__ == "__main__":
    main()
