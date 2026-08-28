#!/usr/bin/env python3
"""Evaluate the V3 projector against V3 Base/Distilled on the frozen test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import config_experiment9 as config
from data import CleanMaskDataset
from models.clean_mask_projector import create_projector

sys.path.insert(0, str(config.EXP8_DIR))
import config_experiment8 as exp8_config  # noqa: E402
from utils.data import disparity_path, read_disparity  # noqa: E402
from utils.metrics import aggregate_metrics, label_panel, mask_panel  # noqa: E402
from utils.postprocess import refine_prediction  # noqa: E402


DEFAULT_DATASET = config.ROOT / "datasets/training/workpiece-seg-grouped-v3"
DEFAULT_COARSE = config.RESULTS_DIR / "coarse_predictions_grouped_v3"
DEFAULT_CHECKPOINT = config.RESULTS_DIR / "clean_mask_projector_grouped_v3/best.pt"
DEFAULT_OUTPUT = config.RESULTS_DIR / "comparison_grouped_v3"
DEFAULT_BASE_MASKS = (
    config.EXP8_DIR / "results/student_base_grouped_v3/evaluation/test_masks"
)
DEFAULT_DISTILLED_MASKS = (
    config.EXP8_DIR / "results/student_distilled_grouped_v3/evaluation/test_masks"
)
CALIBRATION_THRESHOLDS = tuple(
    sorted(
        set(exp8_config.THRESHOLD_CANDIDATES)
        | {round(value / 100.0, 2) for value in range(5, 20)}
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--coarse-dir", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-masks", type=Path, default=DEFAULT_BASE_MASKS)
    parser.add_argument(
        "--distilled-masks", type=Path, default=DEFAULT_DISTILLED_MASKS
    )
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = create_projector(
        in_channels=config.PROJECTOR_INPUT_CHANNELS,
        channels=config.PROJECTOR_CHANNELS,
        bottleneck_channels=config.BOTTLENECK_CHANNELS,
        patch_size=config.PATCH_SIZE,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, np.ndarray]:
    output = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(features)
        probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for name, probability in zip(batch["name"], probabilities):
            output[name] = probability.astype(np.float32)
    return output


def load_disparities(
    dataset: Path, records: list[dict[str, str]]
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        if record.get("disparity"):
            path = dataset / record["disparity"]
        else:
            path = disparity_path(config.ROOT, record)
        output[record["name"]] = read_disparity(path)
    return output


def refine_all(
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    disparities: dict[str, np.ndarray],
    threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    masks, diagnostics = {}, {}
    for record in records:
        name = record["name"]
        masks[name], diagnostics[name] = refine_prediction(
            probabilities[name],
            disparities[name],
            threshold,
            gaussian_sigma=exp8_config.GEOMETRY_GAUSSIAN_SIGMA,
            binary_threshold=exp8_config.GEOMETRY_BINARY_THRESHOLD,
            geometry_closing_radius=exp8_config.GEOMETRY_CLOSING_RADIUS,
            preserve_hole_area=exp8_config.GEOMETRY_PRESERVE_HOLE_AREA,
            enable_topology_repair=exp8_config.TOPOLOGY_REPAIR,
            topology_smooth_sigma=exp8_config.TOPOLOGY_SMOOTH_SIGMA,
            topology_smooth_threshold=exp8_config.TOPOLOGY_SMOOTH_THRESHOLD,
            topology_envelope_min_added_fraction=(
                exp8_config.TOPOLOGY_ENVELOPE_MIN_ADDED_FRACTION
            ),
            topology_envelope_max_added_fraction=(
                exp8_config.TOPOLOGY_ENVELOPE_MAX_ADDED_FRACTION
            ),
            topology_envelope_closing_radius=(
                exp8_config.TOPOLOGY_ENVELOPE_CLOSING_RADIUS
            ),
            reference_threshold=exp8_config.OVERFLOW_REFERENCE_THRESHOLD,
            search_start=exp8_config.OVERFLOW_SEARCH_START,
            search_stop=exp8_config.OVERFLOW_SEARCH_STOP,
            search_step=exp8_config.OVERFLOW_SEARCH_STEP,
            max_reference_area_ratio=exp8_config.OVERFLOW_MAX_REFERENCE_AREA_RATIO,
            min_reference_bbox_fraction=(
                exp8_config.OVERFLOW_MIN_REFERENCE_BBOX_FRACTION
            ),
            max_step_area_ratio=exp8_config.OVERFLOW_MAX_STEP_AREA_RATIO,
            max_bbox_contraction_ratio=(
                exp8_config.OVERFLOW_MAX_BBOX_CONTRACTION_RATIO
            ),
            min_candidate_area_fraction=(
                exp8_config.OVERFLOW_MIN_CANDIDATE_AREA_FRACTION
            ),
            overflow_closing_radius=exp8_config.OVERFLOW_CLOSING_RADIUS,
        )
    return masks, diagnostics


def calibrate_threshold(
    dataset: Path,
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    disparities: dict[str, np.ndarray],
) -> tuple[float, list[dict]]:
    table = []
    for threshold in CALIBRATION_THRESHOLDS:
        masks, _ = refine_all(records, probabilities, disparities, threshold)
        metrics = aggregate_metrics(
            dataset, records, masks, config.BOUNDARY_TOLERANCE
        )["overall"]
        table.append({"threshold": threshold, **metrics})
    eligible = [
        row for row in table if row["recall"] >= config.THRESHOLD_RECALL_FLOOR
    ]
    selected = max(
        eligible or table,
        key=lambda row: (
            row["foreground_iou"],
            row["macro_category_iou"],
            row["boundary_f1"],
        ),
    )
    return float(selected["threshold"]), table


def load_masks(
    directory: Path, records: list[dict[str, str]]
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        path = directory / f"{record['name']}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        output[record["name"]] = mask > 127
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_contact_sheet(
    dataset: Path,
    records: list[dict[str, str]],
    methods: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
    width, height, columns = 112, 200, 3
    scene_width = width * 5
    rows = int(math.ceil(len(records) / columns))
    canvas = np.zeros((rows * height, columns * scene_width, 3), dtype=np.uint8)
    for index, record in enumerate(records):
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            raise FileNotFoundError(record["name"])
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST) > 127
        panels = [label_panel(image, f"{record['category']} {record['name'][-9:]}")]
        panels.append(mask_panel(gt, "GT"))
        for method, label in (
            ("base", "V3 Base"),
            ("distilled", "V3 Distilled"),
            ("experiment9", "Exp9 V3"),
        ):
            mask = cv2.resize(
                methods[method][record["name"]].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            panels.append(mask_panel(mask, label))
        panel = np.hstack(panels)
        row, column = divmod(index, columns)
        y, x = row * height, column * scene_width
        canvas[y : y + height, x : x + scene_width] = panel
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])


def profile_projector() -> dict[str, float]:
    model = create_projector(
        in_channels=config.PROJECTOR_INPUT_CHANNELS,
        channels=config.PROJECTOR_CHANNELS,
        bottleneck_channels=config.BOTTLENECK_CHANNELS,
        patch_size=config.PATCH_SIZE,
    ).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {"projector_params_m": parameters / 1e6}
    try:
        from thop import profile

        features = torch.zeros(
            1,
            config.PROJECTOR_INPUT_CHANNELS,
            config.PROJECTOR_HEIGHT,
            config.PROJECTOR_WIDTH,
        )
        macs, _ = profile(model, inputs=(features,), verbose=False)
        result["projector_flops_g_2x_macs"] = float(2.0 * macs / 1e9)
    except (ImportError, RuntimeError) as error:
        result["profile_error"] = str(error)
    return result


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    coarse_dir = args.coarse_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    datasets = {
        split: CleanMaskDataset(
            split,
            augment=False,
            seed=config.SEED,
            dataset=dataset,
            coarse_dir=coarse_dir,
        )
        for split in ("val", "test")
    }
    loaders = {
        split: DataLoader(
            value,
            batch_size=2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for split, value in datasets.items()
    }
    model = load_model(checkpoint, device)
    probabilities = {
        split: predict(model, loaders[split], device, amp)
        for split in ("val", "test")
    }
    disparities = {
        split: load_disparities(dataset, datasets[split].records)
        for split in ("val", "test")
    }
    threshold, threshold_table = calibrate_threshold(
        dataset,
        datasets["val"].records,
        probabilities["val"],
        disparities["val"],
    )
    experiment9_masks, diagnostics = refine_all(
        datasets["test"].records,
        probabilities["test"],
        disparities["test"],
        threshold,
    )
    methods = {
        "base": load_masks(args.base_masks.resolve(), datasets["test"].records),
        "distilled": load_masks(
            args.distilled_masks.resolve(), datasets["test"].records
        ),
        "experiment9": experiment9_masks,
    }
    evaluations = {
        method: aggregate_metrics(
            dataset, datasets["test"].records, masks, config.BOUNDARY_TOLERANCE
        )
        for method, masks in methods.items()
    }

    mask_dir = output / "test_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in experiment9_masks.items():
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask.astype(np.uint8) * 255)
    write_csv(output / "threshold_sweep.csv", threshold_table)
    write_csv(
        output / "per_category.csv",
        [
            {"category": category, **metrics}
            for category, metrics in evaluations["experiment9"]["per_category"].items()
        ],
    )
    write_csv(
        output / "per_image.csv",
        [
            {
                "name": record["name"],
                "category": record["category"],
                **evaluations["experiment9"]["per_scene"][record["name"]],
                "base_iou": evaluations["base"]["per_scene"][record["name"]][
                    "foreground_iou"
                ],
                "iou_delta_vs_base": (
                    evaluations["experiment9"]["per_scene"][record["name"]][
                        "foreground_iou"
                    ]
                    - evaluations["base"]["per_scene"][record["name"]][
                        "foreground_iou"
                    ]
                ),
            }
            for record in datasets["test"].records
        ],
    )
    save_contact_sheet(
        dataset,
        datasets["test"].records,
        methods,
        output / "test_contact_sheet.jpg",
    )

    profile = profile_projector()
    base_parameters_m = 2.397513
    base_flops_g = 9.048
    comparison = []
    for method in ("base", "distilled", "experiment9"):
        metrics = evaluations[method]["overall"]
        comparison.append(
            {
                "method": method,
                "iou": metrics["foreground_iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "boundary_f1": metrics["boundary_f1"],
                "macro_category_iou": metrics["macro_category_iou"],
                "params_m": (
                    base_parameters_m
                    + (profile["projector_params_m"] if method == "experiment9" else 0.0)
                ),
                "flops_g": (
                    base_flops_g
                    + (
                        profile.get("projector_flops_g_2x_macs", 0.0)
                        if method == "experiment9"
                        else 0.0
                    )
                ),
            }
        )
    write_csv(output / "comparison.csv", comparison)
    improved = sum(
        evaluations["experiment9"]["per_scene"][record["name"]]["foreground_iou"]
        > evaluations["base"]["per_scene"][record["name"]]["foreground_iou"]
        for record in datasets["test"].records
    )
    component_counts = [
        diagnostics[record["name"]]["connected_components"]
        for record in datasets["test"].records
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "coarse_dir": str(coarse_dir),
        "checkpoint": str(checkpoint),
        "validation_count": len(datasets["val"]),
        "test_count": len(datasets["test"]),
        "validation_selected_threshold": threshold,
        "threshold_selection": "validation IoU maximum subject to Recall >= 0.98",
        "projector_profile": profile,
        "metrics": evaluations,
        "comparison": comparison,
        "test_scene_changes_vs_base": {
            "improved": improved,
            "degraded_or_equal": len(datasets["test"]) - improved,
        },
        "continuity": {
            "all_single_component": all(count == 1 for count in component_counts),
            "mean_components": float(np.mean(component_counts)),
            "max_components": int(max(component_counts)),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"threshold": threshold, "comparison": comparison, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
