#!/usr/bin/env python3
"""Evaluate Experiment 9 against Experiments 7.2 and 8 on frozen test data."""

from __future__ import annotations

import argparse
import csv
import json
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
from utils.data import disparity_path  # noqa: E402
from utils.metrics import aggregate_metrics, label_panel, mask_panel  # noqa: E402
from utils.postprocess import refine_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.RUN_DIR / "best.pt")
    parser.add_argument("--output", type=Path, default=config.COMPARISON_DIR)
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


def refine_all(
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    disparities: dict[str, np.ndarray],
    threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    masks = {}
    diagnostics = {}
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
            reference_threshold=exp8_config.OVERFLOW_REFERENCE_THRESHOLD,
            search_start=exp8_config.OVERFLOW_SEARCH_START,
            search_stop=exp8_config.OVERFLOW_SEARCH_STOP,
            search_step=exp8_config.OVERFLOW_SEARCH_STEP,
            max_reference_area_ratio=exp8_config.OVERFLOW_MAX_REFERENCE_AREA_RATIO,
            min_reference_bbox_fraction=exp8_config.OVERFLOW_MIN_REFERENCE_BBOX_FRACTION,
            max_step_area_ratio=exp8_config.OVERFLOW_MAX_STEP_AREA_RATIO,
            max_bbox_contraction_ratio=exp8_config.OVERFLOW_MAX_BBOX_CONTRACTION_RATIO,
            min_candidate_area_fraction=exp8_config.OVERFLOW_MIN_CANDIDATE_AREA_FRACTION,
            overflow_closing_radius=exp8_config.OVERFLOW_CLOSING_RADIUS,
        )
    return masks, diagnostics


def calibrate_threshold(
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    disparities: dict[str, np.ndarray],
) -> tuple[float, list[dict]]:
    table = []
    for threshold in config.THRESHOLD_CANDIDATES:
        masks, _ = refine_all(records, probabilities, disparities, threshold)
        metrics = aggregate_metrics(
            config.DATASET_DIR, records, masks, config.BOUNDARY_TOLERANCE
        )["overall"]
        table.append({"threshold": threshold, **metrics})
    eligible = [item for item in table if item["recall"] >= config.THRESHOLD_RECALL_FLOOR]
    selected = max(
        eligible or table,
        key=lambda item: (
            item["foreground_iou"],
            item["macro_category_iou"],
            item["boundary_f1"],
        ),
    )
    return float(selected["threshold"]), table


def load_saved_masks(method: str, records: list[dict[str, str]]) -> dict[str, np.ndarray]:
    directory = config.EXP8_COMPARISON_DIR / "masks" / method
    output = {}
    for record in records:
        path = directory / f"{record['name']}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        output[record["name"]] = mask > 127
    return output


def profile_projector(model: torch.nn.Module) -> dict[str, float]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {"projector_params_m": parameters / 1e6}
    from thop import profile

    model = model.cpu().eval()
    features = torch.zeros(
        1,
        config.PROJECTOR_INPUT_CHANNELS,
        config.PROJECTOR_HEIGHT,
        config.PROJECTOR_WIDTH,
    )
    macs, _ = profile(model, inputs=(features,), verbose=False)
    result["projector_macs_g"] = float(macs / 1e9)
    result["projector_flops_g_2x_macs"] = float(2.0 * macs / 1e9)
    return result


def save_representatives(
    output: Path,
    records: list[dict[str, str]],
    methods: dict[str, dict[str, np.ndarray]],
    evaluations: dict[str, dict],
) -> None:
    directory = output / "representative_cases"
    directory.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        records,
        key=lambda record: evaluations["experiment9"]["per_scene"][record["name"]][
            "foreground_iou"
        ],
    )
    selected = []
    categories = set()
    for record in ranked:
        if record["category"] not in categories:
            selected.append(record)
            categories.add(record["category"])
    for record in ranked:
        if record not in selected:
            selected.append(record)
        if len(selected) >= 6:
            break
    panels = []
    for record in selected[:6]:
        name = record["name"]
        image = cv2.imread(str(config.DATASET_DIR / record["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(config.DATASET_DIR / record["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        panel = np.hstack(
            [
                label_panel(image, "image"),
                mask_panel(gt, "GT"),
                mask_panel(methods["teacher_7_2"][name], "Exp7.2"),
                mask_panel(methods["student_base"][name], "Exp8 Base"),
                mask_panel(methods["experiment9"][name], "Exp9 clean-mask"),
            ]
        )
        cv2.putText(
            panel,
            f"{name} [{record['category']}]",
            (5, panel.shape[0] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(directory / f"{name}.jpg"), panel)
        panels.append(cv2.resize(panel, (1000, 300), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(output / "representative_6.jpg"), np.vstack(panels))


def print_table(rows: list[dict]) -> None:
    print("Method | IoU | Precision | Recall | Boundary F1 | Params(M) | FLOPs(G)")
    print("--- | --- | --- | --- | --- | --- | ---")
    for row in rows:
        print(
            f"{row['method']} | {row['iou']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['boundary_f1']:.4f} | "
            f"{row['params_m']:.3f} | {row['flops_g']:.3f}"
        )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    datasets = {
        split: CleanMaskDataset(split, augment=False, seed=config.SEED)
        for split in ("val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items()
    }
    records = {split: dataset.records for split, dataset in datasets.items()}
    disparities = {
        split: {
            record["name"]: np.load(disparity_path(config.ROOT, record), allow_pickle=False)
            for record in records[split]
        }
        for split in records
    }
    model = load_model(args.checkpoint.resolve(), device)
    probabilities = {
        split: predict(model, loaders[split], device, amp) for split in ("val", "test")
    }
    threshold, threshold_table = calibrate_threshold(
        records["val"], probabilities["val"], disparities["val"]
    )
    experiment9_masks, diagnostics = refine_all(
        records["test"], probabilities["test"], disparities["test"], threshold
    )
    projector_profile = profile_projector(model)

    methods = {
        "teacher_7_2": load_saved_masks("teacher_7_2", records["test"]),
        "student_base": load_saved_masks("student_base", records["test"]),
        "student_distilled": load_saved_masks("student_distilled", records["test"]),
        "experiment9": experiment9_masks,
    }
    evaluations = {
        method: aggregate_metrics(
            config.DATASET_DIR, records["test"], masks, config.BOUNDARY_TOLERANCE
        )
        for method, masks in methods.items()
    }
    mask_dir = output / "masks/experiment9"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in experiment9_masks.items():
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask.astype(np.uint8) * 255)
    save_representatives(output, records["test"], methods, evaluations)

    exp8_summary = json.loads(
        (config.EXP8_COMPARISON_DIR / "summary.json").read_text(encoding="utf-8")
    )
    exp8_profile = exp8_summary["student_profile"]
    profiles = {
        "teacher_7_2": (40.06, 192.0),
        "student_base": (
            exp8_profile["segmenter_params_m"],
            exp8_profile["segmenter_flops_g_2x_macs"],
        ),
        "student_distilled": (
            exp8_profile["segmenter_params_m"],
            exp8_profile["segmenter_flops_g_2x_macs"],
        ),
        "experiment9": (
            exp8_profile["segmenter_params_m"] + projector_profile["projector_params_m"],
            exp8_profile["segmenter_flops_g_2x_macs"]
            + projector_profile["projector_flops_g_2x_macs"],
        ),
    }
    rows = []
    for method in ("teacher_7_2", "student_base", "student_distilled", "experiment9"):
        metrics = evaluations[method]["overall"]
        rows.append(
            {
                "method": method,
                "iou": metrics["foreground_iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "boundary_f1": metrics["boundary_f1"],
                "params_m": profiles[method][0],
                "flops_g": profiles[method][1],
            }
        )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_transfer": {
            "used": ["direct x/clean-mask prediction", "large-patch low-rank bottleneck"],
            "rejected": ["iterative diffusion", "large ViT"],
        },
        "validation_selected_threshold": threshold,
        "threshold_selection": "highest IoU subject to Recall >= 0.98",
        "threshold_table": threshold_table,
        "projector_profile": projector_profile,
        "metrics": evaluations,
        "continuity": {
            "mean_components": float(
                np.mean([values["connected_components"] for values in diagnostics.values()])
            ),
            "all_single_component": all(
                values["connected_components"] == 1 for values in diagnostics.values()
            ),
            "overflow_triggered": [
                name
                for name, values in diagnostics.items()
                if values["overflow"]["triggered"]
            ],
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_table(rows)
    print(json.dumps({"threshold": threshold, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
