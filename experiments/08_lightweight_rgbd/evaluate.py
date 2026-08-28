#!/usr/bin/env python3
"""Evaluate Teacher 7.2, base student and distilled student on 21 images."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import config_experiment8 as config
from models.student_network import create_student
from utils.data import WorkpieceStudentDataset, disparity_path, read_records
from utils.metrics import aggregate_metrics, label_panel, mask_panel
from utils.postprocess import refine_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=config.BASE_RUN_DIR / "best.pt")
    parser.add_argument(
        "--distilled", type=Path, default=config.DISTILLED_RUN_DIR / "best.pt"
    )
    parser.add_argument("--output", type=Path, default=config.COMPARISON_DIR)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = create_student(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, np.ndarray]:
    output = {}
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        geometry = batch["geometry"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(rgb, geometry)
        values = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for name, probability in zip(batch["name"], values):
            output[name] = cv2.resize(
                probability,
                (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
    return output


def postprocess_all(
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    threshold: float,
    disparities: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    masks = {}
    diagnostics = {}
    for record in records:
        name = record["name"]
        masks[name], diagnostics[name] = refine_prediction(
            probabilities[name],
            disparities[name],
            threshold,
            gaussian_sigma=config.GEOMETRY_GAUSSIAN_SIGMA,
            binary_threshold=config.GEOMETRY_BINARY_THRESHOLD,
            geometry_closing_radius=config.GEOMETRY_CLOSING_RADIUS,
            preserve_hole_area=config.GEOMETRY_PRESERVE_HOLE_AREA,
            enable_topology_repair=config.TOPOLOGY_REPAIR,
            topology_smooth_sigma=config.TOPOLOGY_SMOOTH_SIGMA,
            topology_smooth_threshold=config.TOPOLOGY_SMOOTH_THRESHOLD,
            topology_envelope_min_added_fraction=(
                config.TOPOLOGY_ENVELOPE_MIN_ADDED_FRACTION
            ),
            topology_envelope_max_added_fraction=(
                config.TOPOLOGY_ENVELOPE_MAX_ADDED_FRACTION
            ),
            topology_envelope_closing_radius=(
                config.TOPOLOGY_ENVELOPE_CLOSING_RADIUS
            ),
            reference_threshold=config.OVERFLOW_REFERENCE_THRESHOLD,
            search_start=config.OVERFLOW_SEARCH_START,
            search_stop=config.OVERFLOW_SEARCH_STOP,
            search_step=config.OVERFLOW_SEARCH_STEP,
            max_reference_area_ratio=config.OVERFLOW_MAX_REFERENCE_AREA_RATIO,
            min_reference_bbox_fraction=config.OVERFLOW_MIN_REFERENCE_BBOX_FRACTION,
            max_step_area_ratio=config.OVERFLOW_MAX_STEP_AREA_RATIO,
            max_bbox_contraction_ratio=config.OVERFLOW_MAX_BBOX_CONTRACTION_RATIO,
            min_candidate_area_fraction=config.OVERFLOW_MIN_CANDIDATE_AREA_FRACTION,
            overflow_closing_radius=config.OVERFLOW_CLOSING_RADIUS,
        )
    return masks, diagnostics


def calibrate_threshold(
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    disparities: dict[str, np.ndarray],
) -> tuple[float, list[dict]]:
    table = []
    for threshold in config.THRESHOLD_CANDIDATES:
        masks, _ = postprocess_all(records, probabilities, threshold, disparities)
        metrics = aggregate_metrics(
            config.DATASET_DIR, records, masks, config.BOUNDARY_TOLERANCE
        )["overall"]
        table.append({"threshold": threshold, **metrics})
    eligible = [item for item in table if item["recall"] >= config.THRESHOLD_RECALL_FLOOR]
    if not eligible:
        eligible = table
    selected = max(
        eligible,
        key=lambda item: (
            item["foreground_iou"],
            item["macro_category_iou"],
            item["boundary_f1"],
        ),
    )
    return float(selected["threshold"]), table


def load_teacher_masks(records: list[dict[str, str]], split: str) -> dict[str, np.ndarray]:
    directory = config.EXP7_OVERFLOW_RUN / split / "masks"
    output = {}
    for record in records:
        path = directory / f"{record['name']}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        output[record["name"]] = mask > 127
    return output


def profile_model(model: torch.nn.Module) -> dict[str, float]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {"segmenter_params_m": parameters / 1e6}
    try:
        from thop import profile

        model = model.cpu().eval()
        rgb = torch.zeros(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        geometry = torch.zeros(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        macs, _ = profile(model, inputs=(rgb, geometry), verbose=False)
        result["segmenter_macs_g"] = float(macs / 1e9)
        result["segmenter_flops_g_2x_macs"] = float(2.0 * macs / 1e9)
    except (ImportError, RuntimeError) as error:
        result["profile_error"] = str(error)
    return result


def save_comparisons(
    output: Path,
    records: list[dict[str, str]],
    methods: dict[str, dict[str, np.ndarray]],
    evaluations: dict[str, dict],
) -> None:
    comparison_dir = output / "representative_cases"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    distilled_scene = evaluations["student_distilled"]["per_scene"]
    ranked = sorted(records, key=lambda record: distilled_scene[record["name"]]["foreground_iou"])
    selected = []
    seen_categories = set()
    for record in ranked:
        if record["category"] not in seen_categories:
            selected.append(record)
            seen_categories.add(record["category"])
    for record in ranked:
        if record not in selected:
            selected.append(record)
        if len(selected) >= 6:
            break
    selected = selected[:6]
    panels = []
    for record in selected:
        name = record["name"]
        image = cv2.imread(str(config.DATASET_DIR / record["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(
            str(config.DATASET_DIR / record["mask"]), cv2.IMREAD_GRAYSCALE
        ) > 127
        panel = np.hstack(
            [
                label_panel(image, "image"),
                mask_panel(gt, "GT"),
                mask_panel(methods["teacher_7_2"][name], "Exp7.2 teacher"),
                mask_panel(methods["student_base"][name], "Student base"),
                mask_panel(methods["student_distilled"][name], "Student distilled"),
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
        path = comparison_dir / f"{name}.jpg"
        cv2.imwrite(str(path), panel, [cv2.IMWRITE_JPEG_QUALITY, 93])
        panels.append(cv2.resize(panel, (1000, 300), interpolation=cv2.INTER_AREA))
    cv2.imwrite(
        str(output / "representative_6.jpg"),
        np.vstack(panels),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def print_table(rows: list[dict]) -> None:
    headers = ["Method", "IoU", "Precision", "Recall", "Boundary F1", "Params(M)", "FLOPs(G)"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
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
    dataset_options = {
        "root": config.ROOT,
        "dataset": config.DATASET_DIR,
        "width": config.IMAGE_WIDTH,
        "height": config.IMAGE_HEIGHT,
        "augment": False,
        "seed": config.SEED,
    }
    datasets = {
        split: WorkpieceStudentDataset(split=split, **dataset_options)
        for split in ("val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items()
    }
    records = {split: datasets[split].records for split in datasets}
    disparities = {
        split: {
            record["name"]: np.load(
                disparity_path(config.ROOT, record), allow_pickle=False
            )
            for record in records[split]
        }
        for split in records
    }

    student_probabilities: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    model_profile = None
    for method, checkpoint in (
        ("student_base", args.base.resolve()),
        ("student_distilled", args.distilled.resolve()),
    ):
        model = load_model(checkpoint, device)
        student_probabilities[method] = {
            split: predict_probabilities(model, loaders[split], device, amp)
            for split in ("val", "test")
        }
        if model_profile is None:
            model_profile = profile_model(model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert model_profile is not None

    thresholds = {}
    threshold_tables = {}
    methods = {"teacher_7_2": load_teacher_masks(records["test"], "test")}
    diagnostics = {}
    for method in ("student_base", "student_distilled"):
        thresholds[method], threshold_tables[method] = calibrate_threshold(
            records["val"], student_probabilities[method]["val"], disparities["val"]
        )
        methods[method], diagnostics[method] = postprocess_all(
            records["test"],
            student_probabilities[method]["test"],
            thresholds[method],
            disparities["test"],
        )
    evaluations = {
        method: aggregate_metrics(
            config.DATASET_DIR, records["test"], masks, config.BOUNDARY_TOLERANCE
        )
        for method, masks in methods.items()
    }

    mask_root = output / "masks"
    for method, masks in methods.items():
        directory = mask_root / method
        directory.mkdir(parents=True, exist_ok=True)
        for name, mask in masks.items():
            cv2.imwrite(str(directory / f"{name}.png"), mask.astype(np.uint8) * 255)
    save_comparisons(output, records["test"], methods, evaluations)

    rows = []
    teacher_profile = {
        "segmenter_params_m": 32.460130,
        "pipeline_params_m": 40.06,
        "reported_flops_g": 192.0,
    }
    for method in ("teacher_7_2", "student_base", "student_distilled"):
        overall = evaluations[method]["overall"]
        profile_values = teacher_profile if method == "teacher_7_2" else model_profile
        rows.append(
            {
                "method": method,
                "iou": overall["foreground_iou"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "boundary_f1": overall["boundary_f1"],
                "params_m": (
                    profile_values["pipeline_params_m"]
                    if method == "teacher_7_2"
                    else profile_values["segmenter_params_m"]
                ),
                "flops_g": (
                    profile_values["reported_flops_g"]
                    if method == "teacher_7_2"
                    else profile_values.get("segmenter_flops_g_2x_macs", float("nan"))
                ),
            }
        )
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "test (21-image fixed development comparison set)",
        "threshold_calibration_split": "val (21 images)",
        "threshold_selection": "highest foreground IoU subject to recall >= 0.98",
        "selected_thresholds": thresholds,
        "threshold_tables": threshold_tables,
        "student_profile": model_profile,
        "teacher_profile": teacher_profile,
        "metrics": evaluations,
        "continuity": {
            method: {
                "mean_components": float(
                    np.mean(
                        [
                            diagnostics[method][name]["connected_components"]
                            for name in diagnostics[method]
                        ]
                    )
                ),
                "overflow_triggered": [
                    name
                    for name, values in diagnostics[method].items()
                    if values["overflow"]["triggered"]
                ],
            }
            for method in ("student_base", "student_distilled")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_table(rows)
    print(json.dumps({"output": str(output), "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()
