#!/usr/bin/env python3
"""Evaluate Experiments 7.2/8 on all 82 human-labelled FDJYP-0 scenes.

FDJYP-0 is absent from the Experiment 7/8 training dataset.  This script keeps
the original validation-selected thresholds fixed, runs both networks on the
full-resolution left image plus cached LiteAnyStereo disparity, and evaluates
against the existing 288 x 512 human outer-contour masks.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

import config_experiment8 as config
from inference_pipeline import preprocess as preprocess_student
from models.student_network import create_student
from utils.metrics import (
    aggregate_metrics,
    label_panel,
    mask_panel,
)
from utils.postprocess import mask_stats, refine_prediction


ANNOTATION_DATASET = (
    config.ROOT / "datasets/annotations/JMP-workpiece-seg-manual-isat-v1"
)
RAW_DATASET = config.ROOT / "datasets/rec_img_set/FDJYP-0-rectified_images"
LAS_ROOT = (
    config.ROOT
    / "experiments/01_stereo_comparison/rec_img_set/results/final_203/outputs/fdjyp0"
)
EXP5_HOLDOUT = (
    config.ROOT
    / "experiments/05_disparity_guided_segmentation/results/fdjyp0_holdout_recall_v2/scenes"
)
EXP7_TRAIN_SCRIPT = config.EXP7_DIR / "scripts/train_rgbd_fusion.py"


def import_experiment7():
    spec = importlib.util.spec_from_file_location("fdjyp0_exp7", EXP7_TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(EXP7_TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP7 = import_experiment7()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR / "fdjyp0_unseen_82_20260823",
    )
    parser.add_argument("--student-batch-size", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def read_fdjyp0_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    pattern = re.compile(r"(\d{12})_(\d{4})$")
    for source_split in ("train", "val"):
        index = ANNOTATION_DATASET / "index" / f"{source_split}.csv"
        with index.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if not row["name"].startswith("fdjyp_0_"):
                continue
            match = pattern.search(row["name"])
            if match is None:
                raise ValueError(f"Cannot parse scene from {row['name']}")
            scene = f"{match.group(1)}-{match.group(2)}"
            record = {
                **row,
                "source_split": source_split,
                "category": row["capture_group"],
                "scene": scene,
                "raw_image": str(RAW_DATASET / scene / "im0.png"),
                "disparity": str(LAS_ROOT / scene / "liteanystereo/disp_full.npy"),
            }
            for key in ("raw_image", "disparity"):
                if not Path(record[key]).is_file():
                    raise FileNotFoundError(record[key])
            if not (ANNOTATION_DATASET / record["mask"]).is_file():
                raise FileNotFoundError(ANNOTATION_DATASET / record["mask"])
            records.append(record)
    records.sort(key=lambda item: item["scene"])
    if len(records) != 82 or len({item["scene"] for item in records}) != 82:
        raise ValueError(f"Expected 82 unique FDJYP-0 scenes, got {len(records)}")
    return records


def load_student(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = create_student(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_student(
    records: list[dict[str, str]],
    checkpoint: Path,
    device: torch.device,
    amp: bool,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model = load_student(checkpoint, device)
    output: dict[str, np.ndarray] = {}
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        rgb_tensors = []
        geometry_tensors = []
        for record in batch_records:
            image = cv2.imread(record["raw_image"], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(record["raw_image"])
            disparity = np.load(record["disparity"], allow_pickle=False).astype(
                np.float32
            )
            rgb, geometry = preprocess_student(image, disparity)
            rgb_tensors.append(rgb)
            geometry_tensors.append(geometry)
        rgb_batch = torch.cat(rgb_tensors).to(device, non_blocking=True)
        geometry_batch = torch.cat(geometry_tensors).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(rgb_batch, geometry_batch)
        probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for record, probability in zip(batch_records, probabilities):
            output[record["name"]] = cv2.resize(
                probability,
                (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def preprocess_teacher(
    image_bgr: np.ndarray,
    raw_disparity: np.ndarray,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image, (width, height), interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0
    disparity, valid = EXP7.normalize_disparity(raw_disparity)
    disparity = cv2.resize(
        disparity, (width, height), interpolation=cv2.INTER_LINEAR
    )
    valid = cv2.resize(valid, (width, height), interpolation=cv2.INTER_NEAREST)
    image = (image - EXP7.RGB_MEAN) / EXP7.RGB_STD
    depth = EXP7.depth_channels(disparity, valid)
    return (
        torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))[None],
        torch.from_numpy(depth.transpose(2, 0, 1).astype(np.float32))[None],
    )


@torch.inference_mode()
def predict_teacher(
    records: list[dict[str, str]], device: torch.device, amp: bool
) -> dict[str, np.ndarray]:
    run_config = json.loads(
        (config.EXP7_SOURCE_RUN / "run_config.json").read_text(encoding="utf-8")
    )
    width = int(run_config["image_size"]["width"])
    height = int(run_config["image_size"]["height"])
    pretrained = torch.load(
        run_config["pretrained"], map_location="cpu", weights_only=True
    )
    model = EXP7.RGBDFusionNet(pretrained).to(device)
    checkpoint = torch.load(
        config.TEACHER_CHECKPOINT, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output: dict[str, np.ndarray] = {}
    for record in records:
        image = cv2.imread(record["raw_image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record["raw_image"])
        disparity = np.load(record["disparity"], allow_pickle=False).astype(np.float32)
        rgb, depth = preprocess_teacher(image, disparity, width, height)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _ = model(rgb.to(device), depth.to(device))
        probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        output[record["name"]] = cv2.resize(
            probability,
            (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def postprocess(
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    masks = {}
    diagnostics = {}
    for record in records:
        disparity = np.load(record["disparity"], allow_pickle=False)
        mask, values = refine_prediction(
            probabilities[record["name"]],
            disparity,
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
        masks[record["name"]] = mask
        diagnostics[record["name"]] = {
            "connected_components": int(values["connected_components"]),
            "overflow_triggered": bool(values["overflow"]["triggered"]),
            "overflow_event_threshold": values["overflow"]["event_threshold"],
        }
    return masks, diagnostics


def load_exp5_holdout(records: list[dict[str, str]]) -> dict[str, np.ndarray]:
    masks = {}
    for record in records:
        path = EXP5_HOLDOUT / record["scene"] / "mask_subject.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        masks[record["name"]] = mask > 127
    return masks


def evaluate(
    records: list[dict[str, str]], predictions: dict[str, np.ndarray]
) -> dict:
    metrics = aggregate_metrics(
        ANNOTATION_DATASET, records, predictions, config.BOUNDARY_TOLERANCE
    )
    per_scene = metrics["per_scene"]
    macro = {
        key: float(np.mean([values[key] for values in per_scene.values()]))
        for key in ("foreground_iou", "precision", "recall", "boundary_f1")
    }
    area_ratios = []
    components = []
    for record in records:
        gt = cv2.imread(
            str(ANNOTATION_DATASET / record["mask"]), cv2.IMREAD_GRAYSCALE
        ) > 127
        prediction = predictions[record["name"]]
        area_ratios.append(float(prediction.sum()) / max(float(gt.sum()), 1.0))
        components.append(mask_stats(prediction)["connected_components"])
    metrics["macro_scene"] = macro
    metrics["mean_area_ratio"] = float(np.mean(area_ratios))
    metrics["single_component_count"] = int(sum(value == 1 for value in components))
    return metrics


def stack_overview(panels: list[np.ndarray], width: int = 1200) -> np.ndarray:
    resized = []
    for panel in panels:
        height = max(1, int(round(panel.shape[0] * width / panel.shape[1])))
        resized.append(cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA))
    return np.vstack(resized)


def colorize_disparity(
    disparity: np.ndarray,
    low: float,
    high: float,
    mask: np.ndarray | None,
) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0)
    normalized = np.clip((disparity - low) / max(high - low, 1e-6), 0.0, 1.0)
    normalized[~valid] = 0.0
    color = cv2.applyColorMap(
        np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    if mask is None:
        color[~valid] = (255, 0, 255)
    else:
        color[~mask] = 0
        color[mask & ~valid] = (255, 0, 255)
    return color


def save_outputs(
    output: Path,
    records: list[dict[str, str]],
    probabilities: dict[str, dict[str, np.ndarray]],
    methods: dict[str, dict[str, np.ndarray]],
    evaluations: dict[str, dict],
    holdout_exp5: dict[str, np.ndarray],
) -> None:
    for method, values in methods.items():
        mask_dir = output / "masks" / method
        probability_dir = output / "probabilities" / method
        mask_dir.mkdir(parents=True, exist_ok=True)
        probability_dir.mkdir(parents=True, exist_ok=True)
        for name, mask in values.items():
            cv2.imwrite(str(mask_dir / f"{name}.png"), mask.astype(np.uint8) * 255)
            np.save(
                probability_dir / f"{name}.npy",
                probabilities[method][name].astype(np.float16),
            )

    comparison_dir = output / "comparisons"
    disparity_dir = output / "subject_disparity"
    holdout_dir = output / "holdout_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    disparity_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    disparity_panels = []
    holdout_panels = []
    base_scene = evaluations["all82"]["student_base"]["per_scene"]
    for record in records:
        name = record["name"]
        image = cv2.imread(
            str(ANNOTATION_DATASET / record["image"]), cv2.IMREAD_COLOR
        )
        gt = cv2.imread(
            str(ANNOTATION_DATASET / record["mask"]), cv2.IMREAD_GRAYSCALE
        ) > 127
        panel = np.hstack(
            [
                label_panel(image, f"RGB {record['scene']}"),
                mask_panel(gt, "human GT"),
                mask_panel(
                    methods["teacher_7_2"][name],
                    f"Exp7.2 IoU {evaluations['all82']['teacher_7_2']['per_scene'][name]['foreground_iou']:.3f}",
                ),
                mask_panel(
                    methods["student_base"][name],
                    f"Exp8 Base IoU {base_scene[name]['foreground_iou']:.3f}",
                ),
                mask_panel(
                    methods["student_distilled"][name],
                    f"Distilled IoU {evaluations['all82']['student_distilled']['per_scene'][name]['foreground_iou']:.3f}",
                ),
            ]
        )
        cv2.imwrite(
            str(comparison_dir / f"{name}.jpg"),
            panel,
            [cv2.IMWRITE_JPEG_QUALITY, 93],
        )
        panels.append(panel)

        disparity = np.load(record["disparity"], allow_pickle=False).astype(np.float32)
        disparity = cv2.resize(
            disparity,
            (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        valid = np.isfinite(disparity) & (disparity > 0)
        values = disparity[valid & gt]
        if values.size < 32:
            values = disparity[valid]
        low, high = (np.percentile(values, (2, 98)) if values.size else (0.0, 1.0))
        if high - low < 1e-3:
            high = low + 1.0
        disparity_panel = np.hstack(
            [
                label_panel(image, "RGB"),
                label_panel(
                    colorize_disparity(disparity, float(low), float(high), None),
                    f"raw LAS {low:.1f}-{high:.1f}px",
                ),
                label_panel(
                    colorize_disparity(disparity, float(low), float(high), gt),
                    "GT subject",
                ),
                label_panel(
                    colorize_disparity(
                        disparity, float(low), float(high), methods["teacher_7_2"][name]
                    ),
                    "Exp7.2 subject",
                ),
                label_panel(
                    colorize_disparity(
                        disparity, float(low), float(high), methods["student_base"][name]
                    ),
                    "Exp8 Base subject",
                ),
                label_panel(
                    colorize_disparity(
                        disparity,
                        float(low),
                        float(high),
                        methods["student_distilled"][name],
                    ),
                    "Distilled subject",
                ),
            ]
        )
        cv2.imwrite(
            str(disparity_dir / f"{name}.jpg"),
            disparity_panel,
            [cv2.IMWRITE_JPEG_QUALITY, 93],
        )
        disparity_panels.append(disparity_panel)

        if record["source_split"] == "val":
            exp5 = holdout_exp5[name]
            holdout_panel = np.hstack(
                [
                    label_panel(image, f"RGB {record['scene']}"),
                    mask_panel(gt, "human GT"),
                    mask_panel(exp5, "Exp5 V2 (trained on FDJYP-0)"),
                    mask_panel(methods["teacher_7_2"][name], "Exp7.2 unseen"),
                    mask_panel(methods["student_base"][name], "Exp8 Base unseen"),
                    mask_panel(methods["student_distilled"][name], "Distilled unseen"),
                ]
            )
            cv2.imwrite(
                str(holdout_dir / f"{name}.jpg"),
                holdout_panel,
                [cv2.IMWRITE_JPEG_QUALITY, 93],
            )
            holdout_panels.append(holdout_panel)

    cv2.imwrite(
        str(output / "overview_82.jpg"),
        stack_overview(panels),
        [cv2.IMWRITE_JPEG_QUALITY, 91],
    )
    cv2.imwrite(
        str(output / "subject_disparity_overview_82.jpg"),
        stack_overview(disparity_panels),
        [cv2.IMWRITE_JPEG_QUALITY, 91],
    )
    cv2.imwrite(
        str(output / "holdout_18_vs_exp5.jpg"),
        stack_overview(holdout_panels),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    ranked_records = sorted(
        records, key=lambda record: base_scene[record["name"]]["foreground_iou"]
    )[:12]
    selected = [panels[records.index(record)] for record in ranked_records]
    cv2.imwrite(
        str(output / "worst_12_base.jpg"),
        stack_overview(selected),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def write_tables(
    output: Path,
    records: list[dict[str, str]],
    evaluations: dict[str, dict],
) -> None:
    rows = []
    for subset, methods in evaluations.items():
        for method, values in methods.items():
            macro = values["macro_scene"]
            overall = values["overall"]
            rows.append(
                {
                    "subset": subset,
                    "method": method,
                    "count": len(values["per_scene"]),
                    "macro_iou": macro["foreground_iou"],
                    "macro_precision": macro["precision"],
                    "macro_recall": macro["recall"],
                    "macro_boundary_f1": macro["boundary_f1"],
                    "pooled_iou": overall["foreground_iou"],
                    "mean_area_ratio": values["mean_area_ratio"],
                    "single_component_count": values["single_component_count"],
                }
            )
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scene_rows = []
    for record in records:
        row = {
            "name": record["name"],
            "scene": record["scene"],
            "capture_group": record["capture_group"],
            "source_split": record["source_split"],
        }
        for method, values in evaluations["all82"].items():
            metrics = values["per_scene"][record["name"]]
            for key in ("foreground_iou", "precision", "recall", "boundary_f1"):
                row[f"{method}_{key}"] = metrics[key]
        scene_rows.append(row)
    with (output / "per_scene.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    records = read_fdjyp0_records()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    probabilities = {
        "teacher_7_2": predict_teacher(records, device, amp),
        "student_base": predict_student(
            records,
            config.BASE_RUN_DIR / "best.pt",
            device,
            amp,
            args.student_batch_size,
        ),
        "student_distilled": predict_student(
            records,
            config.DISTILLED_RUN_DIR / "best.pt",
            device,
            amp,
            args.student_batch_size,
        ),
    }
    thresholds = {
        "teacher_7_2": 0.075,
        "student_base": 0.24,
        "student_distilled": 0.32,
    }
    methods = {}
    diagnostics = {}
    for method in probabilities:
        methods[method], diagnostics[method] = postprocess(
            records, probabilities[method], thresholds[method]
        )
    subsets = {
        "source_train64": [
            record for record in records if record["source_split"] == "train"
        ],
        "old_holdout18": [
            record for record in records if record["source_split"] == "val"
        ],
        "all82": records,
    }
    holdout_exp5 = load_exp5_holdout(subsets["old_holdout18"])
    evaluations = {
        subset: {
            method: evaluate(subset_records, predictions)
            for method, predictions in methods.items()
        }
        for subset, subset_records in subsets.items()
    }
    evaluations["old_holdout18"]["experiment5_recall_v2"] = evaluate(
        subsets["old_holdout18"], holdout_exp5
    )
    save_outputs(
        output, records, probabilities, methods, evaluations, holdout_exp5
    )
    write_tables(output, records, evaluations)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(ANNOTATION_DATASET),
        "count": len(records),
        "evaluation_role": (
            "FDJYP-0 was not used to train Experiments 7/8; thresholds are fixed "
            "from the original validation set. Experiment 5 is a historical, "
            "non-independent reference because it used FDJYP-0 during training."
        ),
        "rgb_input": "full-resolution im0.png resized by each model preprocessing",
        "disparity_input": "cached LiteAnyStereo disp_full.npy",
        "thresholds": thresholds,
        "capture_group_counts": {
            group: sum(record["capture_group"] == group for record in records)
            for group in sorted({record["capture_group"] for record in records})
        },
        "metrics": evaluations,
        "overflow_triggered": {
            method: [
                name
                for name, values in method_diagnostics.items()
                if values["overflow_triggered"]
            ]
            for method, method_diagnostics in diagnostics.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    concise = {
        subset: {
            method: values["macro_scene"] for method, values in subset_values.items()
        }
        for subset, subset_values in evaluations.items()
    }
    print(json.dumps({"output": str(output), "macro_scene": concise}, indent=2))


if __name__ == "__main__":
    main()
