#!/usr/bin/env python3
"""Evaluate current Experiment 7/8 models on every canonical JieTai dataset.

The inventory de-duplicates derived annotation/training copies and covers 353
unique industrial stereo scenes.  Of these, 317 have human foreground masks;
the remaining 36 are reported as qualitative blind-domain checks only.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

import config_experiment8 as config
from evaluate_fdjyp0 import EXP7, preprocess_teacher
from inference_pipeline import preprocess as preprocess_student
from models.student_network import create_student
from utils.data import disparity_path as v2_disparity_path
from utils.data import read_records, scene_name
from utils.metrics import (
    add_boundary_metrics,
    boundary_counts,
    confusion_counts,
    label_panel,
    mask_panel,
    metrics_from_confusion,
)
from utils.postprocess import enclosed_holes, mask_stats, refine_prediction


ROOT = config.ROOT
DATASETS = ROOT / "datasets"
MANUAL_V1 = DATASETS / "annotations/JMP-workpiece-seg-manual-isat-v1"
MULTIDOMAIN_V2 = DATASETS / "training/workpiece-seg-isat-v2"
JMP_STEREO = DATASETS / "training/JMP-LF6020-ETH3D"
REC = DATASETS / "rec_img_set"
FINAL203 = (
    ROOT / "experiments/01_stereo_comparison/rec_img_set/results/final_203/outputs"
)
EXTRA78 = (
    ROOT / "experiments/01_stereo_comparison/rec_img_set/results/extra_78/official"
)
JOP9 = ROOT / "experiments/01_stereo_comparison/jop1/results/final_9"
OUTPUT = config.RESULTS_DIR / "all_datasets_353_20260824"

METHODS = ("teacher_7_2", "student_base", "student_distilled")
THRESHOLDS = {
    "teacher_7_2": 0.075,
    "student_base": 0.24,
    "student_distilled": 0.32,
}
METRIC_KEYS = ("foreground_iou", "precision", "recall", "boundary_f1")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_pfm(path: Path) -> np.ndarray:
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
        data = np.fromfile(stream, dtype=dtype, count=width * height * channels)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"Truncated PFM {path}: {data.size}/{expected}")
    shape = (height, width, channels) if color else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def read_disparity(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".pfm":
        value = read_pfm(path)
    else:
        value = np.load(path, allow_pickle=False)
    if value.ndim == 3:
        value = value[..., 0]
    return np.asarray(value, dtype=np.float32)


def manual_scene(name: str) -> tuple[str, str]:
    match = re.fullmatch(r"fdjyp_(0|2)_\d+_(\d{12})_(\d{4})", name)
    if match is None:
        raise ValueError(f"Unexpected manual-v1 name: {name}")
    return f"fdjyp{match.group(1)}", f"{match.group(2)}-{match.group(3)}"


def manual_records() -> list[dict]:
    records = []
    for split in ("train", "val"):
        for row in read_csv(MANUAL_V1 / "index" / f"{split}.csv"):
            group, scene = manual_scene(row["name"])
            if group == "fdjyp0":
                image = REC / "FDJYP-0-rectified_images" / scene / "im0.png"
                disparity = FINAL203 / "fdjyp0" / scene / "liteanystereo/disp_full.npy"
            else:
                image = JMP_STEREO / row["name"] / "im0.png"
                disparity = JMP_STEREO / row["name"] / "disp0GT.pfm"
            records.append(
                {
                    "name": row["name"],
                    "scene": scene,
                    "group": group,
                    "image": str(image),
                    "disparity": str(disparity),
                    "gt": str(MANUAL_V1 / row["mask"]),
                    "annotation_source": "manual_isat_v1",
                    "annotation_split": split,
                    "experiment8_training_relation": "unseen",
                }
            )
    return records


def multidomain_image(category: str, scene: str) -> Path:
    if category == "fdjyp3":
        return REC / "FDJYP-3-rectified_images" / scene / "im0.png"
    if category == "luowen":
        return REC / "luowen_rectified_images" / scene / "im0.png"
    if category == "general":
        return REC / "rectified_images" / scene / "im0.png"
    if category == "scale":
        return REC / "rectified_images_刻度" / scene / "im0.png"
    if category == "jop1":
        return JOP9 / "preprocessed" / scene / "left.png"
    raise ValueError(category)


def multidomain_records() -> list[dict]:
    records = []
    for split in ("train", "val", "test"):
        for row in read_records(MULTIDOMAIN_V2, split):
            scene = scene_name(row)
            records.append(
                {
                    "name": row["name"],
                    "scene": scene,
                    "group": row["category"],
                    "image": str(multidomain_image(row["category"], scene)),
                    "disparity": str(v2_disparity_path(ROOT, row)),
                    "gt": str(MULTIDOMAIN_V2 / row["mask"]),
                    "annotation_source": "workpiece_seg_isat_v2",
                    "annotation_split": split,
                    "experiment8_training_relation": (
                        "seen_train" if split == "train" else "development"
                    ),
                }
            )
    return records


def blind_records() -> list[dict]:
    records = []
    stereo_manifest = read_csv(JMP_STEREO / "manifest.csv")
    for row in stereo_manifest:
        if not row["name"].startswith("de0548_"):
            continue
        records.append(
            {
                "name": row["name"],
                "scene": row["name"].removeprefix("de0548_camera_"),
                "group": "de0548",
                "image": str(JMP_STEREO / row["left"]),
                "disparity": str(JMP_STEREO / row["disparity"]),
                "gt": "",
                "annotation_source": "none",
                "annotation_split": "blind",
                "experiment8_training_relation": "unseen",
            }
        )
    extra_scene = "202502211433-0099"
    records.append(
        {
            "name": f"de0548_camera_{extra_scene.replace('-', '_')}",
            "scene": extra_scene,
            "group": "de0548",
            "image": str(DATASETS / "tradition_raw/DE0548_right" / extra_scene / "im0.png"),
            "disparity": str(EXTRA78 / "de0548_extra" / extra_scene / "disp.npy"),
            "gt": "",
            "annotation_source": "none",
            "annotation_split": "blind",
            "experiment8_training_relation": "unseen",
        }
    )
    specifications = {
        "jxp": ("JXP", [f"{index:04d}" for index in range(1, 16)]),
        "gongjian_test": ("gongjian_test", [str(index) for index in range(1, 9)]),
        "other_test": ("other_test", ["95", "96", "98", "98_2", "99", "100"]),
    }
    for group, (raw_group, scenes) in specifications.items():
        for scene in scenes:
            records.append(
                {
                    "name": f"{group}_{scene}",
                    "scene": scene,
                    "group": group,
                    "image": str(DATASETS / "tradition_raw" / raw_group / scene / "im0.png"),
                    "disparity": str(EXTRA78 / group.replace("_test", "_test") / scene / "disp.npy"),
                    "gt": "",
                    "annotation_source": "none",
                    "annotation_split": "blind",
                    "experiment8_training_relation": "unseen",
                }
            )
    return records


def inventory() -> list[dict]:
    records = manual_records() + multidomain_records() + blind_records()
    records.sort(key=lambda item: (item["group"], item["name"]))
    names = [record["name"] for record in records]
    if len(records) != 353 or len(set(names)) != 353:
        raise ValueError(f"Expected 353 unique records, got {len(records)}/{len(set(names))}")
    for record in records:
        for key in ("image", "disparity"):
            if not Path(record[key]).is_file():
                raise FileNotFoundError(f"{record['name']} {key}: {record[key]}")
        if record["gt"] and not Path(record["gt"]).is_file():
            raise FileNotFoundError(record["gt"])
    return records


def load_student(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = create_student(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_student(
    records: list[dict], checkpoint: Path, device: torch.device, batch_size: int = 2
) -> dict[str, np.ndarray]:
    model = load_student(checkpoint, device)
    output = {}
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        rgb_values = []
        geometry_values = []
        for record in batch:
            image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(record["image"])
            rgb, geometry = preprocess_student(image, read_disparity(record["disparity"]))
            rgb_values.append(rgb)
            geometry_values.append(geometry)
        rgb = torch.cat(rgb_values).to(device, non_blocking=True)
        geometry = torch.cat(geometry_values).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, _ = model(rgb, geometry)
        probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for record, value in zip(batch, probabilities):
            output[record["name"]] = cv2.resize(
                value,
                (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


@torch.inference_mode()
def predict_teacher(records: list[dict], device: torch.device) -> dict[str, np.ndarray]:
    run_config = json.loads(
        (config.EXP7_SOURCE_RUN / "run_config.json").read_text(encoding="utf-8")
    )
    width = int(run_config["image_size"]["width"])
    height = int(run_config["image_size"]["height"])
    pretrained = torch.load(run_config["pretrained"], map_location="cpu", weights_only=True)
    model = EXP7.RGBDFusionNet(pretrained).to(device)
    checkpoint = torch.load(
        config.TEACHER_CHECKPOINT, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output = {}
    for record in records:
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record["image"])
        rgb, geometry = preprocess_teacher(
            image, read_disparity(record["disparity"]), width, height
        )
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, _ = model(rgb.to(device), geometry.to(device))
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


def postprocess_and_save(
    records: list[dict], probabilities: dict[str, np.ndarray], method: str
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    masks = {}
    diagnostics = {}
    directory = OUTPUT / "masks" / method
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        name = record["name"]
        mask, values = refine_prediction(
            probabilities[name],
            read_disparity(record["disparity"]),
            THRESHOLDS[method],
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
            topology_envelope_closing_radius=config.TOPOLOGY_ENVELOPE_CLOSING_RADIUS,
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
        masks[name] = mask
        diagnostics[name] = values
        cv2.imwrite(str(directory / f"{name}.png"), mask.astype(np.uint8) * 255)
    return masks, diagnostics


def metric_values(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    values = metrics_from_confusion(confusion_counts(gt, prediction))
    add_boundary_metrics(
        values, boundary_counts(gt, prediction, config.BOUNDARY_TOLERANCE)
    )
    return {key: float(values[key]) for key in METRIC_KEYS}


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}


def evaluate(
    records: list[dict], methods: dict[str, dict[str, np.ndarray]]
) -> tuple[dict, list[dict]]:
    per_image = []
    for record in records:
        if not record["gt"]:
            continue
        gt = cv2.imread(record["gt"], cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(record["gt"])
        gt = gt > 127
        row = {
            key: record[key]
            for key in (
                "name",
                "group",
                "annotation_source",
                "annotation_split",
                "experiment8_training_relation",
            )
        }
        for method, masks in methods.items():
            prediction = masks[record["name"]]
            if prediction.shape != gt.shape:
                prediction = cv2.resize(
                    prediction.astype(np.uint8),
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            for key, value in metric_values(gt, prediction).items():
                row[f"{method}_{key}"] = value
        per_image.append(row)

    summary = {"counts": {}, "subsets": {}}
    subsets = {
        "all_human_317": per_image,
        "unseen_fdjyp0_fdjyp2_187": [
            row for row in per_image if row["group"] in ("fdjyp0", "fdjyp2")
        ],
        "multidomain_full_130_includes_train": [
            row for row in per_image if row["annotation_source"] == "workpiece_seg_isat_v2"
        ],
        "multidomain_test_21": [
            row
            for row in per_image
            if row["annotation_source"] == "workpiece_seg_isat_v2"
            and row["annotation_split"] == "test"
        ],
    }
    for name, rows in subsets.items():
        summary["counts"][name] = len(rows)
        summary["subsets"][name] = {}
        for method in methods:
            values = [
                {key: row[f"{method}_{key}"] for key in METRIC_KEYS} for row in rows
            ]
            summary["subsets"][name][method] = mean_metrics(values)
    groups = sorted({row["group"] for row in per_image})
    summary["per_group"] = {}
    for group in groups:
        rows = [row for row in per_image if row["group"] == group]
        summary["per_group"][group] = {
            "count": len(rows),
            "metrics": {
                method: mean_metrics(
                    [
                        {key: row[f"{method}_{key}"] for key in METRIC_KEYS}
                        for row in rows
                    ]
                )
                for method in methods
            },
        }
    for method in methods:
        group_values = [
            summary["per_group"][group]["metrics"][method] for group in groups
        ]
        summary.setdefault("macro_across_7_groups", {})[method] = mean_metrics(group_values)
    return summary, per_image


def pair_iou(first: np.ndarray, second: np.ndarray) -> float:
    return float((first & second).sum()) / max(float((first | second).sum()), 1.0)


def colorize_disparity(disparity: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    value = cv2.resize(
        disparity,
        (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    valid = np.isfinite(value) & (value > 0)
    values = value[valid & mask] if mask is not None else value[valid]
    if values.size < 32:
        values = value[valid]
    low, high = np.percentile(values, (2, 98)) if values.size else (0.0, 1.0)
    normalized = np.clip((value - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    normalized[~valid] = 0.0
    color = cv2.applyColorMap(
        np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    if mask is not None:
        color[~mask] = 0
    return color


def save_visuals(
    records: list[dict],
    methods: dict[str, dict[str, np.ndarray]],
    diagnostics: dict[str, dict[str, dict]],
    per_image: list[dict],
) -> list[dict]:
    metric_lookup = {row["name"]: row for row in per_image}
    group_panels = defaultdict(list)
    blind_rows = []
    for record in records:
        name = record["name"]
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        image = cv2.resize(
            image,
            (config.EVALUATION_WIDTH, config.EVALUATION_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        disparity = read_disparity(record["disparity"])
        base = methods["student_base"][name]
        if record["gt"]:
            gt = cv2.imread(record["gt"], cv2.IMREAD_GRAYSCALE) > 127
            metric = metric_lookup[name]
            panels = [
                label_panel(image, f"RGB {record['group']} {record['scene']}"),
                mask_panel(gt, "human GT"),
                mask_panel(
                    methods["teacher_7_2"][name],
                    f"Exp7.2 IoU {metric['teacher_7_2_foreground_iou']:.3f}",
                ),
                mask_panel(
                    base, f"Exp8 Base IoU {metric['student_base_foreground_iou']:.3f}"
                ),
                mask_panel(
                    methods["student_distilled"][name],
                    f"Distilled IoU {metric['student_distilled_foreground_iou']:.3f}",
                ),
                label_panel(colorize_disparity(disparity, base), "Base subject disparity"),
            ]
        else:
            panels = [
                label_panel(image, f"RGB {record['group']} {record['scene']}"),
                label_panel(colorize_disparity(disparity), "raw disparity"),
                mask_panel(methods["teacher_7_2"][name], "Exp7.2"),
                mask_panel(base, "Exp8 Base"),
                mask_panel(methods["student_distilled"][name], "Distilled"),
                label_panel(colorize_disparity(disparity, base), "Base subject disparity"),
            ]
            blind_rows.append(
                {
                    "name": name,
                    "group": record["group"],
                    "scene": record["scene"],
                    "teacher_base_iou": pair_iou(
                        methods["teacher_7_2"][name], base
                    ),
                    "base_distilled_iou": pair_iou(
                        base, methods["student_distilled"][name]
                    ),
                    "teacher_distilled_iou": pair_iou(
                        methods["teacher_7_2"][name],
                        methods["student_distilled"][name],
                    ),
                }
            )
        panel = np.hstack(panels)
        comparison_dir = OUTPUT / "comparisons" / record["group"]
        comparison_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(comparison_dir / f"{name}.jpg"),
            panel,
            [cv2.IMWRITE_JPEG_QUALITY, 91],
        )
        target_width = 1200
        target_height = max(1, int(round(panel.shape[0] * target_width / panel.shape[1])))
        group_panels[record["group"]].append(
            cv2.resize(panel, (target_width, target_height), interpolation=cv2.INTER_AREA)
        )

    overview_dir = OUTPUT / "overviews"
    overview_dir.mkdir()
    for group, panels in group_panels.items():
        cv2.imwrite(
            str(overview_dir / f"{group}_{len(panels)}.jpg"),
            np.vstack(panels),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
    return blind_rows


def write_report(records: list[dict], summary: dict, blind_rows: list[dict]) -> None:
    group_counts = defaultdict(int)
    for record in records:
        group_counts[record["group"]] += 1
    best = "student_base"
    all_metrics = summary["subsets"]["all_human_317"]
    unseen_metrics = summary["subsets"]["unseen_fdjyp0_fdjyp2_187"]
    rows = []
    for method in METHODS:
        values = all_metrics[method]
        rows.append(
            f"| {method} | {values['foreground_iou']:.4f} | {values['precision']:.4f} | "
            f"{values['recall']:.4f} | {values['boundary_f1']:.4f} |"
        )
    unseen_rows = []
    for method in METHODS:
        values = unseen_metrics[method]
        unseen_rows.append(
            f"| {method} | {values['foreground_iou']:.4f} | {values['precision']:.4f} | "
            f"{values['recall']:.4f} | {values['boundary_f1']:.4f} |"
        )
    group_table = []
    for group, values in summary["per_group"].items():
        metrics = values["metrics"]
        winner = max(METHODS, key=lambda method: metrics[method]["foreground_iou"])
        group_table.append(
            f"| {group} | {values['count']} | {metrics['teacher_7_2']['foreground_iou']:.4f} | "
            f"{metrics['student_base']['foreground_iou']:.4f} | "
            f"{metrics['student_distilled']['foreground_iou']:.4f} | {winner} |"
        )
    blind_mean = {
        key: float(np.mean([row[key] for row in blind_rows]))
        for key in ("teacher_base_iou", "base_distilled_iou", "teacher_distilled_iou")
    }
    report = f"""# All canonical JieTai industrial datasets: Experiment 7/8 comparison

Created 2026-08-24. The inventory de-duplicates annotations, training-format
copies, raw/rectified copies, references and archives. It covers 353 unique
industrial stereo scenes under `JieTai/datasets`: {dict(sorted(group_counts.items()))}.
ETH3D is an external natural-scene stereo benchmark and is not treated as a
workpiece foreground-segmentation dataset.

All methods use the same RGB input, disparity input and topology repair. Human
GT exists for 317 scenes. The remaining 36 blind scenes are visualized but are
not assigned fabricated accuracy metrics.

## All 317 human-labelled scenes (per-image macro)

| Method | IoU | Precision | Recall | Boundary F1 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

This 317-image descriptive result includes 88 images used to train Experiment 8.
The unseen-domain subset below is more informative for generalization.

## Unseen FDJYP-0 + FDJYP-2 (187 images)

| Method | IoU | Precision | Recall | Boundary F1 |
|---|---:|---:|---:|---:|
{chr(10).join(unseen_rows)}

## Per annotated dataset (macro IoU)

| Dataset | Count | Exp7.2 | Exp8 Base | Distilled | Winner |
|---|---:|---:|---:|---:|---|
{chr(10).join(group_table)}

## Blind datasets (36 images)

DE0548, JXP, gongjian_test and other_test have no human foreground mask. Mean
pairwise Mask IoU is Teacher/Base `{blind_mean['teacher_base_iou']:.4f}` and
Base/Distilled `{blind_mean['base_distilled_iou']:.4f}`. These agreement values
are triage signals, not accuracy. Visual review shows widespread background
overflow on DE0548, JXP and other_test; gongjian_test is mostly plausible for
scenes 1--7 but fails on scene 8. Consequently, none of the current models is
reliable on every directory under `datasets`, even though Experiment 8 Base is
the best single model on the 317 labelled scenes. Use the group overviews for
case-level review.

## Artifacts

- `overviews/`: one complete contact sheet per canonical dataset.
- `comparisons/<group>/`: all 353 RGB/Mask/subject-disparity panels.
- `masks/<method>/`: final masks for all three methods.
- `metrics/per_image.csv`: all 317 per-image human-GT metrics.
- `metrics/blind_diagnostics.csv`: pairwise agreement on the 36 unlabelled scenes.
- `inventory.csv`: canonical source image/disparity/GT mapping.

No point clouds are generated here because several blind datasets lack a
verified per-scene calibration manifest. Subject disparity is included in every
comparison panel.
"""
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    records = inventory()
    with (OUTPUT / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = {}
    diagnostics = {}
    predictors = {
        "teacher_7_2": lambda: predict_teacher(records, device),
        "student_base": lambda: predict_student(
            records, config.BASE_RUN_DIR / "best.pt", device
        ),
        "student_distilled": lambda: predict_student(
            records, config.DISTILLED_RUN_DIR / "best.pt", device
        ),
    }
    for method in METHODS:
        print(f"predicting {method} on {len(records)} scenes", flush=True)
        probabilities = predictors[method]()
        masks[method], diagnostics[method] = postprocess_and_save(
            records, probabilities, method
        )
        del probabilities
    summary, per_image = evaluate(records, masks)
    metrics_dir = OUTPUT / "metrics"
    metrics_dir.mkdir()
    with (metrics_dir / "per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    blind_rows = save_visuals(records, masks, diagnostics, per_image)
    with (metrics_dir / "blind_diagnostics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(blind_rows[0]))
        writer.writeheader()
        writer.writerows(blind_rows)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_scene_count": len(records),
        "human_gt_count": len(per_image),
        "blind_count": len(blind_rows),
        "thresholds": THRESHOLDS,
        **summary,
    }
    (metrics_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_report(records, summary, blind_rows)
    print(json.dumps(result["subsets"], indent=2), flush=True)
    print(f"saved={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
