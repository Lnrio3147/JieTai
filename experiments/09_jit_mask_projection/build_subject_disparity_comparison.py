#!/usr/bin/env python3
"""Build aligned subject-disparity comparisons for the 21 development images.

No model inference is performed. The script combines the frozen LiteAnyStereo
float disparity with existing GT/Exp7.2/Exp8/Exp9 masks. All columns for one
scene share the same robust disparity scale so color differences are comparable.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config_experiment9 as config


METHOD_DIRECTORIES = {
    "Exp7.2": config.EXP8_COMPARISON_DIR / "masks/teacher_7_2",
    "Exp8 Base": config.EXP8_COMPARISON_DIR / "masks/student_base",
    "Exp8 Distilled": config.EXP8_COMPARISON_DIR / "masks/student_distilled",
    "Exp9": config.COMPARISON_DIR / "masks/experiment9",
}


def read_records(dataset: Path, split: str) -> list[dict[str, str]]:
    """Read the frozen split without importing the PyTorch training dataset."""
    path = dataset / "index" / f"{split}.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"Empty dataset split: {path}")
    return records


def disparity_path(root: Path, record: dict[str, str]) -> Path:
    """Resolve the cached LiteAnyStereo float disparity for one sample."""
    category = record["category"]
    prefix = f"{category}_"
    if not record["name"].startswith(prefix):
        raise ValueError(f"Unexpected sample name: {record['name']}")
    suffix = record["name"][len(prefix) :]
    if category in ("general", "scale", "jop1"):
        scene = suffix.replace("_", "-")
    else:
        stem, frame = suffix.rsplit("_", 1)
        scene = f"{stem}-{frame}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR / "subject_disparity_comparison",
    )
    return parser.parse_args()


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        mask = cv2.resize(
            mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask > 127


def robust_scale(disparity: np.ndarray, reference_mask: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(disparity) & (disparity > 0)
    values = disparity[valid & reference_mask]
    if values.size < 32:
        values = disparity[valid]
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, (2.0, 98.0))
    if high - low < 1e-3:
        high = low + 1.0
    return float(low), float(high)


def colorize(
    disparity: np.ndarray,
    low: float,
    high: float,
    display_mask: np.ndarray | None,
) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0)
    normalized = np.clip((disparity - low) / max(high - low, 1e-6), 0.0, 1.0)
    normalized[~valid] = 0.0
    color = cv2.applyColorMap(
        np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    if display_mask is None:
        color[~valid] = (255, 0, 255)
    else:
        color[~display_mask] = 0
        color[display_mask & ~valid] = (255, 0, 255)
    return color


def add_header(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        title,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            output,
            subtitle,
            (6, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def mask_overlap(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    tp = int((gt & prediction).sum())
    fp = int((~gt & prediction).sum())
    fn = int((gt & ~prediction).sum())
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "foreground_pixels": int(prediction.sum()),
    }


def make_scene_panel(record: dict[str, str]) -> tuple[np.ndarray, list[dict]]:
    image_path = config.DATASET_DIR / record["image"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    raw_disparity = np.load(
        disparity_path(config.ROOT, record), allow_pickle=False
    ).astype(np.float32)
    disparity = cv2.resize(
        raw_disparity, (width, height), interpolation=cv2.INTER_LINEAR
    )
    gt = load_mask(config.DATASET_DIR / record["mask"], (height, width))
    method_masks = {
        method: load_mask(directory / f"{record['name']}.png", (height, width))
        for method, directory in METHOD_DIRECTORIES.items()
    }
    low, high = robust_scale(disparity, gt)
    scale_text = f"GT scale {low:.1f}-{high:.1f}px"
    panels = [
        add_header(image, "RGB", f"{record['name']} [{record['category']}]")
    ]
    panels.append(add_header(colorize(disparity, low, high, None), "Raw LAS", scale_text))
    valid = np.isfinite(disparity) & (disparity > 0)
    gt_invalid = int((gt & ~valid).sum())
    panels.append(
        add_header(
            colorize(disparity, low, high, gt),
            "GT subject",
            f"area={int(gt.sum())}, invalid={gt_invalid}",
        )
    )
    rows = []
    for method, mask in method_masks.items():
        metrics = mask_overlap(gt, mask)
        invalid = int((mask & ~valid).sum())
        panels.append(
            add_header(
                colorize(disparity, low, high, mask),
                method,
                f"IoU={metrics['iou']:.3f}, R={metrics['recall']:.3f}",
            )
        )
        rows.append(
            {
                "name": record["name"],
                "category": record["category"],
                "method": method,
                **metrics,
                "invalid_subject_pixels": invalid,
                "scale_low_px": low,
                "scale_high_px": high,
            }
        )
    panel = np.hstack(panels)
    cv2.putText(
        panel,
        "black=background, magenta=invalid LAS disparity",
        (5, panel.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel, rows


def stack_overview(panels: list[np.ndarray], target_width: int = 1400) -> np.ndarray:
    resized = []
    for panel in panels:
        height = max(1, int(round(panel.shape[0] * target_width / panel.shape[1])))
        resized.append(cv2.resize(panel, (target_width, height), interpolation=cv2.INTER_AREA))
    return np.vstack(resized)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    scenes_dir = output / "scenes"
    categories_dir = output / "categories"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    categories_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(config.DATASET_DIR, "test")
    all_panels: list[np.ndarray] = []
    category_panels: dict[str, list[np.ndarray]] = defaultdict(list)
    rows: list[dict] = []
    for record in records:
        panel, scene_rows = make_scene_panel(record)
        cv2.imwrite(
            str(scenes_dir / f"{record['name']}.jpg"),
            panel,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
        all_panels.append(panel)
        category_panels[record["category"]].append(panel)
        rows.extend(scene_rows)
    cv2.imwrite(
        str(output / "overview_21.jpg"),
        stack_overview(all_panels),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    for category, panels in sorted(category_panels.items()):
        cv2.imwrite(
            str(categories_dir / f"{category}_overview.jpg"),
            stack_overview(panels),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    with (output / "per_scene.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    readme = """# 21张开发比较图的主体视差横向对比

每一行依次为：RGB、原始 LiteAnyStereo 视差、GT 主体视差、实验7.2、实验8 Base、
实验8 Distilled、实验9。每个场景的所有视差列使用同一个由 GT 主体有效视差
2%～98% 分位数确定的色标；超出范围的值截断显示。黑色是被 Mask 移除的背景，
洋红色是 Mask 内 LiteAnyStereo 无效视差。

这些图只比较“同一张 LAS 视差被不同 Mask 保留后的覆盖范围”。Mask 共同保留的像素
数值完全相同，因此不能把颜色相同解释为分割模型估计了新的深度。

- `overview_21.jpg`：21张完整总览；
- `categories/`：五类数据集分组总览；
- `scenes/`：逐图原分辨率横向对比；
- `per_scene.csv`：逐图 Mask IoU/Precision/Recall、主体像素和无效视差数量。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {len(records)} scene panels to {output}")


if __name__ == "__main__":
    main()
