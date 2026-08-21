#!/usr/bin/env python3
"""Re-evaluate saved RT-IGEV and rerun LAS disparities with one protocol.

RT-IGEV is not inferred here because its phase-I code/checkpoint is absent.
The script reads its full-resolution floating-point output from
``tradition_stereo/igev_output/<scene>/disp.npy``.  LiteAnyStereo predictions
are read from a fresh ``evaluate_stereo.py --save_vis`` run, which now also
saves ``disp.npy``.  Both arrays are evaluated by the same function against
the same reference, fixed ROI, valid mask, scene set, and aggregation rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data import TRADITION_CROP, read_rgb
from training.visualization import save_validation_vis


EXCLUDED_SCENES = {
    "202506281607-0012",
    "202506281609-0019",
    "202506281609-0020",
    "202506281619-0053",
}
METRICS = ("epe", "d1", "bad1", "bad2", "bad3")
HIGH_REFLECTION_SCENE_COUNT = 15


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tradition-root", default="../tradition_stereo")
    parser.add_argument(
        "--las-output-root",
        default="./runs/evaluation/jmp_unified_rerun_73/liteanystereo",
    )
    parser.add_argument("--image-root", default="./data/datasets/JMP-LF6020-ETH3D")
    parser.add_argument(
        "--output-dir",
        default="./runs/evaluation/jmp_unified_rerun_73",
    )
    parser.add_argument("--save-comparisons", action="store_true")
    return parser.parse_args()


def compute_metrics(prediction, reference, extra_mask=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction/reference mismatch: {prediction.shape} vs {reference.shape}")
    valid = np.isfinite(reference) & (reference > 0.0)
    if extra_mask is not None:
        extra_mask = np.asarray(extra_mask, dtype=bool)
        if extra_mask.shape != reference.shape:
            raise ValueError(f"Extra mask/reference mismatch: {extra_mask.shape} vs {reference.shape}")
        valid &= extra_mask
    if not valid.any():
        raise ValueError("Reference has no valid pixels")
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("Prediction contains non-finite values inside the reference mask")
    error = np.abs(prediction - reference)
    relative = error / np.maximum(np.abs(reference), 1e-12)
    return {
        "epe": float(error[valid].mean()),
        "d1": float(100.0 * ((error > 3.0) & (relative > 0.05))[valid].mean()),
        "bad1": float(100.0 * (error[valid] > 1.0).mean()),
        "bad2": float(100.0 * (error[valid] > 2.0).mean()),
        "bad3": float(100.0 * (error[valid] > 3.0).mean()),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(reference.size),
    }


def reflection_statistics(image):
    """Return a model-independent proxy for strong metallic highlights.

    A highlight pixel is clipped in at least one channel, has very high mean
    intensity, or is bright and nearly neutral.  Scene ranking uses a weighted
    score so a few clipped points do not outrank a broad reflective surface.
    """
    image = np.asarray(image, dtype=np.float32)
    maximum = image.max(axis=2)
    minimum = image.min(axis=2)
    mean = image.mean(axis=2)
    clipped = maximum >= 250.0
    very_bright = mean >= 220.0
    bright_neutral = (mean >= 200.0) & ((maximum - minimum) <= 35.0)
    highlight = clipped | very_bright | bright_neutral
    clipped_percent = float(100.0 * clipped.mean())
    very_bright_percent = float(100.0 * very_bright.mean())
    bright_neutral_percent = float(100.0 * bright_neutral.mean())
    score = clipped_percent + 0.5 * very_bright_percent + 0.25 * bright_neutral_percent
    return {
        "reflection_score": score,
        "clipped_percent": clipped_percent,
        "very_bright_percent": very_bright_percent,
        "bright_neutral_percent": bright_neutral_percent,
        "highlight_percent": float(100.0 * highlight.mean()),
        "highlight_mask": highlight,
    }


def aggregate(rows):
    result = {
        algorithm: {
            metric: float(np.mean([row[f"{algorithm}_{metric}"] for row in rows]))
            for metric in METRICS
        }
        for algorithm in ("rt_igev", "liteanystereo")
    }
    result["scene_count"] = len(rows)
    result["valid_pixels"] = int(sum(row["valid_pixels"] for row in rows))
    result["rt_igev_epe_median"] = float(np.median([row["rt_igev_epe"] for row in rows]))
    result["liteanystereo_epe_median"] = float(
        np.median([row["liteanystereo_epe"] for row in rows])
    )
    result["rt_igev_epe_wins"] = sum(
        row["rt_igev_epe"] < row["liteanystereo_epe"] for row in rows
    )
    result["liteanystereo_epe_wins"] = sum(
        row["liteanystereo_epe"] < row["rt_igev_epe"] for row in rows
    )
    return result


def aggregate_highlight_pixels(rows):
    result = {
        algorithm: {
            metric: float(np.mean([row[f"{algorithm}_highlight_{metric}"] for row in rows]))
            for metric in METRICS
        }
        for algorithm in ("rt_igev", "liteanystereo")
    }
    result["scene_count"] = len(rows)
    result["highlight_valid_pixels"] = int(sum(row["highlight_valid_pixels"] for row in rows))
    result["mean_highlight_coverage_percent"] = float(
        np.mean([row["highlight_percent"] for row in rows])
    )
    result["aggregation"] = "scene macro average restricted to highlight pixels"
    return result


def create_chart(path, summary, title):
    labels = ["EPE (px)", "D1 (%)", "Bad1 (%)", "Bad2 (%)", "Bad3 (%)"]
    old = [summary["rt_igev"][key] for key in METRICS]
    new = [summary["liteanystereo"][key] for key in METRICS]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.5, 5.8), dpi=180)
    old_bars = axis.bar(positions - width / 2, old, width, label="RT-IGEV saved output", color="#7B8794")
    new_bars = axis.bar(positions + width / 2, new, width, label="LiteAnyStereo rerun", color="#2878B5")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Lower is better")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(old_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.bar_label(new_bars, fmt="%.2f", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def create_reflection_contact_sheet(path, selected_rows, image_lookup):
    tile_width, tile_height, label_height, gap = 200, 320, 24, 8
    columns = 5
    rows = int(np.ceil(len(selected_rows) / columns))
    canvas = Image.new(
        "RGB",
        (columns * (tile_width + gap) + gap, rows * (tile_height + label_height + gap) + gap),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    y0, y1, x0, x1 = TRADITION_CROP
    for index, row in enumerate(selected_rows):
        scene = row["scene"]
        image = Image.open(image_lookup[scene] / "im0.png").convert("RGB").crop((x0, y0, x1, y1))
        image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        column = index % columns
        line = index // columns
        x = gap + column * (tile_width + gap)
        y = gap + line * (tile_height + label_height + gap)
        canvas.paste(image, (x, y))
        draw.text(
            (x + 4, y + tile_height + 4),
            f"{scene[-4:]}  score={row['reflection_score']:.2f}",
            fill="white",
        )
    canvas.save(path)


def image_scene_lookup(image_root):
    lookup = {}
    for scene in image_root.iterdir():
        if not scene.is_dir():
            continue
        parts = scene.name.rsplit("_", 2)
        if len(parts) >= 3:
            lookup[f"{parts[-2]}-{parts[-1]}"] = scene
    return lookup


def save_comparison(path, left, old, new, reference):
    valid = np.isfinite(reference) & (reference > 0.0)
    save_validation_vis(
        path,
        left=torch.from_numpy(np.ascontiguousarray(left)).permute(2, 0, 1).float(),
        prediction=torch.from_numpy(new[None]).float(),
        target=torch.from_numpy(reference[None]).float(),
        valid=torch.from_numpy(valid[None]),
        evaluation_protocol="standard",
        traditional=torch.from_numpy(old[None]).float(),
        traditional_label="RT-IGEV saved output",
        disparity_max=192.0,
        error_max=20.0,
    )


def main():
    args = parse_args()
    tradition_root = Path(args.tradition_root).expanduser().resolve()
    las_root = Path(args.las_output_root).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    old_root = tradition_root / "igev_output"
    reference_root = tradition_root / "datasets/FDJYP-3"
    scenes = sorted(scene.name for scene in old_root.iterdir() if scene.is_dir())
    if len(scenes) != 73:
        raise ValueError(f"Expected 73 saved RT-IGEV scenes, found {len(scenes)}")
    image_lookup = image_scene_lookup(image_root)
    y0, y1, x0, x1 = TRADITION_CROP
    rows = []

    for scene in scenes:
        old_full = np.load(old_root / scene / "disp.npy").astype(np.float32)
        old = old_full[y0:y1, x0:x1]
        new = np.load(las_root / scene / "disp.npy").astype(np.float32)
        reference = np.load(reference_root / scene / "disp_cropped.npy").astype(np.float32)
        image_scene = image_lookup.get(scene)
        if image_scene is None:
            raise FileNotFoundError(f"No image scene mapped for {scene}")
        left = read_rgb(image_scene / "im0.png")[y0:y1, x0:x1]
        reflection = reflection_statistics(left)
        old_metrics = compute_metrics(old, reference)
        new_metrics = compute_metrics(new, reference)
        old_highlight_metrics = compute_metrics(old, reference, reflection["highlight_mask"])
        new_highlight_metrics = compute_metrics(new, reference, reflection["highlight_mask"])
        row = {
            "scene": scene,
            **{f"rt_igev_{key}": old_metrics[key] for key in METRICS},
            **{f"liteanystereo_{key}": new_metrics[key] for key in METRICS},
            "epe_delta_las_minus_rt": new_metrics["epe"] - old_metrics["epe"],
            "epe_winner": "LiteAnyStereo" if new_metrics["epe"] < old_metrics["epe"] else "RT-IGEV",
            "valid_pixels": old_metrics["valid_pixels"],
            "total_pixels": old_metrics["total_pixels"],
            "fixed69_status": "excluded" if scene in EXCLUDED_SCENES else "kept",
            **{
                key: reflection[key]
                for key in (
                    "reflection_score",
                    "clipped_percent",
                    "very_bright_percent",
                    "bright_neutral_percent",
                    "highlight_percent",
                )
            },
            **{f"rt_igev_highlight_{key}": old_highlight_metrics[key] for key in METRICS},
            **{f"liteanystereo_highlight_{key}": new_highlight_metrics[key] for key in METRICS},
            "highlight_valid_pixels": old_highlight_metrics["valid_pixels"],
        }
        rows.append(row)

        if args.save_comparisons:
            save_comparison(
                output_dir / "comparisons" / scene / "vis.png",
                left,
                old,
                new,
                reference,
            )

    csv_path = metrics_dir / "unified_scene_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fixed_rows = [row for row in rows if row["fixed69_status"] == "kept"]
    high_reflection_rows = sorted(
        rows, key=lambda row: row["reflection_score"], reverse=True
    )[:HIGH_REFLECTION_SCENE_COUNT]
    high_reflection_fixed_rows = [
        row for row in high_reflection_rows if row["fixed69_status"] == "kept"
    ]
    summary = {
        "protocol": {
            "rt_igev_prediction": "../tradition_stereo/igev_output/<scene>/disp.npy then fixed crop [234:1052,126:638]",
            "liteanystereo_prediction": "fresh LAS1 rerun output <scene>/disp.npy",
            "reference": "../tradition_stereo/datasets/FDJYP-3/<scene>/disp_cropped.npy",
            "valid_mask": "finite reference and reference > 0",
            "metrics": "same implementation for both algorithms",
            "aggregation": "scene macro average",
            "epe_scene_filter": None,
            "old_model_rerun": False,
            "old_model_rerun_reason": "phase-I RT-IGEV code/checkpoint not found locally; raw float prediction is available",
            "liteanystereo_rerun": True,
        },
        "all_73_scenes": aggregate(rows),
        "fixed_69_scenes": aggregate(fixed_rows),
        "fixed_excluded_scenes": sorted(EXCLUDED_SCENES),
        "high_reflection_selection": {
            "scene_count": HIGH_REFLECTION_SCENE_COUNT,
            "selection": "top 15 of 73 scenes by image-only metallic highlight score, then visually checked",
            "score_formula": "clipped_percent + 0.5*very_bright_percent + 0.25*bright_neutral_percent",
            "clipped_definition": "max(R,G,B) >= 250",
            "very_bright_definition": "mean(R,G,B) >= 220",
            "bright_neutral_definition": "mean(R,G,B) >= 200 and max(R,G,B)-min(R,G,B) <= 35",
            "minimum_selected_score": float(high_reflection_rows[-1]["reflection_score"]),
            "scenes": [row["scene"] for row in high_reflection_rows],
        },
        "high_reflection_15_scenes_full_roi": aggregate(high_reflection_rows),
        "high_reflection_pixels_in_15_scenes": aggregate_highlight_pixels(high_reflection_rows),
        "high_reflection_fixed_13_scenes_full_roi": aggregate(high_reflection_fixed_rows),
        "high_reflection_pixels_in_fixed_13_scenes": aggregate_highlight_pixels(
            high_reflection_fixed_rows
        ),
    }
    summary_path = metrics_dir / "unified_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    high_csv_path = metrics_dir / "high_reflection_scene_metrics.csv"
    with high_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(high_reflection_rows[0]))
        writer.writeheader()
        writer.writerows(high_reflection_rows)
    create_chart(
        output_dir / "unified_comparison_fixed69.png",
        summary["fixed_69_scenes"],
        "Unified fixed-69 comparison: same GT, ROI, mask, metrics, and scene set",
    )
    create_chart(
        output_dir / "unified_comparison_all73.png",
        summary["all_73_scenes"],
        "Unified all-73 comparison: no scene exclusion or EPE filtering",
    )
    create_chart(
        output_dir / "high_reflection_scene_comparison.png",
        summary["high_reflection_15_scenes_full_roi"],
        "High-reflection 15-scene subset: full-ROI unified metrics",
    )
    create_chart(
        output_dir / "high_reflection_pixel_comparison.png",
        summary["high_reflection_pixels_in_15_scenes"],
        "High-reflection pixels only in selected 15 scenes",
    )
    create_chart(
        output_dir / "high_reflection_fixed13_scene_comparison.png",
        summary["high_reflection_fixed_13_scenes_full_roi"],
        "High-reflection fixed-protocol 13-scene subset: full ROI",
    )
    create_chart(
        output_dir / "high_reflection_fixed13_pixel_comparison.png",
        summary["high_reflection_pixels_in_fixed_13_scenes"],
        "High-reflection pixels only in fixed-protocol 13-scene subset",
    )
    create_reflection_contact_sheet(
        output_dir / "high_reflection_scene_contact_sheet.png",
        high_reflection_rows,
        image_lookup,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
