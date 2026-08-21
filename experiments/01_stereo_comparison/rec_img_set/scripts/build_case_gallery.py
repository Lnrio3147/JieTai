#!/usr/bin/env python3
"""Build a small, evidence-oriented gallery from the unified stereo run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[2]
TRADITION = ROOT / "projects/tradition_stereo"
RESULT = EXPERIMENT_ROOT / "results/final_203"
OUTPUT = RESULT / "representative_gallery"
CROP = (234, 1052, 126, 638)

CASES = {
    "01_holes": [
        {
            "group": "luowen",
            "scene": "656565-0004",
            "slug": "igev_holes_las_fills",
            "note": "IGEV loses more cloud-valid pixels; no reference and geometry warning",
        },
        {
            "group": "general_1221",
            "scene": "camera-202512281402-0162",
            "slug": "las_holes_igev_fills",
            "note": "Opposite-direction example; no reference and epipolar geometry is high risk",
        },
    ],
    "02_edges": [
        {
            "group": "fdjyp3",
            "scene": "202506281608-0018",
            "slug": "las_large_edge_and_global_gain",
            "note": "LAS strongly reduces IGEV catastrophic failure",
        },
        {
            "group": "fdjyp3",
            "scene": "202506281615-0038",
            "slug": "igev_edge_counterexample",
            "note": "IGEV preserves the reference boundary better in this counterexample",
        },
    ],
    "03_exposure": [
        {
            "group": "fdjyp0",
            "scene": "202506261704-0028",
            "slug": "severe_underexposure",
            "note": "Large dark/clipped area and severe model disagreement; no reference",
        },
        {
            "group": "fdjyp3",
            "scene": "202506281614-0035",
            "slug": "metal_highlights",
            "note": "Bright metal highlight case with quantitative reference",
        },
    ],
    "04_scale_details": [
        {
            "group": "scale_1221",
            "scene": "camera-202512081522-0005",
            "slug": "printed_ticks",
            "note": "IGEV keeps more high-frequency tick responses; no depth reference",
        },
        {
            "group": "scale_1221",
            "scene": "camera-202512081732-0018",
            "slug": "ticks_and_bosses",
            "note": "IGEV shows stronger tick responses while both recover large bosses",
        },
        {
            "group": "scale_1221",
            "scene": "camera-202512081522-0004",
            "slug": "agreement_control",
            "note": "Control example where both models agree closely",
        },
    ],
}


def colorize(values: np.ndarray, maximum: float, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    normalized = np.clip(values / maximum, 0.0, 1.0)
    normalized = np.where(finite, normalized, 0.0)
    image = cv2.applyColorMap(np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    mask = finite if valid is None else finite & valid
    image[~mask] = 0
    return image


def label(image: np.ndarray, text: str) -> np.ndarray:
    header = np.zeros((42, image.shape[1], 3), dtype=np.uint8)
    scale = 0.50 if len(text) < 58 else 0.42
    cv2.putText(header, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def title(image: np.ndarray, text: str) -> np.ndarray:
    header = np.full((58, image.shape[1], 3), (21, 25, 32), dtype=np.uint8)
    cv2.putText(header, text, (12, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def grid(panels: list[np.ndarray], heading: str) -> np.ndarray:
    if len(panels) != 6:
        raise ValueError("Gallery diagnostic grids require six panels")
    return title(np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:]))), heading)


def gradient_image(disparity: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(disparity.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disparity.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    return colorize(magnitude, 12.0)


def cloud_valid_mask(disparity: np.ndarray, left: np.ndarray, q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y0, _, x0, _ = CROP
    q_crop = q_full.copy().astype(np.float32)
    q_crop[0, 3] += x0
    q_crop[1, 3] += y0
    filtered = cv2.bilateralFilter(disparity.astype(np.float32), 5, 50, 50)
    points = cv2.reprojectImageTo3D(filtered, q_crop, handleMissingValues=True)
    z = points[..., 2]
    eligible = np.any(left > 50, axis=2)
    valid = (
        np.isfinite(filtered)
        & (filtered >= 5.0)
        & (filtered <= 192.0)
        & np.isfinite(points).all(axis=2)
        & (z > 0.0)
        & (z < 200.0)
        & eligible
    )
    return valid, eligible


def hole_overlay(left: np.ndarray, valid: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    overlay = left.copy()
    holes = eligible & ~valid
    excluded = ~eligible
    overlay[excluded] = (32, 32, 32)
    overlay[holes] = (20, 20, 255)
    return cv2.addWeighted(left, 0.42, overlay, 0.58, 0.0)


def hole_difference(first: np.ndarray, second: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    # White: both valid; red: IGEV-only hole; cyan: LAS-only hole; dark: both invalid/excluded.
    image = np.zeros((*first.shape, 3), dtype=np.uint8)
    image[first & second] = (235, 235, 235)
    image[eligible & ~first & second] = (20, 20, 255)
    image[eligible & first & ~second] = (255, 220, 0)
    image[eligible & ~first & ~second] = (85, 85, 85)
    return image


def clipping_map(gray: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image = (image.astype(np.float32) * 0.38).astype(np.uint8)
    image[gray < 10] = (20, 20, 255)
    image[gray > 245] = (0, 235, 255)
    return image


def histogram_panel(gray: np.ndarray, metrics: dict[str, float]) -> np.ndarray:
    height, width = gray.shape
    image = np.full((height, width, 3), (22, 26, 32), dtype=np.uint8)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    histogram /= max(histogram.max(), 1.0)
    base = height - 84
    usable = height - 180
    points = np.column_stack(
        (
            np.linspace(36, width - 36, 256),
            base - histogram * usable,
        )
    ).astype(np.int32)
    cv2.polylines(image, [points], False, (95, 190, 255), 2, cv2.LINE_AA)
    cv2.line(image, (36, base), (width - 36, base), (170, 170, 170), 1)
    lines = [
        f"mean intensity: {metrics['mean']:.1f}",
        f"dark <25: {metrics['dark_pct']:.2f}%",
        f"bright >235: {metrics['bright_pct']:.2f}%",
        f"clipped <10 or >245: {metrics['clipped_pct']:.2f}%",
    ]
    for index, text in enumerate(lines):
        cv2.putText(image, text, (36, 44 + index * 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return image


def load_case(row: dict[str, str], summary: dict) -> dict:
    group, scene = row["group"], row["scene"]
    source = TRADITION / "rec_img_set" / row["source_dir"] / scene
    left_full = cv2.imread(str(source / "im0.png"), cv2.IMREAD_COLOR)
    right_full = cv2.imread(str(source / "im1.png"), cv2.IMREAD_COLOR)
    y0, y1, x0, x1 = CROP
    left = left_full[y0:y1, x0:x1]
    right = right_full[y0:y1, x0:x1]
    output = RESULT / "outputs" / group / scene
    igev = np.load(output / "igev_rt/disp_crop.npy").astype(np.float32, copy=False)
    las = np.load(output / "liteanystereo/disp_crop.npy").astype(np.float32, copy=False)
    reference_path = TRADITION / "datasets/FDJYP-3" / scene / "disp_cropped.npy"
    reference = np.load(reference_path).astype(np.float32, copy=False) if reference_path.is_file() else None
    q = np.asarray(summary["calibrations"][group]["Q"], dtype=np.float32)
    return {"left": left, "right": right, "igev": igev, "las": las, "reference": reference, "q": q}


def holes_diagnostic(data: dict, heading: str) -> tuple[np.ndarray, dict]:
    first, eligible = cloud_valid_mask(data["igev"], data["left"], data["q"])
    second, _ = cloud_valid_mask(data["las"], data["left"], data["q"])
    stats = {
        "igev_cloud_valid_pct": float(100.0 * first.mean()),
        "las_cloud_valid_pct": float(100.0 * second.mean()),
        "igev_only_hole_pct": float(100.0 * (eligible & ~first & second).mean()),
        "las_only_hole_pct": float(100.0 * (eligible & first & ~second).mean()),
    }
    panels = [
        label(data["left"], "Rectified left ROI"),
        label(colorize(data["igev"], 192.0), "IGEV++ RT disparity"),
        label(colorize(data["las"], 192.0), "LiteAnyStereo disparity"),
        label(hole_overlay(data["left"], first, eligible), f"IGEV holes red; valid {stats['igev_cloud_valid_pct']:.2f}%"),
        label(hole_overlay(data["left"], second, eligible), f"LAS holes red; valid {stats['las_cloud_valid_pct']:.2f}%"),
        label(hole_difference(first, second, eligible), "red: IGEV hole; cyan: LAS hole; white: both valid"),
    ]
    return grid(panels, heading), stats


def edges_diagnostic(data: dict, heading: str) -> tuple[np.ndarray, dict]:
    reference = data["reference"]
    if reference is None:
        raise ValueError("Edge diagnostics require a reference disparity")
    valid = np.isfinite(reference) & (reference > 0.0)
    gx = cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3)
    edge = (np.hypot(gx, gy) > 4.0) & valid
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    edge &= valid
    overlay = data["left"].copy()
    overlay[edge] = (20, 30, 255)
    overlay = cv2.addWeighted(data["left"], 0.42, overlay, 0.58, 0.0)
    first_error = np.abs(data["igev"] - reference)
    second_error = np.abs(data["las"] - reference)
    stats = {
        "edge_band_pct": float(100.0 * edge.mean()),
        "igev_edge_epe": float(first_error[edge].mean()),
        "las_edge_epe": float(second_error[edge].mean()),
        "igev_all_epe": float(first_error[valid].mean()),
        "las_all_epe": float(second_error[valid].mean()),
    }
    panels = [
        label(overlay, "Rectified left; evaluated edge band in red"),
        label(colorize(data["igev"], 192.0), "IGEV++ RT disparity"),
        label(colorize(data["las"], 192.0), "LiteAnyStereo disparity"),
        label(colorize(reference, 192.0, valid), "Foundation Stereo reference"),
        label(colorize(first_error, 20.0, valid), f"IGEV abs error; edge EPE {stats['igev_edge_epe']:.2f}px"),
        label(colorize(second_error, 20.0, valid), f"LAS abs error; edge EPE {stats['las_edge_epe']:.2f}px"),
    ]
    return grid(panels, heading), stats


def exposure_diagnostic(data: dict, heading: str) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(data["left"], cv2.COLOR_BGR2GRAY)
    stats = {
        "mean": float(gray.mean()),
        "dark_pct": float(100.0 * (gray < 25).mean()),
        "bright_pct": float(100.0 * (gray > 235).mean()),
        "clipped_pct": float(100.0 * ((gray < 10) | (gray > 245)).mean()),
        "inter_model_mae_px": float(np.mean(np.abs(data["igev"].astype(np.float64) - data["las"]))),
    }
    panels = [
        label(data["left"], "Rectified left ROI"),
        label(colorize(data["igev"], 192.0), "IGEV++ RT disparity"),
        label(colorize(data["las"], 192.0), "LiteAnyStereo disparity"),
        label(clipping_map(gray), "red: <10 dark clip; yellow: >245 bright clip"),
        label(histogram_panel(gray, stats), "Left-image intensity histogram"),
        label(colorize(np.abs(data["igev"] - data["las"]), 20.0), f"Inter-model difference; MAE {stats['inter_model_mae_px']:.2f}px"),
    ]
    return grid(panels, heading), stats


def scale_diagnostic(data: dict, heading: str) -> tuple[np.ndarray, dict]:
    first_gradient = np.hypot(
        cv2.Sobel(data["igev"], cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(data["igev"], cv2.CV_32F, 0, 1, ksize=3),
    )
    second_gradient = np.hypot(
        cv2.Sobel(data["las"], cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(data["las"], cv2.CV_32F, 0, 1, ksize=3),
    )
    stats = {
        "inter_model_mae_px": float(np.mean(np.abs(data["igev"].astype(np.float64) - data["las"]))),
        "igev_gradient_mean": float(first_gradient.mean()),
        "las_gradient_mean": float(second_gradient.mean()),
        "igev_gradient_p95": float(np.percentile(first_gradient, 95)),
        "las_gradient_p95": float(np.percentile(second_gradient, 95)),
    }
    panels = [
        label(data["left"], "Rectified left ROI"),
        label(colorize(data["igev"], 192.0), "IGEV++ RT disparity"),
        label(colorize(data["las"], 192.0), "LiteAnyStereo disparity"),
        label(data["right"], "Rectified right ROI"),
        label(gradient_image(data["igev"]), f"IGEV disparity gradient; mean {stats['igev_gradient_mean']:.2f}"),
        label(gradient_image(data["las"]), f"LAS disparity gradient; mean {stats['las_gradient_mean']:.2f}"),
    ]
    return grid(panels, heading), stats


def make_overview(images: list[Path], output: Path, columns: int = 2) -> None:
    thumbnails = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        scale = 900.0 / image.shape[1]
        thumbnails.append(cv2.resize(image, (900, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    height = max(image.shape[0] for image in thumbnails)
    rows = int(np.ceil(len(thumbnails) / columns))
    canvas = np.zeros((rows * height, columns * 900, 3), dtype=np.uint8)
    for index, image in enumerate(thumbnails):
        y = (index // columns) * height
        x = (index % columns) * 900
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (RESULT / "metrics/per_scene.csv").open(encoding="utf-8", newline="") as handle:
        rows = {(row["group"], row["scene"]): row for row in csv.DictReader(handle)}
    summary = json.loads((RESULT / "metrics/summary.json").read_text(encoding="utf-8"))
    builders = {
        "01_holes": holes_diagnostic,
        "02_edges": edges_diagnostic,
        "03_exposure": exposure_diagnostic,
        "04_scale_details": scale_diagnostic,
    }
    manifest = []
    category_overviews = []
    for category, cases in CASES.items():
        directory = OUTPUT / category
        directory.mkdir(parents=True, exist_ok=True)
        generated = []
        for index, case in enumerate(cases, start=1):
            row = rows[(case["group"], case["scene"])]
            data = load_case(row, summary)
            heading = f"{case['group']}/{case['scene']} - {case['note']}"
            image, stats = builders[category](data, heading)
            filename = f"{index:02d}_{case['slug']}__{case['scene']}.png"
            path = directory / filename
            cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            generated.append(path)
            manifest.append(
                {
                    "category": category,
                    "group": case["group"],
                    "scene": case["scene"],
                    "file": str(path.relative_to(OUTPUT)),
                    "note": case["note"],
                    "geometry_status": row["geometry_status"],
                    "has_reference": row["has_foundation_reference"],
                    "stats": stats,
                }
            )
        overview = OUTPUT / f"{category}_overview.jpg"
        make_overview(generated, overview)
        category_overviews.append(overview)
    make_overview(category_overviews, OUTPUT / "gallery_overview.jpg", columns=2)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("category", "group", "scene", "file", "note", "geometry_status", "has_reference", "stats"),
        )
        writer.writeheader()
        for item in manifest:
            writer.writerow({**item, "stats": json.dumps(item["stats"], ensure_ascii=False)})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
