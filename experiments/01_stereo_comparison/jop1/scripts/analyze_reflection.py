#!/usr/bin/env python3
"""Compare final IGEV++ RT and LAS1 depth in the Jop1 reflective band."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[2]
RESULT = EXPERIMENT_ROOT / "results/final_9"
SCENE = "camera-202412091816-0108"
OUTPUT = RESULT / "reflection_depth_analysis" / SCENE
CALIBRATION = ROOT / "projects/tradition_stereo/config/stereo.yml"
CHECKPOINT = ROOT / "projects/IGEV-plusplus/pretrained_models/igev_rt/sceneflow.pth"

# x0, y0, x1, y1. This rectangle covers the horizontal specular band on the plate.
REFLECTION_RECT = (180, 690, 540, 850)
BRIGHT_THRESHOLD = 235
DEPTH_LIMIT = 200.0
PLANE_INLIER_THRESHOLD = 1.0

if str(EXPERIMENT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))
from run_comparison import build_igev, run_igev  # noqa: E402


def load_q(path: Path) -> np.ndarray:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(path)
    q = storage.getNode("Q").mat()
    storage.release()
    if q is None or q.shape != (4, 4):
        raise ValueError(f"Invalid Q matrix: {path}")
    return q.astype(np.float64)


def run_final_igev(left: np.ndarray, right: np.ndarray, output: Path) -> tuple[np.ndarray, float | None]:
    path = output / "igev_final_128_16_disp.npy"
    if path.is_file():
        return np.load(path).astype(np.float32, copy=False), None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to rerun final IGEV++ RT")
    device = torch.device("cuda")
    model, model_args = build_igev(CHECKPOINT, device, max_disp=128, mixed_precision=False)
    disparity, seconds = run_igev(model, model_args, left, right, device, valid_iters=16)
    output.mkdir(parents=True, exist_ok=True)
    np.save(path, disparity.astype(np.float32))
    return disparity, seconds


def depth_from_disparity(disparity: np.ndarray, fb: float) -> np.ndarray:
    return np.divide(
        fb,
        disparity,
        out=np.full(disparity.shape, np.nan, dtype=np.float64),
        where=np.isfinite(disparity) & (disparity > 0),
    )


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def fit_depth_plane_ransac(
    depth: np.ndarray,
    mask: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    threshold: float = PLANE_INLIER_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    valid = mask & np.isfinite(depth) & (depth > 0) & (depth < DEPTH_LIMIT)
    design = np.column_stack((xx[valid], yy[valid], np.ones(valid.sum()))).astype(np.float64)
    target = depth[valid].astype(np.float64)
    rng = np.random.default_rng(20260819)
    count = min(5000, len(target))
    selected = rng.choice(len(target), count, replace=False)
    sample_x = design[selected]
    sample_y = target[selected]
    best_count = -1
    best_coefficients = None
    for _ in range(600):
        indices = rng.choice(count, 3, replace=False)
        try:
            coefficients = np.linalg.solve(sample_x[indices], sample_y[indices])
        except np.linalg.LinAlgError:
            continue
        inlier_count = int((np.abs(sample_x @ coefficients - sample_y) < threshold).sum())
        if inlier_count > best_count:
            best_count = inlier_count
            best_coefficients = coefficients
    if best_coefficients is None:
        raise RuntimeError("Could not fit the local depth plane")
    inliers = np.abs(design @ best_coefficients - target) < threshold
    coefficients = np.linalg.lstsq(design[inliers], target[inliers], rcond=None)[0]
    fit_residual = np.abs(design[inliers] @ coefficients - target[inliers])
    metadata = {
        "candidate_pixels": int(len(target)),
        "inlier_pixels": int(inliers.sum()),
        "inlier_ratio_pct": float(100.0 * inliers.mean()),
        "inlier_median_abs_residual": float(np.median(fit_residual)),
        "threshold": threshold,
        "coefficients_z_ax_by_c": coefficients.tolist(),
    }
    return coefficients, metadata


def apply_plane(coefficients: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    return coefficients[0] * xx + coefficients[1] * yy + coefficients[2]


def colorize_depth(depth: np.ndarray, low: float = 20.0, high: float = 100.0) -> np.ndarray:
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0)
    image = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~np.isfinite(depth) | (depth <= 0) | (depth >= DEPTH_LIMIT)] = 0
    return image


def colorize_residual(residual: np.ndarray, limit: float = 20.0) -> np.ndarray:
    normalized = np.clip((residual + limit) / (2.0 * limit), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0)
    image = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~np.isfinite(residual)] = 0
    return image


def label(image: np.ndarray, text: str) -> np.ndarray:
    header = np.zeros((46, image.shape[1], 3), dtype=np.uint8)
    scale = 0.52 if len(text) < 54 else 0.44
    cv2.putText(header, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def resize_panel(image: np.ndarray, width: int = 520) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_diagnostic(
    left: np.ndarray,
    bright: np.ndarray,
    depths: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    metrics: dict,
    path: Path,
) -> None:
    x0, y0, x1, y1 = 140, 560, 570, 950
    region = np.s_[y0:y1, x0:x1]
    overlay = left.copy()
    red = overlay.copy()
    red[bright] = (20, 20, 255)
    overlay = cv2.addWeighted(overlay, 0.48, red, 0.52, 0.0)
    panels = [
        label(resize_panel(overlay[region]), f"Bright core in red: gray>{BRIGHT_THRESHOLD}"),
        label(resize_panel(colorize_depth(depths["igev"])[region]), "Final IGEV++ RT depth [20,100] mm"),
        label(resize_panel(colorize_depth(depths["las"])[region]), "LiteAnyStereo depth [20,100] mm"),
        label(resize_panel(colorize_depth(depths["reference"])[region]), "Supplied PLY depth [20,100] mm; not ground truth"),
        label(
            resize_panel(colorize_residual(residuals["igev"], 20.0)[region]),
            f"IGEV local-plane residual +/-20 mm; median {metrics['plane_residual']['igev']['median']:.2f}",
        ),
        label(
            resize_panel(colorize_residual(residuals["las"], 20.0)[region]),
            f"LAS local-plane residual +/-20 mm; median {metrics['plane_residual']['las']['median']:.2f}",
        ),
    ]
    comparison = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    cv2.imwrite(str(path), comparison, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def make_difference_diagnostic(
    left: np.ndarray,
    bright: np.ndarray,
    depths: dict[str, np.ndarray],
    metrics: dict,
    path: Path,
) -> None:
    x0, y0, x1, y1 = 140, 560, 570, 950
    region = np.s_[y0:y1, x0:x1]
    overlay = left.copy()
    red = overlay.copy()
    red[bright] = (20, 20, 255)
    overlay = cv2.addWeighted(overlay, 0.48, red, 0.52, 0.0)
    difference = depths["igev"] - depths["las"]
    igev_reference = depths["igev"] - depths["reference"]
    las_reference = depths["las"] - depths["reference"]
    panels = [
        label(resize_panel(overlay[region]), f"Reflection mask: {metrics['reflection_definition']['pixels']} pixels"),
        label(
            resize_panel(colorize_residual(difference, 100.0)[region]),
            f"IGEV - LAS depth +/-100 mm; median +{metrics['final_igev_minus_las_depth']['median']:.2f}",
        ),
        label(
            resize_panel(colorize_residual(igev_reference, 100.0)[region]),
            f"IGEV - supplied PLY +/-100 mm; median +{metrics['supplied_ply_consistency_not_ground_truth']['igev']['signed']['median']:.2f}",
        ),
        label(
            resize_panel(colorize_residual(las_reference, 100.0)[region]),
            f"LAS - supplied PLY +/-100 mm; median {metrics['supplied_ply_consistency_not_ground_truth']['las']['signed']['median']:.2f}",
        ),
    ]
    comparison = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    cv2.imwrite(str(path), comparison, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def row_medians(values: np.ndarray, mask: np.ndarray, y0: int, y1: int) -> np.ndarray:
    result = np.full(y1 - y0, np.nan, dtype=np.float64)
    for index, y in enumerate(range(y0, y1)):
        valid = mask[y] & np.isfinite(values[y]) & (values[y] > 0) & (values[y] < DEPTH_LIMIT)
        if valid.any():
            result[index] = np.median(values[y, valid])
    return result


def make_profile(gray: np.ndarray, depths: dict[str, np.ndarray], xx: np.ndarray, path: Path) -> None:
    x0, _, x1, _ = REFLECTION_RECT
    y0, y1 = 560, 950
    ys = np.arange(y0, y1)
    horizontal = (xx >= x0) & (xx < x1)
    bright_fraction = np.asarray([100.0 * (gray[y, x0:x1] > BRIGHT_THRESHOLD).mean() for y in ys])
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    axes[0].plot(ys, bright_fraction, color="#d62728", linewidth=1.4)
    axes[0].axvspan(REFLECTION_RECT[1], REFLECTION_RECT[3], color="#ffcc66", alpha=0.22)
    axes[0].set_ylabel("pixels >235 (%)")
    axes[0].set_title(f"{SCENE}: reflective-band depth profile")
    colors = {"reference": "black", "igev": "#e67e22", "las": "#2878b5"}
    labels = {"reference": "Supplied PLY", "igev": "Final IGEV++ RT", "las": "LiteAnyStereo LAS1"}
    for key in ("reference", "igev", "las"):
        profile = row_medians(depths[key], horizontal, y0, y1)
        axes[1].plot(ys, profile, color=colors[key], label=labels[key], linewidth=1.25)
    axes[1].axvspan(REFLECTION_RECT[1], REFLECTION_RECT[3], color="#ffcc66", alpha=0.22, label="reflection rectangle")
    axes[1].set_ylim(20, 200)
    axes[1].set_xlabel("image y (pixel)")
    axes[1].set_ylabel("row-median depth (mm)")
    axes[1].grid(alpha=0.22)
    axes[1].legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_zoomed_profile(depths: dict[str, np.ndarray], path: Path) -> None:
    x0, _, x1, _ = REFLECTION_RECT
    y0, y1 = 560, 950
    ys = np.arange(y0, y1)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    colors = {"reference": "black", "las": "#2878b5"}
    labels = {"reference": "Supplied PLY", "las": "LiteAnyStereo LAS1"}
    for key in ("reference", "las"):
        medians = np.full(len(ys), np.nan, dtype=np.float64)
        lower = np.full(len(ys), np.nan, dtype=np.float64)
        upper = np.full(len(ys), np.nan, dtype=np.float64)
        for index, y in enumerate(ys):
            values = depths[key][y, x0:x1]
            values = values[np.isfinite(values) & (values > 0) & (values < DEPTH_LIMIT)]
            if values.size:
                lower[index], medians[index], upper[index] = np.percentile(values, (25, 50, 75))
        valid = np.isfinite(medians)
        coefficients = np.polyfit(ys[valid], medians[valid], 1)
        trend = np.polyval(coefficients, ys)
        axes[0].plot(ys, medians, color=colors[key], label=labels[key], linewidth=1.4)
        axes[0].fill_between(ys, lower, upper, color=colors[key], alpha=0.12)
        axes[1].plot(
            ys,
            medians - trend,
            color=colors[key],
            label=f"{labels[key]} detrended",
            linewidth=1.2,
        )
    for axis in axes:
        axis.axvspan(REFLECTION_RECT[1], REFLECTION_RECT[3], color="#ffcc66", alpha=0.22)
        axis.grid(alpha=0.22)
        axis.legend(loc="upper left")
    axes[0].set_ylim(24, 36)
    axes[0].set_ylabel("row depth (mm)")
    axes[0].set_title(f"{SCENE}: zoomed LAS / supplied-PLY profile; band = row IQR")
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_ylabel("detrended median (mm)")
    axes[1].set_xlabel("image y (pixel)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    left = cv2.imread(str(RESULT / "preprocessed" / SCENE / "left.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(RESULT / "preprocessed" / SCENE / "right.png"), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Missing preprocessed pair for {SCENE}")
    igev_disparity, inference_seconds = run_final_igev(left, right, OUTPUT)
    las_disparity = np.load(RESULT / "liteanystereo" / SCENE / "disp.npy").astype(np.float32, copy=False)
    reference_disparity = np.load(RESULT / "reference" / SCENE / "disp.npy").astype(np.float32, copy=False)
    early_igev_disparity = np.load(RESULT / "igev_rt" / SCENE / "disp.npy").astype(np.float32, copy=False)
    q = load_q(CALIBRATION)
    fb = float(q[2, 3] / q[3, 2])
    depths = {
        "igev": depth_from_disparity(igev_disparity, fb),
        "las": depth_from_disparity(las_disparity, fb),
        "reference": depth_from_disparity(reference_disparity, fb),
        "early_igev_192_8": depth_from_disparity(early_igev_disparity, fb),
    }
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    yy, xx = np.indices(gray.shape)
    rx0, ry0, rx1, ry1 = REFLECTION_RECT
    rectangle = (xx >= rx0) & (xx < rx1) & (yy >= ry0) & (yy < ry1)
    bright = rectangle & (gray > BRIGHT_THRESHOLD)
    adjacent = (
        (xx >= rx0)
        & (xx < rx1)
        & (((yy >= 560) & (yy < 670)) | ((yy >= 870) & (yy < 980)))
        & (gray >= 25)
        & (gray < 200)
    )

    coefficients = {}
    plane_fit = {}
    residuals = {}
    plane_residual = {}
    for key in ("reference", "igev", "las"):
        coefficients[key], plane_fit[key] = fit_depth_plane_ransac(depths[key], adjacent, xx, yy)
        residuals[key] = depths[key] - apply_plane(coefficients[key], xx, yy)
        valid = bright & np.isfinite(depths[key]) & (depths[key] > 0) & (depths[key] < DEPTH_LIMIT)
        plane_residual[key] = stats(residuals[key][valid])
        plane_residual[key]["median_abs"] = float(np.median(np.abs(residuals[key][valid])))
        plane_residual[key]["p95_abs"] = float(np.percentile(np.abs(residuals[key][valid]), 95))
        plane_residual[key]["valid_under_200_pct"] = float(100.0 * valid.sum() / bright.sum())

    reflection_depth = {}
    for key in ("reference", "igev", "las", "early_igev_192_8"):
        valid = bright & np.isfinite(depths[key]) & (depths[key] > 0)
        reflection_depth[key] = stats(depths[key][valid])
        reflection_depth[key]["over_200_pct"] = float(100.0 * (bright & (depths[key] >= DEPTH_LIMIT)).sum() / bright.sum())

    both = bright & np.isfinite(depths["igev"]) & np.isfinite(depths["las"])
    difference = depths["igev"][both] - depths["las"][both]
    direct_difference = stats(difference)
    direct_difference["median_abs"] = float(np.median(np.abs(difference)))
    direct_difference["p95_abs"] = float(np.percentile(np.abs(difference), 95))

    reference_valid = bright & (reference_disparity > 0) & np.isfinite(depths["reference"])
    reference_consistency = {}
    for key in ("igev", "las"):
        valid = reference_valid & np.isfinite(depths[key]) & (depths[key] > 0)
        error = depths[key][valid] - depths["reference"][valid]
        reference_consistency[key] = {
            "pixels": int(valid.sum()),
            "signed": stats(error),
            "median_abs": float(np.median(np.abs(error))),
            "mean_abs": float(np.mean(np.abs(error))),
            "p95_abs": float(np.percentile(np.abs(error), 95)),
        }

    metrics = {
        "scene": SCENE,
        "models": {
            "old_final": "IGEV++ RT SceneFlow, max_disp=128, valid_iters=16, FP32",
            "new": "LiteAnyStereo LAS1 official, max_disp=192, FP32",
            "early_jop_output_not_used_as_old_final": "IGEV++ RT max_disp=192, valid_iters=8, FP32",
        },
        "inference_seconds_if_rerun": inference_seconds,
        "calibration": str(CALIBRATION),
        "depth_formula": "Z = (Q[2,3] / Q[3,2]) / disparity",
        "focal_baseline": fb,
        "depth_unit": "mm according to tradition_stereo/POINTMAP_FORMAT.md and calibration convention",
        "reflection_definition": {
            "rectangle_xyxy": list(REFLECTION_RECT),
            "gray_threshold": f">{BRIGHT_THRESHOLD}",
            "pixels": int(bright.sum()),
            "rectangle_pixels": int(rectangle.sum()),
            "bright_fraction_pct": float(100.0 * bright.sum() / rectangle.sum()),
        },
        "reflection_depth": reflection_depth,
        "final_igev_minus_las_depth": direct_difference,
        "adjacent_plane_fit": plane_fit,
        "plane_residual": plane_residual,
        "supplied_ply_consistency_not_ground_truth": reference_consistency,
        "interpretation_scope": "Relative behavior on one reflective Jop1 scene; supplied PLY is a consistency reference, not certified ground truth.",
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_diagnostic(left, bright, depths, residuals, metrics, OUTPUT / "reflection_depth_comparison.png")
    make_difference_diagnostic(left, bright, depths, metrics, OUTPUT / "reflection_model_difference.png")
    make_profile(gray, depths, xx, OUTPUT / "reflection_vertical_profile.png")
    make_zoomed_profile(depths, OUTPUT / "reflection_vertical_profile_zoomed.png")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
