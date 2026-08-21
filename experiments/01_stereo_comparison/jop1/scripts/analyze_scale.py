#!/usr/bin/env python3
"""Inspect whether ruler markings appear in geometry or only in RGB texture."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from run_sgbm_reference import add_label, load_calibration


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[2]
RESULT_ROOT = EXPERIMENT_ROOT / "results/final_9"
OUTPUT_ROOT = RESULT_ROOT / "scale_geometry_analysis"

# (x0, y0, x1, y1, profile_x): regions containing the clearest ruler marks.
SCALE_REGIONS = {
    "camera-202412091814-0105": (250, 280, 540, 1080, 370),
    "camera-202412091815-0107": (180, 220, 520, 1060, 330),
    "camera-202412091816-0108": (140, 220, 570, 980, 350),
    "camera-202412091818-0110": (180, 220, 490, 1140, 350),
    "camera-202412091822-0111": (20, 220, 250, 1030, 105),
    "camera-202412091822-0112": (0, 100, 260, 960, 95),
}


def weighted_smooth(values: np.ndarray, valid: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    weights = valid.astype(np.float32)
    numerator = cv2.GaussianBlur(values.astype(np.float32) * weights, (0, 0), sigma)
    denominator = cv2.GaussianBlur(weights, (0, 0), sigma)
    return numerator / np.maximum(denominator, 1e-6)


def local_residual(values: np.ndarray, valid: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    return values.astype(np.float32) - weighted_smooth(values, valid, sigma)


def residual_color(values: np.ndarray, valid: np.ndarray, limit: float) -> np.ndarray:
    residual = local_residual(values, valid)
    normalized = np.clip((residual + limit) / (2.0 * limit), 0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def normal_shading(disparity: np.ndarray, valid: np.ndarray, q: np.ndarray) -> np.ndarray:
    safe_disparity = np.where(valid, disparity, 1.0).astype(np.float32)
    points = cv2.reprojectImageTo3D(safe_disparity, q)
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.maximum(length, 1e-8)
    light = np.asarray([0.35, -0.45, 0.82], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.abs(np.sum(normal * light, axis=2))
    shade = np.clip(0.18 + 0.82 * diffuse, 0.0, 1.0)
    image = np.repeat((shade[..., None] * 255).astype(np.uint8), 3, axis=2)
    safe = valid & np.isfinite(points).all(axis=2) & (points[..., 2] > 0) & (points[..., 2] < 200)
    safe = cv2.erode(safe.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    image[~safe] = 0
    return image


def crop(image: np.ndarray, region) -> np.ndarray:
    x0, y0, x1, y1, _ = region
    return image[y0:y1, x0:x1]


def resize_panel(image: np.ndarray, width: int = 380) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def texture_metrics(gray, disparity, valid, region):
    x0, y0, x1, y1, _ = region
    roi = np.zeros_like(valid, dtype=bool)
    roi[y0:y1, x0:x1] = True
    interior = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    mask = roi & interior
    if mask.sum() < 100:
        return None

    image_high = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
    disp_high = local_residual(disparity, valid, sigma=12.0)
    image_strength = np.abs(image_high)
    depth_strength = np.abs(disp_high)
    edge_threshold = np.percentile(image_strength[mask], 85)
    flat_threshold = np.percentile(image_strength[mask], 50)
    edge = mask & (image_strength >= edge_threshold)
    flat = mask & (image_strength <= flat_threshold)
    correlation = float(np.corrcoef(image_strength[mask], depth_strength[mask])[0, 1])
    edge_response = float(np.median(depth_strength[edge]))
    flat_response = float(np.median(depth_strength[flat]))
    return {
        "pixels": int(mask.sum()),
        "texture_depth_correlation": correlation,
        "edge_response_px": edge_response,
        "flat_response_px": flat_response,
        "edge_to_flat_ratio": edge_response / max(flat_response, 1e-6),
    }


def make_profile(stem, region, gray, disparities, validities, q):
    x0, y0, x1, y1, profile_x = region
    fb = float(q[2, 3] / q[3, 2])
    ys = np.arange(y0, y1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(ys, gray[y0:y1, profile_x], color="black", linewidth=1.0)
    axes[0].set_ylabel("gray value")
    axes[0].set_title(f"{stem}: vertical profile at x={profile_x}")
    colors = {"PLY reference": "black", "IGEV++ RT": "tab:orange", "LAS1": "tab:blue"}
    for label, disparity in disparities.items():
        valid = validities[label]
        depth = np.divide(fb, disparity, out=np.full_like(disparity, np.nan), where=valid & (disparity > 0))
        trend = weighted_smooth(np.nan_to_num(depth), np.isfinite(depth), sigma=18.0)
        residual = depth - trend
        profile = residual[y0:y1, profile_x]
        profile[~valid[y0:y1, profile_x]] = np.nan
        axes[1].plot(ys, profile, label=label, color=colors[label], linewidth=1.0, alpha=0.9)
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[1].set_xlabel("image y (pixel)")
    axes[1].set_ylabel("local depth residual (Q units)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    path = OUTPUT_ROOT / stem / "vertical_profile.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    q = load_calibration(ROOT / "projects/tradition_stereo/config/stereo.yml")["Q"].astype(np.float32)
    rows = []
    overview = []

    for stem, region in SCALE_REGIONS.items():
        scene_out = OUTPUT_ROOT / stem
        scene_out.mkdir(parents=True, exist_ok=True)
        left = cv2.imread(str(RESULT_ROOT / "preprocessed" / stem / "left.png"), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        disparities = {
            "PLY reference": np.load(RESULT_ROOT / "reference" / stem / "disp.npy").astype(np.float32),
            "IGEV++ RT": np.load(RESULT_ROOT / "igev_rt" / stem / "disp.npy").astype(np.float32),
            "LAS1": np.load(RESULT_ROOT / "liteanystereo" / stem / "disp.npy").astype(np.float32),
        }
        validities = {
            "PLY reference": disparities["PLY reference"] > 0,
            "IGEV++ RT": np.isfinite(disparities["IGEV++ RT"]) & (disparities["IGEV++ RT"] > 0),
            "LAS1": np.isfinite(disparities["LAS1"]) & (disparities["LAS1"] > 0),
        }

        rgb = crop(left.copy(), region)
        profile_x = region[4] - region[0]
        cv2.line(rgb, (profile_x, 0), (profile_x, rgb.shape[0] - 1), (0, 0, 255), 2)
        relief_panels = [add_label(resize_panel(rgb), "RGB crop; red = profile")]
        normal_panels = [add_label(resize_panel(np.full_like(rgb, 96)), "Geometry-only normal shading")]
        for label in ("PLY reference", "IGEV++ RT", "LAS1"):
            disparity = disparities[label]
            valid = validities[label]
            relief = crop(residual_color(disparity, valid, limit=2.0), region)
            normals = crop(normal_shading(disparity, valid, q), region)
            relief_panels.append(add_label(resize_panel(relief), f"{label}: local disparity +/-2 px"))
            normal_panels.append(add_label(resize_panel(normals), f"{label}: no RGB texture"))
            metrics = texture_metrics(gray, disparity, valid, region)
            row = {"scene": stem, "source": label}
            if metrics is not None:
                row.update(metrics)
            rows.append(row)

        comparison = np.vstack([np.hstack(relief_panels), np.hstack(normal_panels)])
        cv2.imwrite(str(scene_out / "geometry_comparison.png"), comparison)
        make_profile(stem, region, gray, disparities, validities, q)
        overview.append(cv2.resize(comparison, (1200, int(comparison.shape[0] * 1200 / comparison.shape[1])), interpolation=cv2.INTER_AREA))

    with (OUTPUT_ROOT / "texture_geometry_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(OUTPUT_ROOT / "overview.jpg"), np.vstack(overview), [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"Wrote {len(SCALE_REGIONS)} scale-region analyses to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
