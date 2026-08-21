#!/usr/bin/env python3
"""Run LiteAnyStereo and the tradition_stereo SGBM baseline on a Jop archive.

The archive contains raw RGBA ``*_L.png``/``*_R.png`` pairs and a PLY for each
pair.  The existing tradition_stereo pipeline expects portrait images, so the
raw images are rotated counter-clockwise and rectified with config/stereo.yml.

Outputs are written below ``--output-root``:

  preprocessed/<scene>/left.png, right.png
  liteanystereo/<scene>/disp.npy, vis.png, disparity_color.png, cloud.ply
  tradition_sgbm/<scene>/disp.npy, vis.png, disparity_color.png, cloud.ply
  reference/<scene>/disp.npy, disparity_color.png
  comparison/<scene>/comparison.png
  metrics/per_scene.csv, summary.json

The supplied PLY is sparse, so its projection is used only as a *PLY reference
consistency* check.  It is not treated as dense ground-truth disparity.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
LAS_ROOT = REPO_ROOT / "projects/LiteAnyStereo"
if str(LAS_ROOT) not in sys.path:
    sys.path.insert(0, str(LAS_ROOT))

from core.models import build_model, load_model_weights  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


SCENE_PREFIX = "camera-"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "datasets/Jop_1/raw",
        help="Directory containing *_L.png, *_R.png and matching .ply files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "results/sgbm_reference",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=LAS_ROOT / "checkpoints/LiteAnyStereo.pth",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "projects/tradition_stereo/config/stereo.yml",
    )
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--rotation",
        choices=["ccw90", "cw90", "none"],
        default="ccw90",
        help="Match tradition_stereo/save_rawimg.py's 90-degree preprocessing.",
    )
    parser.add_argument(
        "--no-rectify",
        action="store_true",
        help="Skip stereo rectification; useful only for diagnostic runs.",
    )
    parser.add_argument(
        "--no-pointcloud",
        action="store_true",
        help="Do not write predicted point clouds.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_calibration(path: Path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration: {path}")
    values = {
        key: fs.getNode(key).mat()
        for key in ["M1", "D1", "M2", "D2", "R1", "R2", "P1", "P2", "Q"]
    }
    fs.release()
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Calibration is missing keys: {missing}")
    return values


def discover_scenes(input_root: Path):
    scenes = []
    for left in sorted(input_root.glob("*_L.png")):
        stem = left.name[:-len("_L.png")]
        right = input_root / f"{stem}_R.png"
        ply = input_root / f"{stem}.ply"
        if not right.is_file():
            logging.warning("Skipping %s: missing right image", stem)
            continue
        if not ply.is_file():
            logging.warning("Skipping %s: missing PLY reference", stem)
            continue
        scenes.append((stem, left, right, ply))
    if not scenes:
        raise FileNotFoundError(f"No complete *_L.png/*_R.png/.ply triplets in {input_root}")
    return scenes


def read_and_prepare(path: Path, rotation: str):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if rotation == "ccw90":
        image = np.rot90(image, 1)
    elif rotation == "cw90":
        image = np.rot90(image, 3)
    return np.ascontiguousarray(image)


def rectify_pair(left, right, calibration):
    height, width = left.shape[:2]
    size = (width, height)
    left_map = cv2.initUndistortRectifyMap(
        calibration["M1"],
        calibration["D1"],
        calibration["R1"],
        calibration["P1"],
        size,
        cv2.CV_16SC2,
    )
    right_map = cv2.initUndistortRectifyMap(
        calibration["M2"],
        calibration["D2"],
        calibration["R2"],
        calibration["P2"],
        size,
        cv2.CV_16SC2,
    )
    left = cv2.remap(left, left_map[0], left_map[1], cv2.INTER_LINEAR)
    right = cv2.remap(right, right_map[0], right_map[1], cv2.INTER_LINEAR)
    return left, right


def colorize(disparity, maximum, valid=None):
    disparity = np.asarray(disparity, dtype=np.float32)
    finite = np.isfinite(disparity)
    normalized = np.clip(disparity / float(maximum), 0.0, 1.0)
    normalized = np.where(finite, normalized, 0.0)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if valid is None:
        valid = finite
    color[~(finite & np.asarray(valid, dtype=bool))] = 0
    return color


def add_label(image, label):
    header = np.zeros((38, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def save_disparity_visuals(directory: Path, left, disparity, label, max_disp):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "disp.npy", disparity.astype(np.float32))
    auto_max = float(np.nanpercentile(disparity[np.isfinite(disparity)], 99.0)) if np.isfinite(disparity).any() else 1.0
    auto_max = max(auto_max, 1.0)
    auto = colorize(disparity, auto_max)
    fixed = colorize(disparity, max_disp)
    cv2.imwrite(str(directory / "disparity_color.png"), fixed)
    cv2.imwrite(str(directory / "vis.png"), np.hstack([left, auto]))
    cv2.imwrite(str(directory / "vis_fixed.png"), np.hstack([left, fixed]))


def make_reference_disparity(ply_path: Path, calibration, shape):
    # PLY is binary and Open3D is available in the LiteAnyStereo environment.
    import open3d as o3d

    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Empty PLY: {ply_path}")
    height, width = shape
    q = calibration["Q"].astype(np.float32)
    f = float(q[2, 3])
    cx = float(-q[0, 3])
    cy = float(-q[1, 3])
    inv_baseline = float(q[3, 2])
    x, y, z = points.T
    valid = np.isfinite(points).all(axis=1) & (z > 0) & (inv_baseline != 0)
    u = np.rint(x[valid] * f / z[valid] + cx).astype(np.int32)
    v = np.rint(y[valid] * f / z[valid] + cy).astype(np.int32)
    disparity = f / (z[valid] * inv_baseline)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & np.isfinite(disparity) & (disparity > 0)
    u, v, disparity = u[inside], v[inside], disparity[inside]
    reference = np.zeros((height, width), dtype=np.float32)
    # Keep the largest disparity when several 3D points hit one pixel.
    np.maximum.at(reference, (v, u), disparity)
    return reference


def evaluate_against_reference(prediction, reference):
    valid = np.isfinite(reference) & (reference > 0) & np.isfinite(prediction)
    if not valid.any():
        return {"reference_pixels": 0, "prediction_coverage": 0.0}
    error = np.abs(prediction[valid] - reference[valid])
    ref = reference[valid]
    relative = error / np.maximum(np.abs(ref), 1e-6)
    predicted_valid = prediction[valid] > 0
    return {
        "reference_pixels": int(valid.sum()),
        "prediction_coverage": float(predicted_valid.mean()),
        "epe_px": float(error.mean()),
        "bad1_pct": float((error > 1.0).mean() * 100.0),
        "bad2_pct": float((error > 2.0).mean() * 100.0),
        "bad3_pct": float((error > 3.0).mean() * 100.0),
        "d1_pct": float(((error > 3.0) & (relative > 0.05)).mean() * 100.0),
    }


def save_point_cloud(directory: Path, disparity, left_bgr, q):
    import open3d as o3d

    points_3d = cv2.reprojectImageTo3D(disparity.astype(np.float32), q)
    z = points_3d[..., 2]
    valid = (
        np.isfinite(disparity)
        & (disparity > 0)
        & (disparity <= 192)
        & np.isfinite(points_3d).all(axis=2)
        & (z > 0)
        & (z < 200)
    )
    points = points_3d[valid].astype(np.float64)
    colors = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)[valid].astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    directory.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(directory / "cloud.ply"), cloud, write_ascii=False, compressed=False)
    return int(len(points))


def run_sgbm(left, right):
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=192,
        blockSize=9,
        P1=8 * 3 * 9**2,
        P2=32 * 3 * 9**2,
        disp12MaxDiff=5,
        uniquenessRatio=10,
        speckleWindowSize=200,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    started = time.perf_counter()
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    elapsed = time.perf_counter() - started
    disparity[disparity < 0] = 0
    return disparity, elapsed


def build_las(checkpoint: Path, device, max_disp):
    model = build_model("las1", fnet_pretrained=False, max_disp=max_disp).to(device).eval()
    checkpoint_data = torch.load(str(checkpoint), map_location=device)
    load_model_weights(model, checkpoint_data, strict=True)
    return model


@torch.inference_mode()
def run_las(model, left, right, device, max_disp):
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    left_tensor = torch.from_numpy(left_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    right_tensor = torch.from_numpy(right_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    padder = InputPadder(left_tensor.shape, divis_by=32)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = model(left_tensor, right_tensor, max_disp=max_disp, test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction = padder.unpad(prediction.float())[0, 0].cpu().numpy()
    return prediction.astype(np.float32), elapsed


def save_comparison(
    directory,
    left,
    baseline_disparity,
    las,
    reference,
    max_disp,
    baseline_label="tradition_stereo SGBM",
):
    directory.mkdir(parents=True, exist_ok=True)
    baseline_color = colorize(baseline_disparity, max_disp)
    las_color = colorize(las, max_disp)
    ref_color = colorize(reference, max_disp, reference > 0)
    baseline_error = colorize(np.abs(baseline_disparity - reference), 20.0, reference > 0)
    las_error = colorize(np.abs(las - reference), 20.0, reference > 0)
    panels = [
        add_label(left, "Rectified left"),
        add_label(baseline_color, baseline_label),
        add_label(las_color, "LiteAnyStereo LAS1"),
        add_label(ref_color, "Supplied PLY reference"),
        add_label(baseline_error, f"{baseline_label} abs error vs PLY [0,20]"),
        add_label(las_error, "LAS1 abs error vs PLY [0,20]"),
    ]
    top = np.hstack(panels[:3])
    bottom = np.hstack(panels[3:])
    cv2.imwrite(str(directory / "comparison.png"), np.vstack([top, bottom]))


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output_root.exists() and not args.overwrite and (output_root / "metrics/summary.json").exists():
        raise FileExistsError(f"{output_root} already contains results; pass --overwrite")

    scenes = discover_scenes(input_root)
    calibration = load_calibration(calibration_path)
    device = torch.device(args.device)
    model = build_las(checkpoint, device, args.max_disp)
    logging.info("Loaded LAS1 checkpoint on %s; processing %d scenes", device, len(scenes))

    # Warm up CUDA without contaminating the reported per-scene LAS timings.
    warmup_left = read_and_prepare(scenes[0][1], args.rotation)
    warmup_right = read_and_prepare(scenes[0][2], args.rotation)
    if not args.no_rectify:
        warmup_left, warmup_right = rectify_pair(warmup_left, warmup_right, calibration)
    run_las(model, warmup_left, warmup_right, device, args.max_disp)

    rows = []
    las_times = []
    sgbm_times = []
    started_all = time.perf_counter()
    for index, (stem, left_path, right_path, ply_path) in enumerate(scenes, start=1):
        left = read_and_prepare(left_path, args.rotation)
        right = read_and_prepare(right_path, args.rotation)
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in {stem}: {left.shape} vs {right.shape}")
        if not args.no_rectify:
            left, right = rectify_pair(left, right, calibration)

        pre_dir = output_root / "preprocessed" / stem
        pre_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(pre_dir / "left.png"), left)
        cv2.imwrite(str(pre_dir / "right.png"), right)

        sgbm, sgbm_seconds = run_sgbm(left, right)
        las, las_seconds = run_las(model, left, right, device, args.max_disp)
        sgbm_times.append(sgbm_seconds)
        las_times.append(las_seconds)

        reference = make_reference_disparity(ply_path, calibration, left.shape[:2])
        reference_dir = output_root / "reference" / stem
        reference_dir.mkdir(parents=True, exist_ok=True)
        np.save(reference_dir / "disp.npy", reference)
        cv2.imwrite(str(reference_dir / "disparity_color.png"), colorize(reference, args.max_disp, reference > 0))
        save_disparity_visuals(output_root / "tradition_sgbm" / stem, left, sgbm, "SGBM", args.max_disp)
        save_disparity_visuals(output_root / "liteanystereo" / stem, left, las, "LAS1", args.max_disp)
        save_comparison(output_root / "comparison" / stem, left, sgbm, las, reference, args.max_disp)

        sgbm_metrics = evaluate_against_reference(sgbm, reference)
        las_metrics = evaluate_against_reference(las, reference)
        sgbm_points = None
        las_points = None
        if not args.no_pointcloud:
            sgbm_points = save_point_cloud(output_root / "tradition_sgbm" / stem, sgbm, left, calibration["Q"])
            las_points = save_point_cloud(output_root / "liteanystereo" / stem, las, left, calibration["Q"])

        row = {
            "scene": stem,
            "height": left.shape[0],
            "width": left.shape[1],
            "sgbm_ms": sgbm_seconds * 1000.0,
            "sgbm_fps": 1.0 / sgbm_seconds if sgbm_seconds else 0.0,
            "las_ms": las_seconds * 1000.0,
            "las_fps": 1.0 / las_seconds if las_seconds else 0.0,
            "sgbm_cloud_points": sgbm_points,
            "las_cloud_points": las_points,
        }
        for key, value in sgbm_metrics.items():
            row[f"sgbm_{key}"] = value
        for key, value in las_metrics.items():
            row[f"las_{key}"] = value
        rows.append(row)
        print(
            f"[{index:02d}/{len(scenes):02d}] {stem} "
            f"SGBM={sgbm_seconds * 1000:.1f}ms LAS={las_seconds * 1000:.1f}ms "
            f"EPE(PLY)={sgbm_metrics.get('epe_px', float('nan')):.3f}/"
            f"{las_metrics.get('epe_px', float('nan')):.3f}",
            flush=True,
        )

    write_csv(output_root / "metrics/per_scene.csv", rows)
    def stats(values):
        return {
            "mean_ms": float(np.mean(values) * 1000.0),
            "median_ms": float(np.median(values) * 1000.0),
            "mean_fps": float(1.0 / np.mean(values)),
            "min_ms": float(np.min(values) * 1000.0),
            "max_ms": float(np.max(values) * 1000.0),
        }

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "checkpoint": str(checkpoint),
        "calibration": str(calibration_path),
        "rotation": args.rotation,
        "rectified": not args.no_rectify,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scene_count": len(rows),
        "image_shape": [rows[0]["height"], rows[0]["width"]],
        "timing_scope": "core disparity inference only; shared image preparation and file writes excluded",
        "tradition_method": "OpenCV StereoSGBM matching the existing tradition_stereo/SGBM.py settings",
        "liteanystereo_method": "LiteAnyStereo LAS1 with official LiteAnyStereo.pth, FP32 inference",
        "sgbm_timing": stats(sgbm_times),
        "las_timing": stats(las_times),
        "total_seconds": time.perf_counter() - started_all,
        "reference_note": "PLY projection is sparse reference consistency, not dense or human ground truth.",
        "metrics_file": str(output_root / "metrics/per_scene.csv"),
    }
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "metrics/summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
