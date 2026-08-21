#!/usr/bin/env python3
"""Unified IGEV++ RT / LiteAnyStereo feasibility test on rec_img_set.

The script uses exactly the same rectified pair for both models, evaluates the
FDJYP-3 scenes against the phase-I Foundation Stereo reference, and exports
full/cropped disparities, fixed-scale images, filtered point clouds, comparison
images, geometry audits, per-scene metrics, and a machine-readable summary.

``rec_img_set/kedu`` is not inferred separately because it is a byte-identical
duplicate of ``rec_img_set/rectified_images_刻度``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

# Make deterministic CUDA matrix multiplication available before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import open3d as o3d
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[2]
TRADITION_ROOT = ROOT / "projects/tradition_stereo"
LAS_ROOT = ROOT / "projects/LiteAnyStereo"
IGEV_ROOT = ROOT / "projects/IGEV-plusplus"

if str(LAS_ROOT) not in sys.path:
    sys.path.insert(0, str(LAS_ROOT))
from core.models import build_model as build_las_model, load_model_weights  # noqa: E402
from core.utils.utils import InputPadder as LASPadder  # noqa: E402

if str(IGEV_ROOT) not in sys.path:
    sys.path.insert(0, str(IGEV_ROOT))
from core_rt.rt_igev_stereo import IGEVStereo, autocast  # noqa: E402
from core_rt.utils.utils import InputPadder as IGEVPadder  # noqa: E402


CROP = (234, 1052, 126, 638)  # y0, y1, x0, x1; H=818, W=512
METRICS = ("epe", "d1", "bad1", "bad2", "bad3")
EXCLUDED_SCENES = {
    "202506281607-0012",
    "202506281609-0019",
    "202506281609-0020",
    "202506281619-0053",
}


@dataclass(frozen=True)
class GroupSpec:
    key: str
    source_dir: str
    calibration: str
    reference_dir: str | None = None
    calibration_basis: str = ""


GROUPS = (
    GroupSpec(
        "fdjyp0",
        "FDJYP-0-rectified_images",
        "stereo_gongjian.yml",
        calibration_basis="Matches the FDJYP camera calibration retained in JMP-LF6020-ETH3D.",
    ),
    GroupSpec(
        "fdjyp3",
        "FDJYP-3-rectified_images",
        "stereo_gongjian.yml",
        "FDJYP-3",
        "Rectified PNGs reproduce gongjian_map exactly; Foundation Stereo reference is available.",
    ),
    GroupSpec(
        "luowen",
        "luowen_rectified_images",
        "stereo_luowen.yml",
        calibration_basis="Uses the named luowen calibration/map retained by tradition_stereo.",
    ),
    GroupSpec(
        "general_1221",
        "rectified_images",
        "stereo.yml",
        calibration_basis="Uses the 1221 calibration selected by the historical rectification script.",
    ),
    GroupSpec(
        "scale_1221",
        "rectified_images_刻度",
        "stereo.yml",
        calibration_basis="Uses the 1221 calibration selected by the historical rectification script.",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "datasets/rec_img_set")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "datasets/tradition_raw")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "results/final_203",
    )
    parser.add_argument(
        "--igev-checkpoint",
        type=Path,
        default=IGEV_ROOT / "pretrained_models/igev_rt/sceneflow.pth",
    )
    parser.add_argument(
        "--las-checkpoint",
        type=Path,
        default=LAS_ROOT / "checkpoints/LiteAnyStereo.pth",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--igev-max-disp", type=int, default=128)
    parser.add_argument("--igev-iters", type=int, default=16)
    parser.add_argument("--las-max-disp", type=int, default=192)
    parser.add_argument("--visual-max-disp", type=float, default=192.0)
    parser.add_argument("--error-max", type=float, default=20.0)
    parser.add_argument("--cloud-min-disp", type=float, default=5.0)
    parser.add_argument("--cloud-max-disp", type=float, default=192.0)
    parser.add_argument("--cloud-max-z", type=float, default=200.0)
    parser.add_argument("--black-threshold", type=int, default=50)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-pointcloud", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_q(path: Path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(path)
    q = storage.getNode("Q").mat()
    storage.release()
    if q is None or q.shape != (4, 4):
        raise ValueError(f"Invalid Q matrix: {path}")
    return q.astype(np.float32)


def discover(input_root: Path, reference_root: Path):
    samples = []
    for group in GROUPS:
        directory = input_root / group.source_dir
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for scene_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
            left = scene_dir / "im0.png"
            right = scene_dir / "im1.png"
            if not left.is_file() or not right.is_file():
                continue
            reference = (
                reference_root / group.reference_dir / scene_dir.name / "disp_cropped.npy"
                if group.reference_dir
                else None
            )
            if reference is not None and not reference.is_file():
                raise FileNotFoundError(reference)
            samples.append((group, scene_dir.name, left, right, reference))
    return samples


def validate_duplicate_archive(input_root: Path):
    first = input_root / "kedu"
    second = input_root / "rectified_images_刻度"
    names_a = sorted(path.name for path in first.iterdir() if path.is_dir())
    names_b = sorted(path.name for path in second.iterdir() if path.is_dir())
    if names_a != names_b:
        return {"identical": False, "reason": "scene lists differ", "scene_count": 0}
    checked = 0
    for scene in names_a:
        for filename in ("im0.png", "im1.png"):
            if (first / scene / filename).read_bytes() != (second / scene / filename).read_bytes():
                return {
                    "identical": False,
                    "reason": f"content differs at {scene}/{filename}",
                    "scene_count": checked,
                }
        checked += 1
    return {
        "identical": True,
        "reason": "byte-identical duplicate excluded from inference counts",
        "scene_count": checked,
        "canonical": str(second),
        "excluded_duplicate": str(first),
    }


def build_igev(checkpoint: Path, device, max_disp):
    args = SimpleNamespace(
        hidden_dim=96,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=max_disp,
        mixed_precision=False,
        precision_dtype="float32",
    )
    model = IGEVStereo(args).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval(), args


def build_las(checkpoint: Path, device):
    model = build_las_model("las1", fnet_pretrained=False).to(device).eval()
    load_model_weights(model, torch.load(str(checkpoint), map_location=device), strict=True)
    return model


def tensor_pair(left_bgr, right_bgr, device):
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    left = torch.from_numpy(left_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    right = torch.from_numpy(right_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    return left, right


@torch.inference_mode()
def infer_igev(model, model_args, left_bgr, right_bgr, device, iters):
    left, right = tensor_pair(left_bgr, right_bgr, device)
    padder = IGEVPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with autocast(enabled=False, dtype=torch.float32):
        prediction = model(left, right, iters=iters, test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction = padder.unpad(prediction.float())[0, 0].cpu().numpy()
    return prediction.astype(np.float32), elapsed


@torch.inference_mode()
def infer_las(model, left_bgr, right_bgr, device, max_disp):
    left, right = tensor_pair(left_bgr, right_bgr, device)
    padder = LASPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = model(left, right, max_disp=max_disp, test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction = padder.unpad(prediction.float())[0, 0].cpu().numpy()
    return prediction.astype(np.float32), elapsed


def epipolar_audit(left_bgr, right_bgr):
    gray_left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 720.0 / max(gray_left.shape))
    if scale < 1.0:
        gray_left = cv2.resize(gray_left, None, fx=scale, fy=scale)
        gray_right = cv2.resize(gray_right, None, fx=scale, fy=scale)
    sift = cv2.SIFT_create(nfeatures=2500)
    keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)
    matches = []
    if descriptors_left is not None and descriptors_right is not None:
        for pair in cv2.BFMatcher().knnMatch(descriptors_left, descriptors_right, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                matches.append(pair[0])
    residuals = np.asarray(
        [
            abs(keypoints_left[m.queryIdx].pt[1] - keypoints_right[m.trainIdx].pt[1]) / scale
            for m in matches
        ],
        dtype=np.float32,
    )
    if residuals.size < 8:
        return {
            "feature_matches": int(residuals.size),
            "median_vertical_residual_px": None,
            "p90_vertical_residual_px": None,
            "geometry_status": "unknown",
        }
    median = float(np.median(residuals))
    p90 = float(np.percentile(residuals, 90))
    status = "good" if median <= 1.0 else "warning" if median <= 5.0 else "high_risk"
    return {
        "feature_matches": int(residuals.size),
        "median_vertical_residual_px": median,
        "p90_vertical_residual_px": p90,
        "geometry_status": status,
    }


def compute_metrics(prediction, reference):
    valid = np.isfinite(reference) & (reference > 0.0)
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction/reference mismatch: {prediction.shape} vs {reference.shape}")
    if not valid.any():
        raise ValueError("Reference has no valid pixels")
    error = np.abs(prediction.astype(np.float64) - reference.astype(np.float64))
    relative = error / np.maximum(np.abs(reference), 1e-12)
    return {
        "epe": float(error[valid].mean()),
        "d1": float(100.0 * ((error > 3.0) & (relative > 0.05))[valid].mean()),
        "bad1": float(100.0 * (error[valid] > 1.0).mean()),
        "bad2": float(100.0 * (error[valid] > 2.0).mean()),
        "bad3": float(100.0 * (error[valid] > 3.0).mean()),
        "valid_pixels": int(valid.sum()),
        "reference_coverage_pct": float(100.0 * valid.mean()),
    }


def disparity_statistics(disparity):
    finite = np.isfinite(disparity)
    values = disparity[finite]
    return {
        "finite_pct": float(100.0 * finite.mean()),
        "positive_pct": float(100.0 * (values > 0.0).mean()),
        "disp_min": float(values.min()),
        "disp_p01": float(np.percentile(values, 1)),
        "disp_median": float(np.median(values)),
        "disp_p99": float(np.percentile(values, 99)),
        "disp_max": float(values.max()),
    }


def inter_model_statistics(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    difference = np.abs(first[valid].astype(np.float64) - second[valid].astype(np.float64))
    return {
        "inter_model_mae_px": float(difference.mean()),
        "inter_model_median_px": float(np.median(difference)),
        "inter_model_p95_px": float(np.percentile(difference, 95)),
        "inter_model_bad3_pct": float(100.0 * (difference > 3.0).mean()),
        "inter_model_correlation": float(np.corrcoef(first[valid], second[valid])[0, 1]),
    }


def colorize(values, maximum, valid=None):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    normalized = np.clip(values / float(maximum), 0.0, 1.0)
    normalized = np.where(finite, normalized, 0.0)
    color = cv2.applyColorMap(np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    mask = finite if valid is None else finite & np.asarray(valid, dtype=bool)
    color[~mask] = 0
    return color


def label(image, text):
    header = np.zeros((40, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text, (9, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def save_model_output(directory, full_disparity, crop_disparity, visual_max):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "disp_full.npy", full_disparity.astype(np.float32))
    np.save(directory / "disp_crop.npy", crop_disparity.astype(np.float32))
    cv2.imwrite(str(directory / "disp_full_color.png"), colorize(full_disparity, visual_max))
    cv2.imwrite(str(directory / "disp_crop_color.png"), colorize(crop_disparity, visual_max))


def save_comparison(path, left_crop, right_crop, igev, las, reference, visual_max, error_max):
    if reference is not None:
        valid = np.isfinite(reference) & (reference > 0.0)
        panels = [
            label(left_crop, "Rectified left ROI"),
            label(colorize(igev, visual_max), f"IGEV++ RT disparity [0,{visual_max:g}] px"),
            label(colorize(las, visual_max), f"LiteAnyStereo disparity [0,{visual_max:g}] px"),
            label(colorize(reference, visual_max, valid), f"Foundation Stereo reference [0,{visual_max:g}] px"),
            label(colorize(np.abs(igev - reference), error_max, valid), f"IGEV++ RT abs error [0,{error_max:g}] px"),
            label(colorize(np.abs(las - reference), error_max, valid), f"LiteAnyStereo abs error [0,{error_max:g}] px"),
        ]
    else:
        difference = np.abs(igev - las)
        panels = [
            label(left_crop, "Rectified left ROI"),
            label(colorize(igev, visual_max), f"IGEV++ RT disparity [0,{visual_max:g}] px"),
            label(colorize(las, visual_max), f"LiteAnyStereo disparity [0,{visual_max:g}] px"),
            label(right_crop, "Rectified right ROI"),
            label(colorize(difference, error_max), f"Inter-model abs difference [0,{error_max:g}] px"),
            label(np.zeros_like(left_crop), "No reference disparity: difference is not error"),
        ]
    montage = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), montage, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def save_cloud(
    path,
    disparity_crop,
    left_crop,
    q_full,
    *,
    min_disp,
    max_disp,
    max_z,
    black_threshold,
):
    y0, _, x0, _ = CROP
    q_crop = q_full.copy()
    q_crop[0, 3] += x0
    q_crop[1, 3] += y0
    filtered = cv2.bilateralFilter(
        disparity_crop.astype(np.float32),
        d=5,
        sigmaColor=50,
        sigmaSpace=50,
    )
    points_3d = cv2.reprojectImageTo3D(filtered, q_crop, handleMissingValues=True)
    z = points_3d[..., 2]
    not_black = np.any(left_crop > black_threshold, axis=2)
    valid = (
        np.isfinite(filtered)
        & (filtered >= min_disp)
        & (filtered <= max_disp)
        & np.isfinite(points_3d).all(axis=2)
        & (z > 0.0)
        & (z < max_z)
        & not_black
    )
    points = points_3d[valid].astype(np.float64)
    colors = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)[valid].astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise IOError(f"Could not write {path}")
    stats = {
        "cloud_points": int(valid.sum()),
        "cloud_coverage_pct": float(100.0 * valid.mean()),
        "cloud_z_p01": float(np.percentile(points[:, 2], 1)) if len(points) else None,
        "cloud_z_median": float(np.median(points[:, 2])) if len(points) else None,
        "cloud_z_p99": float(np.percentile(points[:, 2], 99)) if len(points) else None,
    }
    return stats


def aggregate_metrics(rows, prefix):
    return {
        metric: {
            "macro_mean": float(np.mean([row[f"{prefix}_{metric}"] for row in rows])),
            "median": float(np.median([row[f"{prefix}_{metric}"] for row in rows])),
            "min": float(np.min([row[f"{prefix}_{metric}"] for row in rows])),
            "max": float(np.max([row[f"{prefix}_{metric}"] for row in rows])),
        }
        for metric in METRICS
    }


def timing(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(1000.0 * values.mean()),
        "median_ms": float(1000.0 * np.median(values)),
        "p95_ms": float(1000.0 * np.percentile(values, 95)),
        "min_ms": float(1000.0 * values.min()),
        "max_ms": float(1000.0 * values.max()),
        "fps_from_mean": float(1.0 / values.mean()),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    igev_checkpoint = args.igev_checkpoint.expanduser().resolve()
    las_checkpoint = args.las_checkpoint.expanduser().resolve()
    summary_path = output_root / "metrics/summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in (input_root, reference_root, igev_checkpoint, las_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic
    torch.use_deterministic_algorithms(args.deterministic)
    output_root.mkdir(parents=True, exist_ok=True)
    duplicate_audit = validate_duplicate_archive(input_root)
    if not duplicate_audit["identical"]:
        raise ValueError(f"kedu duplicate audit failed: {duplicate_audit}")
    samples = discover(input_root, reference_root)
    if len(samples) != 203:
        raise RuntimeError(f"Expected 203 unique rectified pairs, found {len(samples)}")
    reference_count = sum(reference is not None for *_, reference in samples)
    if reference_count != 73:
        raise RuntimeError(f"Expected 73 reference scenes, found {reference_count}")

    q_matrices = {
        group.key: load_q(TRADITION_ROOT / "config" / group.calibration)
        for group in GROUPS
    }
    device = torch.device(args.device)
    igev, igev_args = build_igev(igev_checkpoint, device, args.igev_max_disp)
    las = build_las(las_checkpoint, device)
    model_info = {
        "igev_parameters": int(sum(parameter.numel() for parameter in igev.parameters())),
        "las_parameters": int(sum(parameter.numel() for parameter in las.parameters())),
    }

    warm_left = cv2.imread(str(samples[0][2]), cv2.IMREAD_COLOR)
    warm_right = cv2.imread(str(samples[0][3]), cv2.IMREAD_COLOR)
    infer_igev(igev, igev_args, warm_left, warm_right, device, args.igev_iters)
    infer_las(las, warm_left, warm_right, device, args.las_max_disp)

    rows = []
    igev_times = []
    las_times = []
    started_all = time.perf_counter()
    for index, (group, scene, left_path, right_path, reference_path) in enumerate(samples, start=1):
        started_scene = time.perf_counter()
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"Invalid pair: {group.key}/{scene}")
        if left.shape[:2] != (1280, 720):
            raise ValueError(f"Unexpected image shape {left.shape}: {group.key}/{scene}")
        audit = epipolar_audit(left, right)
        igev_full, igev_seconds = infer_igev(
            igev, igev_args, left, right, device, args.igev_iters
        )
        las_full, las_seconds = infer_las(las, left, right, device, args.las_max_disp)
        igev_times.append(igev_seconds)
        las_times.append(las_seconds)

        y0, y1, x0, x1 = CROP
        region = np.s_[y0:y1, x0:x1]
        left_crop = left[region]
        right_crop = right[region]
        igev_crop = igev_full[region]
        las_crop = las_full[region]
        reference = (
            np.load(reference_path).astype(np.float32, copy=False)
            if reference_path is not None
            else None
        )
        scene_root = output_root / "outputs" / group.key / scene
        save_model_output(scene_root / "igev_rt", igev_full, igev_crop, args.visual_max_disp)
        save_model_output(scene_root / "liteanystereo", las_full, las_crop, args.visual_max_disp)
        save_comparison(
            scene_root / "comparison.png",
            left_crop,
            right_crop,
            igev_crop,
            las_crop,
            reference,
            args.visual_max_disp,
            args.error_max,
        )

        row = {
            "group": group.key,
            "source_dir": group.source_dir,
            "scene": scene,
            "has_foundation_reference": reference is not None,
            "fixed69_status": "excluded" if scene in EXCLUDED_SCENES else "kept",
            "height": left.shape[0],
            "width": left.shape[1],
            "crop_y0": y0,
            "crop_y1": y1,
            "crop_x0": x0,
            "crop_x1": x1,
            **audit,
            "igev_ms": igev_seconds * 1000.0,
            "las_ms": las_seconds * 1000.0,
        }
        row.update({f"igev_{key}": value for key, value in disparity_statistics(igev_crop).items()})
        row.update({f"las_{key}": value for key, value in disparity_statistics(las_crop).items()})
        row.update(inter_model_statistics(igev_crop, las_crop))
        if reference is not None:
            igev_metrics = compute_metrics(igev_crop, reference)
            las_metrics = compute_metrics(las_crop, reference)
            row.update({f"igev_{key}": value for key, value in igev_metrics.items()})
            row.update({f"las_{key}": value for key, value in las_metrics.items()})
        else:
            for prefix in ("igev", "las"):
                for key in (*METRICS, "valid_pixels", "reference_coverage_pct"):
                    row[f"{prefix}_{key}"] = None

        if args.no_pointcloud:
            for prefix in ("igev", "las"):
                for key in ("cloud_points", "cloud_coverage_pct", "cloud_z_p01", "cloud_z_median", "cloud_z_p99"):
                    row[f"{prefix}_{key}"] = None
        else:
            cloud_options = {
                "min_disp": args.cloud_min_disp,
                "max_disp": args.cloud_max_disp,
                "max_z": args.cloud_max_z,
                "black_threshold": args.black_threshold,
            }
            igev_cloud = save_cloud(
                scene_root / "igev_rt/cloud.ply",
                igev_crop,
                left_crop,
                q_matrices[group.key],
                **cloud_options,
            )
            las_cloud = save_cloud(
                scene_root / "liteanystereo/cloud.ply",
                las_crop,
                left_crop,
                q_matrices[group.key],
                **cloud_options,
            )
            row.update({f"igev_{key}": value for key, value in igev_cloud.items()})
            row.update({f"las_{key}": value for key, value in las_cloud.items()})
        row["end_to_end_ms"] = (time.perf_counter() - started_scene) * 1000.0
        rows.append(row)
        metric_text = (
            f" EPE={row['igev_epe']:.3f}/{row['las_epe']:.3f}"
            if reference is not None
            else " EPE=n/a"
        )
        print(
            f"[{index:03d}/{len(samples):03d}] {group.key}/{scene} "
            f"IGEV={row['igev_ms']:.1f}ms LAS={row['las_ms']:.1f}ms "
            f"geometry={row['geometry_status']}{metric_text}",
            flush=True,
        )

    metrics_dir = output_root / "metrics"
    write_csv(metrics_dir / "per_scene.csv", rows)
    quantitative = [row for row in rows if row["has_foundation_reference"]]
    fixed69 = [row for row in quantitative if row["fixed69_status"] == "kept"]
    group_counts = Counter(row["group"] for row in rows)
    geometry_by_group = defaultdict(Counter)
    for row in rows:
        geometry_by_group[row["group"]][row["geometry_status"]] += 1
    summary = {
        "verdict_scope": "algorithm feasibility through disparity and point-cloud generation",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "unique_scene_count": len(rows),
        "quantitative_reference_scene_count": len(quantitative),
        "qualitative_only_scene_count": len(rows) - len(quantitative),
        "group_counts": dict(group_counts),
        "duplicate_audit": duplicate_audit,
        "protocol": {
            "same_pair_for_both_models": True,
            "full_image_shape": [1280, 720],
            "fixed_evaluation_crop_yx": [234, 1052, 126, 638],
            "crop_shape": [818, 512],
            "reference": "FDJYP-3 Foundation Stereo disp_cropped.npy; not human ground truth",
            "valid_mask": "finite reference and reference > 0",
            "aggregation": "per-scene metrics followed by scene macro average",
            "primary_set": "all 73 FDJYP-3 scenes; no scene or EPE filtering",
            "secondary_set": "historical fixed-69 scenes; four declared exclusions only",
            "pointcloud": {
                "disparity_filter": "bilateral d=5 sigmaColor=50 sigmaSpace=50",
                "disparity_range_px": [args.cloud_min_disp, args.cloud_max_disp],
                "z_range": [0.0, args.cloud_max_z],
                "black_threshold": args.black_threshold,
                "q_crop_adjustment": "Q[0,3]+=126; Q[1,3]+=234",
            },
        },
        "models": {
            "igev_rt": {
                "checkpoint": str(igev_checkpoint),
                "checkpoint_sha256": sha256(igev_checkpoint),
                "parameters": model_info["igev_parameters"],
                "max_disp": args.igev_max_disp,
                "valid_iters": args.igev_iters,
                "precision": "FP32",
            },
            "liteanystereo": {
                "checkpoint": str(las_checkpoint),
                "checkpoint_sha256": sha256(las_checkpoint),
                "parameters": model_info["las_parameters"],
                "max_disp": args.las_max_disp,
                "precision": "FP32",
            },
        },
        "calibrations": {
            group.key: {
                "file": str(TRADITION_ROOT / "config" / group.calibration),
                "basis": group.calibration_basis,
                "Q": q_matrices[group.key].tolist(),
            }
            for group in GROUPS
        },
        "geometry_status_all_203": dict(Counter(row["geometry_status"] for row in rows)),
        "geometry_status_by_group": {
            group: dict(counts) for group, counts in geometry_by_group.items()
        },
        "all_73_metrics": {
            "igev_rt": aggregate_metrics(quantitative, "igev"),
            "liteanystereo": aggregate_metrics(quantitative, "las"),
            "igev_epe_wins": int(sum(row["igev_epe"] < row["las_epe"] for row in quantitative)),
            "liteanystereo_epe_wins": int(sum(row["las_epe"] < row["igev_epe"] for row in quantitative)),
        },
        "fixed_69_metrics": {
            "igev_rt": aggregate_metrics(fixed69, "igev"),
            "liteanystereo": aggregate_metrics(fixed69, "las"),
            "igev_epe_wins": int(sum(row["igev_epe"] < row["las_epe"] for row in fixed69)),
            "liteanystereo_epe_wins": int(sum(row["las_epe"] < row["igev_epe"] for row in fixed69)),
            "excluded_scenes": sorted(EXCLUDED_SCENES),
        },
        "timing_scope": "core model forward pass only; CUDA synchronized; warm-up excluded",
        "timing_all_203": {
            "igev_rt": timing(igev_times),
            "liteanystereo": timing(las_times),
            "las_speedup_vs_igev": float(np.mean(igev_times) / np.mean(las_times)),
            "end_to_end_mean_ms": float(np.mean([row["end_to_end_ms"] for row in rows])),
        },
        "runtime_environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "opencv": cv2.__version__,
            "deterministic_algorithms": args.deterministic,
        },
        "total_wall_seconds": float(time.perf_counter() - started_all),
        "metrics_file": str(metrics_dir / "per_scene.csv"),
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
