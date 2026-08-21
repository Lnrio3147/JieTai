#!/usr/bin/env python3
"""Compare official RT-IGEV (IGEV++) with LiteAnyStereo on the Jop archive.

This is deliberately separate from ``run_sgbm_reference.py``: the latter is
the already-completed SGBM diagnostic baseline, while this script implements
the project's final tradition_stereo algorithm, IGEV++ RT, using the official
``core_rt`` model and RT SceneFlow checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[2]
LAS_ROOT = ROOT / "projects/LiteAnyStereo"
IGEV_ROOT = ROOT / "projects/IGEV-plusplus"
if str(LAS_ROOT) not in sys.path:
    sys.path.insert(0, str(LAS_ROOT))

from core.models import build_model, load_model_weights  # noqa: E402
from core.utils.utils import InputPadder as LASPadder  # noqa: E402

if str(IGEV_ROOT) not in sys.path:
    sys.path.insert(0, str(IGEV_ROOT))

from core_rt.rt_igev_stereo import IGEVStereo, autocast  # noqa: E402
from core_rt.utils.utils import InputPadder as IGEVPadder  # noqa: E402
from run_sgbm_reference import (  # noqa: E402
    colorize,
    discover_scenes,
    evaluate_against_reference,
    load_calibration,
    make_reference_disparity,
    read_and_prepare,
    rectify_pair,
    save_comparison,
    save_disparity_visuals,
    save_point_cloud,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "datasets/Jop_1/raw")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "results/final_9")
    parser.add_argument("--igev-checkpoint", type=Path, default=IGEV_ROOT / "pretrained_models/igev_rt/sceneflow.pth")
    parser.add_argument("--las-checkpoint", type=Path, default=LAS_ROOT / "checkpoints/LiteAnyStereo.pth")
    parser.add_argument("--calibration", type=Path, default=ROOT / "projects/tradition_stereo/config/stereo.yml")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--valid-iters", type=int, default=8, help="RT-IGEV update iterations")
    parser.add_argument("--rotation", choices=["ccw90", "cw90", "none"], default="ccw90")
    parser.add_argument("--no-rectify", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true", help="Use RT-IGEV AMP; default is FP32")
    parser.add_argument("--no-pointcloud", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_igev(checkpoint: Path, device, max_disp: int, mixed_precision: bool):
    args = SimpleNamespace(
        hidden_dim=96,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=max_disp,
        mixed_precision=mixed_precision,
        precision_dtype="float16" if mixed_precision else "float32",
    )
    model = IGEVStereo(args).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval(), args


@torch.inference_mode()
def run_igev(model, model_args, left, right, device, valid_iters):
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    left_tensor = torch.from_numpy(left_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    right_tensor = torch.from_numpy(right_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    padder = IGEVPadder(left_tensor.shape, divis_by=32)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with autocast(
        enabled=model_args.mixed_precision,
        dtype=getattr(torch, model_args.precision_dtype, torch.float16),
    ):
        prediction = model(left_tensor, right_tensor, iters=valid_iters, test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction = padder.unpad(prediction.float())[0, 0].cpu().numpy()
    return prediction.astype(np.float32), elapsed


@torch.inference_mode()
def run_las(model, left, right, device, max_disp):
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    left_tensor = torch.from_numpy(left_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    right_tensor = torch.from_numpy(right_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    padder = LASPadder(left_tensor.shape, divis_by=32)
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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def timing_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "median_ms": float(np.median(values) * 1000.0),
        "mean_fps": float(1.0 / values.mean()),
        "min_ms": float(values.min() * 1000.0),
        "max_ms": float(values.max() * 1000.0),
    }


def metric_stats(rows, prefix):
    keys = ("epe_px", "d1_pct", "bad1_pct", "bad2_pct", "bad3_pct", "prediction_coverage")
    result = {}
    for key in keys:
        values = np.asarray([float(row[f"{prefix}_{key}"]) for row in rows], dtype=np.float64)
        result[key] = {
            "macro_mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def save_overview(output_root, scenes):
    thumbnails = []
    for stem, *_ in scenes:
        image = cv2.imread(str(output_root / "comparison" / stem / "comparison.png"), cv2.IMREAD_COLOR)
        if image is None:
            continue
        width = 540
        height = int(round(image.shape[0] * width / image.shape[1]))
        thumbnails.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    if not thumbnails:
        return None
    columns = 3
    rows = []
    black = np.zeros_like(thumbnails[0])
    for start in range(0, len(thumbnails), columns):
        row = thumbnails[start:start + columns]
        row.extend([black] * (columns - len(row)))
        rows.append(np.hstack(row))
    path = output_root / "comparison" / "overview.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    igev_checkpoint = args.igev_checkpoint.expanduser().resolve()
    las_checkpoint = args.las_checkpoint.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in [igev_checkpoint, las_checkpoint, calibration_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (output_root / "metrics/summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output_root} already contains results; pass --overwrite")

    device = torch.device(args.device)
    scenes = discover_scenes(input_root)
    calibration = load_calibration(calibration_path)
    igev, igev_args = build_igev(igev_checkpoint, device, args.max_disp, args.mixed_precision)
    las = build_model("las1", fnet_pretrained=False, max_disp=args.max_disp).to(device).eval()
    load_model_weights(las, torch.load(str(las_checkpoint), map_location=device), strict=True)

    warm_left = read_and_prepare(scenes[0][1], args.rotation)
    warm_right = read_and_prepare(scenes[0][2], args.rotation)
    if not args.no_rectify:
        warm_left, warm_right = rectify_pair(warm_left, warm_right, calibration)
    run_igev(igev, igev_args, warm_left, warm_right, device, args.valid_iters)
    run_las(las, warm_left, warm_right, device, args.max_disp)

    rows = []
    igev_times = []
    las_times = []
    started_all = time.perf_counter()
    for index, (stem, left_path, right_path, ply_path) in enumerate(scenes, start=1):
        left = read_and_prepare(left_path, args.rotation)
        right = read_and_prepare(right_path, args.rotation)
        if not args.no_rectify:
            left, right = rectify_pair(left, right, calibration)
        pre_dir = output_root / "preprocessed" / stem
        pre_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(pre_dir / "left.png"), left)
        cv2.imwrite(str(pre_dir / "right.png"), right)

        igev_disp, igev_seconds = run_igev(igev, igev_args, left, right, device, args.valid_iters)
        las_disp, las_seconds = run_las(las, left, right, device, args.max_disp)
        igev_times.append(igev_seconds)
        las_times.append(las_seconds)

        reference = make_reference_disparity(ply_path, calibration, left.shape[:2])
        reference_dir = output_root / "reference" / stem
        reference_dir.mkdir(parents=True, exist_ok=True)
        np.save(reference_dir / "disp.npy", reference)
        cv2.imwrite(str(reference_dir / "disparity_color.png"), colorize(reference, args.max_disp, reference > 0))

        save_disparity_visuals(output_root / "igev_rt" / stem, left, igev_disp, "IGEV++ RT", args.max_disp)
        save_disparity_visuals(output_root / "liteanystereo" / stem, left, las_disp, "LAS1", args.max_disp)
        save_comparison(
            output_root / "comparison" / stem,
            left,
            igev_disp,
            las_disp,
            reference,
            args.max_disp,
            baseline_label="tradition_stereo IGEV++ RT",
        )

        igev_metrics = evaluate_against_reference(igev_disp, reference)
        las_metrics = evaluate_against_reference(las_disp, reference)
        igev_points = None
        las_points = None
        if not args.no_pointcloud:
            igev_points = save_point_cloud(output_root / "igev_rt" / stem, igev_disp, left, calibration["Q"])
            las_points = save_point_cloud(output_root / "liteanystereo" / stem, las_disp, left, calibration["Q"])

        row = {
            "scene": stem,
            "height": left.shape[0],
            "width": left.shape[1],
            "igev_ms": igev_seconds * 1000.0,
            "igev_fps": 1.0 / igev_seconds,
            "las_ms": las_seconds * 1000.0,
            "las_fps": 1.0 / las_seconds,
            "igev_cloud_points": igev_points,
            "las_cloud_points": las_points,
        }
        row.update({f"igev_{key}": value for key, value in igev_metrics.items()})
        row.update({f"las_{key}": value for key, value in las_metrics.items()})
        rows.append(row)
        print(
            f"[{index:02d}/{len(scenes):02d}] {stem} "
            f"IGEV++RT={igev_seconds * 1000:.1f}ms LAS={las_seconds * 1000:.1f}ms "
            f"EPE(PLY)={igev_metrics.get('epe_px', float('nan')):.3f}/"
            f"{las_metrics.get('epe_px', float('nan')):.3f}",
            flush=True,
        )

    metrics_path = output_root / "metrics/per_scene.csv"
    write_csv(metrics_path, rows)
    overview_path = save_overview(output_root, scenes)
    igev_timing = timing_stats(igev_times)
    las_timing = timing_stats(las_times)
    igev_metrics = metric_stats(rows, "igev")
    las_metrics = metric_stats(rows, "las")
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "igev_checkpoint": str(igev_checkpoint),
        "las_checkpoint": str(las_checkpoint),
        "calibration": str(calibration_path),
        "rotation": args.rotation,
        "rectified": not args.no_rectify,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scene_count": len(rows),
        "image_shape": [rows[0]["height"], rows[0]["width"]],
        "igev_model": "official IGEV++ RT core_rt model, SceneFlow checkpoint",
        "igev_valid_iters": args.valid_iters,
        "igev_mixed_precision": args.mixed_precision,
        "las_model": "LiteAnyStereo LAS1 official checkpoint, FP32 inference",
        "timing_scope": "core disparity inference only; shared image preparation and file writes excluded",
        "igev_timing": igev_timing,
        "las_timing": las_timing,
        "speed_comparison": {
            "las_speedup_vs_igev": float(igev_timing["mean_ms"] / las_timing["mean_ms"]),
            "las_latency_reduction_percent": float(
                (igev_timing["mean_ms"] - las_timing["mean_ms"]) / igev_timing["mean_ms"] * 100.0
            ),
        },
        "igev_ply_reference_metrics": igev_metrics,
        "las_ply_reference_metrics": las_metrics,
        "ply_reference_comparison": {
            "las_epe_reduction_px": float(
                igev_metrics["epe_px"]["macro_mean"] - las_metrics["epe_px"]["macro_mean"]
            ),
            "las_epe_reduction_percent": float(
                (igev_metrics["epe_px"]["macro_mean"] - las_metrics["epe_px"]["macro_mean"])
                / igev_metrics["epe_px"]["macro_mean"]
                * 100.0
            ),
        },
        "total_seconds": time.perf_counter() - started_all,
        "reference_note": "PLY projection is sparse reference consistency, not dense or human ground truth.",
        "metrics_file": str(metrics_path),
        "overview_image": str(overview_path) if overview_path is not None else None,
    }
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "metrics/summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
