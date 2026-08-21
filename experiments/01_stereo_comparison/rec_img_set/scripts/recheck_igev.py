#!/usr/bin/env python3
"""Reproduce the 73 historical IGEV++ RT disparities and compare them exactly.

The historical output was produced from the rectified FDJYP-3 image pairs.
This script deliberately writes to this experiment's ``results`` directory and
never modifies ``results/igev_legacy_73``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

# Required by torch's deterministic CUDA matrix multiplication mode. Set it
# before torch initializes CUDA so repeated comparisons are reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[2]
IGEV_ROOT = ROOT / "projects/IGEV-plusplus"
if str(IGEV_ROOT) not in sys.path:
    sys.path.insert(0, str(IGEV_ROOT))

from core_rt.rt_igev_stereo import IGEVStereo, autocast  # noqa: E402
from core_rt.utils.utils import InputPadder  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "datasets/rec_img_set/FDJYP-3-rectified_images",
    )
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=EXPERIMENT_ROOT / "results/igev_legacy_73",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=IGEV_ROOT / "pretrained_models/igev_rt/sceneflow.pth",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "results/igev_recheck_73",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-disp", type=int, default=128)
    parser.add_argument("--valid-iters", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint: Path, device: torch.device, max_disp: int):
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


@torch.inference_mode()
def infer(model, model_args, left_bgr, right_bgr, device, valid_iters):
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    left = torch.from_numpy(left_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    right = torch.from_numpy(right_rgb.copy()).permute(2, 0, 1).float()[None].to(device)
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with autocast(enabled=False, dtype=torch.float32):
        prediction = model(left, right, iters=valid_iters, test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction = padder.unpad(prediction.float())[0, 0].cpu().numpy()
    return prediction.astype(np.float32), elapsed


def jet(disparity, maximum):
    normalized = np.clip(disparity / maximum, 0.0, 1.0)
    return cv2.applyColorMap(np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_JET)


def labelled(image, label):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(output, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def save_visuals(directory: Path, left, current, historical, max_disp, row):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "disp.npy", current)
    cv2.imwrite(str(directory / "vis.png"), jet(current, max_disp))

    absolute = np.abs(current - historical)
    difference = cv2.applyColorMap(
        np.round(np.clip(absolute / 3.0, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    panels = [
        labelled(left, "Rectified left"),
        labelled(jet(historical, max_disp), "Historical igev_output"),
        labelled(jet(current, max_disp), "Current IGEV++ RT"),
        labelled(difference, "Absolute difference (0-3 px)"),
    ]
    montage = np.hstack(panels)
    caption = (
        f"EPE {row['epe_px']:.6f} px | median {row['median_abs_px']:.6f} px | "
        f"<=0.1 px {row['within_0_1_pct']:.3f}% | exact {row['array_equal']}"
    )
    cv2.rectangle(montage, (0, montage.shape[0] - 44), (montage.shape[1], montage.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        montage,
        caption,
        (12, montage.shape[0] - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(directory / "comparison.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])


def compare(scene, current, historical, elapsed):
    if current.shape != historical.shape:
        raise ValueError(f"{scene}: shape mismatch {current.shape} != {historical.shape}")
    valid = np.isfinite(current) & np.isfinite(historical)
    absolute = np.abs(current[valid].astype(np.float64) - historical[valid].astype(np.float64))
    squared = absolute * absolute
    exact_pixels = current[valid] == historical[valid]
    return {
        "scene": scene,
        "shape": "x".join(map(str, current.shape)),
        "array_equal": bool(np.array_equal(current, historical)),
        "exact_pixel_pct": float(exact_pixels.mean() * 100.0),
        "epe_px": float(absolute.mean()),
        "rmse_px": float(np.sqrt(squared.mean())),
        "median_abs_px": float(np.median(absolute)),
        "p95_abs_px": float(np.percentile(absolute, 95)),
        "p99_abs_px": float(np.percentile(absolute, 99)),
        "max_abs_px": float(absolute.max()),
        "within_0_01_pct": float((absolute <= 0.01).mean() * 100.0),
        "within_0_1_pct": float((absolute <= 0.1).mean() * 100.0),
        "within_0_5_pct": float((absolute <= 0.5).mean() * 100.0),
        "within_1_pct": float((absolute <= 1.0).mean() * 100.0),
        "bad3_pct": float((absolute > 3.0).mean() * 100.0),
        "correlation": float(np.corrcoef(current[valid].ravel(), historical[valid].ravel())[0, 1]),
        "historical_min": float(historical[valid].min()),
        "historical_max": float(historical[valid].max()),
        "current_min": float(current[valid].min()),
        "current_max": float(current[valid].max()),
        "runtime_ms": float(elapsed * 1000.0),
    }


def aggregate(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "macro_mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_overview(output_root: Path, scenes):
    thumbs = []
    for scene in scenes:
        image = cv2.imread(str(output_root / "current_output" / scene / "comparison.jpg"))
        if image is None:
            continue
        width = 960
        height = round(image.shape[0] * width / image.shape[1])
        thumbs.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    rows = []
    for index in range(0, len(thumbs), 2):
        pair = thumbs[index:index + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows.append(np.hstack(pair))
    if rows:
        cv2.imwrite(str(output_root / "overview.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    input_root = args.input_root.expanduser().resolve()
    historical_root = args.historical_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    summary_path = output_root / "metrics/summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite to rerun")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in (input_root, historical_root, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    scenes = sorted(path.name for path in historical_root.iterdir() if path.is_dir())
    if len(scenes) != 73:
        raise RuntimeError(f"Expected 73 historical scenes, found {len(scenes)}")
    missing = [
        str(path)
        for scene in scenes
        for path in (input_root / scene / "im0.png", input_root / scene / "im1.png", historical_root / scene / "disp.npy")
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, model_args = build_model(checkpoint, device, args.max_disp)

    warm_left = cv2.imread(str(input_root / scenes[0] / "im0.png"), cv2.IMREAD_COLOR)
    warm_right = cv2.imread(str(input_root / scenes[0] / "im1.png"), cv2.IMREAD_COLOR)
    infer(model, model_args, warm_left, warm_right, device, args.valid_iters)

    rows = []
    started_all = time.perf_counter()
    for index, scene in enumerate(scenes, start=1):
        left = cv2.imread(str(input_root / scene / "im0.png"), cv2.IMREAD_COLOR)
        right = cv2.imread(str(input_root / scene / "im1.png"), cv2.IMREAD_COLOR)
        historical = np.load(historical_root / scene / "disp.npy").astype(np.float32, copy=False)
        current, elapsed = infer(model, model_args, left, right, device, args.valid_iters)
        row = compare(scene, current, historical, elapsed)
        rows.append(row)
        save_visuals(output_root / "current_output" / scene, left, current, historical, args.max_disp, row)
        print(
            f"[{index:02d}/73] {scene} EPE={row['epe_px']:.6f}px "
            f"<=0.1px={row['within_0_1_pct']:.3f}% exact={row['array_equal']}"
        )

    csv_path = output_root / "metrics/per_scene.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = (
        "exact_pixel_pct",
        "epe_px",
        "rmse_px",
        "median_abs_px",
        "p95_abs_px",
        "p99_abs_px",
        "max_abs_px",
        "within_0_01_pct",
        "within_0_1_pct",
        "within_0_5_pct",
        "within_1_pct",
        "bad3_pct",
        "correlation",
        "runtime_ms",
    )
    all_exact = all(row["array_equal"] for row in rows)
    summary = {
        "verdict": "exactly_identical" if all_exact else "not_bitwise_identical",
        "all_73_arrays_equal": all_exact,
        "scene_count": len(rows),
        "input_root": str(input_root),
        "historical_root": str(historical_root),
        "output_root": str(output_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "configuration": {
            "model": "IGEV++ RT core_rt",
            "max_disp": args.max_disp,
            "valid_iters": args.valid_iters,
            "mixed_precision": False,
            "deterministic_algorithms": True,
            "input_color": "RGB",
            "padding_divisor": 32,
        },
        "runtime_environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "metrics": {key: aggregate(rows, key) for key in metric_keys},
        "total_wall_seconds": float(time.perf_counter() - started_all),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_overview(output_root, scenes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
