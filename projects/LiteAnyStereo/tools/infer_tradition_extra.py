#!/usr/bin/env python3
"""Run LiteAnyStereo on tradition_stereo pairs absent from JMP-LF6020-ETH3D."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.models import build_model, load_model_weights, normalize_model_size, normalize_version
from core.utils.utils import InputPadder
from training.checkpoint import safe_torch_load
from training.data import read_rgb
from training.visualization import save_inference_vis


@dataclass(frozen=True)
class ExtraSample:
    group: str
    name: str
    left: Path
    right: Path
    source_kind: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer tradition_stereo image pairs not present in JMP-LF6020-ETH3D."
    )
    parser.add_argument("--tradition_root", default="../tradition_stereo")
    parser.add_argument("--current_data_root", default="./data/datasets/JMP-LF6020-ETH3D")
    parser.add_argument("--restore_ckpt", default="./checkpoints/LiteAnyStereo.pth")
    parser.add_argument("--output_dir", default="./runs/inference/tradition_extra/official")
    parser.add_argument("--version", default="las1")
    parser.add_argument("--model_size", "--model-size", default=None)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _pair_directories(root: Path):
    if not root.is_dir():
        return []
    return sorted(
        scene
        for scene in root.iterdir()
        if scene.is_dir() and (scene / "im0.png").is_file() and (scene / "im1.png").is_file()
    )


def discover_samples(tradition_root: Path, current_data_root: Path):
    current_de0548 = {
        "-".join(scene.name.rsplit("_", 2)[-2:])
        for scene in current_data_root.glob("de0548_*")
        if scene.is_dir()
    }
    sources = [
        ("de0548_extra", tradition_root / "datasets/DE0548_right", "historical_pair"),
        ("jxp", tradition_root / "datasets/JXP", "historical_pair"),
        ("gongjian_test", tradition_root / "datasets/gongjian_test", "historical_pair"),
        ("other_test", tradition_root / "datasets/other_test", "historical_pair"),
        ("luowen", tradition_root / "rec_img_set/luowen_rectified_images", "rectified_archive"),
        ("dec_scale", tradition_root / "rec_img_set/rectified_images_刻度", "rectified_archive"),
        ("dec_general", tradition_root / "rec_img_set/rectified_images", "rectified_archive"),
    ]
    samples = []
    for group, root, source_kind in sources:
        for scene in _pair_directories(root):
            if group == "de0548_extra" and scene.name in current_de0548:
                continue
            samples.append(
                ExtraSample(
                    group=group,
                    name=scene.name,
                    left=scene / "im0.png",
                    right=scene / "im1.png",
                    source_kind=source_kind,
                )
            )
    names = [(sample.group, sample.name) for sample in samples]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate group/scene names in extra inference inputs")
    return samples


def epipolar_audit(left: np.ndarray, right: np.ndarray):
    gray_left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, 720.0 / max(gray_left.shape))
    if scale < 1.0:
        gray_left = cv2.resize(gray_left, None, fx=scale, fy=scale)
        gray_right = cv2.resize(gray_right, None, fx=scale, fy=scale)
    sift = cv2.SIFT_create(nfeatures=2500)
    left_keypoints, left_descriptors = sift.detectAndCompute(gray_left, None)
    right_keypoints, right_descriptors = sift.detectAndCompute(gray_right, None)
    good = []
    if left_descriptors is not None and right_descriptors is not None:
        for pair in cv2.BFMatcher().knnMatch(left_descriptors, right_descriptors, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
    residuals = np.asarray(
        [
            abs(
                left_keypoints[match.queryIdx].pt[1]
                - right_keypoints[match.trainIdx].pt[1]
            )
            / scale
            for match in good
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


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.max_disp <= 0:
        raise ValueError("--max_disp must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")

    tradition_root = Path(args.tradition_root).expanduser().resolve()
    current_data_root = Path(args.current_data_root).expanduser().resolve()
    checkpoint = Path(args.restore_ckpt).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite or choose another output")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    samples = discover_samples(tradition_root, current_data_root)
    if not samples:
        raise FileNotFoundError("No extra tradition_stereo image pairs found")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    version = normalize_version(args.version)
    model_size = normalize_model_size(version, args.model_size)
    model = build_model(
        version,
        fnet_pretrained=False,
        model_size=model_size,
        max_disp=args.max_disp,
    ).to(device)
    load_model_weights(
        model,
        safe_torch_load(checkpoint, map_location=device),
        strict=True,
    )
    model.eval()

    rows = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        left = read_rgb(sample.left)
        right = read_rgb(sample.right)
        if left.shape != right.shape:
            raise ValueError(f"Left/right mismatch for {sample.group}/{sample.name}")
        audit = epipolar_audit(left, right)
        left_tensor = torch.from_numpy(left.copy()).permute(2, 0, 1).float()[None].to(device)
        right_tensor = torch.from_numpy(right.copy()).permute(2, 0, 1).float()[None].to(device)
        padder = InputPadder(left_tensor.shape, divis_by=32)
        padded_left, padded_right = padder.pad(left_tensor, right_tensor)
        with torch.no_grad():
            prediction = model(
                padded_left,
                padded_right,
                max_disp=args.max_disp,
                test_mode=True,
            )
        prediction = padder.unpad(prediction.float())[0, 0].cpu().numpy()
        scene_output = output_dir / sample.group / sample.name
        scene_output.mkdir(parents=True, exist_ok=True)
        np.save(scene_output / "disp.npy", prediction.astype(np.float32))
        save_inference_vis(
            scene_output,
            left=left,
            right=right,
            prediction=prediction,
            disparity_max=args.max_disp,
        )
        row = {
            "group": sample.group,
            "scene": sample.name,
            "source_kind": sample.source_kind,
            "left": str(sample.left),
            "right": str(sample.right),
            "height": left.shape[0],
            "width": left.shape[1],
            **audit,
            "disp_min": float(np.nanmin(prediction)),
            "disp_median": float(np.nanmedian(prediction)),
            "disp_max": float(np.nanmax(prediction)),
            "output_dir": str(scene_output),
        }
        rows.append(row)
        print(
            f"[{index:03d}/{len(samples):03d}] {sample.group}/{sample.name} "
            f"geometry={audit['geometry_status']} median_dy={audit['median_vertical_residual_px']}"
        )

    write_csv(manifest_path, rows)
    group_counts = {}
    geometry_counts = {}
    for row in rows:
        group_counts[row["group"]] = group_counts.get(row["group"], 0) + 1
        status = row["geometry_status"]
        geometry_counts[status] = geometry_counts.get(status, 0) + 1
    summary = {
        "checkpoint": str(checkpoint),
        "current_data_root": str(current_data_root),
        "tradition_root": str(tradition_root),
        "sample_count": len(rows),
        "group_counts": group_counts,
        "geometry_counts": geometry_counts,
        "outputs_per_scene": ["disp.npy", "vis.png", "vis_fixed.png", "comparison.png"],
        "metrics_available": False,
        "metrics_note": "These extra scenes have no unified reference disparity; no EPE/D1/Bad metrics were calculated.",
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logging.info("Wrote %d extra-scene predictions to %s", len(rows), output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
