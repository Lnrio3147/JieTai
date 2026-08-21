#!/usr/bin/env python3
"""Add previous/current disparity comparison images to extra-scene inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.data import read_rgb
from training.visualization import save_algorithm_comparison_vis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare saved LiteAnyStereo predictions with historical disparity/point clouds."
    )
    parser.add_argument("--tradition_root", default="../tradition_stereo")
    parser.add_argument("--inference_dir", default="./runs/inference/tradition_extra/official")
    parser.add_argument("--max_disp", type=float, default=192.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_q(path: Path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(path)
    q = storage.getNode("Q").mat()
    storage.release()
    if q is None or q.shape != (4, 4):
        raise ValueError(f"Invalid Q matrix in {path}")
    return q.astype(np.float64)


def ply_to_disparity(path: Path, q: np.ndarray, shape):
    # PCL files used here have a fixed binary vertex layout: float XYZ + uchar RGB.
    # Import Open3D lazily because only the luowen group needs point-cloud conversion.
    import open3d as o3d

    points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if points.size == 0:
        raise ValueError(f"No points in {path}")
    finite = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    x, y, z = points[finite].T
    focal = float(q[2, 3])
    cx, cy = -float(q[0, 3]), -float(q[1, 3])
    inverse_baseline = float(q[3, 2])
    u = np.rint(x * focal / z + cx).astype(np.int32)
    v = np.rint(y * focal / z + cy).astype(np.int32)
    disparity_values = focal / (z * inverse_baseline)
    height, width = shape
    valid = (
        (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
        & np.isfinite(disparity_values)
        & (disparity_values > 0.0)
    )
    u, v, disparity_values = u[valid], v[valid], disparity_values[valid]
    # Write far points first so nearer/larger disparity wins on collisions.
    order = np.argsort(disparity_values)
    disparity = np.zeros(shape, dtype=np.float32)
    disparity[v[order], u[order]] = disparity_values[order]
    return disparity


def find_previous_result(row, tradition_root: Path, shape, luowen_q):
    group, scene = row["group"], row["scene"]
    if group == "jxp":
        path = tradition_root / "datasets/JXP" / scene / f"{scene}.npy"
        result_kind = "historical_npy"
    elif group in {"gongjian_test", "other_test"}:
        path = tradition_root / "datasets" / group / scene / f"{scene}_disp.npy"
        result_kind = "historical_npy"
    elif group == "luowen":
        number = scene.rsplit("-", 1)[-1]
        path = tradition_root / "datasets/luowen" / scene / f"{number}_old.ply"
        result_kind = "historical_point_cloud"
    else:
        return None, None, "not_available"

    if not path.is_file():
        return None, str(path), "not_available"
    if path.suffix.lower() == ".npy":
        disparity = np.load(path).astype(np.float32)
    else:
        disparity = ply_to_disparity(path, luowen_q, shape)
        # The saved luowen *_old.ply point clouds use the opposite image
        # orientation from the rectified PNG inputs.  Align them before the
        # pixel-wise visualization; otherwise the old result is upside down.
        disparity = np.rot90(disparity, 2).copy()
        result_kind = "historical_point_cloud_rot180"
    if disparity.shape != shape:
        return None, str(path), f"shape_mismatch:{disparity.shape}"
    return disparity, str(path), result_kind


def main():
    args = parse_args()
    if args.max_disp <= 0:
        raise ValueError("--max_disp must be positive")
    tradition_root = Path(args.tradition_root).expanduser().resolve()
    inference_dir = Path(args.inference_dir).expanduser().resolve()
    manifest_path = inference_dir / "manifest.csv"
    output_manifest = inference_dir / "traditional_comparison_manifest.csv"
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"{output_manifest} exists; pass --overwrite")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    luowen_q = load_q(tradition_root / "config/stereo_luowen.yml")

    output_rows = []
    generated = 0
    for index, row in enumerate(rows, start=1):
        shape = (int(row["height"]), int(row["width"]))
        prediction_path = Path(row["output_dir"]) / "disp.npy"
        prediction = np.load(prediction_path).astype(np.float32)
        traditional, source_path, status = find_previous_result(
            row,
            tradition_root,
            shape,
            luowen_q,
        )
        output_path = Path(row["output_dir"]) / "traditional_comparison.png"
        if traditional is not None:
            left = read_rgb(Path(row["left"]))
            save_algorithm_comparison_vis(
                output_path,
                left=left,
                traditional=traditional,
                prediction=prediction,
                disparity_max=args.max_disp,
            )
            np.save(Path(row["output_dir"]) / "traditional_disp.npy", traditional)
            valid_ratio = float(
                (np.isfinite(traditional) & (traditional > 0.0)).mean() * 100.0
            )
            difference_mask = (
                np.isfinite(traditional)
                & (traditional > 0.0)
                & np.isfinite(prediction)
            )
            mean_absolute_difference = (
                float(np.abs(prediction[difference_mask] - traditional[difference_mask]).mean())
                if difference_mask.any()
                else None
            )
            generated += 1
        else:
            valid_ratio = None
            mean_absolute_difference = None
        output_rows.append(
            {
                "group": row["group"],
                "scene": row["scene"],
                "geometry_status": row["geometry_status"],
                "comparison_status": status,
                "traditional_source": source_path,
                "traditional_valid_ratio": valid_ratio,
                "mean_absolute_difference_not_error": mean_absolute_difference,
                "output": str(output_path) if traditional is not None else "",
            }
        )
        print(f"[{index:03d}/{len(rows):03d}] {row['group']}/{row['scene']}: {status}")

    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    status_counts = {}
    group_counts = {}
    for row in output_rows:
        status_counts[row["comparison_status"]] = status_counts.get(row["comparison_status"], 0) + 1
        if row["output"]:
            group_counts[row["group"]] = group_counts.get(row["group"], 0) + 1
    summary = {
        "input_scene_count": len(rows),
        "comparison_count": generated,
        "group_counts": group_counts,
        "status_counts": status_counts,
        "comparison_file": "traditional_comparison.png",
        "traditional_disparity_file": "traditional_disp.npy",
        "difference_note": "Absolute difference compares the two algorithms; without reference GT it is not prediction error.",
    }
    (inference_dir / "traditional_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
