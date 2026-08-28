#!/usr/bin/env python3
"""Prepare exact-layout RKNN INT8 calibration inputs for both models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "training" / "JMP-LF6020-ETH3D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to <dataset-root>/manifest.csv.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("build/calibration"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--source-height", type=int, default=1280)
    parser.add_argument("--source-width", type=int, default=720)
    parser.add_argument("--bisenet-height", type=int, default=512)
    parser.add_argument("--bisenet-width", type=int, default=288)
    parser.add_argument(
        "--las-layout",
        choices=["nchw", "nhwc"],
        default="nchw",
        help="Layout of LAS .npy calibration tensors; must match the LAS ONNX.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def pad_to_32(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    pad_height = (-height) % 32
    pad_width = (-width) % 32
    return cv2.copyMakeBorder(
        rgb,
        pad_height // 2,
        pad_height - pad_height // 2,
        pad_width // 2,
        pad_width - pad_width // 2,
        cv2.BORDER_REPLICATE,
    )


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def read_pairs(root: Path, manifest: Path, split: str) -> list[tuple[Path, Path]]:
    pairs = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"left", "right", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Manifest {manifest} must contain columns {sorted(required)}"
            )
        for row in reader:
            if row["split"].strip().lower() != split.strip().lower():
                continue
            left = (root / row["left"]).resolve()
            right = (root / row["right"]).resolve()
            if not left.is_file() or not right.is_file():
                raise FileNotFoundError(f"Missing calibration pair: {left}, {right}")
            pairs.append((left, right))
    return sorted(pairs)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else root / "manifest.csv"
    )
    output_dir = args.output_dir.expanduser().resolve()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Calibration output is not empty: {output_dir}; pass --force"
        )

    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    pairs = read_pairs(root, manifest, args.split)
    if not pairs:
        raise ValueError(f"No {args.split!r} pairs found in {manifest}")
    selected_left = evenly_spaced([pair[0] for pair in pairs], args.samples)
    by_left = {left: right for left, right in pairs}

    las_dir = output_dir / "las"
    bisenet_dir = output_dir / "bisenet"
    las_dir.mkdir(parents=True, exist_ok=True)
    bisenet_dir.mkdir(parents=True, exist_ok=True)
    las_lines = []
    bisenet_lines = []
    selected = []
    for index, left_path in enumerate(selected_left):
        right_path = by_left[left_path]
        left_rgb = read_rgb(left_path)
        right_rgb = read_rgb(right_path)
        if left_rgb.shape != right_rgb.shape:
            raise ValueError(
                f"Left/right shape mismatch: {left_path} {left_rgb.shape}, "
                f"{right_path} {right_rgb.shape}"
            )
        if left_rgb.shape[:2] != (args.source_height, args.source_width):
            raise ValueError(
                f"Expected {(args.source_height, args.source_width)}, got "
                f"{left_rgb.shape[:2]} for {left_path}"
            )

        left_las = pad_to_32(left_rgb)[None]
        right_las = pad_to_32(right_rgb)[None]
        if args.las_layout == "nchw":
            left_las = left_las.transpose(0, 3, 1, 2)
            right_las = right_las.transpose(0, 3, 1, 2)
        left_las = np.ascontiguousarray(left_las)
        right_las = np.ascontiguousarray(right_las)
        bisenet = cv2.resize(
            left_rgb,
            (args.bisenet_width, args.bisenet_height),
            interpolation=cv2.INTER_AREA,
        )[None]
        bisenet = np.ascontiguousarray(bisenet)

        stem = f"{index:03d}_{left_path.parent.name}"
        left_npy = las_dir / f"{stem}_left.npy"
        right_npy = las_dir / f"{stem}_right.npy"
        bisenet_npy = bisenet_dir / f"{stem}.npy"
        np.save(left_npy, left_las, allow_pickle=False)
        np.save(right_npy, right_las, allow_pickle=False)
        np.save(bisenet_npy, bisenet, allow_pickle=False)
        las_lines.append(f"{left_npy} {right_npy}")
        bisenet_lines.append(str(bisenet_npy))
        selected.append(
            {
                "name": left_path.parent.name,
                "left": str(left_path),
                "right": str(right_path),
            }
        )

    las_list = output_dir / "dataset_las.txt"
    bisenet_list = output_dir / "dataset_bisenet.txt"
    las_list.write_text("\n".join(las_lines) + "\n", encoding="utf-8")
    bisenet_list.write_text("\n".join(bisenet_lines) + "\n", encoding="utf-8")
    metadata = {
        "dataset_root": str(root),
        "manifest": str(manifest),
        "split": args.split,
        "sample_count": len(selected),
        "selection": "deterministic evenly spaced over sorted scene paths",
        "las": {
            "dataset_file": str(las_list),
            "layout": f"two uint8 {args.las_layout.upper()} .npy inputs per line",
            "shape": list(left_las.shape),
            "preprocess": "BGR->RGB, symmetric replicate padding to /32",
        },
        "bisenet": {
            "dataset_file": str(bisenet_list),
            "layout": "one uint8 NHWC .npy input per line",
            "shape": list(bisenet.shape),
            "preprocess": "BGR->RGB, resize with INTER_AREA; normalization in RKNN config",
        },
        "selected": selected,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
