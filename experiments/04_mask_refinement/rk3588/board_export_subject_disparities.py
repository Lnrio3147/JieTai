#!/usr/bin/env python3
"""Export the final Experiment 4 subject disparity for every listed pair."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from board_benchmark import (
    LiteRunner,
    foreground_probability,
    read_bgr,
    sha256_file,
    summarize_ms,
)
from postprocess import TRADITION_CROP, crop_array, refine_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bisenet-model", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--full-disparity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--core", choices=["auto", "0", "1", "2", "0_1_2"], default="0_1_2"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--source-height", type=int, default=1280)
    parser.add_argument("--source-width", type=int, default=720)
    return parser.parse_args()


def read_pairs(path: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = shlex.split(line)
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected left and right paths")
        resolved = []
        for field in fields:
            item = Path(field).expanduser()
            if not item.is_absolute():
                item = path.parent / item
            item = item.resolve()
            if not item.is_file():
                raise FileNotFoundError(item)
            resolved.append(item)
        pairs.append((resolved[0], resolved[1]))
    if not pairs:
        raise ValueError(f"No pairs in {path}")
    return pairs


def prepare_bisenet_input(left_bgr: np.ndarray) -> np.ndarray:
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(left_rgb, (288, 512), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized[None])


def save_subject_disparity(
    scene_dir: Path,
    subject_disparity: np.ndarray,
    subject_mask: np.ndarray,
) -> dict:
    scene_dir.mkdir(parents=True, exist_ok=False)
    valid = np.asarray(subject_mask, dtype=bool) & np.isfinite(subject_disparity)
    if not valid.any():
        raise ValueError(f"Subject mask is empty for {scene_dir.name}")

    npy_path = scene_dir / "subject_disparity.npy"
    np.save(
        npy_path,
        subject_disparity.astype(np.float32, copy=False),
        allow_pickle=False,
    )

    fixed_u16 = np.zeros(subject_disparity.shape, dtype=np.uint16)
    fixed_u16[valid] = np.rint(
        np.clip(subject_disparity[valid], 0.0, 255.996) * 256.0
    ).astype(np.uint16)
    fixed_path = scene_dir / "subject_disparity_x256.png"
    if not cv2.imwrite(str(fixed_path), fixed_u16):
        raise RuntimeError(f"Failed to write {fixed_path}")

    valid_values = subject_disparity[valid]
    minimum = float(valid_values.min())
    maximum = float(valid_values.max())
    preview_u8 = np.zeros(subject_disparity.shape, dtype=np.uint8)
    if maximum > minimum:
        preview_u8[valid] = np.rint(
            (valid_values - minimum) * (255.0 / (maximum - minimum))
        ).astype(np.uint8)
    preview = cv2.applyColorMap(preview_u8, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    preview_path = scene_dir / "subject_disparity_preview.png"
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Failed to write {preview_path}")

    mask_path = scene_dir / "subject_mask.png"
    if not cv2.imwrite(str(mask_path), valid.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write {mask_path}")

    return {
        "shape": list(subject_disparity.shape),
        "foreground_pixels": int(valid.sum()),
        "foreground_fraction": float(valid.mean()),
        "finite_foreground": bool(np.isfinite(subject_disparity[valid]).all()),
        "nan_background": bool(np.isnan(subject_disparity[~valid]).all()),
        "min_px": minimum,
        "max_px": maximum,
        "mean_px": float(valid_values.mean()),
        "subject_disparity_npy_sha256": sha256_file(npy_path),
        "subject_disparity_x256_png_sha256": sha256_file(fixed_path),
        "subject_disparity_preview_png_sha256": sha256_file(preview_path),
        "subject_mask_png_sha256": sha256_file(mask_path),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    bisenet_model = args.bisenet_model.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    full_disparity_dir = args.full_disparity_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (bisenet_model, pairs_file, full_disparity_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(pairs_file)

    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError("Run this script on the RK3588 with Lite2 installed") from exc

    expected_image_shape = (args.source_height, args.source_width, 3)
    expected_disparity_shape = expected_image_shape[:2]
    records = []
    bisenet_times = []
    postprocess_times = []
    save_times = []
    runner = LiteRunner(RKNNLite, bisenet_model, args.core)
    try:
        for index, (left_path, right_path) in enumerate(pairs):
            left_bgr = read_bgr(left_path)
            if left_bgr.shape != expected_image_shape:
                raise ValueError(
                    f"Unexpected source shape for {left_path}: {left_bgr.shape}"
                )
            scene = left_path.parent.name
            disparity_path = full_disparity_dir / scene / "disparity.npy"
            if not disparity_path.is_file():
                raise FileNotFoundError(disparity_path)
            disparity = np.load(disparity_path, allow_pickle=False)
            if (
                disparity.shape != expected_disparity_shape
                or disparity.dtype != np.float32
                or not np.isfinite(disparity).all()
            ):
                raise ValueError(f"Invalid full disparity: {disparity_path}")

            bisenet_input = prepare_bisenet_input(left_bgr)
            start = time.perf_counter_ns()
            probability_output = runner.infer([bisenet_input])[0]
            bisenet_ms = (time.perf_counter_ns() - start) / 1e6
            probability = foreground_probability(probability_output)

            postprocess_start = time.perf_counter_ns()
            disparity_crop = crop_array(disparity)
            _, refined_mask, refinement = refine_mask(
                probability,
                left_bgr.shape[:2],
                disparity_crop,
                threshold=args.threshold,
            )
            subject_mask = crop_array(refined_mask)
            subject_disparity = np.where(
                subject_mask, disparity_crop, np.nan
            ).astype(np.float32)
            postprocess_ms = (time.perf_counter_ns() - postprocess_start) / 1e6

            save_start = time.perf_counter_ns()
            stats = save_subject_disparity(
                output_dir / scene, subject_disparity, subject_mask
            )
            save_ms = (time.perf_counter_ns() - save_start) / 1e6
            bisenet_times.append(bisenet_ms)
            postprocess_times.append(postprocess_ms)
            save_times.append(save_ms)
            records.append(
                {
                    "index": index,
                    "scene": scene,
                    "left": str(left_path),
                    "right": str(right_path),
                    "input_disparity_sha256": sha256_file(disparity_path),
                    "bisenet_ms": bisenet_ms,
                    "postprocess_ms": postprocess_ms,
                    "save_ms": save_ms,
                    "filled_hole_count": refinement["filled_hole_count"],
                    "refined_foreground_pixels_full": refinement[
                        "refined_foreground_pixels"
                    ],
                    **stats,
                }
            )
            print(
                f"[{index + 1:02d}/{len(pairs):02d}] {scene}: "
                f"BiSeNet={bisenet_ms:.2f} ms post={postprocess_ms:.2f} ms "
                f"subject={stats['foreground_pixels']} px"
            )
    finally:
        runner.release()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bisenet_model": str(bisenet_model),
        "bisenet_model_sha256": sha256_file(bisenet_model),
        "input_full_disparity_dir": str(full_disparity_dir),
        "core": args.core,
        "threshold": args.threshold,
        "crop_y0_y1_x0_x1": list(TRADITION_CROP),
        "pair_count": len(pairs),
        "timing_scope": (
            "Uses the already exported RKNN LAS disparity; times BiSeNet, "
            "Experiment 4 post-processing, and file saving separately"
        ),
        "timing": {
            "bisenet": summarize_ms(bisenet_times),
            "postprocess": summarize_ms(postprocess_times),
            "save": summarize_ms(save_times),
        },
        "formats": {
            "subject_disparity.npy": (
                "float32 pixel disparity in the 818x512 Experiment 4 ROI; "
                "background is NaN"
            ),
            "subject_disparity_x256.png": (
                "uint16 fixed scale; subject disparity_px = value / 256; "
                "background is 0 and subject_mask.png disambiguates it"
            ),
            "subject_disparity_preview.png": (
                "per-scene normalized TURBO subject preview; black background; "
                "visualization only"
            ),
            "subject_mask.png": "uint8 ROI mask; subject=255, background=0",
        },
        "records": records,
    }
    (output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "export_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report["timing"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
