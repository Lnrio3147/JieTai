#!/usr/bin/env python3
"""Export full-resolution LiteAnyStereo disparities for every listed pair."""

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
    pad_to_32,
    read_bgr,
    sha256_file,
    summarize_ms,
    unpad_disparity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--core", choices=["auto", "0", "1", "2", "0_1_2"], default="0_1_2"
    )
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


def prepare_las_inputs(
    left_bgr: np.ndarray, right_bgr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    left_rgb = np.ascontiguousarray(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB))
    right_rgb = np.ascontiguousarray(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB))
    left_padded, padding = pad_to_32(left_rgb)
    right_padded, right_padding = pad_to_32(right_rgb)
    if padding != right_padding:
        raise ValueError(f"Left/right padding mismatch: {padding} vs {right_padding}")
    return left_padded[None], right_padded[None], padding


def save_disparity(scene_dir: Path, disparity: np.ndarray) -> dict:
    scene_dir.mkdir(parents=True, exist_ok=False)
    npy_path = scene_dir / "disparity.npy"
    np.save(npy_path, disparity.astype(np.float32, copy=False), allow_pickle=False)

    fixed_u16 = np.rint(np.clip(disparity, 0.0, 255.996) * 256.0).astype(np.uint16)
    fixed_path = scene_dir / "disparity_x256.png"
    if not cv2.imwrite(str(fixed_path), fixed_u16):
        raise RuntimeError(f"Failed to write {fixed_path}")

    preview_u8 = cv2.normalize(
        disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    preview = cv2.applyColorMap(preview_u8, cv2.COLORMAP_TURBO)
    preview_path = scene_dir / "disparity_preview.png"
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Failed to write {preview_path}")

    return {
        "shape": list(disparity.shape),
        "finite": bool(np.isfinite(disparity).all()),
        "min_px": float(np.nanmin(disparity)),
        "max_px": float(np.nanmax(disparity)),
        "mean_px": float(np.nanmean(disparity)),
        "disparity_npy_sha256": sha256_file(npy_path),
        "disparity_x256_png_sha256": sha256_file(fixed_path),
        "preview_png_sha256": sha256_file(preview_path),
    }


def main() -> None:
    args = parse_args()
    model = args.model.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model.is_file() or not pairs_file.is_file():
        raise FileNotFoundError(model if not model.is_file() else pairs_file)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(pairs_file)

    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError("Run this script on the RK3588 with Lite2 installed") from exc

    expected_shape = (args.source_height, args.source_width, 3)
    records = []
    inference_times = []
    runner = LiteRunner(RKNNLite, model, args.core)
    try:
        for index, (left_path, right_path) in enumerate(pairs):
            left_bgr = read_bgr(left_path)
            right_bgr = read_bgr(right_path)
            if left_bgr.shape != expected_shape or right_bgr.shape != expected_shape:
                raise ValueError(
                    f"Unexpected source shape for {left_path}: "
                    f"{left_bgr.shape}, {right_bgr.shape}"
                )
            left_input, right_input, padding = prepare_las_inputs(
                left_bgr, right_bgr
            )
            start = time.perf_counter_ns()
            output = runner.infer([left_input, right_input])[0]
            inference_ms = (time.perf_counter_ns() - start) / 1e6
            disparity = unpad_disparity(output, padding)
            if not np.isfinite(disparity).all():
                raise FloatingPointError(f"Non-finite disparity for {left_path}")

            scene = left_path.parent.name
            save_start = time.perf_counter_ns()
            stats = save_disparity(output_dir / scene, disparity)
            save_ms = (time.perf_counter_ns() - save_start) / 1e6
            inference_times.append(inference_ms)
            records.append(
                {
                    "index": index,
                    "scene": scene,
                    "left": str(left_path),
                    "right": str(right_path),
                    "inference_ms": inference_ms,
                    "save_ms": save_ms,
                    **stats,
                }
            )
            print(
                f"[{index + 1:02d}/{len(pairs):02d}] {scene}: "
                f"infer={inference_ms:.2f} ms save={save_ms:.2f} ms"
            )
    finally:
        runner.release()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model),
        "model_sha256": sha256_file(model),
        "core": args.core,
        "pair_count": len(pairs),
        "timing_scope": "RKNNLite LAS call only; image load, preprocessing, and disk output excluded",
        "inference": summarize_ms(inference_times),
        "formats": {
            "disparity.npy": "float32 pixels; authoritative numeric output",
            "disparity_x256.png": "uint16 fixed scale; disparity_px = value / 256",
            "disparity_preview.png": "per-scene normalized TURBO preview; visualization only",
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
    print(json.dumps(report["inference"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
