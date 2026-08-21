#!/usr/bin/env python3
"""Apply BiSeNetV2 workpiece masks to LiteAnyStereo disparity predictions.

The stereo model always receives the original rectified RGB pair.  The left
foreground mask is resized with nearest-neighbour interpolation and applied to
the disparity only after stereo inference.  This avoids creating artificial
black boundaries in the images used for correspondence matching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRADITION_CROP = (234, 1052, 126, 638)
METRICS = ("epe", "d1", "bad1", "bad2", "bad3")
FIXED_EXCLUDED_SCENES = {
    "202506281607-0012",
    "202506281609-0019",
    "202506281609-0020",
    "202506281619-0053",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="./data/datasets/JMP-LF6020-ETH3D/manifest.csv",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene-prefix", default="fdjyp_3_")
    parser.add_argument("--mask-dir", required=True, help="BiSeNetV2 0/255 PNG masks")
    parser.add_argument("--bisenet-model", default=None, help="PB path recorded for provenance")
    parser.add_argument(
        "--tradition-reference-root",
        default="../tradition_stereo/datasets/FDJYP-3",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stereo-mode",
        choices=["saved", "live"],
        default="saved",
        help="reuse a verified LAS disparity directory or run the LAS network now",
    )
    parser.add_argument(
        "--las-output-root",
        default="./runs/evaluation/jmp_unified_rerun_73/liteanystereo",
    )
    parser.add_argument("--version", default="las1")
    parser.add_argument("--model-size", default=None)
    parser.add_argument("--restore-ckpt", default="./checkpoints/LiteAnyStereo.pth")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--contact-sheet-samples", type=int, default=18)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(base, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def scene_id(name):
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot map dataset name to FDJYP-3 scene: {name!r}")
    return f"{parts[-2]}-{parts[-1]}"


def read_manifest(manifest_path, split, prefix):
    rows = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"left", "right", "split", "name"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for row in reader:
            if row["split"].strip().lower() != split.lower():
                continue
            if not row["name"].startswith(prefix):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No manifest rows match split={split!r}, prefix={prefix!r}")
    return sorted(rows, key=lambda row: row["name"])


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_metrics(prediction, reference, extra_mask=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction/reference mismatch: {prediction.shape} vs {reference.shape}")
    valid = np.isfinite(reference) & (reference > 0.0) & np.isfinite(prediction)
    if extra_mask is not None:
        extra_mask = np.asarray(extra_mask, dtype=bool)
        if extra_mask.shape != valid.shape:
            raise ValueError(f"Mask/reference mismatch: {extra_mask.shape} vs {valid.shape}")
        valid &= extra_mask
    count = int(valid.sum())
    if count == 0:
        return {key: None for key in METRICS} | {"valid_pixels": 0}
    error = np.abs(prediction - reference)
    relative = error / np.maximum(np.abs(reference), 1e-12)
    return {
        "epe": float(error[valid].mean()),
        "d1": float(100.0 * ((error > 3.0) & (relative > 0.05))[valid].mean()),
        "bad1": float(100.0 * (error[valid] > 1.0).mean()),
        "bad2": float(100.0 * (error[valid] > 2.0).mean()),
        "bad3": float(100.0 * (error[valid] > 3.0).mean()),
        "valid_pixels": count,
    }


def aggregate(rows, prefix):
    available = [row for row in rows if int(row[f"{prefix}_valid_pixels"]) > 0]
    return {
        "scene_count": len(available),
        "valid_pixels": int(sum(int(row[f"{prefix}_valid_pixels"]) for row in available)),
        "aggregation": "scene_macro",
        **{
            key: float(np.mean([float(row[f"{prefix}_{key}"]) for row in available]))
            for key in METRICS
        },
    }


def colorize(values, valid, maximum):
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    normalized = np.clip(values / float(maximum), 0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def label_panel(panel, text):
    result = panel.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 25), (0, 0, 0), thickness=-1)
    cv2.putText(
        result,
        text,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def save_comparison(path, left_rgb, prediction, reference, subject_mask, max_disp):
    reference_valid = np.isfinite(reference) & (reference > 0.0)
    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
    overlay = left_bgr.copy()
    overlay[subject_mask] = (
        0.45 * overlay[subject_mask]
        + 0.55 * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    raw_vis = colorize(prediction, np.isfinite(prediction), max_disp)
    subject_vis = colorize(prediction, subject_mask, max_disp)
    reference_vis = colorize(reference, reference_valid, max_disp)
    error = np.abs(prediction - reference)
    error_vis = colorize(error, reference_valid & subject_mask, 20.0)
    panels = [
        label_panel(left_bgr, "rectified left RGB"),
        label_panel(overlay, "BiSeNetV2 foreground overlay"),
        label_panel(raw_vis, f"LiteAnyStereo disparity 0-{max_disp}px"),
        label_panel(subject_vis, "subject disparity after post-mask"),
        label_panel(reference_vis, f"reference disparity 0-{max_disp}px"),
        label_panel(error_vis, "subject absolute error 0-20px"),
    ]
    height = 409
    width = 256
    panels = [cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA) for panel in panels]
    montage = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), montage, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Failed to write {path}")


def save_contact_sheet(paths, output_path, sample_count):
    count = min(sample_count, len(paths))
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int32)
    tiles = []
    for index in indices:
        image = cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(paths[int(index)])
        tiles.append(cv2.resize(image, (384, 409), interpolation=cv2.INTER_AREA))
    columns = 3
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    if not cv2.imwrite(str(output_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Failed to write {output_path}")


def load_live_model(args):
    import torch

    from core.models import (
        build_model,
        load_model_weights,
        normalize_model_size,
        normalize_version,
        require_checkpoint,
    )

    version = normalize_version(args.version)
    model_size = normalize_model_size(version, args.model_size)
    checkpoint_path = Path(args.restore_ckpt).expanduser().resolve()
    require_checkpoint(checkpoint_path)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = build_model(
        version,
        fnet_pretrained=False,
        model_size=model_size,
        max_disp=args.max_disp,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_weights(model, checkpoint, strict=True)
    return model.to(device).eval(), device, checkpoint_path


def infer_live(model, device, left_rgb, right_rgb, max_disp):
    import torch

    from core.utils.utils import InputPadder

    left = torch.as_tensor(left_rgb, device=device).float()[None].permute(0, 3, 1, 2)
    right = torch.as_tensor(right_rgb, device=device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        prediction = model(left, right, max_disp=max_disp, test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    prediction = padder.unpad(prediction.float()).cpu().numpy().reshape(left_rgb.shape[:2])
    return prediction.astype(np.float32), elapsed


def write_result_readme(path, summary):
    baseline = summary["metrics"]["all_reference_valid"]
    subject = summary["metrics"]["subject_reference_valid"]
    background = summary["metrics"]["background_reference_valid"]
    fixed_subject = summary["fixed_69_metrics"]["subject_reference_valid"]
    text = f"""# BiSeNetV2 + LiteAnyStereo FDJYP-3 试验结果

- 场景数：{summary['scene_count']}
- 接入方式：原始校正左右图进入 LiteAnyStereo；BiSeNetV2 左掩码只在视差输出端过滤。
- 主体掩码平均覆盖 ROI：{summary['coverage']['mean_roi_foreground_percent']:.2f}%
- 参考有效像素平均保留：{summary['coverage']['mean_reference_valid_retained_percent']:.2f}%
- 全参考有效区 EPE：{baseline['epe']:.4f} px
- 主体参考有效区 EPE：{subject['epe']:.4f} px
- 主体参考有效区 D1：{subject['d1']:.2f}%
- 背景参考有效区 EPE：{background['epe']:.4f} px
- 固定 69 场主体 EPE：{fixed_subject['epe']:.4f} px

注意：全区域和主体区域不是同一像素集合，不能把两者差值解释为“分割提高了视差网络精度”。后置掩码不改变主体像素的视差值，只删除非主体输出。可信的端到端结论仍需要 FDJYP-3 人工分割 GT。
"""
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    if not 0 <= args.mask_threshold <= 255:
        raise ValueError("mask-threshold must be in [0,255]")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_root = manifest_path.parent
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    reference_root = Path(args.tradition_reference_root).expanduser().resolve()
    las_output_root = Path(args.las_output_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "scenes").mkdir()

    source_rows = read_manifest(manifest_path, args.split, args.scene_prefix)
    if args.limit is not None:
        source_rows = source_rows[:args.limit]

    model = device = checkpoint_path = None
    if args.stereo_mode == "live":
        model, device, checkpoint_path = load_live_model(args)

    y0, y1, x0, x1 = TRADITION_CROP
    metric_rows = []
    comparison_paths = []
    inference_times = []

    for source_row in source_rows:
        name = source_row["name"]
        scene = scene_id(name)
        left_path = resolve(manifest_root, source_row["left"])
        right_path = resolve(manifest_root, source_row["right"])
        left_rgb = read_rgb(left_path)
        right_rgb = read_rgb(right_path)
        if left_rgb.shape != right_rgb.shape:
            raise ValueError(f"Stereo shape mismatch for {name}")

        mask_path = mask_dir / f"{name}.png"
        mask_small = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_small is None:
            raise FileNotFoundError(f"BiSeNetV2 mask not found: {mask_path}")
        full_height, full_width = left_rgb.shape[:2]
        subject_full = cv2.resize(
            mask_small,
            (full_width, full_height),
            interpolation=cv2.INTER_NEAREST,
        ) > args.mask_threshold
        subject_roi = subject_full[y0:y1, x0:x1]
        left_roi = left_rgb[y0:y1, x0:x1]

        if args.stereo_mode == "live":
            prediction_full, elapsed = infer_live(
                model, device, left_rgb, right_rgb, args.max_disp
            )
            inference_times.append(elapsed)
            prediction = prediction_full[y0:y1, x0:x1]
        else:
            saved_path = las_output_root / scene / "disp.npy"
            prediction = np.load(saved_path).astype(np.float32)
            if prediction.shape == left_rgb.shape[:2]:
                prediction = prediction[y0:y1, x0:x1]

        reference_path = reference_root / scene / "disp_cropped.npy"
        reference = np.load(reference_path).astype(np.float32)
        if prediction.shape != reference.shape or subject_roi.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch for {scene}: prediction={prediction.shape}, "
                f"mask={subject_roi.shape}, reference={reference.shape}"
            )

        reference_valid = np.isfinite(reference) & (reference > 0.0)
        baseline_metrics = compute_metrics(prediction, reference)
        subject_metrics = compute_metrics(prediction, reference, subject_roi)
        background_metrics = compute_metrics(prediction, reference, ~subject_roi)
        masked_disparity = np.where(subject_roi, prediction, np.nan).astype(np.float32)

        scene_dir = output_dir / "scenes" / scene
        scene_dir.mkdir()
        np.save(scene_dir / "disp_subject.npy", masked_disparity)
        cv2.imwrite(str(scene_dir / "foreground_mask.png"), subject_roi.astype(np.uint8) * 255)
        comparison_path = scene_dir / "comparison.jpg"
        save_comparison(
            comparison_path,
            left_roi,
            prediction,
            reference,
            subject_roi,
            args.max_disp,
        )
        comparison_paths.append(comparison_path)

        row = {
            "name": name,
            "scene": scene,
            "left": str(left_path),
            "right": str(right_path),
            "source_mask": str(mask_path),
            "reference": str(reference_path),
            "roi_foreground_percent": float(100.0 * subject_roi.mean()),
            "reference_valid_retained_percent": float(
                100.0 * (reference_valid & subject_roi).sum() / max(reference_valid.sum(), 1)
            ),
            "post_mask_max_abs_change_inside_subject": 0.0,
        }
        for prefix, values in (
            ("all", baseline_metrics),
            ("subject", subject_metrics),
            ("background", background_metrics),
        ):
            for key, value in values.items():
                row[f"{prefix}_{key}"] = value
        metric_rows.append(row)

    metric_path = output_dir / "scene_metrics.csv"
    with metric_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    fixed_rows = [row for row in metric_rows if row["scene"] not in FIXED_EXCLUDED_SCENES]
    declared_checkpoint = Path(args.restore_ckpt).expanduser().resolve()
    if not declared_checkpoint.is_file():
        declared_checkpoint = None
    all_metrics = aggregate(metric_rows, "all")
    subject_metrics = aggregate(metric_rows, "subject")
    background_metrics = aggregate(metric_rows, "background")
    coverage_values = np.asarray(
        [row["reference_valid_retained_percent"] for row in metric_rows],
        dtype=np.float64,
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(metric_rows),
        "protocol": {
            "pipeline": "BiSeNetV2 left mask -> LiteAnyStereo on original rectified RGB -> post-mask disparity",
            "stereo_mode": args.stereo_mode,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "scene_prefix": args.scene_prefix,
            "mask_dir": str(mask_dir),
            "mask_resize": "nearest neighbour to full image, then fixed tradition ROI",
            "tradition_crop": list(TRADITION_CROP),
            "reference": str(reference_root / "<scene>/disp_cropped.npy"),
            "reference_valid": "finite and > 0",
            "metrics_aggregation": "scene macro average",
            "max_disp": args.max_disp,
        },
        "models": {
            "bisenet_pb": str(Path(args.bisenet_model).resolve()) if args.bisenet_model else None,
            "bisenet_pb_sha256": (
                sha256_file(Path(args.bisenet_model).resolve()) if args.bisenet_model else None
            ),
            "liteanystereo_checkpoint": (
                str(checkpoint_path or declared_checkpoint)
                if checkpoint_path or declared_checkpoint
                else None
            ),
            "liteanystereo_checkpoint_sha256": (
                sha256_file(checkpoint_path or declared_checkpoint)
                if checkpoint_path or declared_checkpoint
                else None
            ),
            "liteanystereo_checkpoint_role": (
                "loaded in this live run"
                if args.stereo_mode == "live"
                else "declared provenance of the verified saved disparity run"
            ),
            "saved_liteanystereo_output": (
                str(las_output_root) if args.stereo_mode == "saved" else None
            ),
        },
        "coverage": {
            "mean_roi_foreground_percent": float(
                np.mean([row["roi_foreground_percent"] for row in metric_rows])
            ),
            "median_roi_foreground_percent": float(
                np.median([row["roi_foreground_percent"] for row in metric_rows])
            ),
            "mean_reference_valid_retained_percent": float(
                np.mean([row["reference_valid_retained_percent"] for row in metric_rows])
            ),
            "median_reference_valid_retained_percent": float(
                np.median(coverage_values)
            ),
            "min_reference_valid_retained_percent": float(np.min(coverage_values)),
            "max_reference_valid_retained_percent": float(np.max(coverage_values)),
            "scenes_with_no_background_reference_pixels": int(
                sum(int(row["background_valid_pixels"]) == 0 for row in metric_rows)
            ),
        },
        "metrics": {
            "all_reference_valid": all_metrics,
            "subject_reference_valid": subject_metrics,
            "background_reference_valid": background_metrics,
        },
        "fixed_69_metrics": {
            "excluded_scenes": sorted(FIXED_EXCLUDED_SCENES),
            "all_reference_valid": aggregate(fixed_rows, "all"),
            "subject_reference_valid": aggregate(fixed_rows, "subject"),
            "background_reference_valid": aggregate(fixed_rows, "background"),
        },
        "selection_effect_not_model_improvement": {
            "subject_epe_relative_difference_vs_all_percent": float(
                100.0 * (1.0 - subject_metrics["epe"] / all_metrics["epe"])
            ),
            "subject_d1_relative_difference_vs_all_percent": float(
                100.0 * (1.0 - subject_metrics["d1"] / all_metrics["d1"])
            ),
            "post_mask_max_abs_change_inside_subject": float(
                max(row["post_mask_max_abs_change_inside_subject"] for row in metric_rows)
            ),
        },
        "inference_runtime_seconds": (
            {
                "count": len(inference_times),
                "mean": float(np.mean(inference_times)),
                "median": float(np.median(inference_times)),
            }
            if inference_times
            else None
        ),
        "interpretation": (
            "All-region and subject-region metrics use different pixel sets. "
            "Their difference is not an accuracy gain caused by masking. Post-masking "
            "preserves LiteAnyStereo values inside the subject exactly and removes output outside it."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_contact_sheet(
        comparison_paths,
        output_dir / "contact_sheet.jpg",
        args.contact_sheet_samples,
    )
    write_result_readme(output_dir / "README.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
