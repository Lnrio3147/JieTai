#!/usr/bin/env python3
"""Refine BiSeNetV2 masks with a single-workpiece topology prior.

The refinement never uses disparity ground truth.  It keeps one foreground
component, removes isolated foreground islands, and selectively repairs
enclosed background holes whose LiteAnyStereo disparity is continuous with
the surrounding workpiece.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


TRADITION_CROP = (234, 1052, 126, 638)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="./data/datasets/JMP-LF6020-ETH3D/manifest.csv",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene-prefix", default="fdjyp_3_")
    parser.add_argument("--probability-dir", required=True)
    parser.add_argument(
        "--las-output-root",
        default="./runs/evaluation/jmp_unified_rerun_73/liteanystereo",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--closing-radius", type=int, default=3)
    parser.add_argument("--hole-ring-radius", type=int, default=7)
    parser.add_argument("--hole-absolute-tolerance", type=float, default=1.5)
    parser.add_argument("--hole-mad-scale", type=float, default=1.0)
    parser.add_argument("--max-fill-hole-fraction", type=float, default=0.025)
    parser.add_argument("--small-hole-area", type=int, default=1000)
    parser.add_argument("--contact-sheet-samples", type=int, default=24)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(base, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def scene_id(name):
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot map dataset name to scene: {name!r}")
    return f"{parts[-2]}-{parts[-1]}"


def read_manifest(path, split, prefix):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"].strip().lower() == split.lower() and row["name"].startswith(prefix):
                rows.append(row)
    if not rows:
        raise ValueError(f"No manifest rows for split={split!r}, prefix={prefix!r}")
    return sorted(rows, key=lambda row: row["name"])


def keep_largest_component(mask):
    """Return one 8-connected foreground component and removal statistics."""
    mask = np.asarray(mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask), max(count - 1, 0), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    result = labels == largest_label
    return result, count - 1, int(mask.sum() - result.sum())


def enclosed_holes(mask):
    """Yield (label, hole mask, area) for background components not touching a border."""
    inverse = (~np.asarray(mask, dtype=bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_border:
            yield label, labels == label, area


def hole_disparity_decision(
    hole,
    foreground,
    disparity,
    crop,
    ring_radius,
    absolute_tolerance,
    mad_scale,
    max_fill_area,
    small_hole_area,
):
    """Decide whether an enclosed hole is likely a dark patch of the workpiece."""
    area = int(hole.sum())
    if area > max_fill_area:
        return False, {"reason": "area_limit", "area": area}

    y0, y1, x0, x1 = crop
    hole_roi = hole[y0:y1, x0:x1]
    if not hole_roi.any():
        fill = area <= small_hole_area
        return fill, {
            "reason": "small_outside_disparity_roi" if fill else "outside_disparity_roi",
            "area": area,
        }

    kernel_size = 2 * ring_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ring = cv2.dilate(hole.astype(np.uint8), kernel).astype(bool)
    ring &= foreground
    ring_roi = ring[y0:y1, x0:x1]
    finite = np.isfinite(disparity)
    hole_values = disparity[hole_roi & finite]
    ring_values = disparity[ring_roi & finite]
    if hole_values.size < 32 or ring_values.size < 32:
        fill = area <= small_hole_area
        return fill, {
            "reason": "small_without_disparity_support" if fill else "insufficient_disparity_support",
            "area": area,
            "hole_disparity_pixels": int(hole_values.size),
            "ring_disparity_pixels": int(ring_values.size),
        }

    hole_median = float(np.median(hole_values))
    ring_median = float(np.median(ring_values))
    ring_mad = float(np.median(np.abs(ring_values - ring_median)))
    median_difference = abs(hole_median - ring_median)
    tolerance = max(absolute_tolerance, mad_scale * ring_mad)
    fill = median_difference <= tolerance
    return fill, {
        "reason": "disparity_continuous" if fill else "disparity_discontinuous",
        "area": area,
        "hole_median_disparity": hole_median,
        "ring_median_disparity": ring_median,
        "ring_mad_disparity": ring_mad,
        "median_disparity_difference": median_difference,
        "allowed_disparity_difference": tolerance,
        "hole_disparity_pixels": int(hole_values.size),
        "ring_disparity_pixels": int(ring_values.size),
    }


def refine_mask(
    probability,
    full_shape,
    disparity,
    threshold=0.5,
    closing_radius=3,
    hole_ring_radius=7,
    hole_absolute_tolerance=1.5,
    hole_mad_scale=1.0,
    max_fill_hole_fraction=0.025,
    small_hole_area=1000,
    crop=TRADITION_CROP,
):
    """Refine one model-resolution probability map at original image resolution."""
    full_height, full_width = full_shape
    probability_full = cv2.resize(
        np.asarray(probability, dtype=np.float32),
        (full_width, full_height),
        interpolation=cv2.INTER_LINEAR,
    )
    raw = probability_full >= threshold
    if closing_radius > 0:
        size = 2 * closing_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    else:
        closed = raw.copy()
    main, initial_components, removed_island_pixels = keep_largest_component(closed)
    if not main.any():
        raise ValueError("No foreground remains after connected-component filtering")

    decisions = []
    max_fill_area = int(round(max_fill_hole_fraction * full_height * full_width))
    for label, hole, area in list(enclosed_holes(main)):
        fill, details = hole_disparity_decision(
            hole,
            main,
            disparity,
            crop,
            hole_ring_radius,
            hole_absolute_tolerance,
            hole_mad_scale,
            max_fill_area,
            small_hole_area,
        )
        if fill:
            main[hole] = True
        decisions.append({"label": label, "fill": bool(fill), **details})

    refined, final_components_before_filter, final_removed_pixels = keep_largest_component(main)
    final_count, _, _, _ = cv2.connectedComponentsWithStats(
        refined.astype(np.uint8), connectivity=8
    )
    stats = {
        "raw_foreground_pixels": int(raw.sum()),
        "refined_foreground_pixels": int(refined.sum()),
        "foreground_pixel_change": int(refined.sum() - raw.sum()),
        "initial_components_after_closing": initial_components,
        "removed_island_pixels": removed_island_pixels + final_removed_pixels,
        "hole_count": len(decisions),
        "filled_hole_count": sum(item["fill"] for item in decisions),
        "filled_hole_pixels": sum(item["area"] for item in decisions if item["fill"]),
        "preserved_hole_count": sum(not item["fill"] for item in decisions),
        "final_components_before_filter": final_components_before_filter,
        "final_foreground_components": final_count - 1,
        "hole_decisions": decisions,
    }
    return raw, refined, stats


def overlay(image, mask, color):
    result = image.copy()
    result[mask] = (
        0.45 * result[mask] + 0.55 * np.asarray(color, dtype=np.float32)
    ).astype(np.uint8)
    return result


def label_panel(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(
        result, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return result


def disparity_color(disparity):
    normalized = np.clip(np.asarray(disparity, dtype=np.float32) / 192.0, 0.0, 1.0)
    result = cv2.applyColorMap((255.0 * normalized).astype(np.uint8), cv2.COLORMAP_TURBO)
    result[~np.isfinite(disparity)] = 0
    return result


def save_diagnostic(path, image, raw, refined, disparity, crop, stats):
    y0, y1, x0, x1 = crop
    image_roi = image[y0:y1, x0:x1]
    raw_roi = raw[y0:y1, x0:x1]
    refined_roi = refined[y0:y1, x0:x1]
    panels = [
        label_panel(image_roi, "rectified left RGB"),
        label_panel(overlay(image_roi, raw_roi, (0, 0, 255)), "raw BiSeNetV2 mask"),
        label_panel(overlay(image_roi, refined_roi, (0, 255, 0)), "refined one-component mask"),
        label_panel(disparity_color(disparity), "LiteAnyStereo disparity (no GT used)"),
        label_panel(raw_roi.astype(np.uint8) * 255, "raw mask"),
        label_panel(
            refined_roi.astype(np.uint8) * 255,
            f"refined: fill {stats['filled_hole_count']} / keep {stats['preserved_hole_count']}",
        ),
    ]
    panels = [
        cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR) if panel.ndim == 2 else panel
        for panel in panels
    ]
    panels = [cv2.resize(panel, (256, 409), interpolation=cv2.INTER_AREA) for panel in panels]
    montage = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    if not cv2.imwrite(str(path), montage, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Failed to write {path}")


def save_contact_sheet(paths, output_path, sample_count):
    count = min(sample_count, len(paths))
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int32)
    tiles = []
    for index in indices:
        image = cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR)
        tiles.append(cv2.resize(image, (384, 409), interpolation=cv2.INTER_AREA))
    rows = []
    for start in range(0, len(tiles), 3):
        row = tiles[start:start + 3]
        while len(row) < 3:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    if not cv2.imwrite(str(output_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Failed to write {output_path}")


def main():
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    if args.closing_radius < 0 or args.hole_ring_radius <= 0:
        raise ValueError("radii must be non-negative and hole-ring-radius must be positive")

    manifest = Path(args.manifest).expanduser().resolve()
    manifest_root = manifest.parent
    probability_dir = Path(args.probability_dir).expanduser().resolve()
    las_root = Path(args.las_output_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    (output_dir / "masks").mkdir(parents=True)
    (output_dir / "diagnostics").mkdir()

    try:
        rows = read_manifest(manifest, args.split, args.scene_prefix)
        records = []
        diagnostics = []
        all_hole_decisions = {}
        for row in rows:
            name = row["name"]
            scene = scene_id(name)
            image_path = resolve(manifest_root, row["left"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            probability_path = probability_dir / f"{name}.npy"
            probability = np.load(probability_path, allow_pickle=False)
            if probability.ndim != 2 or not np.isfinite(probability).all():
                raise ValueError(f"Invalid probability map: {probability_path}")
            disparity_path = las_root / scene / "disp.npy"
            disparity = np.load(disparity_path, allow_pickle=False).astype(np.float32)
            expected_disparity_shape = (
                TRADITION_CROP[1] - TRADITION_CROP[0],
                TRADITION_CROP[3] - TRADITION_CROP[2],
            )
            if disparity.shape != expected_disparity_shape:
                raise ValueError(f"Unexpected disparity shape {disparity.shape}: {disparity_path}")

            raw, refined, stats = refine_mask(
                probability,
                image.shape[:2],
                disparity,
                threshold=args.threshold,
                closing_radius=args.closing_radius,
                hole_ring_radius=args.hole_ring_radius,
                hole_absolute_tolerance=args.hole_absolute_tolerance,
                hole_mad_scale=args.hole_mad_scale,
                max_fill_hole_fraction=args.max_fill_hole_fraction,
                small_hole_area=args.small_hole_area,
            )
            if stats["final_foreground_components"] != 1:
                raise AssertionError(f"Final mask is not one component: {name}")
            mask_path = output_dir / "masks" / f"{name}.png"
            if not cv2.imwrite(str(mask_path), refined.astype(np.uint8) * 255):
                raise OSError(f"Failed to write {mask_path}")
            diagnostic_path = output_dir / "diagnostics" / f"{name}.jpg"
            save_diagnostic(
                diagnostic_path, image, raw, refined, disparity, TRADITION_CROP, stats
            )
            diagnostics.append(diagnostic_path)
            all_hole_decisions[name] = stats.pop("hole_decisions")
            records.append(
                {
                    "name": name,
                    "scene": scene,
                    "source_image": str(image_path),
                    "source_probability": str(probability_path),
                    "mask": str(mask_path.relative_to(output_dir)),
                    "mask_sha256": sha256_file(mask_path),
                    **stats,
                }
            )

        csv_path = output_dir / "refinement.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (output_dir / "hole_decisions.json").write_text(
            json.dumps(all_hole_decisions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        save_contact_sheet(
            diagnostics,
            output_dir / "refinement_contact_sheet.jpg",
            args.contact_sheet_samples,
        )
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": (
                "bilinear probability upsample -> threshold -> morphological close -> "
                "largest 8-connected foreground component -> disparity-consistent enclosed-hole "
                "repair -> final largest-component invariant"
            ),
            "ground_truth_used_for_refinement": False,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "probability_dir": str(probability_dir),
            "las_output_root": str(las_root),
            "scene_count": len(records),
            "parameters": {
                "threshold": args.threshold,
                "closing_radius": args.closing_radius,
                "hole_ring_radius": args.hole_ring_radius,
                "hole_absolute_tolerance": args.hole_absolute_tolerance,
                "hole_mad_scale": args.hole_mad_scale,
                "max_fill_hole_fraction": args.max_fill_hole_fraction,
                "small_hole_area": args.small_hole_area,
                "crop": list(TRADITION_CROP),
            },
            "totals": {
                "initial_components_after_closing": int(
                    sum(record["initial_components_after_closing"] for record in records)
                ),
                "removed_island_pixels": int(
                    sum(record["removed_island_pixels"] for record in records)
                ),
                "holes": int(sum(record["hole_count"] for record in records)),
                "filled_holes": int(sum(record["filled_hole_count"] for record in records)),
                "filled_hole_pixels": int(
                    sum(record["filled_hole_pixels"] for record in records)
                ),
                "preserved_holes": int(
                    sum(record["preserved_hole_count"] for record in records)
                ),
                "final_masks_with_exactly_one_component": int(
                    sum(record["final_foreground_components"] == 1 for record in records)
                ),
            },
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
