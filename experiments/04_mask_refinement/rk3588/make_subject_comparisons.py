#!/usr/bin/env python3
"""Render before/after subject-disparity comparisons with a shared color scale."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from postprocess import TRADITION_CROP, crop_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--full-disparity-dir", type=Path, required=True)
    parser.add_argument("--subject-disparity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse already rendered scene images."
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def colorize(
    disparity: np.ndarray,
    valid: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    normalized = np.zeros(disparity.shape, dtype=np.uint8)
    if maximum > minimum:
        values = np.clip(disparity[valid], minimum, maximum)
        normalized[valid] = np.rint(
            (values - minimum) * (255.0 / (maximum - minimum))
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float = 0.62,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def camera_panel(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Fit the full left-camera input and mark the Experiment 4 ROI."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unexpected camera image shape: {image.shape}")
    marked = image.copy()
    y0, y1, x0, x1 = TRADITION_CROP
    cv2.rectangle(marked, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 5)
    output_height, output_width = output_shape
    scale = min(output_width / image.shape[1], output_height / image.shape[0])
    resized_width = int(round(image.shape[1] * scale))
    resized_height = int(round(image.shape[0] * scale))
    resized = cv2.resize(
        marked, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    panel = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    x_offset = (output_width - resized_width) // 2
    y_offset = (output_height - resized_height) // 2
    panel[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return panel


def render_comparison(
    scene: str,
    camera_bgr: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    finite_before = np.isfinite(before)
    valid_after = mask & np.isfinite(after)
    if not finite_before.all() or not valid_after.any():
        raise ValueError(f"Invalid disparity values for {scene}")
    if not np.isnan(after[~mask]).all():
        raise ValueError(f"Subject background is not NaN for {scene}")
    if not np.array_equal(after[valid_after], before[valid_after]):
        raise ValueError(f"Subject disparity differs from input ROI for {scene}")

    minimum = float(before.min())
    maximum = float(before.max())
    before_color = colorize(before, finite_before, minimum, maximum)
    after_color = colorize(after, valid_after, minimum, maximum)
    mask_color = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    panel_height, panel_width = before.shape
    input_panel = camera_panel(camera_bgr, (panel_height, panel_width))
    gap = 4
    header_height = 70
    footer_height = 55
    canvas_width = panel_width * 4 + gap * 3
    canvas_height = header_height + panel_height + footer_height
    canvas = np.full((canvas_height, canvas_width, 3), 24, dtype=np.uint8)
    x0 = 0
    x1 = panel_width + gap
    x2 = (panel_width + gap) * 2
    x3 = (panel_width + gap) * 3
    canvas[header_height : header_height + panel_height, x0 : x0 + panel_width] = (
        input_panel
    )
    canvas[header_height : header_height + panel_height, x1 : x1 + panel_width] = (
        before_color
    )
    canvas[header_height : header_height + panel_height, x2 : x2 + panel_width] = (
        after_color
    )
    canvas[header_height : header_height + panel_height, x3 : x3 + panel_width] = (
        mask_color
    )

    text(canvas, scene, (12, 24), 0.56)
    text(canvas, "Input: left camera (red ROI)", (12, 55))
    text(canvas, "Before: LAS disparity ROI", (x1 + 12, 55))
    text(canvas, "After: final subject disparity", (x2 + 12, 55))
    text(canvas, "Final subject mask", (x3 + 12, 55))

    bar_left = x1 + 12
    bar_top = header_height + panel_height + 12
    bar_width = panel_width * 2 + gap - 24
    gradient = np.linspace(0, 255, bar_width, dtype=np.uint8)[None]
    gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    canvas[bar_top : bar_top + 14, bar_left : bar_left + bar_width] = gradient
    text(canvas, f"{minimum:.3f} px", (bar_left, bar_top + 38), 0.48)
    maximum_label = f"{maximum:.3f} px"
    label_width = cv2.getTextSize(
        maximum_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    )[0][0]
    text(
        canvas,
        maximum_label,
        (bar_left + bar_width - label_width, bar_top + 38),
        0.48,
    )
    coverage = float(valid_after.mean())
    text(
        canvas,
        f"Subject: {int(valid_after.sum())} px ({coverage * 100.0:.2f}%)",
        (x3 + 12, bar_top + 28),
        0.54,
    )
    y0, y1, x0_roi, x1_roi = TRADITION_CROP
    text(
        canvas,
        f"ROI: y={y0}:{y1}, x={x0_roi}:{x1_roi}",
        (12, bar_top + 28),
        0.50,
    )
    return canvas, {
        "shape": list(after.shape),
        "shared_scale_min_px": minimum,
        "shared_scale_max_px": maximum,
        "subject_pixels": int(valid_after.sum()),
        "subject_fraction": coverage,
        "background_pixels": int((~valid_after).sum()),
    }


def make_contact_sheet(
    rendered: list[tuple[str, Path]], output: Path, columns: int
) -> None:
    if columns <= 0:
        raise ValueError("--contact-sheet-columns must be positive")
    thumb_width = 480
    label_height = 30
    gap = 8
    thumbnails = []
    for scene, path in rendered:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        thumb_height = int(round(image.shape[0] * thumb_width / image.shape[1]))
        thumb = cv2.resize(image, (thumb_width, thumb_height), cv2.INTER_AREA)
        tile = np.full((label_height + thumb_height, thumb_width, 3), 24, np.uint8)
        text(tile, scene, (7, 21), 0.44)
        tile[label_height:] = thumb
        thumbnails.append(tile)

    rows = (len(thumbnails) + columns - 1) // columns
    tile_height = thumbnails[0].shape[0]
    sheet_width = columns * thumb_width + (columns - 1) * gap
    sheet_height = rows * tile_height + (rows - 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), 24, np.uint8)
    for index, tile in enumerate(thumbnails):
        row, column = divmod(index, columns)
        x = column * (thumb_width + gap)
        y = row * (tile_height + gap)
        sheet[y : y + tile_height, x : x + thumb_width] = tile
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Failed to write {output}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    full_root = args.full_disparity_dir.expanduser().resolve()
    subject_root = args.subject_disparity_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    report_path = subject_root / "export_report.json"
    for required in (dataset_root, full_root, report_path):
        if not required.exists():
            raise FileNotFoundError(required)
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    subject_report = json.loads(report_path.read_text(encoding="utf-8"))
    records = []
    rendered = []
    for record in subject_report["records"]:
        scene = record["scene"]
        full_path = full_root / scene / "disparity.npy"
        subject_path = subject_root / scene / "subject_disparity.npy"
        mask_path = subject_root / scene / "subject_mask.png"
        camera_path = dataset_root / scene / "im0.png"
        camera_bgr = cv2.imread(str(camera_path), cv2.IMREAD_COLOR)
        if camera_bgr is None:
            raise FileNotFoundError(camera_path)
        before = crop_array(np.load(full_path, allow_pickle=False))
        after = np.load(subject_path, allow_pickle=False)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            raise FileNotFoundError(mask_path)
        mask = mask_image > 0
        comparison, stats = render_comparison(
            scene, camera_bgr, before, after, mask
        )
        scene_dir = output_root / scene
        scene_dir.mkdir(parents=True, exist_ok=args.resume)
        output_path = scene_dir / "before_after_subject.png"
        if not output_path.exists():
            if not cv2.imwrite(str(output_path), comparison):
                raise RuntimeError(f"Failed to write {output_path}")
        elif not args.resume:
            raise FileExistsError(output_path)
        else:
            existing = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            if existing is None or existing.shape != comparison.shape:
                raise ValueError(f"Invalid existing comparison: {output_path}")
        rendered.append((scene, output_path))
        records.append(
            {
                "index": record["index"],
                "scene": scene,
                "input_camera_sha256": sha256_file(camera_path),
                "input_full_disparity_sha256": sha256_file(full_path),
                "input_subject_disparity_sha256": sha256_file(subject_path),
                "input_subject_mask_sha256": sha256_file(mask_path),
                "comparison_sha256": sha256_file(output_path),
                **stats,
            }
        )
        print(f"[{len(records):02d}/{subject_report['pair_count']:02d}] {scene}")

    contact_sheet = output_root / "all_subject_before_after.jpg"
    make_contact_sheet(rendered, contact_sheet, args.contact_sheet_columns)
    report = {
        "pair_count": len(records),
        "dataset_root": str(dataset_root),
        "crop_y0_y1_x0_x1": list(TRADITION_CROP),
        "layout": [
            "Original full left-camera input with the Experiment 4 ROI marked in red",
            "Before: full LAS disparity in the Experiment 4 ROI",
            "After: final subject disparity with black background",
            "Final subject mask",
        ],
        "color_scale": (
            "Shared before/after scale per scene using the actual minimum and "
            "maximum of the full disparity ROI"
        ),
        "contact_sheet": contact_sheet.name,
        "contact_sheet_sha256": sha256_file(contact_sheet),
        "records": records,
    }
    (output_root / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "comparison_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"pair_count": len(records), "contact_sheet": str(contact_sheet)}, indent=2))


if __name__ == "__main__":
    main()
