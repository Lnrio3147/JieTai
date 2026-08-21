#!/usr/bin/env python3
"""Convert JMP-LF6020.zip into a rectified, ETH3D-style training dataset.

The source archive is never modified. Measurement images are rotated and rectified,
and the enhanced PLY point clouds are projected into left-view pseudo-disparity maps.
Each output scene contains im0.png, im1.png, disp0GT.pfm, mask0nocc.png,
and calib.txt, matching the layout expected for ETH3D-style stereo data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image


TRAIN_GROUPS = {"DE0548", "FDJYP-0", "FDJYP-2"}
VAL_GROUPS = {"FDJYP-3"}
SUPPORTED_GROUPS = TRAIN_GROUPS | VAL_GROUPS


class UnusableSampleError(ValueError):
    """A source sample is structurally valid but contains no usable geometry."""


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the JMP-LF6020 stereo dataset.")
    parser.add_argument("--archive", required=True, help="path to JMP-LF6020.zip")
    parser.add_argument("--output", required=True, help="new training-ready output directory")
    parser.add_argument("--max_disp", type=float, default=192.0, help="used for quality statistics only")
    parser.add_argument(
        "--archive_sha256",
        default=None,
        help="previously verified archive SHA-256; if omitted it is calculated before conversion",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="development only; 0 converts all")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not value:
        raise ValueError("Cannot create an empty safe sample name")
    return value


def write_pfm(path, disparity):
    """Write a single-channel float32 PFM without importing the model package."""
    disparity = np.asarray(disparity, dtype=np.float32)
    if disparity.ndim != 2:
        raise ValueError(f"PFM disparity must be 2D, got {disparity.shape}")
    height, width = disparity.shape
    with Path(path).open("wb") as handle:
        handle.write(f"Pf\n{width} {height}\n-1\n".encode("ascii"))
        handle.write(np.flipud(disparity).astype("<f4", copy=False).tobytes())


def eth3d_calibration_text(calibration, width, height, max_disp):
    """Render the rectified projection matrices in ETH3D calib.txt syntax."""
    p1 = calibration["P1"]
    p2 = calibration["P2"]
    cam0 = p1[:, :3]
    cam1 = p2[:, :3]
    doffs = float(p1[0, 2] - p2[0, 2])
    baseline = float(-p2[0, 3] / p2[0, 0])

    def matrix_line(name, matrix):
        rows = [" ".join(f"{float(value):.12g}" for value in row) for row in matrix]
        return f"{name}=[{'; '.join(rows)}]"

    return (
        f"{matrix_line('cam0', cam0)}\n"
        f"{matrix_line('cam1', cam1)}\n"
        f"doffs={doffs:.12g}\n"
        f"baseline={baseline:.12g}\n"
        f"width={int(width)}\n"
        f"height={int(height)}\n"
        f"ndisp={int(np.ceil(max_disp))}\n"
    )


def parse_opencv_matrix(text, key, shape):
    match = re.search(rf"(?m)^{re.escape(key)}:.*?data:\s*\[(.*?)\]", text, re.S)
    if match is None:
        raise ValueError(f"Calibration matrix {key} is missing")
    values = [
        float(value)
        for value in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))
    ]
    if len(values) != int(np.prod(shape)):
        raise ValueError(f"Calibration matrix {key} has {len(values)} values, expected {np.prod(shape)}")
    return np.asarray(values, dtype=np.float64).reshape(shape)


def load_calibration(archive, info):
    raw = archive.read(info)
    text = raw.decode("utf-8")
    values = {
        "M1": parse_opencv_matrix(text, "M1", (3, 3)),
        "D1": parse_opencv_matrix(text, "D1", (1, 5)),
        "M2": parse_opencv_matrix(text, "M2", (3, 3)),
        "D2": parse_opencv_matrix(text, "D2", (1, 5)),
        "R1": parse_opencv_matrix(text, "R1", (3, 3)),
        "R2": parse_opencv_matrix(text, "R2", (3, 3)),
        "P1": parse_opencv_matrix(text, "P1", (3, 4)),
        "P2": parse_opencv_matrix(text, "P2", (3, 4)),
    }
    values["source"] = info.filename
    values["raw"] = raw
    return values


def make_rectification_maps(calibration, width=720, height=1280):
    size = (width, height)
    left_maps = cv2.initUndistortRectifyMap(
        calibration["M1"],
        calibration["D1"],
        calibration["R1"],
        calibration["P1"][:, :3],
        size,
        cv2.CV_32FC1,
    )
    right_maps = cv2.initUndistortRectifyMap(
        calibration["M2"],
        calibration["D2"],
        calibration["R2"],
        calibration["P2"][:, :3],
        size,
        cv2.CV_32FC1,
    )
    return left_maps, right_maps


def read_rgb_from_zip(archive, name):
    with Image.open(io.BytesIO(archive.read(name))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def read_binary_ply(raw, source_name):
    header_end = raw.find(b"end_header\n")
    if header_end < 0:
        raise ValueError(f"PLY header terminator is missing: {source_name}")
    header_end += len(b"end_header\n")
    header = raw[:header_end].decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"Only binary little-endian PLY is supported: {source_name}")
    count_match = re.search(r"element vertex (\d+)", header)
    if count_match is None:
        raise ValueError(f"PLY vertex count is missing: {source_name}")
    count = int(count_match.group(1))
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    expected_size = header_end + count * dtype.itemsize
    if len(raw) != expected_size:
        raise ValueError(f"Unexpected PLY size for {source_name}: {len(raw)} vs {expected_size}")
    return np.frombuffer(raw, dtype=dtype, count=count, offset=header_end)


def project_disparity(points, calibration, height=1280, width=720):
    p1 = calibration["P1"]
    p2 = calibration["P2"]
    x = points["x"].astype(np.float64)
    y = points["y"].astype(np.float64)
    z = points["z"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = p1[0, 0] * x / z + p1[0, 2]
        v = p1[1, 1] * y / z + p1[1, 2]
        disparity_values = -p2[0, 3] / z
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(disparity_values)
    ui = np.zeros(len(points), dtype=np.int64)
    vi = np.zeros(len(points), dtype=np.int64)
    ui[finite] = np.rint(u[finite]).astype(np.int64)
    vi[finite] = np.rint(v[finite]).astype(np.int64)
    target_u = u - disparity_values

    valid = (
        finite
        & (z > 0.0)
        & (disparity_values > 0.0)
        & (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
        & (target_u >= 0.0)
        & (target_u < width)
    )
    if not valid.any():
        raise UnusableSampleError("point cloud projection produced no valid stereo pixels")

    projection_residual = float(
        max(np.max(np.abs(u[finite] - np.rint(u[finite]))), np.max(np.abs(v[finite] - np.rint(v[finite]))))
    )
    if projection_residual > 0.51:
        raise ValueError(f"Point cloud does not align to the rectified pixel grid (residual={projection_residual})")

    pixel_index = vi[valid] * width + ui[valid]
    disparity = np.zeros(height * width, dtype=np.float32)
    # In the unlikely event of duplicate projections, keep the closer point.
    np.maximum.at(disparity, pixel_index, disparity_values[valid].astype(np.float32))
    collision_count = int(len(pixel_index) - len(np.unique(pixel_index)))
    disparity = disparity.reshape(height, width)
    valid_mask = disparity > 0.0
    return disparity, valid_mask, projection_residual, collision_count


def find_calibration_infos(infos):
    nonempty_yml = [info for info in infos if info.filename.lower().endswith(".yml") and info.file_size]
    de = next((info for info in nonempty_yml if info.filename.startswith("DE0548/")), None)
    fd = next((info for info in nonempty_yml if info.filename.startswith("FDJYP-0/")), None)
    if de is None or fd is None:
        raise FileNotFoundError("Could not locate both DE0548 and FDJYP calibration YAML files")
    return {"DE0548": de, "FDJYP": fd}


def choose_samples(infos, info_by_name):
    dat_stems = sorted(info.filename[:-4] for info in infos if info.filename.lower().endswith(".dat"))
    kept = []
    excluded = []
    seen_signatures = {}
    for stem in dat_stems:
        group = stem.split("/", 1)[0]
        required = [stem + suffix for suffix in ("_L.png", "_R.png", ".ply")]
        missing = [name for name in required if name not in info_by_name]
        if missing:
            excluded.append({"source_stem": stem, "reason": "missing_companion", "detail": ";".join(missing)})
            continue
        signature = tuple(
            (info_by_name[name].CRC, info_by_name[name].file_size) for name in required
        )
        if signature in seen_signatures:
            excluded.append(
                {
                    "source_stem": stem,
                    "reason": "exact_duplicate",
                    "detail": seen_signatures[signature],
                }
            )
            continue
        seen_signatures[signature] = stem
        if group not in SUPPORTED_GROUPS:
            reason = "missing_valid_calibration" if group == "JM" else "unsupported_group"
            excluded.append({"source_stem": stem, "reason": reason, "detail": group})
            continue
        kept.append(stem)
    return kept, excluded


def save_preview(preview_dir, sample_id, left, right, disparity, max_disp):
    preview_dir.mkdir(parents=True, exist_ok=True)
    target_height = 640
    scale = target_height / left.shape[0]
    size = (max(1, int(round(left.shape[1] * scale))), target_height)
    left_small = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    normalized = np.clip(disparity / max_disp, 0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[disparity <= 0] = 0
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color_small = cv2.resize(color, size, interpolation=cv2.INTER_NEAREST)
    montage = np.concatenate([left_small, right_small, color_small], axis=1)
    Image.fromarray(montage).save(preview_dir / f"{sample_id}.jpg", quality=92)


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    archive_path = Path(args.archive).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    staging_path = output_path.with_name(output_path.name + ".preparing")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output_path}")
    if staging_path.exists():
        raise FileExistsError(f"Partial staging directory already exists: {staging_path}")
    if args.max_disp <= 0 or args.max_samples < 0:
        raise ValueError("--max_disp must be positive and --max_samples cannot be negative")

    archive_sha256 = args.archive_sha256 or sha256_file(archive_path)
    started = time.perf_counter()
    print(f"archive={archive_path}", flush=True)
    print(f"archive_sha256={archive_sha256}", flush=True)

    for relative in ("calibration", "metadata", "previews", "splits"):
        (staging_path / relative).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    rename_rows = []
    sample_reports = []
    previewed_groups = set()
    sampled_disparities = []
    group_counts = Counter()
    split_counts = Counter()
    total_pixels = 0
    total_valid = 0
    total_over_max = 0
    max_projection_residual = 0.0
    total_collisions = 0

    with ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        info_by_name = {info.filename: info for info in infos}
        calibration_infos = find_calibration_infos(infos)
        calibrations = {
            key: load_calibration(archive, info) for key, info in calibration_infos.items()
        }
        maps = {key: make_rectification_maps(value) for key, value in calibrations.items()}
        for key, calibration in calibrations.items():
            filename = "de0548_stereo.yml" if key == "DE0548" else "fdjyp_stereo.yml"
            (staging_path / "calibration" / filename).write_bytes(calibration["raw"])

        sample_stems, excluded_rows = choose_samples(infos, info_by_name)
        if args.max_samples:
            sample_stems = sample_stems[: args.max_samples]
        print(
            f"selected_samples={len(sample_stems)} excluded_samples={len(excluded_rows)}",
            flush=True,
        )

        for index, stem in enumerate(sample_stems, start=1):
            group = stem.split("/", 1)[0]
            split = "train" if group in TRAIN_GROUPS else "val"
            calibration_key = "DE0548" if group == "DE0548" else "FDJYP"
            calibration = calibrations[calibration_key]
            left_maps, right_maps = maps[calibration_key]
            source_base = PurePosixPath(stem).name
            sample_id = f"{safe_name(group)}_{safe_name(source_base)}"

            # The calibration metadata explicitly places source *_R as camera 1
            # and source *_L as camera 2. Rotate landscape captures back to the
            # portrait calibration orientation before rectification.
            source_left = read_rgb_from_zip(archive, stem + "_R.png")
            source_right = read_rgb_from_zip(archive, stem + "_L.png")
            if source_left.shape != (720, 1280, 3) or source_right.shape != (720, 1280, 3):
                raise ValueError(f"Unexpected measurement image shape for {stem}")
            rotated_left = cv2.rotate(source_left, cv2.ROTATE_90_CLOCKWISE)
            rotated_right = cv2.rotate(source_right, cv2.ROTATE_90_CLOCKWISE)
            rectified_left = cv2.remap(rotated_left, *left_maps, interpolation=cv2.INTER_LINEAR)
            rectified_right = cv2.remap(rotated_right, *right_maps, interpolation=cv2.INTER_LINEAR)

            points = read_binary_ply(archive.read(stem + ".ply"), stem + ".ply")
            try:
                disparity, valid, projection_residual, collisions = project_disparity(
                    points, calibration, height=rectified_left.shape[0], width=rectified_left.shape[1]
                )
            except UnusableSampleError as error:
                excluded_rows.append(
                    {"source_stem": stem, "reason": "invalid_point_cloud", "detail": str(error)}
                )
                print(f"excluded sample={stem} reason=invalid_point_cloud", flush=True)
                continue
            values = disparity[valid]
            if values.size == 0:
                raise ValueError(f"No valid disparity remained for {stem}")

            scene_dir = staging_path / sample_id
            scene_dir.mkdir()
            left_relative = Path(sample_id) / "im0.png"
            right_relative = Path(sample_id) / "im1.png"
            disparity_relative = Path(sample_id) / "disp0GT.pfm"
            valid_relative = Path(sample_id) / "mask0nocc.png"
            Image.fromarray(rectified_left).save(staging_path / left_relative)
            Image.fromarray(rectified_right).save(staging_path / right_relative)
            write_pfm(staging_path / disparity_relative, disparity)
            Image.fromarray(valid.astype(np.uint8) * 255).save(staging_path / valid_relative)
            (scene_dir / "calib.txt").write_text(
                eth3d_calibration_text(
                    calibration,
                    width=rectified_left.shape[1],
                    height=rectified_left.shape[0],
                    max_disp=args.max_disp,
                ),
                encoding="utf-8",
            )

            if group not in previewed_groups:
                save_preview(
                    staging_path / "previews",
                    sample_id,
                    rectified_left,
                    rectified_right,
                    disparity,
                    args.max_disp,
                )
                previewed_groups.add(group)

            manifest_rows.append(
                {
                    "left": left_relative.as_posix(),
                    "right": right_relative.as_posix(),
                    "disparity": disparity_relative.as_posix(),
                    "valid": valid_relative.as_posix(),
                    "split": split,
                    "name": sample_id,
                    "disp_scale": "1",
                }
            )
            rename_rows.append(
                {
                    "sample_id": sample_id,
                    "group": group,
                    "split": split,
                    "source_named_L": stem + "_L.png",
                    "source_named_R": stem + "_R.png",
                    "training_left_source": stem + "_R.png",
                    "training_right_source": stem + "_L.png",
                    "source_point_cloud": stem + ".ply",
                    "calibration_source": calibration["source"],
                }
            )
            sample_reports.append(
                {
                    "sample_id": sample_id,
                    "group": group,
                    "split": split,
                    "points": int(len(points)),
                    "valid_pixels": int(valid.sum()),
                    "valid_fraction": float(valid.mean()),
                    "disparity_min": float(values.min()),
                    "disparity_median": float(np.median(values)),
                    "disparity_p99": float(np.percentile(values, 99)),
                    "disparity_max": float(values.max()),
                    "over_max_disp_pixels": int((values >= args.max_disp).sum()),
                    "projection_residual": projection_residual,
                    "projection_collisions": collisions,
                }
            )
            stride = max(1, values.size // 5000)
            sampled_disparities.append(values[::stride])
            group_counts[group] += 1
            split_counts[split] += 1
            total_pixels += int(disparity.size)
            total_valid += int(valid.sum())
            total_over_max += int((values >= args.max_disp).sum())
            max_projection_residual = max(max_projection_residual, projection_residual)
            total_collisions += collisions
            if index == 1 or index % 10 == 0 or index == len(sample_stems):
                print(
                    f"processed={index}/{len(sample_stems)} sample={sample_id} "
                    f"valid={valid.mean():.3f} disp_median={np.median(values):.2f}",
                    flush=True,
                )

    write_csv(
        staging_path / "manifest.csv",
        ["left", "right", "disparity", "valid", "split", "name", "disp_scale"],
        manifest_rows,
    )
    write_csv(
        staging_path / "metadata/rename_map.csv",
        [
            "sample_id",
            "group",
            "split",
            "source_named_L",
            "source_named_R",
            "training_left_source",
            "training_right_source",
            "source_point_cloud",
            "calibration_source",
        ],
        rename_rows,
    )
    write_csv(
        staging_path / "metadata/excluded_samples.csv",
        ["source_stem", "reason", "detail"],
        excluded_rows,
    )
    for split in ("train", "val"):
        split_names = [row["name"] for row in manifest_rows if row["split"] == split]
        (staging_path / "splits" / f"{split}.txt").write_text(
            "".join(f"{name}\n" for name in split_names), encoding="utf-8"
        )

    sampled = np.concatenate(sampled_disparities) if sampled_disparities else np.asarray([])
    report = {
        "status": "completed",
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "output": str(output_path),
        "processing": {
            "layout": "ETH3D-style scene directories",
            "scene_files": ["im0.png", "im1.png", "disp0GT.pfm", "mask0nocc.png", "calib.txt"],
            "source_named_R_becomes_training_left": True,
            "source_named_L_becomes_training_right": True,
            "rotation": "90_degrees_clockwise",
            "rectification": "OpenCV M/D/R/P matrices from stereo.yml",
            "label": "enhanced PLY projected to P1; disparity = -P2[0,3] / Z",
            "right_visibility_required": True,
            "point_clouds_copied": False,
        },
        "samples": {
            "converted": len(manifest_rows),
            "excluded": len(excluded_rows),
            "by_group": dict(sorted(group_counts.items())),
            "by_split": dict(sorted(split_counts.items())),
            "excluded_by_reason": dict(sorted(Counter(row["reason"] for row in excluded_rows).items())),
        },
        "quality": {
            "image_height": 1280,
            "image_width": 720,
            "total_pixels": total_pixels,
            "valid_pixels": total_valid,
            "valid_fraction": total_valid / total_pixels if total_pixels else 0.0,
            "over_max_disp_pixels": total_over_max,
            "over_max_disp_fraction_of_valid": total_over_max / total_valid if total_valid else 0.0,
            "max_disp_for_statistics": args.max_disp,
            "sampled_disparity_min": float(sampled.min()) if sampled.size else None,
            "sampled_disparity_median": float(np.median(sampled)) if sampled.size else None,
            "sampled_disparity_p99": float(np.percentile(sampled, 99)) if sampled.size else None,
            "sampled_disparity_max": float(sampled.max()) if sampled.size else None,
            "max_projection_residual_pixels": max_projection_residual,
            "projection_collisions": total_collisions,
        },
        "seconds": time.perf_counter() - started,
        "sample_reports": sample_reports,
    }
    (staging_path / "metadata/conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging_path / "metadata/README_zh-CN.txt").write_text(
        "本目录由 tools/prepare_jmp_lf6020.py 自动生成。\n"
        "原始 ZIP 未修改；训练图像已交换左右、顺时针旋转并完成极线校正。\n"
        "每个样本采用 ETH3D 目录形式：im0.png、im1.png、disp0GT.pfm、mask0nocc.png、calib.txt。\n"
        "disp0GT.pfm 是由增强后 PLY 点云和标定矩阵生成的伪视差，不是人工真值。\n"
        "mask0nocc.png 是伪视差有效性掩码，不是工件分割；稠密 PLY 的有效区域会显示为白色矩形。\n"
        "训练和验证样本分别记录在 splits/train.txt 与 splits/val.txt。\n"
        "原始文件名与新 sample_id 的对应关系见 rename_map.csv。\n"
        "被排除的重复或缺标定样本见 excluded_samples.csv。\n",
        encoding="utf-8",
    )
    staging_path.rename(output_path)
    print(f"completed output={output_path} seconds={report['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
