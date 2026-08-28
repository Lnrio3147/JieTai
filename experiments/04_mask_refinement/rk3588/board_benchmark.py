#!/usr/bin/env python3
"""Run and time the Experiment 1-4 pipeline on an RK3588 board."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from pointcloud import FDJYP3_Q, adjusted_q_for_crop, reconstruct_point_cloud, write_binary_ply
from postprocess import TRADITION_CROP, crop_array, refine_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--las-model", type=Path, required=True)
    parser.add_argument("--bisenet-model", type=Path, required=True)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument(
        "--pairs-file",
        type=Path,
        help="Optional text file with one 'left right' pair per line.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/board_run"))
    parser.add_argument("--source-height", type=int, default=1280)
    parser.add_argument("--source-width", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--las-core",
        choices=["auto", "0", "1", "2", "0_1_2"],
        default="0",
    )
    parser.add_argument(
        "--bisenet-core",
        choices=["auto", "0", "1", "2", "0_1_2"],
        default="0",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-pointcloud",
        action="store_true",
        help=(
            "Include fixed-ROI FDJYP-3 XYZRGB reconstruction in pipeline timing "
            "and save a binary PLY after timing."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read {path}")
    return image


def resolve_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.pairs_file is not None:
        if args.left is not None or args.right is not None:
            raise ValueError("Use either --pairs-file or --left/--right, not both")
        pairs_file = args.pairs_file.expanduser().resolve()
        if not pairs_file.is_file():
            raise FileNotFoundError(pairs_file)
        pairs = []
        for line_number, raw_line in enumerate(
            pairs_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = shlex.split(line)
            if len(fields) != 2:
                raise ValueError(
                    f"{pairs_file}:{line_number}: expected two paths, got {fields}"
                )
            resolved = []
            for field in fields:
                path = Path(field).expanduser()
                if not path.is_absolute():
                    path = pairs_file.parent / path
                resolved.append(path.resolve())
            pairs.append((resolved[0], resolved[1]))
        if not pairs:
            raise ValueError(f"No image pairs in {pairs_file}")
        return pairs

    if args.left is None or args.right is None:
        raise ValueError("Provide --left and --right together, or use --pairs-file")
    return [
        (args.left.expanduser().resolve(), args.right.expanduser().resolve())
    ]


def pad_to_32(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = rgb.shape[:2]
    pad_height = (-height) % 32
    pad_width = (-width) % 32
    padding = (
        pad_width // 2,
        pad_width - pad_width // 2,
        pad_height // 2,
        pad_height - pad_height // 2,
    )
    left, right, top, bottom = padding
    padded = cv2.copyMakeBorder(
        rgb, top, bottom, left, right, cv2.BORDER_REPLICATE
    )
    return padded, padding


def prepare_inputs(left_bgr: np.ndarray, right_bgr: np.ndarray):
    left_rgb = np.ascontiguousarray(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB))
    right_rgb = np.ascontiguousarray(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB))
    left_las, padding = pad_to_32(left_rgb)
    right_las, right_padding = pad_to_32(right_rgb)
    if padding != right_padding:
        raise ValueError(f"Left/right padding mismatch: {padding} vs {right_padding}")
    bisenet = cv2.resize(left_rgb, (288, 512), interpolation=cv2.INTER_AREA)
    return (
        np.ascontiguousarray(left_las[None]),
        np.ascontiguousarray(right_las[None]),
        np.ascontiguousarray(bisenet[None]),
        padding,
    )


def unpad_disparity(disparity: np.ndarray, padding) -> np.ndarray:
    array = np.asarray(disparity)
    while array.ndim > 2 and 1 in array.shape:
        axis = next(index for index, size in enumerate(array.shape) if size == 1)
        array = np.squeeze(array, axis=axis)
    if array.ndim != 2:
        raise ValueError(f"Unexpected LAS output shape: {np.asarray(disparity).shape}")
    left, right, top, bottom = padding
    height, width = array.shape
    return np.asarray(
        array[top : height - bottom if bottom else height, left : width - right if right else width],
        dtype=np.float32,
    )


def foreground_probability(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (512, 288, 2):
        probability = array[:, :, 1]
    elif array.shape == (2, 512, 288):
        probability = array[1]
    else:
        raise ValueError(f"Unexpected BiSeNet output shape: {np.asarray(output).shape}")
    probability = np.asarray(probability, dtype=np.float32)
    if not np.isfinite(probability).all():
        raise FloatingPointError("BiSeNet probability contains non-finite values")
    return probability


def summarize_ms(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean_ms = float(array.mean())
    return {
        "count": int(array.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "fps_from_mean": float(1000.0 / mean_ms),
    }


def core_mask(rknn_lite_class, name: str):
    attribute = {
        "auto": "NPU_CORE_AUTO",
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "0_1_2": "NPU_CORE_0_1_2",
    }[name]
    if not hasattr(rknn_lite_class, attribute):
        raise RuntimeError(
            f"Installed RKNNLite does not provide {attribute}; use a matching Lite2 wheel"
        )
    return getattr(rknn_lite_class, attribute)


class LiteRunner:
    def __init__(self, rknn_lite_class, model_path: Path, core: str):
        self.runtime = rknn_lite_class(verbose=False)
        result = self.runtime.load_rknn(str(model_path))
        if result != 0:
            raise RuntimeError(f"load_rknn failed for {model_path}: {result}")
        result = self.runtime.init_runtime(
            core_mask=core_mask(rknn_lite_class, core)
        )
        if result != 0:
            raise RuntimeError(f"init_runtime failed for {model_path}: {result}")

    def infer(self, inputs: list[np.ndarray]):
        outputs = self.runtime.inference(
            inputs=inputs, data_format=["nhwc"] * len(inputs)
        )
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs")
        return outputs

    def release(self) -> None:
        self.runtime.release()


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def system_info() -> dict:
    devfreq = {}
    for path in Path("/sys/class/devfreq").glob("*npu*/cur_freq"):
        devfreq[str(path)] = read_optional(str(path))
    npu_governors = {}
    for path in Path("/sys/class/devfreq").glob("*npu*/governor"):
        npu_governors[str(path)] = read_optional(str(path))
    cpu_governors = {}
    for path in Path("/sys/devices/system/cpu/cpufreq").glob(
        "policy*/scaling_governor"
    ):
        cpu_governors[str(path)] = read_optional(str(path))
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "device_compatible": read_optional("/proc/device-tree/compatible"),
        "npu_cur_freq_hz": devfreq,
        "npu_governor": npu_governors,
        "cpu_governor": cpu_governors,
    }


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0:
        raise ValueError("--runs must be positive and --warmup non-negative")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")

    las_model = args.las_model.expanduser().resolve()
    bisenet_model = args.bisenet_model.expanduser().resolve()
    image_pairs = resolve_pairs(args)
    output_dir = args.output_dir.expanduser().resolve()
    for path in (las_model, bisenet_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    for pair in image_pairs:
        for path in pair:
            if not path.is_file():
                raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_shape = (args.source_height, args.source_width, 3)
    loaded_pairs = []
    for left_path, right_path in image_pairs:
        left_bgr = read_bgr(left_path)
        right_bgr = read_bgr(right_path)
        if left_bgr.shape != expected_shape or right_bgr.shape != expected_shape:
            raise ValueError(
                f"The fixed RKNN models expect source shape {expected_shape}; got "
                f"left={left_bgr.shape}, right={right_bgr.shape} for {left_path}"
            )
        if left_bgr.shape != right_bgr.shape:
            raise ValueError("Left/right source image shapes differ")
        loaded_pairs.append((left_bgr, right_bgr))

    try:
        import rknnlite
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit-lite2 is not installed. Run this script on the RK3588 board."
        ) from exc

    first_left_bgr, first_right_bgr = loaded_pairs[0]
    left_input, right_input, bisenet_input, padding = prepare_inputs(
        first_left_bgr, first_right_bgr
    )
    las_runner = LiteRunner(RKNNLite, las_model, args.las_core)
    bisenet_runner = LiteRunner(RKNNLite, bisenet_model, args.bisenet_core)
    try:
        for _ in range(args.warmup):
            las_runner.infer([left_input, right_input])
            bisenet_runner.infer([bisenet_input])

        las_times = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            las_runner.infer([left_input, right_input])
            las_times.append((time.perf_counter_ns() - start) / 1e6)

        bisenet_times = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            bisenet_runner.infer([bisenet_input])
            bisenet_times.append((time.perf_counter_ns() - start) / 1e6)

        sequential_times = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            las_runner.infer([left_input, right_input])
            bisenet_runner.infer([bisenet_input])
            sequential_times.append((time.perf_counter_ns() - start) / 1e6)

        stage_times = {
            "preprocess": [],
            "las": [],
            "bisenet": [],
            "postprocess": [],
            "end_to_end": [],
        }
        if args.include_pointcloud:
            stage_times["pointcloud"] = []
            stage_times["end_to_end_pointcloud"] = []
        pipeline_samples = []
        last = None
        for _ in range(max(1, min(args.warmup, 3))):
            prepared = prepare_inputs(first_left_bgr, first_right_bgr)
            las_output = las_runner.infer([prepared[0], prepared[1]])[0]
            probability_output = bisenet_runner.infer([prepared[2]])[0]
            disparity = unpad_disparity(las_output, prepared[3])
            probability = foreground_probability(probability_output)
            _, refined_mask, _ = refine_mask(
                probability,
                first_left_bgr.shape[:2],
                crop_array(disparity),
                threshold=args.threshold,
            )
            if args.include_pointcloud:
                subject_disparity = np.where(
                    crop_array(refined_mask), crop_array(disparity), np.nan
                ).astype(np.float32)
                reconstruct_point_cloud(
                    subject_disparity, crop_array(first_left_bgr)
                )

        last_pair_index = 0
        for run_index in range(args.runs):
            pair_index = run_index % len(loaded_pairs)
            left_bgr, right_bgr = loaded_pairs[pair_index]
            full_start = time.perf_counter_ns()
            preprocess_start = full_start
            prepared = prepare_inputs(left_bgr, right_bgr)
            preprocess_end = time.perf_counter_ns()

            las_output = las_runner.infer([prepared[0], prepared[1]])[0]
            las_end = time.perf_counter_ns()
            probability_output = bisenet_runner.infer([prepared[2]])[0]
            bisenet_end = time.perf_counter_ns()

            disparity = unpad_disparity(las_output, prepared[3])
            probability = foreground_probability(probability_output)
            disparity_crop = crop_array(disparity)
            raw_mask, refined_mask, refinement = refine_mask(
                probability,
                left_bgr.shape[:2],
                disparity_crop,
                threshold=args.threshold,
            )
            subject_disparity = np.where(
                crop_array(refined_mask), disparity_crop, np.nan
            ).astype(np.float32)
            postprocess_end = time.perf_counter_ns()

            pointcloud_points = None
            pointcloud_rgb = None
            pointcloud_valid = None
            if args.include_pointcloud:
                pointcloud_points, pointcloud_rgb, pointcloud_valid = (
                    reconstruct_point_cloud(
                        subject_disparity, crop_array(left_bgr)
                    )
                )
                full_end = time.perf_counter_ns()
            else:
                full_end = postprocess_end
            stage_times["preprocess"].append((preprocess_end - preprocess_start) / 1e6)
            stage_times["las"].append((las_end - preprocess_end) / 1e6)
            stage_times["bisenet"].append((bisenet_end - las_end) / 1e6)
            stage_times["postprocess"].append(
                (postprocess_end - bisenet_end) / 1e6
            )
            stage_times["end_to_end"].append(
                (postprocess_end - full_start) / 1e6
            )
            if args.include_pointcloud:
                stage_times["pointcloud"].append(
                    (full_end - postprocess_end) / 1e6
                )
                stage_times["end_to_end_pointcloud"].append(
                    (full_end - full_start) / 1e6
                )
            sample_timings = {
                "preprocess": stage_times["preprocess"][-1],
                "las": stage_times["las"][-1],
                "bisenet": stage_times["bisenet"][-1],
                "postprocess": stage_times["postprocess"][-1],
                "end_to_end": stage_times["end_to_end"][-1],
            }
            if args.include_pointcloud:
                sample_timings["pointcloud"] = stage_times["pointcloud"][-1]
                sample_timings["end_to_end_pointcloud"] = stage_times[
                    "end_to_end_pointcloud"
                ][-1]
            pipeline_samples.append(
                {
                    "run_index": run_index,
                    "pair_index": pair_index,
                    "left": str(image_pairs[pair_index][0]),
                    "right": str(image_pairs[pair_index][1]),
                    "timings_ms": sample_timings,
                }
            )
            last = (
                pair_index,
                disparity,
                probability,
                raw_mask,
                refined_mask,
                subject_disparity,
                refinement,
                pointcloud_points,
                pointcloud_rgb,
                pointcloud_valid,
            )
    finally:
        las_runner.release()
        bisenet_runner.release()

    assert last is not None
    (
        last_pair_index,
        disparity,
        probability,
        raw_mask,
        refined_mask,
        subject_disparity,
        refinement,
        pointcloud_points,
        pointcloud_rgb,
        pointcloud_valid,
    ) = last
    np.save(output_dir / "disparity.npy", disparity, allow_pickle=False)
    np.save(output_dir / "foreground_probability.npy", probability, allow_pickle=False)
    np.save(output_dir / "subject_disparity.npy", subject_disparity, allow_pickle=False)
    cv2.imwrite(str(output_dir / "raw_mask.png"), raw_mask.astype(np.uint8) * 255)
    cv2.imwrite(
        str(output_dir / "refined_mask.png"), refined_mask.astype(np.uint8) * 255
    )

    pointcloud_report = None
    if args.include_pointcloud:
        assert pointcloud_points is not None
        assert pointcloud_rgb is not None
        assert pointcloud_valid is not None
        ply_start = time.perf_counter_ns()
        write_binary_ply(
            output_dir / "pointcloud_xyzrgb_binary.ply",
            pointcloud_points,
            pointcloud_rgb,
        )
        ply_write_ms = (time.perf_counter_ns() - ply_start) / 1e6
        pointcloud_report = {
            "calibration": "FDJYP-3 JXP",
            "crop_y0_y1_x0_x1": list(TRADITION_CROP),
            "q_original": FDJYP3_Q.tolist(),
            "q_adjusted": adjusted_q_for_crop().tolist(),
            "valid_points": int(pointcloud_points.shape[0]),
            "valid_fraction": float(pointcloud_valid.mean()),
            "binary_ply_bytes": int(
                (output_dir / "pointcloud_xyzrgb_binary.ply").stat().st_size
            ),
            "binary_ply_write_ms_outside_pipeline_timing": float(ply_write_ms),
        }

    try:
        lite_version = importlib.metadata.version("rknn-toolkit-lite2")
    except importlib.metadata.PackageNotFoundError:
        lite_version = getattr(rknnlite, "__version__", "unknown")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing_scope": (
            "steady-state blocking RKNNLite calls; model load and disk I/O excluded; "
            "end_to_end includes in-memory preprocessing, both models, output conversion, "
            "Experiment 4 refinement, and subject disparity generation; when "
            "enabled, end_to_end_pointcloud additionally includes in-memory "
            "FDJYP-3 XYZRGB reconstruction but excludes PLY disk write and camera I/O"
        ),
        "warmup": args.warmup,
        "runs": args.runs,
        "core_assignment": {"las": args.las_core, "bisenet": args.bisenet_core},
        "models": {
            "las": {"path": str(las_model), "sha256": sha256_file(las_model)},
            "bisenet": {
                "path": str(bisenet_model),
                "sha256": sha256_file(bisenet_model),
            },
        },
        "inputs": {
            "pair_count": len(image_pairs),
            "pairs": [
                {"left": str(left), "right": str(right)}
                for left, right in image_pairs
            ],
            "artifact_pair_index": last_pair_index,
            "source_shape_hwc": list(first_left_bgr.shape),
            "las_padding_left_right_top_bottom": list(padding),
        },
        "model_only": {
            "liteanystereo": summarize_ms(las_times),
            "bisenetv2": summarize_ms(bisenet_times),
            "sequential_both": summarize_ms(sequential_times),
        },
        "pipeline": {
            name: summarize_ms(values) for name, values in stage_times.items()
        },
        "pipeline_samples": pipeline_samples,
        "refinement": refinement,
        "pointcloud": pointcloud_report,
        "rknn_toolkit_lite2_version": lite_version,
        "system": system_info(),
    }
    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
