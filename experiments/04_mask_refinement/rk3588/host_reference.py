#!/usr/bin/env python3
"""Generate same-input FP32 references before comparing RKNN accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from board_benchmark import prepare_inputs, unpad_disparity
from postprocess import crop_array, refine_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    las = subparsers.add_parser("las", help="Run the FP32 LAS ONNX reference")
    las.add_argument("--onnx", type=Path, required=True)
    las.add_argument("--left", type=Path, required=True)
    las.add_argument("--right", type=Path, required=True)
    las.add_argument("--output-dir", type=Path, required=True)

    bisenet = subparsers.add_parser(
        "bisenet", help="Run the FP32 TensorFlow PB reference"
    )
    bisenet.add_argument("--pb", type=Path, required=True)
    bisenet.add_argument("--left", type=Path, required=True)
    bisenet.add_argument("--output-dir", type=Path, required=True)

    postprocess = subparsers.add_parser(
        "postprocess", help="Apply Experiment 4 to saved FP32 model outputs"
    )
    postprocess.add_argument("--output-dir", type=Path, required=True)
    postprocess.add_argument("--threshold", type=float, default=0.5)
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
        raise FileNotFoundError(path)
    return image


def metadata_path(output_dir: Path) -> Path:
    return output_dir / "reference_metadata.json"


def update_metadata(output_dir: Path, stage: str, values: dict) -> None:
    path = metadata_path(output_dir)
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    metadata[stage] = values
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_las(args: argparse.Namespace, output_dir: Path) -> None:
    import onnxruntime as ort

    onnx_path = args.onnx.expanduser().resolve()
    left_path = args.left.expanduser().resolve()
    right_path = args.right.expanduser().resolve()
    for path in (onnx_path, left_path, right_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    left_bgr = read_bgr(left_path)
    right_bgr = read_bgr(right_path)
    if left_bgr.shape != right_bgr.shape:
        raise ValueError("Left/right image shape mismatch")
    left, right, _, padding = prepare_inputs(left_bgr, right_bgr)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input_shape = session.get_inputs()[0].shape
    if len(onnx_input_shape) != 4:
        raise ValueError(f"Unexpected LAS ONNX input shape: {onnx_input_shape}")
    if onnx_input_shape[1] == 3:
        input_layout = "nchw"
        left = np.ascontiguousarray(left.transpose(0, 3, 1, 2))
        right = np.ascontiguousarray(right.transpose(0, 3, 1, 2))
    elif onnx_input_shape[-1] == 3:
        input_layout = "nhwc"
    else:
        raise ValueError(f"Cannot determine LAS ONNX layout: {onnx_input_shape}")
    output = session.run(
        ["disparity"],
        {"left": left.astype(np.float32), "right": right.astype(np.float32)},
    )[0]
    disparity = unpad_disparity(output, padding)
    np.save(output_dir / "disparity.npy", disparity, allow_pickle=False)
    update_metadata(
        output_dir,
        "las",
        {
            "onnx": str(onnx_path),
            "onnx_sha256": sha256_file(onnx_path),
            "left": str(left_path),
            "right": str(right_path),
            "input_shape": list(left_bgr.shape),
            "onnx_input_layout": input_layout,
            "padding_left_right_top_bottom": list(padding),
        },
    )


def run_bisenet(args: argparse.Namespace, output_dir: Path) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    pb_path = args.pb.expanduser().resolve()
    left_path = args.left.expanduser().resolve()
    for path in (pb_path, left_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    left_bgr = read_bgr(left_path)
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    prepared = cv2.resize(left_rgb, (288, 512), interpolation=cv2.INTER_AREA)
    batch = ((prepared.astype(np.float32) / 255.0 - 0.5) / 0.5)[None]

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(pb_path.read_bytes())
        tf.import_graph_def(graph_def, name="")
    input_tensor = graph.get_tensor_by_name("input_tensor:0")
    probability_tensor = graph.get_tensor_by_name("final_probability:0")
    with tf.Session(graph=graph) as session:
        probability = session.run(
            probability_tensor, feed_dict={input_tensor: batch}
        )[0, :, :, 1]
    probability = np.asarray(probability, dtype=np.float32)
    np.save(
        output_dir / "foreground_probability.npy", probability, allow_pickle=False
    )
    update_metadata(
        output_dir,
        "bisenet",
        {
            "pb": str(pb_path),
            "pb_sha256": sha256_file(pb_path),
            "left": str(left_path),
            "source_shape": list(left_bgr.shape),
            "input_shape": list(batch.shape),
        },
    )


def run_postprocess(args: argparse.Namespace, output_dir: Path) -> None:
    disparity_path = output_dir / "disparity.npy"
    probability_path = output_dir / "foreground_probability.npy"
    if not disparity_path.is_file() or not probability_path.is_file():
        raise FileNotFoundError(
            "Run the las and bisenet stages before the postprocess stage"
        )
    disparity = np.load(disparity_path, allow_pickle=False)
    probability = np.load(probability_path, allow_pickle=False)
    raw, refined, stats = refine_mask(
        probability,
        disparity.shape,
        crop_array(disparity),
        threshold=args.threshold,
    )
    subject_disparity = np.where(
        crop_array(refined), crop_array(disparity), np.nan
    ).astype(np.float32)
    cv2.imwrite(str(output_dir / "raw_mask.png"), raw.astype(np.uint8) * 255)
    cv2.imwrite(
        str(output_dir / "refined_mask.png"), refined.astype(np.uint8) * 255
    )
    np.save(
        output_dir / "subject_disparity.npy", subject_disparity, allow_pickle=False
    )
    update_metadata(
        output_dir,
        "postprocess",
        {"threshold": args.threshold, "stats": stats},
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "las":
        run_las(args, output_dir)
    elif args.stage == "bisenet":
        run_bisenet(args, output_dir)
    else:
        run_postprocess(args, output_dir)
    print(f"Completed {args.stage} reference stage: {output_dir}")


if __name__ == "__main__":
    main()
