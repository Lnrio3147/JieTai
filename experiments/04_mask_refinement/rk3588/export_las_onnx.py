#!/usr/bin/env python3
"""Export the Experiment 1 LAS1 checkpoint as a fixed-shape ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAS_ROOT = REPO_ROOT / "projects" / "LiteAnyStereo"
DEFAULT_CHECKPOINT = LAS_ROOT / "checkpoints" / "LiteAnyStereo.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def padding_for(height: int, width: int, divisor: int = 32) -> tuple[int, int, int, int]:
    pad_height = (-height) % divisor
    pad_width = (-width) % divisor
    return (
        pad_width // 2,
        pad_width - pad_width // 2,
        pad_height // 2,
        pad_height - pad_height // 2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/liteanystereo_las1_1280x736.onnx"),
    )
    parser.add_argument(
        "--source-height",
        type=int,
        default=1280,
        help="Unpadded rectified image height.",
    )
    parser.add_argument(
        "--source-width",
        type=int,
        default=720,
        help="Unpadded rectified image width.",
    )
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument(
        "--input-layout",
        choices=["nchw", "nhwc"],
        default="nchw",
        help=(
            "ONNX input layout. RKNN-Toolkit2 expects image-model ONNX inputs "
            "in NCHW when channel mean/std preprocessing is configured."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
    if args.source_height <= 0 or args.source_width <= 0:
        raise ValueError("Source dimensions must be positive")
    if args.max_disp != 192:
        raise ValueError(
            "The LAS1 aggregation network has 48 disparity channels, so the "
            "deployable Experiment 1 checkpoint requires --max-disp 192."
        )

    left_pad, right_pad, top_pad, bottom_pad = padding_for(
        args.source_height, args.source_width
    )
    padded_height = args.source_height + top_pad + bottom_pad
    padded_width = args.source_width + left_pad + right_pad

    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(LAS_ROOT))
    from export_onnx import export  # pylint: disable=import-error,import-outside-toplevel

    export(
        version="las1",
        model_size=None,
        restore_ckpt=str(checkpoint),
        width=padded_width,
        height=padded_height,
        max_disp=args.max_disp,
        output_name=str(output),
        simplify=False,
        input_layout=args.input_layout,
    )

    import onnx  # pylint: disable=import-outside-toplevel

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    input_shapes = {}
    for value in graph.graph.input:
        input_shapes[value.name] = [
            dimension.dim_value for dimension in value.type.tensor_type.shape.dim
        ]
    if args.input_layout == "nchw":
        input_shape = [1, 3, padded_height, padded_width]
    else:
        input_shape = [1, padded_height, padded_width, 3]
    expected_shapes = {"left": input_shape, "right": input_shape}
    if input_shapes != expected_shapes:
        raise ValueError(f"Unexpected ONNX inputs: {input_shapes}")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "LiteAnyStereo LAS1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "source_shape_hwc": [args.source_height, args.source_width, 3],
        "padding_left_right_top_bottom": [
            left_pad,
            right_pad,
            top_pad,
            bottom_pad,
        ],
        "input_layout": args.input_layout,
        "input_shapes": expected_shapes,
        "output": {"name": "disparity", "shape": [1, 1, padded_height, padded_width]},
        "input_range": "RGB float32 values in [0, 255]; normalization is inside the graph",
        "max_disparity_px": args.max_disp,
        "onnx_opset": 18,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
