#!/usr/bin/env python3
"""Convert LiteAnyStereo ONNX or BiSeNetV2 frozen PB to RK3588 RKNN."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BISENET = (
    REPO_ROOT
    / "experiments"
    / "03_manual_segmentation"
    / "fdjyp3"
    / "results"
    / "model_manual"
    / "bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["las", "bisenet"], required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help="LAS ONNX or BiSeNet frozen PB. BiSeNet defaults to Experiment 3.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["fp16", "int8"], default="fp16")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="RKNN calibration list; mandatory for --dtype int8.",
    )
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_ret(operation: str, result: int) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN {operation} failed with return code {result}")


def package_version() -> str:
    for name in ("rknn-toolkit2", "rknn_toolkit2"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unknown"


def main() -> None:
    args = parse_args()
    source = args.source
    if source is None:
        if args.model != "bisenet":
            raise ValueError("--source is required for --model las")
        source = DEFAULT_BISENET
    source = source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_suffix = ".onnx" if args.model == "las" else ".pb"
    if source.suffix.lower() != expected_suffix:
        raise ValueError(f"--model {args.model} expects a {expected_suffix} source")
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
    if args.dtype == "int8" and (dataset is None or not dataset.is_file()):
        raise FileNotFoundError(
            "INT8 conversion requires a valid --dataset calibration list"
        )

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit2 is not installed. Run this script in the x86_64 "
            "conversion environment, not with rknn-toolkit-lite2 on the board."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=args.verbose)
    try:
        common_config = {
            "target_platform": args.target,
            "optimization_level": args.optimization_level,
            "quant_img_RGB2BGR": False,
        }
        if args.dtype == "fp16":
            common_config["float_dtype"] = "float16"
        else:
            common_config["quantized_dtype"] = "w8a8"

        if args.model == "las":
            check_ret(
                "config",
                rknn.config(
                    mean_values=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    std_values=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    **common_config,
                ),
            )
            check_ret("load_onnx", rknn.load_onnx(model=str(source)))
        else:
            check_ret(
                "config",
                rknn.config(
                    mean_values=[[127.5, 127.5, 127.5]],
                    std_values=[[127.5, 127.5, 127.5]],
                    **common_config,
                ),
            )
            check_ret(
                "load_tensorflow",
                rknn.load_tensorflow(
                    tf_pb=str(source),
                    inputs=["input_tensor"],
                    outputs=["final_probability"],
                    input_size_list=[[1, 512, 288, 3]],
                ),
            )

        build_args = {"do_quantization": args.dtype == "int8"}
        if dataset is not None:
            build_args["dataset"] = str(dataset)
        check_ret("build", rknn.build(**build_args))
        check_ret("export_rknn", rknn.export_rknn(str(output)))
    finally:
        rknn.release()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "dtype": args.dtype,
        "target": args.target,
        "optimization_level": args.optimization_level,
        "dataset": str(dataset) if dataset else None,
        "rknn_toolkit2_version": package_version(),
        "preprocess": (
            "RGB uint8 NHWC -> unchanged 0..255; LAS normalizes inside graph"
            if args.model == "las"
            else "RGB uint8 NHWC -> (x - 127.5) / 127.5 in RKNN input config"
        ),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
