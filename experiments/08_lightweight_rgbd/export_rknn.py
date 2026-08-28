#!/usr/bin/env python3
"""Export Experiment 8 to ONNX and convert it with RKNN mixed quantization.

The mixed-precision path follows RKNN-Toolkit2's two-step hybrid quantization:
step 1 profiles an INT8 model, geometry-related nodes are overridden to float16,
and step 2 builds the final RKNN file from the edited quantization config.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch

import config_experiment8 as cfg
from models.student_network import create_student
from utils.data import WorkpieceStudentDataset


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor):
        return self.model(rgb, geometry)


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_student(pretrained=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    return model.eval()


def export_onnx(checkpoint_path: Path, onnx_path: Path) -> None:
    model = ExportWrapper(load_model(checkpoint_path))
    rgb = torch.zeros(1, 3, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    geometry = torch.zeros(1, 3, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (rgb, geometry),
        str(onnx_path),
        input_names=["rgb", "geometry"],
        output_names=["mask_logits", "boundary_logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"ONNX exported: {onnx_path}")


def prepare_calibration_data(output_dir: Path, sample_count: int) -> Path:
    dataset = WorkpieceStudentDataset(
        root=cfg.ROOT,
        dataset=cfg.DATASET_DIR,
        split="train",
        width=cfg.IMAGE_WIDTH,
        height=cfg.IMAGE_HEIGHT,
        augment=False,
        seed=cfg.SEED,
        require_teachers=False,
    )
    calibration_dir = output_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    indices = np.linspace(0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=int)
    for number, index in enumerate(indices):
        sample = dataset[int(index)]
        rgb_path = calibration_dir / f"rgb_{number:03d}.npy"
        geometry_path = calibration_dir / f"geometry_{number:03d}.npy"
        np.save(rgb_path, sample["rgb"].numpy()[None].astype(np.float32))
        np.save(geometry_path, sample["geometry"].numpy()[None].astype(np.float32))
        lines.append(f"{rgb_path} {geometry_path}\n")
    dataset_file = output_dir / "calibration_dataset.txt"
    dataset_file.write_text("".join(lines), encoding="utf-8")
    print(f"Calibration set: {dataset_file} ({len(lines)} samples)")
    return dataset_file


def import_rknn():
    try:
        from rknn.api import RKNN
    except ImportError as error:
        raise RuntimeError(
            "rknn-toolkit2 is not installed in this Python environment. "
            "ONNX export is still usable; install the RK3588-compatible "
            "rknn-toolkit2 wheel before running a RKNN conversion stage."
        ) from error
    return RKNN


def configure_rknn(rknn) -> None:
    result = rknn.config(
        target_platform="rk3588",
        mean_values=None,
        std_values=None,
        quantized_dtype="asymmetric_quantized-8",
        quantized_algorithm="normal",
        quantized_method="channel",
        optimization_level=3,
    )
    if result != 0:
        raise RuntimeError(f"RKNN config failed: {result}")


def geometry_onnx_node_names(onnx_path: Path) -> list[str]:
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    patterns = ("geometry_encoder", "geometry_refine")
    names: list[str] = []
    for node in graph.node:
        candidates = [node.name, *node.output]
        if node.op_type == "Conv" and any(
            pattern in value for value in candidates for pattern in patterns
        ):
            name = node.name or node.output[0]
            if name and name not in names:
                names.append(name)
    return names


def patch_hybrid_config(config_path: Path, node_names: list[str]) -> None:
    """Keep geometry/gradient operations in FP16 in RKNN's generated YAML."""
    text = config_path.read_text(encoding="utf-8")
    entries = "".join(f"    '{name}': float16\n" for name in node_names)
    if not entries:
        raise RuntimeError("No geometry nodes were found in the exported ONNX graph")

    pattern = re.compile(r"^custom_quantize_layers:\s*(?:\{\})?\s*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub("custom_quantize_layers:\n" + entries.rstrip("\n"), text, count=1)
    else:
        text = text.rstrip() + "\n\ncustom_quantize_layers:\n" + entries
    config_path.write_text(text, encoding="utf-8")
    hint_path = config_path.with_suffix(config_path.suffix + ".geometry_fp16.txt")
    hint_path.write_text("\n".join(node_names) + "\n", encoding="utf-8")
    print(f"Marked {len(node_names)} geometry nodes as FP16 in {config_path}")


def hybrid_step1(onnx_path: Path, output_dir: Path, dataset_file: Path) -> tuple[Path, Path, Path]:
    RKNN = import_rknn()
    output_dir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        rknn = RKNN(verbose=True)
        configure_rknn(rknn)
        if rknn.load_onnx(model=str(onnx_path.resolve())) != 0:
            raise RuntimeError("RKNN failed to load ONNX")
        result = rknn.hybrid_quantization_step1(
            dataset=str(dataset_file.resolve()),
            proposal=True,
            proposal_dataset_size=16,
        )
        rknn.release()
        if result != 0:
            raise RuntimeError(f"RKNN hybrid quantization step 1 failed: {result}")
    finally:
        os.chdir(old_cwd)

    stem = onnx_path.stem
    model_input = output_dir / f"{stem}.model"
    data_input = output_dir / f"{stem}.data"
    quant_config = output_dir / f"{stem}.quantization.cfg"
    missing = [path for path in (model_input, data_input, quant_config) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"RKNN step 1 did not produce: {missing}")
    patch_hybrid_config(quant_config, geometry_onnx_node_names(onnx_path))
    return model_input, data_input, quant_config


def hybrid_step2(output_dir: Path, onnx_path: Path, rknn_path: Path) -> None:
    RKNN = import_rknn()
    stem = onnx_path.stem
    model_input = output_dir / f"{stem}.model"
    data_input = output_dir / f"{stem}.data"
    quant_config = output_dir / f"{stem}.quantization.cfg"
    rknn = RKNN(verbose=True)
    configure_rknn(rknn)
    result = rknn.hybrid_quantization_step2(
        model_input=str(model_input),
        data_input=str(data_input),
        model_quantization_cfg=str(quant_config),
    )
    if result != 0:
        rknn.release()
        raise RuntimeError(f"RKNN hybrid quantization step 2 failed: {result}")
    if rknn.export_rknn(str(rknn_path)) != 0:
        rknn.release()
        raise RuntimeError("RKNN export failed")
    rknn.release()
    print(f"Mixed INT8/FP16 RKNN exported: {rknn_path}")


def full_int8(onnx_path: Path, rknn_path: Path, dataset_file: Path) -> None:
    RKNN = import_rknn()
    rknn = RKNN(verbose=True)
    configure_rknn(rknn)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise RuntimeError("RKNN failed to load ONNX")
    if rknn.build(do_quantization=True, dataset=str(dataset_file)) != 0:
        raise RuntimeError("RKNN INT8 build failed")
    if rknn.export_rknn(str(rknn_path)) != 0:
        raise RuntimeError("RKNN export failed")
    rknn.release()
    print(f"Full INT8 RKNN exported: {rknn_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=cfg.DISTILLED_RUN_DIR / "best.pt"
    )
    parser.add_argument(
        "--stage",
        choices=("onnx", "hybrid-step1", "hybrid-step2", "hybrid-all", "int8"),
        default="onnx",
    )
    parser.add_argument("--output-dir", type=Path, default=cfg.EXPERIMENT_DIR / "exports")
    parser.add_argument("--calibration-samples", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "experiment8_student.onnx"
    rknn_path = args.output_dir / "experiment8_student_mixed.rknn"

    if args.stage in ("onnx", "hybrid-all", "int8") and not onnx_path.exists():
        export_onnx(args.checkpoint, onnx_path)
    elif args.stage == "onnx":
        export_onnx(args.checkpoint, onnx_path)

    if args.stage == "onnx":
        return
    dataset_file = prepare_calibration_data(args.output_dir, args.calibration_samples)
    if args.stage in ("hybrid-step1", "hybrid-all"):
        hybrid_step1(onnx_path, args.output_dir, dataset_file)
    if args.stage in ("hybrid-step2", "hybrid-all"):
        hybrid_step2(args.output_dir, onnx_path, rknn_path)
    if args.stage == "int8":
        full_int8(onnx_path, args.output_dir / "experiment8_student_int8.rknn", dataset_file)


if __name__ == "__main__":
    main()
