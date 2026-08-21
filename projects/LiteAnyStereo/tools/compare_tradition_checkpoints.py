#!/usr/bin/env python3
"""Compare LiteAnyStereo checkpoints with the tradition_stereo evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.models import build_model, load_model_weights, normalize_model_size, normalize_version
from training.checkpoint import safe_torch_load
from training.data import TraditionStereoEvaluationDataset
from training.engine import validate
from training.metrics import TRADITION_EXCLUDED_SCENES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare checkpoints on FDJYP-3 with fixed 818x512 tradition metrics."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="repeatable LABEL=PATH entry; the first checkpoint is the comparison baseline",
    )
    parser.add_argument("--version", default="las1")
    parser.add_argument("--model_size", "--model-size", default=None)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument(
        "--data_root",
        default="../tradition_stereo/datasets/FDJYP-3",
        help="tradition_stereo FDJYP-3 directory",
    )
    parser.add_argument(
        "--image_root",
        default="./data/datasets/JMP-LF6020-ETH3D",
        help="rectified JMP-LF6020 image root used as model input",
    )
    parser.add_argument("--output_dir", default="./runs/evaluation/tradition_checkpoint_comparison")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epe_threshold", type=float, default=20.0)
    parser.add_argument("--no_epe_filter", action="store_true")
    parser.add_argument("--include_excluded", action="store_true")
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="save scene vis.png files under <output_dir>/visualizations/<label>",
    )
    parser.add_argument("--vis_error_max", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_checkpoint_specs(values):
    specs = []
    labels = set()
    for value in values:
        if "=" in value:
            label, path_value = value.split("=", 1)
        else:
            path_value = value
            label = Path(value).stem
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
        if not label or label in labels:
            raise ValueError(f"Checkpoint labels must be non-empty and unique: {label!r}")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        specs.append((label, path))
    return specs


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summaries, protocol):
    lines = [
        "# LiteAnyStereo / tradition_stereo 统一口径对比",
        "",
        f"- 数据：`{protocol['data_root']}`",
        f"- 模型输入图：`{protocol['image_root']}`",
        "- ROI：`disp[234:1052, 126:638]`（818×512）",
        f"- 场景 EPE 过滤：`{protocol['epe_threshold']}`",
        f"- 固定排除场景：`{', '.join(protocol['excluded_scenes']) or '关闭'}`",
        "- 汇总：保留场景的宏平均",
        f"- 各 checkpoint 评价场景集合一致：`{protocol['same_scene_set']}`",
        "",
        "| Checkpoint | EPE(px) | D1(%) | Bad1(%) | Bad2(%) | Bad3(%) | EPE提升 | 场景数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['epe']:.4f} | {row['d1']:.2f} | "
            f"{row['bad1']:.2f} | {row['bad2']:.2f} | {row['bad3']:.2f} | "
            f"{row['epe_improvement_percent']:.2f}% | {row['scene_count']} |"
        )
    if not protocol["same_scene_set"]:
        lines.extend(
            [
                "",
                "> 注意：EPE 阈值按 checkpoint 独立过滤，保留场景集合不同；上表提升率仅复现 tradition_stereo 原策略，严格训练对比请加 `--no_epe_filter`。",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if (
        args.workers < 0
        or args.max_disp <= 0
        or args.epe_threshold <= 0
        or args.vis_error_max <= 0
    ):
        raise ValueError("workers cannot be negative; max_disp/epe_threshold must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")

    specs = parse_checkpoint_specs(args.checkpoint)
    output_dir = Path(args.output_dir).expanduser().resolve()
    comparison_path = output_dir / "comparison.json"
    if comparison_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{comparison_path} exists; choose another --output_dir or pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    version = normalize_version(args.version)
    model_size = normalize_model_size(version, args.model_size)
    dataset = TraditionStereoEvaluationDataset(args.data_root, image_root=args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(
        version,
        fnet_pretrained=False,
        model_size=model_size,
        max_disp=args.max_disp,
    ).to(device)

    excluded_scenes = () if args.include_excluded else TRADITION_EXCLUDED_SCENES
    epe_threshold = None if args.no_epe_filter else args.epe_threshold
    summaries = []
    kept_scene_sets = []
    for label, checkpoint_path in specs:
        load_model_weights(
            model,
            safe_torch_load(checkpoint_path, map_location=device),
            strict=True,
        )
        metrics = validate(
            model,
            loader,
            device=device,
            max_disp=args.max_disp,
            amp=False,
            logger=logging.getLogger("tradition-comparison"),
            evaluation_protocol="tradition",
            excluded_scenes=excluded_scenes,
            epe_threshold=epe_threshold,
            return_scene_metrics=True,
            save_vis_dir=output_dir / "visualizations" / label if args.save_vis else None,
            vis_error_max=args.vis_error_max,
        )
        scene_rows = metrics.pop("scene_metrics")
        excluded_set = set(metrics["excluded_scenes"])
        filtered_set = set(metrics["epe_filtered_scenes"])
        for row in scene_rows:
            row["status"] = (
                "excluded"
                if row["scene"] in excluded_set
                else "epe_filtered"
                if row["scene"] in filtered_set
                else "kept"
            )
        kept_scene_sets.append({row["scene"] for row in scene_rows if row["status"] == "kept"})
        write_csv(output_dir / f"scene_metrics_{label}.csv", scene_rows)
        summaries.append({"label": label, "checkpoint": str(checkpoint_path), **metrics})
        print(
            f"{label}: EPE={metrics['epe']:.4f} D1={metrics['d1']:.2f}% "
            f"Bad1/2/3={metrics['bad1']:.2f}/{metrics['bad2']:.2f}/{metrics['bad3']:.2f}% "
            f"scenes={metrics['scene_count']}"
        )

    baseline_epe = summaries[0]["epe"]
    for row in summaries:
        row["epe_improvement_percent"] = (
            100.0 * (baseline_epe - row["epe"]) / baseline_epe if baseline_epe else 0.0
        )
    summary_columns = (
        "label",
        "checkpoint",
        "epe",
        "d1",
        "bad1",
        "bad2",
        "bad3",
        "epe_improvement_percent",
        "valid_pixels",
        "total_pixels",
        "valid_ratio",
        "scene_count",
        "original_scene_count",
        "excluded_scene_count",
        "epe_filtered_scene_count",
    )
    write_csv(
        output_dir / "checkpoint_metrics.csv",
        [{key: row[key] for key in summary_columns} for row in summaries],
    )
    protocol = {
        "name": "tradition_stereo_512x818",
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "image_root": str(Path(args.image_root).expanduser().resolve()),
        "crop": [234, 1052, 126, 638],
        "mask": "finite GT and GT > 0",
        "aggregation": "scene_macro",
        "epe_threshold": epe_threshold,
        "excluded_scenes": list(excluded_scenes),
        "max_disp": args.max_disp,
        "same_scene_set": all(
            scene_set == kept_scene_sets[0] for scene_set in kept_scene_sets[1:]
        ),
    }
    comparison_path.write_text(
        json.dumps({"protocol": protocol, "checkpoints": summaries}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(
        render_markdown(summaries, protocol), encoding="utf-8"
    )
    print(f"Comparison written to {output_dir}")


if __name__ == "__main__":
    main()
