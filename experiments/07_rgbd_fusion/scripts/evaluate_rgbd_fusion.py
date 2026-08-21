#!/usr/bin/env python3
"""Evaluate a trained RGB-D fusion checkpoint on the fixed val or test split."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT = Path(__file__).resolve().with_name("train_rgbd_fusion.py")
spec = importlib.util.spec_from_file_location("rgbd_training", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(SCRIPT)
RGBD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RGBD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=RGBD.EXPERIMENT / "results/rgbd_fusion_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional result directory. The default preserves the original layout.",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    training_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset"])
    width = int(config["image_size"]["width"])
    height = int(config["image_size"]["height"])
    threshold = float(training_summary["selected_probability_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"] and device.type == "cuda")

    pretrained = torch.load(config["pretrained"], map_location="cpu", weights_only=True)
    model = RGBD.RGBDFusionNet(pretrained).to(device)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    split_set = RGBD.WorkpieceRGBD(
        dataset_root, args.split, (width, height), False, int(config["seed"])
    )
    loader = DataLoader(
        split_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    records = [{**record, "dataset": str(dataset_root)} for record in split_set.records]
    probabilities = RGBD.predict_probabilities(model, loader, device, (288, 512), amp)
    raw = {name: value >= threshold for name, value in probabilities.items()}
    refined = RGBD.refine_with_experiment4(records, probabilities, threshold)
    baseline_root = (
        RGBD.BASELINE_EXPERIMENT
        / f"results/tune01_jop_reflective_rescue_{args.split}_v2/masks/jop_reflective_rescue"
    )
    baseline = RGBD.load_baseline(records, baseline_root)
    methods = {"v4_1": baseline, "rgbd_raw": raw, "rgbd_exp4": refined}
    evaluation = {
        name: RGBD.aggregate(records, masks, args.boundary_tolerance)
        for name, masks in methods.items()
    }

    output = (
        args.output.resolve()
        if args.output is not None
        else run / ("validation" if args.split == "val" else "test_recheck")
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    probability_directory = output / "probabilities"
    probability_directory.mkdir(parents=True)
    for name, probability in probabilities.items():
        np.save(probability_directory / f"{name}.npy", probability.astype(np.float16))
    for method, masks in methods.items():
        directory = output / "masks" / method
        directory.mkdir(parents=True)
        for name, mask in masks.items():
            cv2.imwrite(str(directory / f"{name}.png"), mask.astype(np.uint8) * 255)

    per_scene = []
    panels = []
    for record in records:
        name = record["name"]
        image = cv2.imread(str(dataset_root / record["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(dataset_root / record["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        row = {"name": name, "category": record["category"]}
        for method, masks in methods.items():
            metrics = RGBD.sample_metrics(gt, masks[name])
            for key in ("foreground_iou", "precision", "recall", "dice"):
                row[f"{method}_{key}"] = metrics[key]
        per_scene.append(row)
        panel = np.hstack(
            [
                RGBD.EVAL.label_panel(image, "image"),
                RGBD.EVAL.mask_panel(gt, "human outer contour"),
                RGBD.EVAL.mask_panel(baseline[name], "V4.1"),
                RGBD.EVAL.mask_panel(raw[name], "RGB-D raw"),
                RGBD.EVAL.mask_panel(refined[name], "RGB-D + Exp4"),
            ]
        )
        cv2.putText(
            panel, name, (5, panel.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (0, 255, 0), 1, cv2.LINE_AA,
        )
        comparison = cv2.resize(panel, (900, 320), interpolation=cv2.INTER_AREA)
        panels.append(comparison)
        cv2.imwrite(str(output / f"{name}.jpg"), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with (output / "per_scene.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_scene[0]))
        writer.writeheader()
        writer.writerows(per_scene)
    cv2.imwrite(
        str(output / "contact_sheet.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 91]
    )
    summary = {
        "split": args.split,
        "count": len(records),
        "threshold_selected_on_validation": threshold,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "methods": evaluation,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
