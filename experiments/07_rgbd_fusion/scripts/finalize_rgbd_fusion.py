#!/usr/bin/env python3
"""Rebuild calibrated test artifacts from an existing RGB-D training run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

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
        "--run", type=Path, default=RGBD.EXPERIMENT / "results/rgbd_fusion_v1"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset"])
    size = (int(config["image_size"]["width"]), int(config["image_size"]["height"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"] and device.type == "cuda")
    pretrained = torch.load(config["pretrained"], map_location="cpu", weights_only=True)
    model = RGBD.RGBDFusionNet(pretrained).to(device)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    datasets = {
        split: RGBD.WorkpieceRGBD(dataset_root, split, size, False, int(config["seed"]))
        for split in ("val", "test")
    }
    probabilities = {}
    records = {}
    for split, split_set in datasets.items():
        loader = DataLoader(
            split_set,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        probabilities[split] = RGBD.predict_probabilities(
            model, loader, device, (288, 512), amp
        )
        records[split] = [
            {**record, "dataset": str(dataset_root)} for record in split_set.records
        ]

    threshold, table = RGBD.calibration(
        records["val"], probabilities["val"], args.boundary_tolerance
    )
    raw = {name: value >= threshold for name, value in probabilities["test"].items()}
    refined = RGBD.refine_with_experiment4(records["test"], probabilities["test"], threshold)
    baseline = RGBD.load_baseline(records["test"], RGBD.BASELINE_DEFAULT)
    methods = {"v4_1": baseline, "rgbd_raw": raw, "rgbd_exp4": refined}
    evaluation = {
        method: RGBD.aggregate(records["test"], masks, args.boundary_tolerance)
        for method, masks in methods.items()
    }
    quality_threshold = 0.15
    quality_raw = {
        name: value >= quality_threshold for name, value in probabilities["test"].items()
    }
    quality_refined = RGBD.refine_with_experiment4(
        records["test"], probabilities["test"], quality_threshold
    )
    quality_reference = {
        "threshold": quality_threshold,
        "purpose": "higher-IoU operating-point reference; recall-priority remains the default",
        "rgbd_raw": RGBD.aggregate(
            records["test"], quality_raw, args.boundary_tolerance
        ),
        "rgbd_exp4": RGBD.aggregate(
            records["test"], quality_refined, args.boundary_tolerance
        ),
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "development_comparison_test",
        "warning": "This 21-image split was used by prior engineering comparisons and is not a pristine final test set.",
        "best_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_score": checkpoint["validation"]["macro_category_iou"],
        "selected_probability_threshold": threshold,
        "threshold_selection": "validation macro-category F2 with overall precision >= 0.94",
        "validation_threshold_table": table,
        "test": evaluation,
        "quality_reference": quality_reference,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RGBD.save_outputs(run, records["test"], probabilities["test"], methods, threshold)
    print(json.dumps({"threshold": threshold, "test": evaluation}, indent=2))


if __name__ == "__main__":
    main()
