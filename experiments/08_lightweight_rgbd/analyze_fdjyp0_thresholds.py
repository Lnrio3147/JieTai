#!/usr/bin/env python3
"""Exploratory threshold sweep over saved FDJYP-0 probabilities.

The selected values are oracle diagnostics on FDJYP-0 and must not replace the
fixed-threshold generalization scores reported by evaluate_fdjyp0.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config_experiment8 as config
import evaluate_fdjyp0 as fdjyp0


FIXED_THRESHOLDS = {
    "teacher_7_2": 0.075,
    "student_base": 0.24,
    "student_distilled": 0.32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=config.RESULTS_DIR / "fdjyp0_unseen_82_20260823",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    records = fdjyp0.read_fdjyp0_records()
    holdout = [record for record in records if record["source_split"] == "val"]
    candidates = {
        "teacher_7_2": np.arange(0.025, 0.751, 0.025),
        "student_base": np.arange(0.02, 0.801, 0.02),
        "student_distilled": np.arange(0.02, 0.801, 0.02),
    }
    rows = []
    for method, thresholds in candidates.items():
        probabilities = {
            record["name"]: np.load(
                results / "probabilities" / method / f"{record['name']}.npy",
                allow_pickle=False,
            ).astype(np.float32)
            for record in records
        }
        for raw_threshold in thresholds:
            threshold = float(round(raw_threshold, 3))
            masks, diagnostics = fdjyp0.postprocess(
                records, probabilities, threshold
            )
            all_metrics = fdjyp0.evaluate(records, masks)["macro_scene"]
            holdout_metrics = fdjyp0.evaluate(holdout, masks)["macro_scene"]
            rows.append(
                {
                    "method": method,
                    "threshold": threshold,
                    "is_original_fixed_threshold": abs(
                        threshold - FIXED_THRESHOLDS[method]
                    )
                    < 1e-9,
                    "all82_macro_iou": all_metrics["foreground_iou"],
                    "all82_macro_precision": all_metrics["precision"],
                    "all82_macro_recall": all_metrics["recall"],
                    "all82_macro_boundary_f1": all_metrics["boundary_f1"],
                    "holdout18_macro_iou": holdout_metrics["foreground_iou"],
                    "holdout18_macro_precision": holdout_metrics["precision"],
                    "holdout18_macro_recall": holdout_metrics["recall"],
                    "holdout18_macro_boundary_f1": holdout_metrics["boundary_f1"],
                    "overflow_triggered_count": sum(
                        values["overflow_triggered"]
                        for values in diagnostics.values()
                    ),
                }
            )
    with (results / "threshold_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = {
        method: max(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["all82_macro_iou"],
        )
        for method in candidates
    }
    payload = {
        "warning": (
            "Oracle analysis on FDJYP-0 labels; values are diagnostic and are "
            "not unbiased generalization metrics."
        ),
        "fixed_thresholds": FIXED_THRESHOLDS,
        "best_all82_macro_iou": best,
    }
    (results / "threshold_sweep.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
