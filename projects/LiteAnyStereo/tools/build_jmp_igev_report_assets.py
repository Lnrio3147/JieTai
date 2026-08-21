#!/usr/bin/env python3
"""Build a report from the results already saved in tradition_stereo.

This script does not run or reimplement RT-IGEV.  Its old-model numbers are
read verbatim from ``evaluation_results/IGEV_metrics.csv`` and are joined with
the saved LiteAnyStereo per-scene metrics on their common scene names.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXCLUDED_SCENES = {
    "202506281607-0012",
    "202506281609-0019",
    "202506281609-0020",
    "202506281619-0053",
}
METRIC_KEYS = ("epe", "d1", "bad1", "bad2", "bad3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tradition_root", default="../tradition_stereo")
    parser.add_argument("--evaluation_dir", default="./runs/evaluation/jmp_official_fixed69")
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def aggregate(rows, *, original_count):
    kept = [row for row in rows if row["status"] == "common"]
    result = {key: float(np.mean([row[key] for row in kept])) for key in METRIC_KEYS}
    result.update(
        {
            "algorithm": "RT-IGEV (phase-I saved result)",
            "source": "../tradition_stereo/datasets/FDJYP-3/evaluation_results/IGEV_metrics.csv",
            "visual_source": "../tradition_stereo/igev_output/<scene>/vis.png",
            "scene_count": len(kept),
            "original_scene_count": original_count,
            "excluded_or_unmatched_scene_count": original_count - len(kept),
            "excluded_scenes": sorted(EXCLUDED_SCENES),
            "aggregation": "scene_macro",
            "reference": "Foundation Stereo disp_cropped.npy",
            "epe_filter": "Already applied when tradition_stereo/IGEV_metrics.csv was saved",
            "comparison_rule": "intersection with LiteAnyStereo kept scenes after the four fixed exclusions",
            "valid_pixels": int(sum(row["valid_pixels"] for row in kept)),
            "total_pixels": int(sum(row["total_pixels"] for row in kept)),
            "valid_ratio": float(np.mean([row["valid_ratio"] for row in kept])),
        }
    )
    return result


def create_chart(path: Path, previous, current):
    labels = ["EPE (px)", "D1 (%)", "Bad1 (%)", "Bad2 (%)", "Bad3 (%)"]
    previous_values = [previous[key] for key in METRIC_KEYS]
    current_values = [current[key] for key in METRIC_KEYS]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.5, 5.8), dpi=180)
    previous_bars = axis.bar(
        positions - width / 2,
        previous_values,
        width,
        label="RT-IGEV (phase-I)",
        color="#7B8794",
    )
    current_bars = axis.bar(
        positions + width / 2,
        current_values,
        width,
        label="LiteAnyStereo LAS1",
        color="#2878B5",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Lower is better")
    axis.set_title("Common 68-scene comparison using saved tradition_stereo results")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(previous_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.bar_label(current_bars, fmt="%.2f", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    tradition_root = Path(args.tradition_root).expanduser().resolve()
    evaluation_dir = Path(args.evaluation_dir).expanduser().resolve()
    metrics_dir = evaluation_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    saved_metrics_path = (
        tradition_root / "datasets/FDJYP-3/evaluation_results/IGEV_metrics.csv"
    )
    saved_rows = read_csv(saved_metrics_path)
    if not saved_rows:
        raise ValueError(f"No saved RT-IGEV metrics found in {saved_metrics_path}")

    las_rows = read_csv(metrics_dir / "liteanystereo_scene_metrics.csv")
    las_kept = {row["scene"]: row for row in las_rows if row["status"] == "kept"}
    rows = []
    for saved in saved_rows:
        scene = saved["Scene"]
        status = "common" if scene in las_kept and scene not in EXCLUDED_SCENES else "not_common"
        rows.append(
            {
                "scene": scene,
                "epe": float(saved["EPE"]),
                "d1": float(saved["D1"]),
                "bad1": float(saved["Bad1"]),
                "bad2": float(saved["Bad2"]),
                "bad3": float(saved["Bad3"]),
                "valid_pixels": int(saved["Valid Pixels"]),
                "total_pixels": int(saved["Total Pixels"]),
                "valid_ratio": float(saved["Valid Ratio"]),
                "status": status,
            }
        )
    summary = aggregate(rows, original_count=len(saved_rows))

    # Preserve the previously generated customer/historical point-cloud metrics
    # under an explicit name before replacing the report comparison target.
    previous_summary_path = metrics_dir / "previous_algorithm_summary.json"
    previous_csv_path = metrics_dir / "previous_algorithm_scene_metrics.csv"
    if previous_summary_path.is_file():
        previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("algorithm") == "historical_pointcloud_disparity":
            previous_summary_path.replace(metrics_dir / "customer_historical_summary.json")
            if previous_csv_path.is_file():
                previous_csv_path.replace(metrics_dir / "customer_historical_scene_metrics.csv")

    write_csv(metrics_dir / "igev_rt_scene_metrics.csv", rows)
    write_json(metrics_dir / "igev_rt_summary.json", summary)

    common_names = {row["scene"] for row in rows if row["status"] == "common"}
    las_common = [las_kept[name] for name in sorted(common_names)]
    current = {key: float(np.mean([float(row[key]) for row in las_common])) for key in METRIC_KEYS}
    current.update(
        {
            "label": "official",
            "checkpoint": str((Path.cwd() / "checkpoints/LiteAnyStereo.pth").resolve()),
            "scene_count": len(las_common),
            "valid_pixels": int(sum(int(row["valid_pixels"]) for row in las_common)),
            "total_pixels": int(sum(int(row["total_pixels"]) for row in las_common)),
            "valid_ratio": float(np.mean([float(row["valid_ratio"]) for row in las_common])),
            "comparison_rule": "same common scenes as saved RT-IGEV metrics",
        }
    )
    if current["scene_count"] != summary["scene_count"]:
        raise ValueError("RT-IGEV and LAS1 scene sets differ")

    with (evaluation_dir / "evaluation_summary.json").open(encoding="utf-8") as handle:
        evaluation_summary = json.load(handle)
    evaluation_summary["comparison_protocol"] = {
        "source": "saved tradition_stereo metrics; no RT-IGEV rerun",
        "rt_igev_metrics": str(saved_metrics_path),
        "liteanystereo_metrics": str(metrics_dir / "liteanystereo_scene_metrics.csv"),
        "scene_join": "intersection by scene name after the four fixed exclusions",
        "scene_count": len(common_names),
    }
    evaluation_summary["previous_algorithm"] = summary
    evaluation_summary["liteanystereo_official"] = current
    evaluation_summary.pop("customer_historical_result_not_primary_baseline", None)
    write_json(evaluation_dir / "evaluation_summary.json", evaluation_summary)
    create_chart(evaluation_dir / "performance_comparison.png", summary, current)

    scene_epe = {row["scene"]: row["epe"] for row in rows if row["status"] == "common"}
    las_epe = {row["scene"]: float(row["epe"]) for row in las_common}
    las_wins = sum(las_epe[name] < scene_epe[name] for name in scene_epe)
    summary_output = {
        "rt_igev": summary,
        "liteanystereo": current,
        "las_epe_win_scenes": las_wins,
        "rt_igev_epe_win_scenes": len(scene_epe) - las_wins,
    }
    write_json(metrics_dir / "igev_vs_liteanystereo_summary.json", summary_output)
    print(json.dumps(summary_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
