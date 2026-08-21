#!/usr/bin/env python3
"""Build charts, contact sheets, point-cloud previews, and analysis JSON."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RESULT = EXPERIMENT_ROOT / "results/final_203"
METRICS = ("epe", "d1", "bad1", "bad2", "bad3")


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None, "None") else np.nan


def metrics_chart(path, rows, title):
    old = [np.mean([number(row, f"igev_{metric}") for row in rows]) for metric in METRICS]
    new = [np.mean([number(row, f"las_{metric}") for row in rows]) for metric in METRICS]
    labels = ["EPE (px)", "D1 (%)", "Bad1 (%)", "Bad2 (%)", "Bad3 (%)"]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.5, 5.8), dpi=180)
    bars_old = axis.bar(positions - width / 2, old, width, label="IGEV++ RT", color="#68778A")
    bars_new = axis.bar(positions + width / 2, new, width, label="LiteAnyStereo LAS1", color="#2878B5")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Lower is better")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(bars_old, fmt="%.2f", padding=3, fontsize=8)
    axis.bar_label(bars_new, fmt="%.2f", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def runtime_chart(path, summary):
    labels = ["IGEV++ RT", "LiteAnyStereo LAS1"]
    values = [
        summary["timing_all_203"]["igev_rt"]["mean_ms"],
        summary["timing_all_203"]["liteanystereo"]["mean_ms"],
    ]
    figure, axis = plt.subplots(figsize=(7.6, 5.4), dpi=180)
    bars = axis.bar(labels, values, color=["#68778A", "#2878B5"], width=0.58)
    axis.set_ylabel("Core inference latency (ms / stereo pair)")
    axis.set_title("RTX 4090, 1280×720 input, FP32")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.2f ms", padding=4)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def geometry_chart(path, rows):
    groups = ["fdjyp0", "fdjyp3", "luowen", "general_1221", "scale_1221"]
    statuses = ["good", "warning", "high_risk"]
    colors = ["#4C9F70", "#E3A32D", "#C74A4A"]
    bottom = np.zeros(len(groups), dtype=float)
    figure, axis = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    for status, color in zip(statuses, colors):
        values = [sum(row["group"] == group and row["geometry_status"] == status for row in rows) for group in groups]
        axis.bar(groups, values, bottom=bottom, label=status, color=color)
        bottom += values
    axis.set_ylabel("Scene count")
    axis.set_title("Epipolar geometry audit by rec_img_set group")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def epe_distribution(path, rows):
    ordered = sorted(rows, key=lambda row: number(row, "igev_epe"))
    x = np.arange(1, len(ordered) + 1)
    old = [number(row, "igev_epe") for row in ordered]
    new = [number(row, "las_epe") for row in ordered]
    figure, axis = plt.subplots(figsize=(12, 5.8), dpi=180)
    axis.plot(x, old, marker="o", ms=3, lw=1.3, label="IGEV++ RT", color="#68778A")
    axis.plot(x, new, marker="o", ms=3, lw=1.3, label="LiteAnyStereo LAS1", color="#2878B5")
    axis.set_yscale("log")
    axis.set_xlabel("FDJYP-3 scenes sorted by IGEV++ RT EPE")
    axis.set_ylabel("EPE (px, logarithmic scale)")
    axis.set_title("Per-scene error distribution: LiteAnyStereo reduces the failure tail")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def contact_sheet(path, entries, title):
    tile_w, tile_h = 900, 984
    columns = 2
    rows = int(np.ceil(len(entries) / columns))
    label_h = 54
    canvas = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h) + 56), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 16), title, fill="white")
    for index, (label, source) in enumerate(entries):
        image = Image.open(source).convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_w
        y = 56 + (index // columns) * (tile_h + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 10, y + image.height + 12), label, fill="white")
    canvas.save(path, quality=91)


def cloud_preview(path, cloud_path):
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if len(points) > 120_000:
        indices = np.linspace(0, len(points) - 1, 120_000).astype(np.int64)
        points = points[indices]
        colors = colors[indices]
    center = np.median(points, axis=0)
    centered = points - center
    x = centered[:, 0]
    y = -centered[:, 1]
    z = centered[:, 2]
    depth_order = np.argsort(z)
    figure, axis = plt.subplots(figsize=(6.7, 7.7), dpi=180, facecolor="#151820")
    axis.set_facecolor("#151820")
    axis.scatter(x[depth_order], y[depth_order], s=0.35, c=np.clip(colors[depth_order], 0, 1), linewidths=0)
    axis.set_aspect("equal", adjustable="datalim")
    axis.axis("off")
    axis.set_title(cloud_path.parent.parent.name + " / " + cloud_path.parent.name, color="white", fontsize=10)
    figure.tight_layout(pad=0.2)
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    assets = RESULT / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    with (RESULT / "metrics/per_scene.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((RESULT / "metrics/summary.json").read_text(encoding="utf-8"))
    quantitative = [row for row in rows if row["has_foundation_reference"] == "True"]
    fixed = [row for row in quantitative if row["fixed69_status"] == "kept"]

    metrics_chart(assets / "metrics_all73.png", quantitative, "Unified FDJYP-3 all-73 comparison")
    metrics_chart(assets / "metrics_fixed69.png", fixed, "Historical fixed-69 protocol (declared exclusions only)")
    runtime_chart(assets / "runtime.png", summary)
    geometry_chart(assets / "geometry_audit.png", rows)
    epe_distribution(assets / "epe_distribution.png", quantitative)

    output = RESULT / "outputs"
    contact_sheet(
        assets / "representative_quantitative.jpg",
        [
            ("0018: IGEV catastrophic failure; LAS reduces tail error", output / "fdjyp3/202506281608-0018/comparison.png"),
            ("0038: counterexample where IGEV has lower EPE", output / "fdjyp3/202506281615-0038/comparison.png"),
            ("0001: LAS has clearly lower reference error", output / "fdjyp3/202506281603-0001/comparison.png"),
            ("0040: ordinary counterexample where IGEV is better", output / "fdjyp3/202506281616-0040/comparison.png"),
        ],
        "Representative FDJYP-3 quantitative cases",
    )
    contact_sheet(
        assets / "representative_feasibility.jpg",
        [
            ("Scale sample: models agree closely", output / "scale_1221/camera-202512081522-0004/comparison.png"),
            ("Thread sample: high geometry risk and large model disagreement", output / "luowen/656565-0002/comparison.png"),
            ("General sample: high epipolar risk; do not treat cloud as validated", output / "general_1221/camera-202512281401-0161/comparison.png"),
            ("FDJYP-0 sample: cross-scene inference completed", output / "fdjyp0/202506261651-0001/comparison.png"),
        ],
        "Cross-group feasibility and data-quality cases",
    )

    preview_jobs = [
        ("cloud_0018_igev.png", output / "fdjyp3/202506281608-0018/igev_rt/cloud.ply"),
        ("cloud_0018_las.png", output / "fdjyp3/202506281608-0018/liteanystereo/cloud.ply"),
        ("cloud_scale_igev.png", output / "scale_1221/camera-202512081522-0004/igev_rt/cloud.ply"),
        ("cloud_scale_las.png", output / "scale_1221/camera-202512081522-0004/liteanystereo/cloud.ply"),
    ]
    for filename, source in preview_jobs:
        cloud_preview(assets / filename, source)

    epe = {algorithm: np.asarray([number(row, f"{algorithm}_epe") for row in quantitative]) for algorithm in ("igev", "las")}
    delta = epe["igev"] - epe["las"]
    sorted_improvements = sorted(
        [
            {
                "scene": row["scene"],
                "igev_epe": number(row, "igev_epe"),
                "las_epe": number(row, "las_epe"),
                "las_improvement_px": number(row, "igev_epe") - number(row, "las_epe"),
            }
            for row in quantitative
        ],
        key=lambda item: item["las_improvement_px"],
        reverse=True,
    )
    group_analysis = {}
    for group in sorted(set(row["group"] for row in rows)):
        group_rows = [row for row in rows if row["group"] == group]
        group_analysis[group] = {
            "scene_count": len(group_rows),
            "geometry": dict(Counter(row["geometry_status"] for row in group_rows)),
            "median_vertical_residual_px": float(np.median([number(row, "median_vertical_residual_px") for row in group_rows])),
            "mean_inter_model_mae_px": float(np.mean([number(row, "inter_model_mae_px") for row in group_rows])),
            "median_inter_model_mae_px": float(np.median([number(row, "inter_model_mae_px") for row in group_rows])),
            "mean_inter_model_correlation": float(np.mean([number(row, "inter_model_correlation") for row in group_rows])),
            "mean_igev_cloud_points": float(np.mean([number(row, "igev_cloud_points") for row in group_rows])),
            "mean_las_cloud_points": float(np.mean([number(row, "las_cloud_points") for row in group_rows])),
        }
    analysis = {
        "all73": {
            "igev_epe_std": float(epe["igev"].std()),
            "las_epe_std": float(epe["las"].std()),
            "igev_epe_p90": float(np.percentile(epe["igev"], 90)),
            "las_epe_p90": float(np.percentile(epe["las"], 90)),
            "igev_epe_p95": float(np.percentile(epe["igev"], 95)),
            "las_epe_p95": float(np.percentile(epe["las"], 95)),
            "igev_scenes_epe_over_10": int((epe["igev"] > 10).sum()),
            "las_scenes_epe_over_10": int((epe["las"] > 10).sum()),
            "igev_scenes_epe_le_3": int((epe["igev"] <= 3).sum()),
            "las_scenes_epe_le_3": int((epe["las"] <= 3).sum()),
            "both_epe_le_10_scene_count": int(((epe["igev"] <= 10) & (epe["las"] <= 10)).sum()),
            "both_epe_le_10_igev_mean": float(epe["igev"][(epe["igev"] <= 10) & (epe["las"] <= 10)].mean()),
            "both_epe_le_10_las_mean": float(epe["las"][(epe["igev"] <= 10) & (epe["las"] <= 10)].mean()),
            "las_mean_epe_reduction_px": float(epe["igev"].mean() - epe["las"].mean()),
            "las_mean_epe_reduction_pct": float(100 * (epe["igev"].mean() - epe["las"].mean()) / epe["igev"].mean()),
            "top_las_improvements": sorted_improvements[:10],
            "top_igev_improvements": list(reversed(sorted_improvements[-10:])),
        },
        "group_analysis": group_analysis,
        "memory_benchmark": {
            "scope": "separate fresh process per model; after one warm-up; torch CUDA allocator",
            "igev_rt": {
                "peak_allocated_mib": 711.02783203125,
                "peak_reserved_mib": 944.0,
                "incremental_peak_allocated_mib": 694.90625,
            },
            "liteanystereo": {
                "peak_allocated_mib": 658.8662109375,
                "peak_reserved_mib": 922.0,
                "incremental_peak_allocated_mib": 597.60546875,
            },
        },
    }
    (RESULT / "metrics/analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
