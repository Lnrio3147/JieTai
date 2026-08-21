#!/usr/bin/env python3
"""Build a self-contained, numbered PPT asset directory from existing results."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RESULT = EXPERIMENT_ROOT / "results/final_203"
GALLERY = RESULT / "representative_gallery"
DELIVERY = EXPERIMENT_ROOT / "slides"
MAIN = DELIVERY / "01_主图"
DETAIL = DELIVERY / "02_分类单场详图"
CLOUD = DELIVERY / "03_点云"
DATA = DELIVERY / "04_数据与报告"


COPIES = {
    RESULT / "report_assets/metrics_all73.png": MAIN / "01_全73场精度指标.png",
    RESULT / "report_assets/epe_distribution.png": MAIN / "02_逐场EPE分布.png",
    RESULT / "report_assets/runtime.png": MAIN / "03_核心推理速度.png",
    GALLERY / "01_holes_overview.jpg": MAIN / "04_空洞对比.jpg",
    GALLERY / "02_edges_overview.jpg": MAIN / "05_边缘对比.jpg",
    GALLERY / "03_exposure_overview.jpg": MAIN / "06_曝光对比.jpg",
    GALLERY / "04_scale_details_overview.jpg": MAIN / "07_刻度细节对比.jpg",
    RESULT / "report_assets/geometry_audit.png": MAIN / "08_极线几何风险.png",
    RESULT / "report_assets/representative_feasibility.jpg": MAIN / "09_跨数据组可行性案例.jpg",
    RESULT / "report_assets/representative_quantitative.jpg": MAIN / "10_定量代表案例.jpg",
    GALLERY / "gallery_overview.jpg": MAIN / "11_四类案例总览.jpg",
    RESULT / "report_assets/metrics_fixed69.png": MAIN / "13_固定69场精度指标.png",
    GALLERY / "01_holes/01_igev_holes_las_fills__656565-0004.png": DETAIL / "空洞/01_IGEV空洞更多_656565-0004.png",
    GALLERY / "01_holes/02_las_holes_igev_fills__camera-202512281402-0162.png": DETAIL / "空洞/02_LAS空洞更多_camera-0162.png",
    GALLERY / "02_edges/01_las_large_edge_and_global_gain__202506281608-0018.png": DETAIL / "边缘/01_LAS显著改善_0018.png",
    GALLERY / "02_edges/02_igev_edge_counterexample__202506281615-0038.png": DETAIL / "边缘/02_IGEV边缘反例_0038.png",
    GALLERY / "03_exposure/01_severe_underexposure__202506261704-0028.png": DETAIL / "曝光/01_严重欠曝_0028.png",
    GALLERY / "03_exposure/02_metal_highlights__202506281614-0035.png": DETAIL / "曝光/02_金属高亮_0035.png",
    GALLERY / "04_scale_details/01_printed_ticks__camera-202512081522-0005.png": DETAIL / "刻度/01_印刷刻度_0005.png",
    GALLERY / "04_scale_details/02_ticks_and_bosses__camera-202512081732-0018.png": DETAIL / "刻度/02_刻度与圆台_0018.png",
    GALLERY / "04_scale_details/03_agreement_control__camera-202512081522-0004.png": DETAIL / "刻度/03_两模型接近对照_0004.png",
    RESULT / "report_assets/cloud_0018_igev.png": CLOUD / "01_0018_IGEV点云.png",
    RESULT / "report_assets/cloud_0018_las.png": CLOUD / "02_0018_LAS点云.png",
    RESULT / "report_assets/cloud_scale_igev.png": CLOUD / "03_刻度_IGEV点云.png",
    RESULT / "report_assets/cloud_scale_las.png": CLOUD / "04_刻度_LAS点云.png",
    RESULT / "metrics/summary.json": DATA / "summary.json",
    RESULT / "metrics/analysis.json": DATA / "analysis.json",
    RESULT / "metrics/per_scene.csv": DATA / "per_scene.csv",
    GALLERY / "README.md": DATA / "cases.md",
    GALLERY / "01_holes_overview.jpg": DATA / "01_holes_overview.jpg",
    GALLERY / "02_edges_overview.jpg": DATA / "02_edges_overview.jpg",
    GALLERY / "03_exposure_overview.jpg": DATA / "03_exposure_overview.jpg",
    GALLERY / "04_scale_details_overview.jpg": DATA / "04_scale_details_overview.jpg",
    EXPERIMENT_ROOT / "reports/comparison_report.md": DATA / "report.md",
}


def make_cloud_pair(output: Path) -> None:
    sources = [CLOUD / "01_0018_IGEV点云.png", CLOUD / "02_0018_LAS点云.png"]
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in sources]
    height = min(image.shape[0] for image in images)
    resized = [
        cv2.resize(image, (int(round(image.shape[1] * height / image.shape[0])), height), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    gap = np.full((height, 20, 3), (245, 245, 245), dtype=np.uint8)
    cv2.imwrite(str(output), np.hstack((resized[0], gap, resized[1])), [cv2.IMWRITE_JPEG_QUALITY, 92])


def localize_full_report() -> None:
    report = DATA / "report.md"
    text = report.read_text(encoding="utf-8")
    replacements = {
        "metrics_all73.png": "01_全73场精度指标.png",
        "epe_distribution.png": "02_逐场EPE分布.png",
        "runtime.png": "03_核心推理速度.png",
        "geometry_audit.png": "08_极线几何风险.png",
        "representative_feasibility.jpg": "09_跨数据组可行性案例.jpg",
        "representative_quantitative.jpg": "10_定量代表案例.jpg",
        "metrics_fixed69.png": "13_固定69场精度指标.png",
        "01_holes_overview.jpg": "04_空洞对比.jpg",
        "02_edges_overview.jpg": "05_边缘对比.jpg",
        "03_exposure_overview.jpg": "06_曝光对比.jpg",
        "04_scale_details_overview.jpg": "07_刻度细节对比.jpg",
    }
    import re

    for source_name, local_name in replacements.items():
        text = re.sub(
            rf"\([^)]*{re.escape(source_name)}\)",
            f"(../01_主图/{local_name})",
            text,
        )
    text = re.sub(
        r"\([^)]*cloud_scale_igev\.png\)",
        "(../03_点云/03_刻度_IGEV点云.png)",
        text,
    )
    text = re.sub(
        r"\([^)]*cloud_scale_las\.png\)",
        "(../03_点云/04_刻度_LAS点云.png)",
        text,
    )
    report.write_text(text, encoding="utf-8")


def main() -> None:
    for source, destination in COPIES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    make_cloud_pair(MAIN / "12_0018点云对比.jpg")
    localize_full_report()
    print(f"Prepared {len(COPIES) + 1} files under {DELIVERY}")


if __name__ == "__main__":
    main()
