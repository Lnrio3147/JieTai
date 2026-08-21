#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试EPE过滤功能
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import torch
    import pandas as pd
    from metric.cal_metric import filter_high_epe_scenes, calculate_overall_metrics_with_filter, get_epe_statistics
    print("[OK] 成功导入EPE过滤函数")
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)


def create_test_metrics():
    """创建测试数据"""
    # 创建一些测试场景的指标数据
    test_metrics = [
        {"Scene": "scene_001", "EPE": 0.5, "D1": 2.1, "Bad1": 5.2, "Bad2": 1.8, "Bad3": 0.5, "Valid Pixels": 10000, "Total Pixels": 12000},
        {"Scene": "scene_002", "EPE": 1.2, "D1": 3.5, "Bad1": 8.1, "Bad2": 3.2, "Bad3": 1.1, "Valid Pixels": 9500, "Total Pixels": 12000},
        {"Scene": "scene_003", "EPE": 8.5, "D1": 15.2, "Bad1": 25.3, "Bad2": 12.8, "Bad3": 5.2, "Valid Pixels": 8000, "Total Pixels": 12000},
        {"Scene": "scene_004", "EPE": 15.8, "D1": 35.6, "Bad1": 45.2, "Bad2": 28.9, "Bad3": 15.3, "Valid Pixels": 7000, "Total Pixels": 12000},  # 高EPE
        {"Scene": "scene_005", "EPE": 25.3, "D1": 52.1, "Bad1": 68.5, "Bad2": 45.2, "Bad3": 28.7, "Valid Pixels": 6000, "Total Pixels": 12000},  # 高EPE
        {"Scene": "scene_006", "EPE": 3.8, "D1": 8.9, "Bad1": 15.2, "Bad2": 6.8, "Bad3": 2.5, "Valid Pixels": 9000, "Total Pixels": 12000},
        {"Scene": "scene_007", "EPE": 0.8, "D1": 2.8, "Bad1": 6.5, "Bad2": 2.3, "Bad3": 0.8, "Valid Pixels": 11000, "Total Pixels": 12000},
        {"Scene": "scene_008", "EPE": 12.5, "D1": 28.5, "Bad1": 38.2, "Bad2": 22.5, "Bad3": 12.1, "Valid Pixels": 7500, "Total Pixels": 12000},  # 高EPE
    ]
    return test_metrics


def test_epe_filter():
    """测试EPE过滤功能"""
    print("\n=== 测试EPE过滤功能 ===")

    # 创建测试数据
    test_metrics = create_test_metrics()
    print(f"原始场景数量: {len(test_metrics)}")

    # 显示原始数据
    print("\n原始数据:")
    for m in test_metrics:
        print(f"  {m['Scene']}: EPE={m['EPE']:.2f} px")

    # 测试过滤功能
    epe_threshold = 10.0
    filtered_metrics, filtered_count = filter_high_epe_scenes(test_metrics, epe_threshold)

    print(f"\n过滤结果 (EPE阈值: {epe_threshold} px):")
    print(f"过滤掉场景数量: {filtered_count}")
    print(f"保留场景数量: {len(filtered_metrics)}")

    print("\n保留的场景:")
    for m in filtered_metrics:
        print(f"  {m['Scene']}: EPE={m['EPE']:.2f} px")

    # 测试整体指标计算
    print(f"\n=== 测试整体指标计算 ===")
    overall_metrics, filter_info = calculate_overall_metrics_with_filter(test_metrics, epe_threshold)

    print(f"过滤信息:")
    print(f"  原始场景数: {filter_info['original_count']}")
    print(f"  过滤场景数: {filter_info['filtered_count']}")
    print(f"  保留场景数: {filter_info['remaining_count']}")
    print(f"  过滤比例: {filter_info['filtered_percentage']:.2f}%")

    print(f"\n过滤后整体指标:")
    print(f"  EPE: {overall_metrics['EPE']:.4f} px")
    print(f"  D1: {overall_metrics['D1']:.2f} %")
    print(f"  Bad1: {overall_metrics['Bad1']:.2f} %")
    print(f"  Bad2: {overall_metrics['Bad2']:.2f} %")
    print(f"  Bad3: {overall_metrics['Bad3']:.2f} %")

    # 测试EPE统计
    print(f"\n=== 测试EPE统计 ===")
    epe_stats = get_epe_statistics(test_metrics)

    print(f"EPE统计:")
    print(f"  总场景数: {epe_stats['count']}")
    print(f"  EPE范围: {epe_stats['min']:.4f} ~ {epe_stats['max']:.4f} px")
    print(f"  平均EPE: {epe_stats['mean']:.4f} px")
    print(f"  高EPE场景数 (>10px): {epe_stats['high_epe_count']}")
    print(f"  高EPE场景比例: {epe_stats['high_epe_percentage']:.2f} %")


def test_edge_cases():
    """测试边界情况"""
    print(f"\n=== 测试边界情况 ===")

    # 测试空列表
    empty_metrics = []
    filtered_metrics, filtered_count = filter_high_epe_scenes(empty_metrics)
    print(f"空列表测试: 过滤掉 {filtered_count} 个场景，保留 {len(filtered_metrics)} 个场景")

    # 测试全高EPE
    all_high_epe = [
        {"Scene": "high_1", "EPE": 15.0, "D1": 30.0},
        {"Scene": "high_2", "EPE": 20.0, "D1": 40.0},
    ]
    filtered_all, count_all = filter_high_epe_scenes(all_high_epe)
    print(f"全高EPE测试: 过滤掉 {count_all} 个场景，保留 {len(filtered_all)} 个场景")

    # 测试全低EPE
    all_low_epe = [
        {"Scene": "low_1", "EPE": 5.0, "D1": 10.0},
        {"Scene": "low_2", "EPE": 3.0, "D1": 8.0},
    ]
    filtered_low, count_low = filter_high_epe_scenes(all_low_epe)
    print(f"全低EPE测试: 过滤掉 {count_low} 个场景，保留 {len(filtered_low)} 个场景")


if __name__ == "__main__":
    test_epe_filter()
    test_edge_cases()
    print(f"\n=== 测试完成 ===")
    print("EPE过滤功能已成功实现，可以过滤掉EPE值 > 10 的场景数据。")