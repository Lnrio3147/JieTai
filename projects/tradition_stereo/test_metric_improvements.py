#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的视差评估指标函数
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import torch
    import numpy as np
    from metric.cal_metric import (
        d1_metric, threshold_metric, epe_metric,
        epe_metric_no_filter, get_disp_statistics
    )
    print("[OK] 成功导入所有metric函数")
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)


def create_test_data():
    """创建测试数据"""
    # 创建一个简单的测试视差图
    batch_size, height, width = 1, 100, 100

    # 正常视差范围 (0-10)
    normal_disp = torch.ones(batch_size, height, width) * 5.0

    # 添加一些大视差值 (>10)
    large_disp = torch.ones(batch_size, height, width) * 15.0

    # 混合数据：左上角正常视差，右下角大视差
    gt_disp = torch.zeros(batch_size, height, width)
    gt_disp[:, :height//2, :width//2] = normal_disp[:, :height//2, :width//2]  # 正常区域
    gt_disp[:, height//2:, width//2:] = large_disp[:, height//2:, width//2:]  # 大视差区域

    # 预测视差：在正常区域添加一些误差
    pred_disp = gt_disp.clone()
    pred_disp[:, :height//2, :width//2] += torch.randn(batch_size, height//2, width//2) * 0.5  # 添加噪声
    pred_disp[:, height//2:, width//2:] += torch.randn(batch_size, height//2, width//2) * 2.0  # 大误差

    # 创建掩码
    mask = torch.ones(batch_size, height, width, dtype=torch.bool)

    return gt_disp, pred_disp, mask


def test_metrics():
    """测试指标计算"""
    print("\n=== 测试改进后的视差评估指标 ===")

    # 创建测试数据
    gt_disp, pred_disp, mask = create_test_data()
    print(f"测试数据形状: {gt_disp.shape}")
    print(f"真值视差范围: {gt_disp[mask].min():.2f} ~ {gt_disp[mask].max():.2f}")
    print(f"预测视差范围: {pred_disp[mask].min():.2f} ~ {pred_disp[mask].max():.2f}")

    # 获取统计信息
    stats = get_disp_statistics(gt_disp, pred_disp, mask)
    print(f"\n=== 视差统计信息 ===")
    print(f"总像素数: {stats['total_pixels']}")
    print(f"有效像素数 (0 < disp <= 10): {stats['valid_pixels']}")
    print(f"过滤像素数 (disp > 10): {stats['filtered_pixels']}")
    print(f"有效像素比例: {stats['valid_ratio']:.2f}%")
    print(f"真值视差范围: {stats['gt_disp_range']['min']:.2f} ~ {stats['gt_disp_range']['max']:.2f}")
    print(f"过滤视差范围: {stats['filtered_disp_range']['min']:.2f} ~ {stats['filtered_disp_range']['max']:.2f}")

    # 测试EPE指标对比
    print(f"\n=== EPE指标对比 ===")
    epe_filtered = epe_metric(pred_disp, pred_disp + 1.0, mask)  # 添加1像素误差
    epe_no_filter = epe_metric_no_filter(pred_disp, pred_disp + 1.0, mask)

    print(f"EPE (过滤大视差): {epe_filtered.item():.4f}")
    print(f"EPE (不过滤): {epe_no_filter.item():.4f}")

    # 测试其他指标
    print(f"\n=== 其他指标 ===")
    d1 = d1_metric(pred_disp, pred_disp + 3.1, mask)  # 添加3.1像素误差
    bad1 = threshold_metric(pred_disp, pred_disp + 1.5, mask, 1.0)
    bad2 = threshold_metric(pred_disp, pred_disp + 2.5, mask, 2.0)
    bad3 = threshold_metric(pred_disp, pred_disp + 3.5, mask, 3.0)

    print(f"D1误差率: {d1.item():.2f}%")
    print(f"Bad1误差率 (>1px): {bad1.item():.2f}%")
    print(f"Bad2误差率 (>2px): {bad2.item():.2f}%")
    print(f"Bad3误差率 (>3px): {bad3.item():.2f}%")


def test_edge_cases():
    """测试边界情况"""
    print(f"\n=== 测试边界情况 ===")

    # 测试空掩码
    empty_mask = torch.zeros(1, 10, 10, dtype=torch.bool)
    gt_disp = torch.ones(1, 10, 10) * 5.0
    pred_disp = torch.ones(1, 10, 10) * 5.0

    epe = epe_metric(pred_disp, gt_disp, empty_mask)
    print(f"空掩码EPE: {epe.item():.4f} (应该为0)")

    # 测试全大视差
    large_disp = torch.ones(1, 10, 10) * 15.0
    full_mask = torch.ones(1, 10, 10, dtype=torch.bool)
    epe_large = epe_metric(large_disp, large_disp + 1.0, full_mask)
    print(f"全大视差EPE: {epe_large.item():.4f} (应该为0，因为全被过滤)")


if __name__ == "__main__":
    test_metrics()
    test_edge_cases()
    print(f"\n=== 测试完成 ===")
    print("所有指标函数现在都会过滤掉视差值 > 10 的像素，避免引入大的误差。")