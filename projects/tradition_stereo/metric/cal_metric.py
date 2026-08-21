# 视差评估指标函数
# 原始指标函数，用于计算单个场景的视差评估指标

import numpy as np
import cv2
import open3d as o3d
import torch
import pandas as pd


def d1_metric(disp_pred, disp_gt, mask):
    """
    D1指标：计算误差 > 3px 且相对误差 > 5% 的像素比例
    """
    E = torch.abs(disp_gt - disp_pred)
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)
    err_mask = err_mask & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid = mask.sum(dim=[1, 2])
    d1 = torch.where(num_valid > 0,
                     num_errors.float() / num_valid.float() * 100,
                     torch.zeros_like(num_errors, dtype=torch.float))
    return d1


def threshold_metric(disp_pred, disp_gt, mask, threshold):
    """
    阈值指标：计算误差 > threshold 的像素比例
    """
    E = torch.abs(disp_gt - disp_pred)
    err_mask = (E > threshold) & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid = mask.sum(dim=[1, 2])
    bad = torch.where(num_valid > 0,
                      num_errors.float() / num_valid.float() * 100,
                      torch.zeros_like(num_errors, dtype=torch.float))
    return bad


def epe_metric(disp_pred, disp_gt, mask):
    """
    EPE (End Point Error) 指标：计算平均端点误差
    """
    E = torch.abs(disp_gt - disp_pred)
    E_masked = torch.where(mask, E, torch.zeros_like(E))
    E_sum = E_masked.sum(dim=[1, 2])
    num_valid = mask.sum(dim=[1, 2])
    epe = torch.where(num_valid > 0,
                      E_sum.float() / num_valid.float(),
                      torch.zeros_like(E_sum, dtype=torch.float))
    return epe


def filter_high_epe_scenes(metrics_list, epe_threshold=10.0):
    """
    过滤掉EPE值过高的场景数据

    参数:
        metrics_list: 包含多个场景评估指标的列表，每个元素是包含指标的字典
        epe_threshold: EPE阈值，默认为10.0

    返回:
        filtered_metrics: 过滤后的指标列表
        filtered_count: 被过滤的场景数量
    """
    if not metrics_list:
        return [], 0

    original_count = len(metrics_list)
    filtered_metrics = []

    for metrics in metrics_list:
        epe_value = metrics.get('EPE', 0.0)

        # 如果EPE值小于等于阈值，保留该场景
        if epe_value <= epe_threshold:
            filtered_metrics.append(metrics)

    filtered_count = original_count - len(filtered_metrics)

    return filtered_metrics, filtered_count


def calculate_overall_metrics_with_filter(metrics_list, epe_threshold=10.0):
    """
    计算整体指标，并过滤掉EPE值过高的场景

    参数:
        metrics_list: 包含多个场景评估指标的列表
        epe_threshold: EPE阈值，默认为10.0

    返回:
        overall_metrics: 整体指标
        filter_info: 过滤信息
    """
    if not metrics_list:
        return {}, {'original_count': 0, 'filtered_count': 0, 'filtered_percentage': 0.0}

    # 过滤高EPE场景
    filtered_metrics, filtered_count = filter_high_epe_scenes(metrics_list, epe_threshold)

    original_count = len(metrics_list)
    filtered_percentage = (filtered_count / original_count * 100.0) if original_count > 0 else 0.0

    # 计算过滤后的整体指标
    if filtered_metrics:
        df_metrics = pd.DataFrame(filtered_metrics)
        overall_metrics = {
            'EPE': df_metrics['EPE'].mean(),
            'D1': df_metrics['D1'].mean(),
            'Bad1': df_metrics['Bad1'].mean(),
            'Bad2': df_metrics['Bad2'].mean(),
            'Bad3': df_metrics['Bad3'].mean(),
            'Valid Pixels': df_metrics['Valid Pixels'].sum(),
            'Total Pixels': df_metrics['Total Pixels'].sum(),
            'Scene Count': len(filtered_metrics)
        }
    else:
        overall_metrics = {
            'EPE': 0.0, 'D1': 0.0, 'Bad1': 0.0, 'Bad2': 0.0, 'Bad3': 0.0,
            'Valid Pixels': 0, 'Total Pixels': 0, 'Scene Count': 0
        }

    filter_info = {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'remaining_count': len(filtered_metrics),
        'filtered_percentage': filtered_percentage,
        'epe_threshold': epe_threshold
    }

    return overall_metrics, filter_info


def get_epe_statistics(metrics_list):
    """
    获取EPE统计信息，帮助分析EPE分布

    参数:
        metrics_list: 包含多个场景评估指标的列表

    返回:
        stats: EPE统计信息
    """
    if not metrics_list:
        return {}

    epe_values = [m.get('EPE', 0.0) for m in metrics_list]

    stats = {
        'count': len(epe_values),
        'min': min(epe_values),
        'max': max(epe_values),
        'mean': sum(epe_values) / len(epe_values),
        'high_epe_count': sum(1 for epe in epe_values if epe > 10.0),
        'high_epe_percentage': sum(1 for epe in epe_values if epe > 10.0) / len(epe_values) * 100.0
    }

    return stats
