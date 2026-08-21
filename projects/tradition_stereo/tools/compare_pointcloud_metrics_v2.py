"""
点云滤波效果对比脚本（修正版）
- GT视差图来自: D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3
- 点云文件来自: D:\Desktop\原始ply点云\原始ply点云

将不同滤波参数的点云转换为视差图，并与ground truth进行对比

作者: Claude
日期: 2025-12-15
"""

import numpy as np
import cv2
import open3d as o3d
import os
import glob
import torch
import pandas as pd
from tabulate import tabulate
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric.cal_metric import d1_metric, threshold_metric, epe_metric

# ==================== 配置参数 ====================
# 数据路径
POINTCLOUD_DIR = r"D:\Desktop\原始ply点云\原始ply点云"  # 点云文件夹
GT_DIR = r"D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3"  # GT视差图文件夹
RESULTS_DIR = os.path.join(os.path.dirname(POINTCLOUD_DIR), "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Q矩阵（工件相机配置，来自stereo_gongjian.yml）
Q_MATRIX = np.array([
    [1., 0., 0., -3.1274110031127930e+02],
    [0., 1., 0., -6.6352561950683594e+02],
    [0., 0., 0., 8.7770274018144380e+02],
    [0., 0., 3.9768562155237530e-01, 0.]
])

# 要对比的点云文件名
FILTER_METHODS = [
    "morph_kernel7_gauss7",
    "morph_kernel15_gauss7",
    "morph_kernel31_gauss7"
]

GT_FILENAME = "disp_cropped.npy"  # Ground Truth视差图文件名
# ================================================


def pointcloud_to_disparity(ply_path, reference_shape, Q_matrix):
    """
    将点云文件转换为视差图

    参数:
        ply_path: 点云文件路径
        reference_shape: 参考视差图的形状 (H, W)
        Q_matrix: Q矩阵，用于3D到视差的转换

    返回:
        disparity: 视差图 (H, W)，无效位置填充0
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)  # (N, 3) -> [X, Y, Z]

    if len(points) == 0:
        print(f"    警告: 点云为空 {ply_path}")
        return np.zeros(reference_shape, dtype=np.float32)

    # 提取3D坐标
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # 根据cv2.reprojectImageTo3D的Q矩阵定义：
    # Z = f / W, W = -d/Tx，所以 Z = -f*Tx/d = f*B/d
    # 因此 d = f*B/Z，其中 f*B = Q[2,3] / Q[3,2]
    cx = -Q_matrix[0, 3]
    cy = -Q_matrix[1, 3]
    f = Q_matrix[2, 3]
    focal_baseline = Q_matrix[2, 3] / Q_matrix[3, 2]  # f * B

    # 反推像素坐标
    # x = X * f / Z + cx
    # y = Y * f / Z + cy
    x = X * f / Z + cx
    y = Y * f / Z + cy

    # 反推视差
    # d = f*B / Z
    disparity_values = focal_baseline / Z

    # 创建空白视差图
    H, W = reference_shape
    disparity = np.zeros((H, W), dtype=np.float32)

    # 投影到像素坐标
    x_int = np.round(x).astype(np.int32)
    y_int = np.round(y).astype(np.int32)

    # 过滤超出边界的点
    valid_mask = (x_int >= 0) & (x_int < W) & (y_int >= 0) & (y_int < H) & (Z > 0)

    x_valid = x_int[valid_mask]
    y_valid = y_int[valid_mask]
    d_valid = disparity_values[valid_mask]

    # 填充视差图（如果有多个点映射到同一像素，取平均值）
    pixel_coords = y_valid * W + x_valid
    unique_pixels, inverse_indices = np.unique(pixel_coords, return_inverse=True)

    # 对每个唯一像素，计算平均视差
    for i, pixel_idx in enumerate(unique_pixels):
        mask = (inverse_indices == i)
        avg_disparity = np.mean(d_valid[mask])
        y_pix = pixel_idx // W
        x_pix = pixel_idx % W
        disparity[y_pix, x_pix] = avg_disparity

    return disparity


def evaluate_scene(scene_name, pointcloud_scene_dir, gt_scene_dir):
    """
    评估单个场景的点云滤波效果

    参数:
        scene_name: 场景名称
        pointcloud_scene_dir: 点云文件夹路径
        gt_scene_dir: GT视差图文件夹路径

    返回:
        results: 每个滤波方法的指标字典列表
    """
    print(f"\n评估场景: {scene_name}")

    # 检查ground truth视差图是否存在
    gt_disp_path = os.path.join(gt_scene_dir, GT_FILENAME)
    if not os.path.exists(gt_disp_path):
        print(f"  警告: 未找到GT视差图 {GT_FILENAME}")
        return None

    # 加载GT视差图
    gt_disp = np.load(gt_disp_path).astype(np.float32)
    reference_shape = gt_disp.shape
    print(f"  GT视差图尺寸: {reference_shape}")

    # 创建有效像素掩码（GT视差>0的区域）
    valid_mask = (gt_disp > 0)
    valid_pixels = valid_mask.sum()

    if valid_pixels == 0:
        print(f"  警告: GT视差图无有效像素")
        return None

    print(f"  GT有效像素数: {valid_pixels:,} / {gt_disp.size:,} ({valid_pixels/gt_disp.size*100:.2f}%)")
    print(f"  GT视差范围: {gt_disp[valid_mask].min():.2f} ~ {gt_disp[valid_mask].max():.2f}")

    # 评估每个滤波方法
    results = []
    for method_name in FILTER_METHODS:
        ply_filename = f"{method_name}.ply"
        ply_path = os.path.join(pointcloud_scene_dir, ply_filename)

        if not os.path.exists(ply_path):
            print(f"  警告: 未找到 {ply_filename}，跳过")
            continue

        print(f"  评估方法: {method_name}")

        # 转换预测点云为视差图
        pred_disp = pointcloud_to_disparity(ply_path, reference_shape, Q_MATRIX)

        # 统计预测视差的有效像素
        pred_valid_mask = (pred_disp > 0)
        pred_valid_pixels = pred_valid_mask.sum()
        print(f"    预测有效像素数: {pred_valid_pixels:,} ({pred_valid_pixels/pred_disp.size*100:.2f}%)")

        if pred_valid_pixels > 0:
            print(f"    预测视差范围: {pred_disp[pred_valid_mask].min():.2f} ~ {pred_disp[pred_valid_mask].max():.2f}")

        # 转换为PyTorch张量
        pred_tensor = torch.from_numpy(pred_disp).unsqueeze(0)  # (1, H, W)
        gt_tensor = torch.from_numpy(gt_disp).unsqueeze(0)
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0)

        # 计算指标
        try:
            metrics = {
                "Scene": scene_name,
                "Method": method_name,
                "EPE": epe_metric(pred_tensor, gt_tensor, mask_tensor).item(),
                "D1": d1_metric(pred_tensor, gt_tensor, mask_tensor).item(),
                "Bad1": threshold_metric(pred_tensor, gt_tensor, mask_tensor, threshold=1.0).item(),
                "Bad2": threshold_metric(pred_tensor, gt_tensor, mask_tensor, threshold=2.0).item(),
                "Bad3": threshold_metric(pred_tensor, gt_tensor, mask_tensor, threshold=3.0).item(),
                "Valid Pixels": valid_pixels,
                "Total Pixels": pred_disp.size,
                "Valid Ratio": valid_pixels / pred_disp.size * 100
            }

            print(f"    EPE: {metrics['EPE']:.4f} px")
            print(f"    D1: {metrics['D1']:.2f} %")

            results.append(metrics)

        except Exception as e:
            print(f"    计算指标时出错: {str(e)}")
            continue

    return results


def visualize_results(all_results):
    """
    可视化结果并生成统计报告

    参数:
        all_results: 所有场景的评估结果列表
    """
    if not all_results:
        print("错误: 没有可用的评估结果")
        return

    # 转换为DataFrame
    df = pd.DataFrame(all_results)

    # 按方法分组统计
    print("\n" + "=" * 80)
    print("按滤波方法分组的统计结果")
    print("=" * 80)

    summary_data = []
    for method in FILTER_METHODS:
        method_df = df[df["Method"] == method]

        if len(method_df) == 0:
            continue

        summary = {
            "Method": method,
            "Scenes": len(method_df),
            "EPE (mean)": method_df["EPE"].mean(),
            "EPE (std)": method_df["EPE"].std(),
            "D1 (mean)": method_df["D1"].mean(),
            "Bad1 (mean)": method_df["Bad1"].mean(),
            "Bad2 (mean)": method_df["Bad2"].mean(),
            "Bad3 (mean)": method_df["Bad3"].mean(),
        }
        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)

    # 打印汇总表格
    table = tabulate(
        summary_df,
        headers=["滤波方法", "场景数", "EPE均值", "EPE标准差", "D1(%)", "Bad1(%)", "Bad2(%)", "Bad3(%)"],
        floatfmt=(".0s", ".0f", ".4f", ".4f", ".2f", ".2f", ".2f", ".2f"),
        tablefmt="grid"
    )
    print(table)

    # 生成详细报告
    report = f"""
点云滤波效果对比评估报告
========================
日期: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
点云数据集: {POINTCLOUD_DIR}
GT数据集: {GT_DIR}
评估场景总数: {len(df['Scene'].unique())}
滤波方法数: {len(FILTER_METHODS)}

方法汇总:
{table}

最佳方法（按EPE排序）:
{summary_df.nsmallest(3, 'EPE (mean)')[['Method', 'EPE (mean)', 'D1 (mean)']].to_string(index=False)}

详细场景结果:
"""

    # 添加每个场景的详细表格
    detailed_table = tabulate(
        df[["Scene", "Method", "EPE", "D1", "Bad1", "Bad2", "Bad3"]],
        headers=["场景", "方法", "EPE(px)", "D1(%)", "Bad1(%)", "Bad2(%)", "Bad3(%)"],
        floatfmt=(".0s", ".0s", ".4f", ".2f", ".2f", ".2f", ".2f"),
        tablefmt="grid"
    )
    report += detailed_table

    print(report)

    # 保存结果到文件
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(RESULTS_DIR, f"pointcloud_filter_comparison_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    # 保存为CSV
    csv_file = os.path.join(RESULTS_DIR, f"pointcloud_filter_metrics_{timestamp}.csv")
    df.to_csv(csv_file, index=False, float_format="%.4f", encoding="utf-8-sig")

    # 保存汇总CSV
    summary_csv_file = os.path.join(RESULTS_DIR, f"pointcloud_filter_summary_{timestamp}.csv")
    summary_df.to_csv(summary_csv_file, index=False, float_format="%.4f", encoding="utf-8-sig")

    print(f"\n评估结果已保存到:")
    print(f"  报告: {report_file}")
    print(f"  详细CSV: {csv_file}")
    print(f"  汇总CSV: {summary_csv_file}")


def main():
    """
    主函数：批量评估所有场景
    """
    print("=" * 80)
    print("点云滤波效果对比评估（修正版）")
    print("=" * 80)
    print(f"点云数据集: {POINTCLOUD_DIR}")
    print(f"GT数据集: {GT_DIR}")
    print(f"Q矩阵:\n{Q_MATRIX}")
    print(f"滤波方法: {', '.join(FILTER_METHODS)}")
    print(f"Ground Truth文件: {GT_FILENAME}")
    print("=" * 80)

    # 获取所有场景文件夹（从点云文件夹）
    pointcloud_scenes = []
    for item in os.listdir(POINTCLOUD_DIR):
        full_path = os.path.join(POINTCLOUD_DIR, item)
        if os.path.isdir(full_path) and item.startswith("202") and "-" in item:
            pointcloud_scenes.append(item)

    if not pointcloud_scenes:
        print("错误: 在点云文件夹中未找到任何场景文件夹")
        return

    pointcloud_scenes.sort()
    print(f"找到 {len(pointcloud_scenes)} 个场景文件夹\n")

    # 评估每个场景
    all_results = []
    success_count = 0
    skipped_count = 0

    for i, scene_name in enumerate(pointcloud_scenes, 1):
        print(f"\n{'='*80}")
        print(f"进度: {i}/{len(pointcloud_scenes)}")
        print(f"{'='*80}")

        # 检查对应的GT文件夹是否存在
        pointcloud_scene_dir = os.path.join(POINTCLOUD_DIR, scene_name)
        gt_scene_dir = os.path.join(GT_DIR, scene_name)

        if not os.path.exists(gt_scene_dir):
            print(f"警告: GT文件夹不存在，跳过场景 {scene_name}")
            skipped_count += 1
            continue

        results = evaluate_scene(scene_name, pointcloud_scene_dir, gt_scene_dir)
        if results:
            all_results.extend(results)
            success_count += 1

    print(f"\n{'='*80}")
    print(f"成功评估: {success_count} 个场景")
    print(f"跳过场景: {skipped_count} 个")
    print(f"总计评估记录: {len(all_results)} 条")
    print(f"{'='*80}")

    if not all_results:
        print("错误: 未完成任何评估")
        return

    # 可视化结果
    visualize_results(all_results)


if __name__ == "__main__":
    main()
