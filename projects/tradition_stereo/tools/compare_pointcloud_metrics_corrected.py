"""
点云滤波效果对比脚本（修正版 - 正确处理裁剪后的点云）

重要：点云是从裁剪后的818×512图像生成的，需要使用调整后的Q矩阵

作者: Claude
日期: 2025-12-15
"""

import numpy as np
import cv2
import open3d as o3d
import os
import torch
import pandas as pd
from tabulate import tabulate
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric.cal_metric import d1_metric, threshold_metric, epe_metric

# ==================== 配置参数 ====================
POINTCLOUD_DIR = r"D:\Desktop\原始ply点云\原始ply点云"
# GT使用点云文件夹里的disp.npy（和out.ply同时生成的原始IGEV视差）
GT_DIR = POINTCLOUD_DIR
RESULTS_DIR = os.path.join(os.path.dirname(POINTCLOUD_DIR), "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Q矩阵（工件相机配置 - 原始的，未调整的）
Q_MATRIX_ORIGINAL = np.array([
    [1., 0., 0., -3.1274110031127930e+02],
    [0., 1., 0., -6.6352561950683594e+02],
    [0., 0., 0., 8.7770274018144380e+02],
    [0., 0., 3.9768562155237530e-01, 0.]
])

# 裁剪参数（来自batch_process_igev.py）
MIN_DISPARITY_PARAM = -104
NUM_DISPARITIES = 208

# 滤波方法
FILTER_METHODS = [
    "morph_kernel7_gauss7",
    "morph_kernel15_gauss7",
    "morph_kernel31_gauss7"
]

GT_FILENAME = "disp.npy"  # 使用点云文件夹里的disp.npy作为GT
# ================================================


def get_adjusted_Q_matrix(original_shape=(1280, 720)):
    """
    计算调整后的Q矩阵（参考batch_process_igev.py的裁剪逻辑）

    返回:
        Q_adjusted: 调整后的Q矩阵
        crop_params: 裁剪参数字典
    """
    h_orig, w_orig = original_shape

    # 裁剪参数计算（精确匹配batch_process_igev.py）
    minDisparity = MIN_DISPARITY_PARAM
    numDisparities = NUM_DISPARITIES
    edge = abs(minDisparity) // 2
    edgeL = minDisparity + numDisparities
    start_x = edgeL
    start_y = edge // 2

    # 计算动态调整的长宽比裁剪
    roi_width = w_orig - 2 * edgeL
    roi_height = h_orig - edge

    k = roi_width / roi_height
    if k > 1.8:
        h = (roi_height * 16 // 10) // 2 * 2
        offset = (roi_width - h) // 4 * 2
        start_x += offset
        roi_width = h
    elif 1 / k > 1.8:
        h = (roi_width * 16 // 10) // 2 * 2
        offset = (roi_height - h) // 4 * 2
        start_y += offset
        roi_height = h

    # 调整Q矩阵（和batch_process_igev.py一致）
    Q_adjusted = Q_MATRIX_ORIGINAL.copy()
    Q_adjusted[0, 3] += start_x
    Q_adjusted[1, 3] += start_y

    crop_params = {
        'start_x': start_x,
        'start_y': start_y,
        'width': roi_width,
        'height': roi_height
    }

    return Q_adjusted, crop_params


def ply_to_disparity_cropped(ply_file, Q_matrix, target_shape):
    """
    从PLY点云生成裁剪后尺寸的视差图

    参数:
        ply_file: PLY点云文件路径
        Q_matrix: 调整后的Q矩阵（已经考虑了裁剪偏移）
        target_shape: 目标形状 (H, W)，应该是(818, 512)

    返回:
        disparity: 视差图 (H, W)
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_file)
    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        return np.zeros(target_shape, dtype=np.float32)

    # 解析Q矩阵参数
    f = Q_matrix[2, 3]
    cx, cy = -Q_matrix[0, 3], -Q_matrix[1, 3]
    invB = Q_matrix[3, 2]

    # 过滤无效点 (Z<=0)
    valid_mask = pts[:, 2] > 0
    X, Y, Z = pts[valid_mask, 0], pts[valid_mask, 1], pts[valid_mask, 2]

    # 计算投影坐标（相对于裁剪后的图像坐标系）
    u = np.round(X * f / Z + cx).astype(int)
    v = np.round(Y * f / Z + cy).astype(int)

    # 创建视差图容器（裁剪后的尺寸）
    H, W = target_shape
    disparity = np.zeros((H, W), dtype=np.float32)

    # 生成视差值
    valid_z = Z.copy()
    valid_z[valid_z == 0] = 1e-6
    disp_values = f / (valid_z * invB)

    # 过滤边界外的点
    valid_proj = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_proj]
    v = v[valid_proj]
    disp_values = disp_values[valid_proj]

    # 投影点到视差图
    disparity[v, u] = disp_values

    return disparity


def evaluate_scene(scene_name, pointcloud_scene_dir, gt_scene_dir):
    """评估单个场景"""
    print(f"\n评估场景: {scene_name}")

    # 加载GT视差图
    gt_disp_path = os.path.join(gt_scene_dir, GT_FILENAME)
    if not os.path.exists(gt_disp_path):
        print(f"  警告: 未找到GT视差图")
        return None

    gt_disp = np.load(gt_disp_path).astype(np.float32)
    target_shape = gt_disp.shape
    print(f"  GT视差图尺寸: {target_shape}")

    # 获取调整后的Q矩阵
    Q_adjusted, crop_params = get_adjusted_Q_matrix()
    print(f"  裁剪参数: start=({crop_params['start_y']},{crop_params['start_x']}), size=({crop_params['height']}×{crop_params['width']})")

    # 创建有效像素掩码（只在GT和预测都有值的区域计算）
    # 注意：由于黑色滤波，点云只覆盖部分区域，所以要同时考虑GT和预测
    valid_mask = (gt_disp > 0)
    valid_pixels = valid_mask.sum()

    if valid_pixels == 0:
        print(f"  警告: GT视差图无有效像素")
        return None

    print(f"  GT有效像素: {valid_pixels:,} ({valid_pixels/gt_disp.size*100:.2f}%)")
    print(f"  GT视差范围: {gt_disp[valid_mask].min():.2f} ~ {gt_disp[valid_mask].max():.2f}")

    # 评估每个滤波方法
    results = []
    for method_name in FILTER_METHODS:
        ply_filename = f"{method_name}.ply"
        ply_path = os.path.join(pointcloud_scene_dir, ply_filename)

        if not os.path.exists(ply_path):
            print(f"  警告: 未找到 {ply_filename}")
            continue

        print(f"  评估方法: {method_name}")

        # 点云转视差（使用调整后的Q矩阵，直接转到裁剪尺寸）
        pred_disp = ply_to_disparity_cropped(ply_path, Q_adjusted, target_shape)

        # 检查尺寸
        if pred_disp.shape != gt_disp.shape:
            print(f"    错误: 视差图尺寸不匹配 ({pred_disp.shape} vs {gt_disp.shape})")
            continue

        # 统计预测视差
        pred_valid_mask = (pred_disp > 0)
        pred_valid_pixels = pred_valid_mask.sum()
        print(f"    预测有效像素: {pred_valid_pixels:,} ({pred_valid_pixels/pred_disp.size*100:.2f}%)")

        if pred_valid_pixels > 0:
            print(f"    预测视差范围: {pred_disp[pred_valid_mask].min():.2f} ~ {pred_disp[pred_valid_mask].max():.2f}")

        # 创建联合掩码：只在预测和GT都有值的区域计算指标
        # 这样可以公平地评估滤波效果（忽略黑色背景过滤掉的区域）
        joint_mask = valid_mask & (pred_disp > 0)
        joint_pixels = joint_mask.sum()

        print(f"    联合有效像素: {joint_pixels:,} ({joint_pixels/pred_disp.size*100:.2f}%)")

        if joint_pixels == 0:
            print(f"    警告: 没有联合有效像素，跳过")
            continue

        # 转换为PyTorch张量（使用联合掩码）
        pred_tensor = torch.from_numpy(pred_disp).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_disp).unsqueeze(0)
        mask_tensor = torch.from_numpy(joint_mask).unsqueeze(0)

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
                "Valid Pixels": joint_pixels,  # 使用联合有效像素数
                "Total Pixels": pred_disp.size,
                "Valid Ratio": valid_pixels / pred_disp.size * 100
            }

            print(f"    EPE: {metrics['EPE']:.4f} px, D1: {metrics['D1']:.2f} %")
            results.append(metrics)

        except Exception as e:
            print(f"    计算指标时出错: {str(e)}")
            continue

    return results


def visualize_results(all_results):
    """可视化结果并生成报告"""
    if not all_results:
        print("错误: 没有可用的评估结果")
        return

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
点云滤波效果对比评估报告（修正版）
====================================
日期: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
点云数据集: {POINTCLOUD_DIR}
GT数据集: {GT_DIR}
评估场景总数: {len(df['Scene'].unique())}
滤波方法数: {len(FILTER_METHODS)}

说明: 点云是从裁剪后的818×512图像生成的，使用调整后的Q矩阵

方法汇总:
{table}

最佳方法（按EPE排序）:
{summary_df.nsmallest(3, 'EPE (mean)')[['Method', 'EPE (mean)', 'D1 (mean)']].to_string(index=False)}

详细场景结果:
"""

    detailed_table = tabulate(
        df[["Scene", "Method", "EPE", "D1", "Bad1", "Bad2", "Bad3"]],
        headers=["场景", "方法", "EPE(px)", "D1(%)", "Bad1(%)", "Bad2(%)", "Bad3(%)"],
        floatfmt=(".0s", ".0s", ".4f", ".2f", ".2f", ".2f", ".2f"),
        tablefmt="grid"
    )
    report += detailed_table

    print(report)

    # 保存结果
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(RESULTS_DIR, f"pointcloud_filter_comparison_corrected_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    csv_file = os.path.join(RESULTS_DIR, f"pointcloud_filter_metrics_corrected_{timestamp}.csv")
    df.to_csv(csv_file, index=False, float_format="%.4f", encoding="utf-8-sig")

    summary_csv_file = os.path.join(RESULTS_DIR, f"pointcloud_filter_summary_corrected_{timestamp}.csv")
    summary_df.to_csv(summary_csv_file, index=False, float_format="%.4f", encoding="utf-8-sig")

    print(f"\n评估结果已保存到:")
    print(f"  报告: {report_file}")
    print(f"  详细CSV: {csv_file}")
    print(f"  汇总CSV: {summary_csv_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("点云滤波效果对比评估（修正版 - 使用调整后的Q矩阵）")
    print("=" * 80)
    print(f"点云数据集: {POINTCLOUD_DIR}")
    print(f"GT数据集: {GT_DIR}")
    print(f"滤波方法: {', '.join(FILTER_METHODS)}")
    print("=" * 80)

    # 获取所有场景
    pointcloud_scenes = []
    for item in os.listdir(POINTCLOUD_DIR):
        full_path = os.path.join(POINTCLOUD_DIR, item)
        if os.path.isdir(full_path) and item.startswith("202") and "-" in item:
            pointcloud_scenes.append(item)

    if not pointcloud_scenes:
        print("错误: 未找到任何场景文件夹")
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
