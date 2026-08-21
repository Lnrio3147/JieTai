"""
从PLY点云反投影生成视差图并与GT计算指标

功能：
1. 读取 D:\Desktop\过滤后\原始ply点云 中的三种PLY文件
   - morph_kernel7_gauss7.ply
   - morph_kernel15_gauss7.ply
   - morph_kernel31_gauss7.ply
2. 将PLY点云转换为视差图
3. 与 D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3 中的GT视差图(disp_cropped.npy)计算指标
"""

import numpy as np
import cv2
import open3d as o3d
import os
import sys
import torch
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric.cal_metric import d1_metric, threshold_metric, epe_metric


# ==================== 配置路径 ====================
# PLY_ROOT = r"D:\Desktop\过滤后\原始ply点云"
# PLY_ROOT = r"D:\Desktop\test_1230"
PLY_ROOT = r"D:\Desktop\原始ply点云_rknn"
GT_ROOT = r"D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3"
RESULTS_DIR = os.path.join(GT_ROOT, "evaluation_results_ply")
os.makedirs(RESULTS_DIR, exist_ok=True)

# PLY文件名列表（四种方法）
PLY_METHODS = [
    "morph_kernel7_gauss7",
    "morph_kernel15_gauss7",
    "morph_kernel25_gauss7",
    "morph_kernel31_gauss7",
]

# 需要过滤的场景列表
EXCLUDE_SCENES = [
    "202506281607-0012",
    "202506281609-0019",
    "202506281609-0020",
    "202506281619-0053",
]

# 原始Q矩阵（与batch_process_igev.py中相同，未裁剪调整的）
Q_ORIGINAL = np.array([
    [1., 0., 0., -2.5134127044677734e+02],
    [0., 1., 0., -6.5667977905273438e+02],
    [0., 0., 0.,  8.8205398705187622e+02],
    [0., 0., 3.8920665588077730e-01, 0.]
])
# ==================================================


def ply_to_disparity(ply_path, Q_matrix, output_shape=None):
    """
    从PLY点云反投影生成视差图

    参数:
        ply_path: PLY点云文件路径
        Q_matrix: 4x4重投影矩阵 (应该是裁剪调整后的Q矩阵)
        output_shape: 输出视差图尺寸 (height, width)，如果为None则自动推断

    返回:
        disparity: 视差图 (H, W), float32
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        print(f"警告: {ply_path} 中没有点")
        return None

    # 从Q矩阵提取参数
    cx = -Q_matrix[0, 3]
    cy = -Q_matrix[1, 3]
    f = Q_matrix[2, 3]       # 焦距
    invB = Q_matrix[3, 2]    # 1/baseline

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    # 过滤无效点 (Z <= 0)
    valid = Z > 0
    X, Y, Z = X[valid], Y[valid], Z[valid]

    # 反投影到像素坐标
    u = np.round(X * f / Z + cx).astype(np.int32)
    v = np.round(Y * f / Z + cy).astype(np.int32)

    # 计算视差值: d = f * B / Z = f / (Z * invB)
    d = f / (Z * invB)

    # 确定输出尺寸
    if output_shape is None:
        h = int(np.max(v)) + 10
        w = int(np.max(u)) + 10
    else:
        h, w = output_shape

    # 过滤超出范围的点
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, d = u[mask], v[mask], d[mask]

    # 创建视差图
    disparity = np.zeros((h, w), dtype=np.float32)

    # 处理遮挡：多个点投影到同一像素时，保留视差最大的（距离最近的）
    sort_idx = np.argsort(d)
    u, v, d = u[sort_idx], v[sort_idx], d[sort_idx]
    disparity[v, u] = d

    return disparity


def get_crop_params(h_orig, w_orig):
    """
    计算裁剪参数（与batch_process_igev.py一致）

    返回:
        start_x, start_y, crop_h, crop_w
    """
    minDisparity = -104
    numDisparities = 208
    edge = abs(minDisparity) // 2
    edgeL = minDisparity + numDisparities
    start_x = edgeL
    start_y = edge // 2

    # 计算动态调整的长宽比裁剪
    roi_width = w_orig - 2 * edgeL
    roi_height = h_orig - edge

    k = roi_width / roi_height
    if k > 1.8:
        h_new = (roi_height * 16 // 10) // 2 * 2
        offset = (roi_width - h_new) // 4 * 2
        start_x += offset
        roi_width = h_new
    elif 1 / k > 1.8:
        h_new = (roi_width * 16 // 10) // 2 * 2
        offset = (roi_height - h_new) // 4 * 2
        start_y += offset
        roi_height = h_new

    return start_x, start_y, roi_height, roi_width


def convert_ply_to_disparity(ply_path, im0_path):
    """
    将PLY文件转换为视差图

    参数:
        ply_path: PLY文件路径
        im0_path: 对应的左图路径（用于获取原始尺寸）

    返回:
        disparity: 视差图
    """
    if not os.path.exists(ply_path):
        return None

    # 获取原始图像尺寸
    if os.path.exists(im0_path):
        img = cv2.imread(im0_path)
        h_orig, w_orig = img.shape[:2]
    else:
        # 默认尺寸
        h_orig, w_orig = 1280, 720

    # 计算裁剪参数
    start_x, start_y, crop_h, crop_w = get_crop_params(h_orig, w_orig)

    # 调整Q矩阵
    Q_adjusted = Q_ORIGINAL.copy()
    Q_adjusted[0, 3] += start_x
    Q_adjusted[1, 3] += start_y

    # 反投影生成视差图
    disparity = ply_to_disparity(ply_path, Q_adjusted, output_shape=(crop_h, crop_w))

    return disparity


def evaluate_scene(scene_name, ply_folder, gt_folder, method_name):
    """
    评估单个场景的视差图

    参数:
        scene_name: 场景名称
        ply_folder: PLY文件夹路径
        gt_folder: GT文件夹路径
        method_name: 方法名称（如morph_kernel7_gauss7）

    返回:
        metrics: 指标字典，如果评估失败则返回None
    """
    # 查找文件
    ply_path = os.path.join(ply_folder, f"{method_name}.ply")
    gt_path = os.path.join(gt_folder, "disp_cropped.npy")
    im0_path = os.path.join(ply_folder, "im0.png")

    if not os.path.exists(ply_path):
        return None

    if not os.path.exists(gt_path):
        return None

    try:
        # 转换PLY为视差图
        pred_disp = convert_ply_to_disparity(ply_path, im0_path)
        if pred_disp is None:
            return None

        # 加载GT视差图
        gt_disp = np.load(gt_path).astype(np.float32)

        # 检查尺寸是否匹配
        if pred_disp.shape != gt_disp.shape:
            # 尝试调整尺寸
            min_h = min(pred_disp.shape[0], gt_disp.shape[0])
            min_w = min(pred_disp.shape[1], gt_disp.shape[1])
            pred_disp = pred_disp[:min_h, :min_w]
            gt_disp = gt_disp[:min_h, :min_w]

        # 创建有效像素掩码：GT有效 且 预测有效
        valid_mask = (gt_disp > 0) & (pred_disp > 0)

        # 转换为PyTorch张量
        pred_tensor = torch.from_numpy(pred_disp).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_disp).unsqueeze(0)
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0)

        # 计算指标
        metrics = {
            "Scene": scene_name,
            "EPE": epe_metric(pred_tensor, gt_tensor, mask_tensor).item(),
            "D1": d1_metric(pred_tensor, gt_tensor, mask_tensor).item(),
            "Bad1": threshold_metric(pred_tensor, gt_tensor, mask_tensor, threshold=1.0).item(),
            "Bad2": threshold_metric(pred_tensor, gt_tensor, mask_tensor, threshold=2.0).item(),
            "Bad3": threshold_metric(pred_tensor, gt_tensor, mask_tensor, threshold=3.0).item(),
            "Valid Pixels": valid_mask.sum(),
            "Total Pixels": pred_disp.size,
            "Valid Ratio": valid_mask.sum() / pred_disp.size * 100
        }

        return metrics

    except Exception as e:
        print(f"  处理 {scene_name} 出错: {str(e)}")
        return None


def evaluate_method(method_name, epe_threshold=10.0):
    """
    评估单个方法的所有场景

    参数:
        method_name: 方法名称
        epe_threshold: EPE阈值，超过此值的场景将被过滤

    返回:
        metrics_df: 指标DataFrame
        avg_metrics: 平均指标
    """
    print(f"\n{'='*60}")
    print(f"评估方法: {method_name}")
    print(f"{'='*60}")

    # 获取所有场景
    ply_scenes = []
    for item in os.listdir(PLY_ROOT):
        folder_path = os.path.join(PLY_ROOT, item)
        if os.path.isdir(folder_path) and item.startswith("202") and "-" in item:
            # 跳过需要过滤的场景
            if item in EXCLUDE_SCENES:
                continue
            ply_scenes.append(item)

    ply_scenes.sort()
    print(f"找到 {len(ply_scenes)} 个PLY场景文件夹 (已排除 {len(EXCLUDE_SCENES)} 个指定场景)")

    # 评估每个场景
    results = []
    filtered_count = 0

    for scene_name in tqdm(ply_scenes, desc=f"评估 {method_name}"):
        ply_folder = os.path.join(PLY_ROOT, scene_name)
        gt_folder = os.path.join(GT_ROOT, scene_name)

        if not os.path.exists(gt_folder):
            continue

        metrics = evaluate_scene(scene_name, ply_folder, gt_folder, method_name)

        if metrics:
            if metrics['EPE'] <= epe_threshold:
                results.append(metrics)
            else:
                filtered_count += 1

    if filtered_count > 0:
        print(f"已过滤 {filtered_count} 个EPE大于{epe_threshold}的场景")

    if not results:
        print("错误: 未完成任何场景的评估")
        return None, None

    # 转换为DataFrame
    metrics_df = pd.DataFrame(results)

    # 计算平均指标
    avg_metrics = {
        "Method": method_name,
        "EPE": metrics_df["EPE"].mean(),
        "D1": metrics_df["D1"].mean(),
        "Bad1": metrics_df["Bad1"].mean(),
        "Bad2": metrics_df["Bad2"].mean(),
        "Bad3": metrics_df["Bad3"].mean(),
        "Valid Pixels": metrics_df["Valid Pixels"].sum(),
        "Total Pixels": metrics_df["Total Pixels"].sum(),
        "Valid Ratio": metrics_df["Valid Ratio"].mean(),
        "Scene Count": len(metrics_df)
    }

    print(f"\n{method_name} 统计:")
    print(f"  评估场景数: {len(metrics_df)}")
    print(f"  EPE: {avg_metrics['EPE']:.4f} px")
    print(f"  D1: {avg_metrics['D1']:.2f} %")
    print(f"  Bad1: {avg_metrics['Bad1']:.2f} %")
    print(f"  Bad2: {avg_metrics['Bad2']:.2f} %")
    print(f"  Bad3: {avg_metrics['Bad3']:.2f} %")

    return metrics_df, avg_metrics


def visualize_results(all_avg_metrics, all_scene_metrics):
    """
    可视化和保存结果

    参数:
        all_avg_metrics: 所有方法的平均指标列表
        all_scene_metrics: 所有方法的场景指标 {method_name: metrics_df}
    """
    # 1. 生成总体对比表
    avg_df = pd.DataFrame(all_avg_metrics)

    table = tabulate(
        avg_df[["Method", "EPE", "D1", "Bad1", "Bad2", "Bad3", "Scene Count"]],
        headers=["方法", "EPE(px)", "D1(%)", "Bad1(%)", "Bad2(%)", "Bad3(%)", "场景数"],
        floatfmt=("", ".4f", ".2f", ".2f", ".2f", ".2f", ".0f"),
        tablefmt="grid"
    )

    # 2. 生成详细报告
    report = f"""
PLY点云反投影视差图评估报告
============================
日期: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

PLY点云路径: {PLY_ROOT}
GT数据路径: {GT_ROOT}

方法对比:
{table}

详细分析:
"""

    for avg in all_avg_metrics:
        method = avg["Method"]
        report += f"""
{method}:
    EPE: {avg['EPE']:.4f} px
    D1: {avg['D1']:.2f} %
    Bad1: {avg['Bad1']:.2f} %
    Bad2: {avg['Bad2']:.2f} %
    Bad3: {avg['Bad3']:.2f} %
    有效像素: {avg['Valid Pixels']:,}
    总像素: {avg['Total Pixels']:,}
    场景数: {avg['Scene Count']}
"""

    print(report)

    # 3. 保存结果
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    # 保存报告
    report_file = os.path.join(RESULTS_DIR, f"ply_evaluation_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    # 保存总体对比CSV
    avg_csv_file = os.path.join(RESULTS_DIR, f"ply_comparison_{timestamp}.csv")
    avg_df.to_csv(avg_csv_file, index=False, float_format="%.4f")

    # 保存每个方法的详细结果
    for method_name, metrics_df in all_scene_metrics.items():
        if metrics_df is not None:
            detail_csv = os.path.join(RESULTS_DIR, f"{method_name}_details_{timestamp}.csv")
            metrics_df.to_csv(detail_csv, index=False, float_format="%.4f")

    print(f"\n评估结果已保存到:")
    print(f"  报告: {report_file}")
    print(f"  对比CSV: {avg_csv_file}")


def evaluate_all_methods():
    """
    评估所有方法
    """
    print(f"开始评估PLY点云反投影视差图指标...")
    print(f"PLY点云路径: {PLY_ROOT}")
    print(f"GT数据路径: {GT_ROOT}")

    all_avg_metrics = []
    all_scene_metrics = {}

    for method_name in PLY_METHODS:
        metrics_df, avg_metrics = evaluate_method(method_name)

        if avg_metrics:
            all_avg_metrics.append(avg_metrics)
            all_scene_metrics[method_name] = metrics_df

    if all_avg_metrics:
        visualize_results(all_avg_metrics, all_scene_metrics)
    else:
        print("错误: 没有成功评估任何方法")


if __name__ == "__main__":
    evaluate_all_methods()
    print("\n评估完成!")
