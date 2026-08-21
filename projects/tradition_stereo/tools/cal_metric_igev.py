import numpy as np
import cv2
import os
import glob
import torch
import pandas as pd
from tabulate import tabulate
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric.cal_metric import d1_metric,threshold_metric,epe_metric
# 配置路径
BASE_DIR = r"D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3"
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)



def evaluate_scene(scene_dir):
    """
    评估单个场景的视差图
    """
    scene_name = os.path.basename(scene_dir)
    print(f"\n评估场景: {scene_name}")

    # 查找关键文件
    pred_file = None
    gt_file = os.path.join(scene_dir, "disp_cropped.npy")

    # 查找预测文件（兼容两种格式）
    candidate_files = glob.glob(os.path.join(scene_dir, "*disp_rknn.npy"))   # 传统算法
    if candidate_files:
        pred_file = candidate_files[0]  # 使用第一个匹配的传统算法文件
    else:
        # 如果没有找到传统算法文件，尝试IGEV文件
        igev_file = os.path.join(scene_dir, "disp_rknn.npy")
        if os.path.exists(igev_file):
            pred_file = igev_file
        else:
            print(f"  警告: 在 {scene_dir} 中未找到预测视差图 (*_disp_cropped.npy 或 disp_rknn.npy)")
            return None

    if not os.path.exists(gt_file):
        print(f"  警告: 在 {scene_dir} 中未找到真值视差图 (disp_cropped.npy)")
        return None

    print(f"  预测视差图: {os.path.basename(pred_file)}")
    print(f"  真值视差图: {os.path.basename(gt_file)}")

    # 加载视差图
    try:
        pred_disp = np.load(pred_file).astype(np.float32)
        gt_disp = np.load(gt_file).astype(np.float32)

        # 检查尺寸是否匹配
        if pred_disp.shape != gt_disp.shape:
            print(f"  错误: 视差图尺寸不匹配 ({pred_disp.shape} vs {gt_disp.shape})")
            return None

        # 创建有效像素掩码
        valid_mask = (gt_disp > 0)

        # 转换为PyTorch张量
        pred_tensor = torch.from_numpy(pred_disp).unsqueeze(0)  # 增加batch维度
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

        print(f"  完成评估:")
        print(f"    EPE: {metrics['EPE']:.4f} px")
        print(f"    D1: {metrics['D1']:.2f} %")

        return metrics

    except Exception as e:
        print(f"  处理出错: {str(e)}")
        return None


def visualize_results(metrics_df):
    """
    可视化结果并生成统计报告
    """
    # 1. 表格格式输出
    table = tabulate(
        metrics_df[["Scene", "EPE", "D1", "Bad1", "Bad2", "Bad3"]],
        headers=["场景", "EPE(px)", "D1(%)", "Bad1(%)", "Bad2(%)", "Bad3(%)"],
        floatfmt=(".0f", ".4f", ".2f", ".2f", ".2f", ".2f"),
        tablefmt="grid"
    )

    # 2. 添加平均行
    avg_row = {
        "Scene": "平均",
        "EPE": metrics_df["EPE"].mean(),
        "D1": metrics_df["D1"].mean(),
        "Bad1": metrics_df["Bad1"].mean(),
        "Bad2": metrics_df["Bad2"].mean(),
        "Bad3": metrics_df["Bad3"].mean(),
        "Valid Pixels": metrics_df["Valid Pixels"].sum(),
        "Total Pixels": metrics_df["Total Pixels"].sum(),
        "Valid Ratio": metrics_df["Valid Ratio"].mean()
    }

    # 3. 生成详细报告
    report = f"""
视差图评估报告
================
日期: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
评估场景数: {len(metrics_df)}
总像素数: {avg_row['Total Pixels']:,}
有效像素数: {avg_row['Valid Pixels']:,} ({avg_row['Valid Ratio']:.2f}%)

详细结果:
{table}

总体统计:
    EPE (端点误差): {avg_row['EPE']:.4f} px
    D1 错误率: {avg_row['D1']:.2f} %
    Bad1 错误率: {avg_row['Bad1']:.2f} %
    Bad2 错误率: {avg_row['Bad2']:.2f} %
    Bad3 错误率: {avg_row['Bad3']:.2f} %
"""
    print(report)

    # 4. 保存结果到文件
    report_file = os.path.join(RESULTS_DIR, f"disparity_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, "w") as f:
        f.write(report)

    # 5. 保存为CSV
    csv_file = os.path.join(RESULTS_DIR, "disparity_metrics.csv")
    metrics_df.to_csv(csv_file, index=False, float_format="%.4f")

    print(f"评估结果已保存到:\n  {report_file}\n  {csv_file}")




def evaluate_all_scenes():
    """
    评估所有场景
    """
    print(f"开始评估视差图指标...")
    print(f"数据集路径: {BASE_DIR}")

    # 获取所有场景文件夹
    scene_dirs = []
    for item in os.listdir(BASE_DIR):
        full_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(full_path) and item.startswith("202") and "-" in item:
            scene_dirs.append(full_path)

    if not scene_dirs:
        print("错误: 未找到任何场景文件夹。请检查路径是否正确。")
        return

    scene_dirs.sort()
    print(f"找到 {len(scene_dirs)} 个场景文件夹")

    # 评估每个场景
    results = []
    filtered_count = 0
    for scene_dir in scene_dirs:
        metrics = evaluate_scene(scene_dir)
        if metrics:
            if metrics['EPE'] <= 10.0:
                results.append(metrics)
            else:
                filtered_count += 1
                print(f"  过滤场景 {metrics['Scene']}: EPE = {metrics['EPE']:.4f} > 10.0")

    if filtered_count > 0:
        print(f"已过滤 {filtered_count} 个EPE大于10的场景")

    if not results:
        print("错误: 未完成任何场景的评估。")
        return

    # 转换为DataFrame
    metrics_df = pd.DataFrame(results)

    # 添加场景计数
    metrics_df["Scene Count"] = range(1, len(metrics_df) + 1)

    # 可视化结果
    visualize_results(metrics_df)



if __name__ == "__main__":
    # 开始评估
    evaluate_all_scenes()

    # 等待用户查看结果
    input("\n按Enter键退出...")