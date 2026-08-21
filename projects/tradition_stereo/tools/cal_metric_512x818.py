"""
视差图指标计算脚本 - 512x818 版本
针对已裁剪好的 512x818 视差图，直接计算指标无需额外裁剪
"""
import numpy as np
import cv2
import os
import glob
import torch
import pandas as pd
from tabulate import tabulate
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric.cal_metric import d1_metric, threshold_metric, epe_metric

# 配置路径
PRED_DIR = r"D:\Desktop\scene_demo-imgs_512x832_crop_first"      # 预测视差图目录
# PRED_DIR = r"D:\Desktop\rknn_demo_output_836x736"      # 预测视差图目录
# PRED_DIR = r"D:\Desktop\rknn_demo_output_992x672"
# PRED_DIR = r"D:\Desktop\rknn_demo_output_992x672_distill"
# PRED_DIR = r"D:\Desktop\rknn_demo_output_608x928_margin40"
GT_DIR = r"D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3"  # 真值视差图目录
RESULTS_DIR = None

# 文件名配置
PRED_FILE_NAME = "disp_512x818_crop.npy"   # 预测视差图
GT_FILE_NAME = "disp_cropped.npy"          # 真值视差图
EPE_FILTER_THRESHOLD = 20.0
APPLY_EPE_FILTER = True
APPLY_EXCLUDE_SCENES = True

# 排除的场景列表
EXCLUDE_SCENES = [
    "202506281607-0012",
    "202506281609-0019",
    "202506281609-0020",
    "202506281619-0053",
]


def evaluate_scene(scene_name):
    """
    评估单个场景的视差图
    """
    print(f"\n评估场景: {scene_name}")

    # 查找文件（预测和真值在不同目录）
    pred_file = os.path.join(PRED_DIR, scene_name, PRED_FILE_NAME)
    gt_file = os.path.join(GT_DIR, scene_name, GT_FILE_NAME)

    if not os.path.exists(pred_file):
        print(f"  警告: 未找到预测视差图 ({pred_file})")
        return None

    if not os.path.exists(gt_file):
        print(f"  警告: 未找到真值视差图 ({gt_file})")
        return None

    print(f"  预测视差图: {PRED_FILE_NAME}")
    print(f"  真值视差图: {GT_FILE_NAME}")

    # 加载视差图
    try:
        pred_disp = np.load(pred_file).astype(np.float32)
        gt_disp = np.load(gt_file).astype(np.float32)

        # 检查尺寸是否匹配
        if pred_disp.shape != gt_disp.shape:
            print(f"  错误: 视差图尺寸不匹配 ({pred_disp.shape} vs {gt_disp.shape})")
            return None

        print(f"  视差图尺寸: {pred_disp.shape}")

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
        import traceback
        traceback.print_exc()
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
视差图评估报告 (512x818 版本)
==============================
日期: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
预测目录: {PRED_DIR}
真值目录: {GT_DIR}
预测文件: {PRED_FILE_NAME}
真值文件: {GT_FILE_NAME}
排除指定场景: {"是" if APPLY_EXCLUDE_SCENES else "否"}
EPE过滤: {"<=" + str(EPE_FILTER_THRESHOLD) if APPLY_EPE_FILTER else "关闭"}
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_file = os.path.join(RESULTS_DIR, f"disparity_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, "w", encoding='utf-8') as f:
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
    print(f"预测目录: {PRED_DIR}")
    print(f"真值目录: {GT_DIR}")
    print(f"预测文件: {PRED_FILE_NAME}")
    print(f"真值文件: {GT_FILE_NAME}")

    # 获取所有场景文件夹（从预测目录）
    scene_names = []
    for item in os.listdir(PRED_DIR):
        full_path = os.path.join(PRED_DIR, item)
        if os.path.isdir(full_path) and item.startswith("202") and "-" in item:
            scene_names.append(item)

    if not scene_names:
        print("错误: 未找到任何场景文件夹。请检查路径是否正确。")
        return

    scene_names.sort()
    print(f"找到 {len(scene_names)} 个场景文件夹")

    # 评估每个场景
    results = []
    filtered_count = 0
    excluded_count = 0
    for scene_name in scene_names:
        # 跳过排除的场景
        if APPLY_EXCLUDE_SCENES and scene_name in EXCLUDE_SCENES:
            excluded_count += 1
            print(f"\n跳过排除场景: {scene_name}")
            continue

        metrics = evaluate_scene(scene_name)
        if metrics:
            if (not APPLY_EPE_FILTER) or metrics['EPE'] <= EPE_FILTER_THRESHOLD:
                results.append(metrics)
            else:
                filtered_count += 1
                print(f"  过滤场景 {metrics['Scene']}: EPE = {metrics['EPE']:.4f} > {EPE_FILTER_THRESHOLD:.1f}")

    if excluded_count > 0:
        print(f"\n已排除 {excluded_count} 个指定场景")

    if filtered_count > 0:
        print(f"\n已过滤 {filtered_count} 个EPE大于{EPE_FILTER_THRESHOLD:.1f}的场景")

    if not results:
        print("错误: 未完成任何场景的评估。")
        return

    # 转换为DataFrame
    metrics_df = pd.DataFrame(results)

    # 添加场景计数
    metrics_df["Scene Count"] = range(1, len(metrics_df) + 1)

    # 可视化结果
    visualize_results(metrics_df)


def parse_args():
    parser = argparse.ArgumentParser(description="计算 512x818 视差图指标")
    parser.add_argument("--pred-dir", default=PRED_DIR, help="预测视差图根目录")
    parser.add_argument("--gt-dir", default=GT_DIR, help="GT 视差图根目录")
    parser.add_argument("--pred-file", default=PRED_FILE_NAME, help="每个场景下的预测视差文件名")
    parser.add_argument("--gt-file", default=GT_FILE_NAME, help="每个场景下的 GT 视差文件名")
    parser.add_argument("--results-dir", default=None, help="评估结果输出目录，默认写到预测目录下 evaluation_results")
    parser.add_argument("--epe-threshold", type=float, default=EPE_FILTER_THRESHOLD, help="过滤高 EPE 场景的阈值")
    parser.add_argument("--no-epe-filter", action="store_true", help="关闭 EPE 高值场景过滤")
    parser.add_argument("--include-excluded", action="store_true", help="包含脚本默认排除的场景")
    return parser.parse_args()


def apply_args(args):
    global PRED_DIR, GT_DIR, RESULTS_DIR, PRED_FILE_NAME, GT_FILE_NAME
    global EPE_FILTER_THRESHOLD, APPLY_EPE_FILTER, APPLY_EXCLUDE_SCENES

    PRED_DIR = args.pred_dir
    GT_DIR = args.gt_dir
    PRED_FILE_NAME = args.pred_file
    GT_FILE_NAME = args.gt_file
    RESULTS_DIR = args.results_dir or os.path.join(PRED_DIR, "evaluation_results")
    EPE_FILTER_THRESHOLD = args.epe_threshold
    APPLY_EPE_FILTER = not args.no_epe_filter
    APPLY_EXCLUDE_SCENES = not args.include_excluded


if __name__ == "__main__":
    apply_args(parse_args())
    # 开始评估
    evaluate_all_scenes()
