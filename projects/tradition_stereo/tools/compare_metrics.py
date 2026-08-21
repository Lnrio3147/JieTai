import pandas as pd
import numpy as np
import os
from tabulate import tabulate
# import matplotlib.pyplot as plt
# import seaborn as sns

# 设置路径
RESULTS_DIR = r"D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3\evaluation_results"

def merge_and_compare_metrics():
    """
    合并并对比IGEV和甲方metrics
    """
    print("开始合并并对比IGEV和甲方metrics...")

    # 读取两个CSV文件
    igev_file = os.path.join(RESULTS_DIR, "IGEV_metrics.csv")
    jf_file = os.path.join(RESULTS_DIR, "甲方_metrics.csv")

    if not os.path.exists(igev_file) or not os.path.exists(jf_file):
        print("错误: 找不到metrics文件")
        return

    # 读取数据
    igev_df = pd.read_csv(igev_file)
    jf_df = pd.read_csv(jf_file)

    print(f"IGEV数据: {len(igev_df)} 个场景")
    print(f"甲方数据: {len(jf_df)} 个场景")

    # 重命名列以便区分
    igev_renamed = igev_df.rename(columns={
        'EPE': 'IGEV_EPE',
        'D1': 'IGEV_D1',
        'Bad1': 'IGEV_Bad1',
        'Bad2': 'IGEV_Bad2',
        'Bad3': 'IGEV_Bad3'
    })

    jf_renamed = jf_df.rename(columns={
        'EPE': '甲方_EPE',
        'D1': '甲方_D1',
        'Bad1': '甲方_Bad1',
        'Bad2': '甲方_Bad2',
        'Bad3': '甲方_Bad3'
    })

    # 基于Scene列合并
    merged_df = pd.merge(igev_renamed, jf_renamed, on='Scene', how='inner')

    print(f"合并后数据: {len(merged_df)} 个共同场景")

    # 计算差异
    merged_df['EPE_Diff'] = merged_df['甲方_EPE'] - merged_df['IGEV_EPE']
    merged_df['D1_Diff'] = merged_df['甲方_D1'] - merged_df['IGEV_D1']
    merged_df['Bad1_Diff'] = merged_df['甲方_Bad1'] - merged_df['IGEV_Bad1']
    merged_df['Bad2_Diff'] = merged_df['甲方_Bad2'] - merged_df['IGEV_Bad2']
    merged_df['Bad3_Diff'] = merged_df['甲方_Bad3'] - merged_df['IGEV_Bad3']

    # 计算改进率
    merged_df['EPE_Improvement'] = (merged_df['EPE_Diff'] / merged_df['甲方_EPE'] * 100)
    merged_df['D1_Improvement'] = (merged_df['D1_Diff'] / merged_df['甲方_D1'] * 100)

    return merged_df

def generate_comparison_report(merged_df):
    """
    生成对比报告
    """
    print("\n" + "="*80)
    print("IGEV vs 甲方算法对比分析报告")
    print("="*80)

    # 1. 统计摘要
    print("\n【统计摘要】")
    summary_stats = pd.DataFrame({
        '指标': ['EPE (px)', 'D1 (%)', 'Bad1 (%)', 'Bad2 (%)', 'Bad3 (%)'],
        'IGEV_平均': [
            merged_df['IGEV_EPE'].mean(),
            merged_df['IGEV_D1'].mean(),
            merged_df['IGEV_Bad1'].mean(),
            merged_df['IGEV_Bad2'].mean(),
            merged_df['IGEV_Bad3'].mean()
        ],
        '甲方_平均': [
            merged_df['甲方_EPE'].mean(),
            merged_df['甲方_D1'].mean(),
            merged_df['甲方_Bad1'].mean(),
            merged_df['甲方_Bad2'].mean(),
            merged_df['甲方_Bad3'].mean()
        ],
        '差异': [
            merged_df['EPE_Diff'].mean(),
            merged_df['D1_Diff'].mean(),
            merged_df['Bad1_Diff'].mean(),
            merged_df['Bad2_Diff'].mean(),
            merged_df['Bad3_Diff'].mean()
        ],
        '改进率(%)': [
            merged_df['EPE_Improvement'].mean(),
            merged_df['D1_Improvement'].mean(),
            (merged_df['Bad1_Diff'].mean() / merged_df['甲方_Bad1'].mean() * 100),
            (merged_df['Bad2_Diff'].mean() / merged_df['甲方_Bad2'].mean() * 100),
            (merged_df['Bad3_Diff'].mean() / merged_df['甲方_Bad3'].mean() * 100)
        ]
    })

    print(tabulate(summary_stats, headers='keys', floatfmt=".4f", tablefmt="grid"))

    # 2. 性能对比
    print("\n【性能对比】")
    igev_better_epe = (merged_df['IGEV_EPE'] < merged_df['甲方_EPE']).sum()
    igev_better_d1 = (merged_df['IGEV_D1'] < merged_df['甲方_D1']).sum()

    print(f"IGEV EPE优于甲方的场景: {igev_better_epe}/{len(merged_df)} ({igev_better_epe/len(merged_df)*100:.1f}%)")
    print(f"IGEV D1优于甲方的场景: {igev_better_d1}/{len(merged_df)} ({igev_better_d1/len(merged_df)*100:.1f}%)")

    # 3. 最显著改进场景
    print("\n【最显著改进场景 (Top 5)】")
    top_improvements = merged_df.nsmallest(5, 'IGEV_EPE')[['Scene', 'IGEV_EPE', '甲方_EPE', 'EPE_Diff', 'EPE_Improvement']]
    top_improvements.columns = ['场景', 'IGEV_EPE', '甲方_EPE', '差异', '改进率(%)']
    print(tabulate(top_improvements, headers='keys', floatfmt=".4f", tablefmt="grid"))

    # 4. 详细对比表
    print("\n【详细对比表】")
    detail_cols = ['Scene', 'IGEV_EPE', '甲方_EPE', 'EPE_Diff', 'IGEV_D1', '甲方_D1', 'D1_Diff']
    detail_table = merged_df[detail_cols].copy()
    detail_table.columns = ['场景', 'IGEV_EPE', '甲方_EPE', 'EPE差异', 'IGEV_D1', '甲方_D1', 'D1差异']
    detail_table = detail_table.round(4)

    print(tabulate(detail_table.head(20), headers='keys', tablefmt="grid", showindex=False))
    print(f"... 显示前20个场景，共{len(detail_table)}个场景")

    return summary_stats

def save_comparison_results(merged_df, summary_stats):
    """
    保存对比结果
    """
    # 保存合并后的数据
    merged_file = os.path.join(RESULTS_DIR, "IGEV_甲方对比分析.csv")
    merged_df.to_csv(merged_file, index=False, float_format="%.4f")
    print(f"\n对比数据已保存到: {merged_file}")

    # 保存统计摘要
    summary_file = os.path.join(RESULTS_DIR, "对比统计摘要.csv")
    summary_stats.to_csv(summary_file, index=False, float_format="%.4f")
    print(f"统计摘要已保存到: {summary_file}")

    # 生成文本报告
    report_file = os.path.join(RESULTS_DIR, "IGEV_甲方对比报告.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("IGEV vs 甲方算法对比分析报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估场景数: {len(merged_df)}\n\n")

        f.write("统计摘要:\n")
        f.write(tabulate(summary_stats, headers='keys', floatfmt=".4f", tablefmt="grid"))
        f.write("\n\n")

        f.write("性能对比:\n")
        igev_better_epe = (merged_df['IGEV_EPE'] < merged_df['甲方_EPE']).sum()
        igev_better_d1 = (merged_df['IGEV_D1'] < merged_df['甲方_D1']).sum()
        f.write(f"IGEV EPE优于甲方的场景: {igev_better_epe}/{len(merged_df)} ({igev_better_epe/len(merged_df)*100:.1f}%)\n")
        f.write(f"IGEV D1优于甲方的场景: {igev_better_d1}/{len(merged_df)} ({igev_better_d1/len(merged_df)*100:.1f}%)\n")

    print(f"详细报告已保存到: {report_file}")

if __name__ == "__main__":
    # 合并数据
    merged_df = merge_and_compare_metrics()

    if merged_df is not None:
        # 生成对比报告
        summary_stats = generate_comparison_report(merged_df)

        # 保存结果
        save_comparison_results(merged_df, summary_stats)

        print("\n对比分析完成！")
    else:
        print("对比分析失败！")