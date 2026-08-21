# JieTai 双目实验工作区

本工作区按当前实验逻辑整理为六个连续阶段。查实验时先进入 `experiments/`，数据、代码和实验产物分别集中在 `datasets/`、`projects/` 和 `experiments/`。

```text
JieTai/
├── datasets/                    # 数据、标注、评价参考和原始压缩包
├── experiments/
│   ├── 01_stereo_comparison/    # LiteAnyStereo 与 IGEV 对比
│   ├── 02_initial_segmentation/ # 直接引入分割模型的初始测试
│   ├── 03_manual_segmentation/  # 人工标注训练与跨数据集测试
│   ├── 04_mask_refinement/      # 修复不合理分割并重新测试
│   ├── 05_disparity_guided_segmentation/ # 视差引导主体提取与干净点云
│   └── 06_multidomain_segmentation/ # 多域人工标注训练、验证和类别路由
├── projects/                    # 工程源码与模型权重
├── requirements.txt
└── MANIFEST.txt
```

## 六阶段主线

1. `01_stereo_comparison/`：引入 LiteAnyStereo，与原 IGEV 模型在 rec_img_set 和 Jop1 上对比；LAS 训练记录也放在这里。
2. `02_initial_segmentation/`：直接使用伪标签训练 BiSeNetV2，观察初始分割效果。
3. `03_manual_segmentation/`：使用人工 ISAT 标注重新训练，并在 FDJYP-3、Jop1 和 rec_img_set 上测试。
4. `04_mask_refinement/`：针对孔洞、孤岛和连通域异常修订掩码，再次评价最终效果。
5. `05_disparity_guided_segmentation/`：比较纯几何、颜色细化、语义掩码与语义概率/视差软融合，输出干净主体视差和点云。
6. `06_multidomain_segmentation/`：使用 FDJYP3、螺纹、general、scale、Jop1 共 130 张人工外轮廓标注分层训练和验证，保留旧模型强项并建立召回优先的类别路由。

当前实验 5 已完成另外 130 个场景的冻结跨数据集预测，结果入口和可行性结论见 [跨数据集报告](experiments/05_disparity_guided_segmentation/reports/cross_dataset_report.md)。

实验 5 当前另提供主体保留优先的 [Recall V2](experiments/05_disparity_guided_segmentation/reports/recall_v2_report.md)：软融合只补不删，并用实验 4 的视差连续性规则保留明显背景孔洞。

实验 6 现在按目标保留两条路线：[单一平衡 V2 + 实验 5](experiments/06_multidomain_segmentation/reports/exp4_exp5_report.md) 的开发比较集前景 IoU 为 0.8912，综合质量最好；新 [Recall V4.1 Jop1 自适应救援](experiments/06_multidomain_segmentation/reports/jop_reflective_rescue_report.md) 的 Recall 为 0.9908，主体保留最好。人工标签只覆盖外轮廓，内部孔洞仍需视差连续性规则和独立孔洞评价。

最完整的路径清单见 [实验总览](experiments/README.md)，数据清单见 [数据说明](datasets/README.md)。

## 路径规则

每个具体实验尽量只使用以下短目录：

```text
README.md
reports/
results/
scripts/
slides/       # 仅有汇报材料时出现
```

项目源码内仍保留少量兼容符号链接，使旧命令中的 `runs/`、`igev_output/` 等入口继续可用；这些链接不占用重复空间。已生成 JSON/CSV 中的旧绝对路径作为运行历史保留，当前入口以各阶段 `README.md` 为准。

当前 IGEV++ RT 使用公开 SceneFlow 权重，并非一期缺失的私有目标域权重。Foundation Stereo 和 Jop1 附带 PLY 都不是人工稠密真值，具体评价口径以实验报告为准。

## 仓库存储策略

Git 仓库保存源码、配置、实验脚本、报告和汇报图片。原始/私有数据集、模型权重、实验 `results/` 目录、点云、数值缓存和压缩包仅保留在本地，不纳入 GitHub。数据来源与本地布置见 [数据说明](datasets/README.md)，已使用模型的文件名和校验值见 [工作区清单](MANIFEST.txt)。
