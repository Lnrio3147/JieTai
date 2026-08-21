# 实验 5：视差引导的主体提取

本实验验证一个直接面向最终产物的思路：复用实验 1 已保存的 LiteAnyStereo 浮点视差，把工件从背景中提取出来，再输出干净视差和主体点云，无需重新运行双目模型。

这里必须区分两个概念：

- `geometry` 是基于视差的几何前景提取，能回答“哪个物体更靠近相机”，但不能独立判断它是不是某个语义类别。
- `refined` 用视差自动生成 GrabCut 的前景/背景种子，再根据左图颜色边界细化；它仍不是学习式语义分类，但更符合“单主体干净掩码”的工程目标。
- `semantic+LAS` 复用实验 3 的人工标注 BiSeNet 掩码确定“哪个像素是工件”，再用该掩码过滤 LAS 浮点视差和点云。
- `soft_fusion` 将 BiSeNet 概率作为主体判断，只让可靠的场景自适应视差证据修正低置信边界。这是当前推荐管线。

## 数据划分

| 角色 | 数据 | 数量 | 用途 |
|---|---|---:|---|
| 参数选择 | 人工标注 `train.csv` 中的 FDJYP-0 | 64 | 选择视差阈值和后处理参数 |
| 冻结测试 | 人工标注 `val.csv` 中的 `fdjyp_0_2` | 18 | 参数冻结后一次性评价 |

原数据集把后者命名为 `val`，但在本实验中不再用它选参数，因此其统计角色是留出测试集。FDJYP-2 没有与实验 1 一一对应的缓存 LAS 视差，本轮不纳入。

这里的“冻结测试”只严格适用于新加入的 `geometry/refined` 参数。实验 3 的 BiSeNet checkpoint 曾用同一个 `val.csv` 选择最佳轮次，因此 `semantic+LAS` 的分割数值是工程参考，不是独立泛化测试。

## 运行

```bash
cd /path/to/JieTai
python3 experiments/05_disparity_guided_segmentation/scripts/run_experiment.py --overwrite
```

脚本默认读取已经生成的 `results/bisenet_reference/` 和 `results/bisenet_train_reference/` 概率图。若要从冻结 PB 重新生成 18 张留出概率图：

```bash
cd /path/to/JieTai/projects/bisenetv2-tensorflow
TF_USE_LEGACY_KERAS=1 \
python \
tools/jmp_workpiece/predict_bisenetv2_jmp.py \
  --model_pb ../../experiments/03_manual_segmentation/fdjyp3/results/model_manual/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --input_dir ../../datasets/annotations/JMP-workpiece-seg-manual-isat-v1/images/val \
  --input_glob 'fdjyp_0_2_*.png' \
  --output_dir ../../experiments/05_disparity_guided_segmentation/results/bisenet_reference \
  --width 288 --height 512 --device cpu --save_probabilities
```

训练选参概率图使用同一命令，将 `images/val`、`fdjyp_0_2_*.png` 和 `bisenet_reference` 分别替换为 `images/train`、`fdjyp_0_*.png` 和 `bisenet_train_reference`。预测脚本会拒绝覆盖已有目录；重算时应使用新的版本名，并通过主脚本的 `--semantic-probability-dir` 和 `--train-semantic-probability-dir` 指定。

若只想快速重算掩码和指标，可给主脚本加 `--no-pointcloud`。默认结果位于 `results/fdjyp0_holdout/`：

```text
summary.json               # 训练选择集与冻结测试集汇总
train_metrics.csv          # 64 张参数选择集逐图指标
holdout_metrics.csv        # 18 张冻结测试逐图指标
contact_sheet.jpg          # 18 张四联对比总览
scenes/<scene>/
├── comparison.jpg         # 左图 / GT / 纯视差 / 颜色细化 / 语义 / 软融合
├── mask_geometry.png      # 纯视差主体掩码
├── mask_refined.png       # 视差引导的颜色细化掩码
├── mask_semantic.png      # 原始 BiSeNet 语义掩码
├── mask_soft_fusion.png   # 语义概率 + 视差一致性软融合
├── mask_subject.png       # 推荐主体掩码，与 mask_soft_fusion 相同
├── mask_semantic_depth_intersection.png # 语义与视差硬交集消融
├── subject_disparity.npy  # 原始 LAS 精度；背景为 NaN
├── subject_disparity.png  # 主体视差可视化
└── subject_cloud.ply      # 只含主体的二进制 PLY 点云
```

## 核心结果

| 方法 | mIoU | Dice | Precision | Recall | Boundary F1@2px |
|---|---:|---:|---:|---:|---:|
| 纯视差 `geometry` | 0.523 | 0.663 | 0.622 | 0.807 | 0.430 |
| 视差引导颜色细化 `refined` | 0.539 | 0.677 | 0.639 | 0.818 | 0.452 |
| BiSeNet 掩码 + LAS 视差 | 0.963 | 0.981 | **0.980** | 0.983 | 0.842 |
| 语义概率 + 视差软融合 | **0.966** | **0.982** | 0.978 | **0.988** | **0.861** |
| BiSeNet 与视差掩码硬交集 | 0.796 | 0.876 | 0.987 | 0.805 | 0.651 |

因此，LAS 视差单独能形成高召回几何先验，但料箱的连续深度坡度会与工件粘连；把它与语义掩码做硬交集又会损失约 18% 的主体面积。软融合平均只修改 0.60% 的像素，却提升了边界和召回。当前方案仍以 BiSeNet 判断主体，只让 LAS 修正不确定边界，并负责输出连续、少空洞的视差和点云。

实验结论和限制见 [实验报告](reports/report.md)。

## 跨数据集预测 V1（130 张，历史对照）

最初冻结的 V1 参数已用于 FDJYP-3、luowen、general_1221、scale_1221 和 Jop1 共 130 个其他场景。历史结果保留在 `results/cross_dataset_130/`；每张图包含全分辨率 `mask_subject.png`、背景为 `NaN` 的 `subject_disparity.npz`、视差图和五联对比图，根目录五张 `*_contact_sheet.jpg` 可直接总览全部场景。当前批处理脚本默认运行下文的 Recall V2。

| 数据集 | 数量 | 低风险 | 需复核 | 高风险 |
|---|---:|---:|---:|---:|
| FDJYP-3 | 73 | 1 | 55 | 17 |
| luowen | 37 | 11 | 17 | 9 |
| general_1221 | 6 | 1 | 2 | 3 |
| scale_1221 | 5 | 0 | 5 | 0 |
| Jop1 | 9 | 4 | 5 | 0 |

风险状态只是无真值筛查，不是准确率。当前结论是“固定场景下有条件可行，但不可无审核泛化”；完整数值、目视审查、通孔限制和下一步建议见 [跨数据集可行性报告](reports/cross_dataset_report.md)。

## Recall V2：主体保留优先（当前推荐）

V2 组合实验 4 的视差感知孔洞判断和实验 5 的软融合：不允许软融合删除选择性修复后的语义主体；模糊孔洞填回主体，只有明显大孔或有视差断层证据的孔洞保留为背景。旧 V1 结果保留不变。

```bash
cd /path/to/JieTai
python3 experiments/05_disparity_guided_segmentation/scripts/run_experiment.py \
  --output experiments/05_disparity_guided_segmentation/results/fdjyp0_holdout_recall_v2 \
  --no-pointcloud
python3 experiments/05_disparity_guided_segmentation/scripts/run_cross_dataset.py
```

18 张人工留出图上，V2 的 Recall 为 **0.98908**，高于 V1 的 0.98758；mIoU 为 0.96407，略低于 V1 的 0.96582。它是召回优先版本，不是综合 IoU 版本。130 张跨域输出全部满足“删除选择性语义主体像素比例为 0”，并恢复了 FDJYP-3 `0056～0058` 的明显背景长槽。

完整取舍、实验 4/V1/V2 对齐结果和路径见 [Recall V2 报告](reports/recall_v2_report.md)。
