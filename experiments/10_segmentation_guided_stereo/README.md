# 实验 10：目标分割引导双目匹配

本实验把“先分割、再匹配”落实为一条可复现链路：RGB-only 轻量分割器分别预测
左右目目标概率，随后比较原始 LiteAnyStereo、仅后处理 Mask、共同 ROI 和概率软引导
四种方法。软引导直接作用于 LiteAnyStereo 的代价体，旧调用不传 Mask 时行为保持不变。

## 1. 数据与标注边界

- `workpiece-seg-grouped-v3` 有 317 张人工左目外轮廓，按采集组隔离为
  `218/53/46` 张 train/val/test；
- 双目匹配不能使用视差作为分割器输入，否则会形成“先做双目、再用 Mask 加速双目”
  的循环依赖。因此本实验训练的是 RGB-only MobileNetV4-Conv-S + EMCAD；
- [prepare_stereo_segmentation_dataset.py](prepare_stereo_segmentation_dataset.py)
  用已有视差把训练集左目人工 Mask 投影到右目。218 个候选中 163 个通过
  `valid_fraction >= 0.70` 和图像边界检查，得到 381 张训练样本；
- 右目训练 Mask 是伪标签，不是人工真值。val/test 仍只含 53/46 张人工左目标签；
- 另有 32 张右目已经在 ISAT 中逐张人工修订，按原采集组保持为 15 张验证和
  17 张冻结测试，只用于独立评价，不回流训练。

生成双视图增强训练集：

```bash
cd JieTai/experiments/10_segmentation_guided_stereo
python prepare_stereo_segmentation_dataset.py
```

为了把右目伪标签升级为独立人工评价，本实验还提供 ISAT 复核任务生成器：

```bash
python prepare_right_annotation_tasks.py
```

任务包共 32 张，val/test 分别为 15/17，覆盖全部 7 类；每类按投影质量
低/中/高档确定性抽样。任务生成时 `tasks.csv` 中的 `manual_status` 固定记录为
`pending`；2026-08-25 人工修订完成后，原始任务索引仍保留生成时状态，完成状态由
导出集的新索引记录。任务入口见
[tasks.csv](../../datasets/annotations/workpiece-right-isat-v1/tasks.csv)和
[32 张总览](../../datasets/annotations/workpiece-right-isat-v1/task_contact_sheet.jpg)。

把人工 JSON 做几何 QA、按 ISAT `layer` 栅格化并冻结为评价集：

```bash
python export_right_isat_annotations.py --annotator-confirmed
```

输出为 `datasets/evaluation/workpiece-right-manual-isat-v1`。32/32 份 JSON 与图片
一一对应，36 个多边形均合法，其中 3 个 `__background__` 高层多边形会从工件
Mask 中擦除背景。人工 Mask 与投影预标注的中位 IoU 为 `0.8646`，平均有
`14.20%` 像素发生修改，说明投影标签不能替代人工真值。结构检查不能自动证明语义
边界正确；语义完成状态来自标注者确认，导出总览位于该数据集的 `qa/`。

## 2. 训练 RGB-only 分割器

初始左目模型复用了实验 8 中形状兼容的 562 个编码器/解码器权重，参数量
`1.434M`。在分组 V3 冻结测试 46 张图上的结果为：

| 模型 | IoU | Precision | Recall | Boundary F1 | 类别宏平均 IoU |
|---|---:|---:|---:|---:|---:|
| RGB-only 左目训练 | 0.9202 | 0.9519 | 0.9650 | 0.6318 | 0.9201 |
| RGB-only + 右目伪标签增强 | 0.9141 | 0.9462 | 0.9643 | **0.6973** | 0.9026 |

右目增强版在验证集把 IoU 从 `0.9602` 提升到 `0.9640`，但冻结测试 IoU 下降
`0.0060`，只有 Boundary F1 明显改善。因此它保留为双视图消融模型，不直接替代
左目模型。

人工标签补齐后，两个冻结 checkpoint 的右目结果为：

| 模型（使用原 checkpoint 阈值） | 右目测试 IoU | Precision | Recall | Boundary F1 | 类别宏 IoU |
|---|---:|---:|---:|---:|---:|
| **RGB-only 左目训练，阈值 0.61** | **0.8586** | 0.9042 | **0.9445** | **0.3912** | **0.8758** |
| RGB-only + 右目伪标签，阈值 0.19 | 0.8560 | **0.9045** | 0.9411 | 0.3182 | 0.8676 |

因此伪标签增强没有在独立人工右目测试上超过左目模型，默认 checkpoint 继续使用
`rgb_segmenter_grouped_v3/best.pt`。作为补充，只在 15 张人工右目验证集重新选择阈值
时，两者测试 IoU 分别为 `0.8534`（阈值 0.21）和 `0.8592`（阈值 0.07）；后者
Boundary F1 仍更低（`0.2267` 对 `0.2367`），而且极低阈值导致明显背景扩张，不能据此
替换默认模型。完整结果见 `results/right_manual_evaluation_v2/`，复现命令为：

```bash
python evaluate_right_segmenter.py
```

训练双视图增强版：

```bash
python train_rgb_segmenter.py \
  --dataset ../../datasets/training/workpiece-seg-stereo-v1 \
  --output results/rgb_segmenter_stereo_v1 \
  --rgbd-initialization results/rgb_segmenter_grouped_v3/best.pt \
  --batch-size 2 --workers 4
```

训练程序只用验证集选择 epoch 和概率阈值，测试集不参与选择。每次运行会保存
`run_config.json`、`history.csv`、`threshold_sweep.csv`、`best.pt` 和 `summary.json`。

## 3. 四组双目消融

| 方法 | 代价体计算 | 输出区域 | 用途 |
|---|---|---|---|
| `baseline` | 全图原始 LAS | 全图 | 对照组 |
| `post_mask` | 复用全图 LAS | 左目 Mask 内 | 只清理点云，不减少匹配计算 |
| `roi` | 左右 Mask 共同包围框 | ROI 内 | 目标较小时减少计算 |
| `guided` | ROI；代价体加入左右概率一致性先验 | 左目 Mask 内 | 抑制背景/目标跨界匹配 |

软引导公式为：

```text
cost'(x,d) = cost(x,d) + lambda * log(g(x,d))
g(x,d) = 1 - P_left(x) + P_left(x) * P_right(x-d)
```

左目背景处 `g=1`，因此不会强迫背景视差；目标像素则偏向右目目标区域。

准备 manifest 并运行：

```bash
python prepare_ablation_manifest.py --split val \
  --output inputs/grouped_v3_val.csv
python prepare_ablation_manifest.py --split test \
  --output inputs/grouped_v3_test.csv
python evaluate_ablation.py \
  --manifest inputs/grouped_v3_test.csv \
  --segmenter results/rgb_segmenter_stereo_v1/best.pt \
  --guidance-weight 2.0 \
  --output results/ablation_stereo_v1
```

### 引导权重定标

在验证集中 11 个 FDJYP-3 场景上，以 Foundation Stereo 作为工程参考进行定标；
它不是人工稠密真值。左目模型的首轮结果如下：

| `lambda` | 参考 EPE / px | 右目 Mask 违例率 | 灰度光度误差 |
|---:|---:|---:|---:|
| 0（baseline） | 1.1090 | 0.08870 | 21.7420 |
| 0.25 | 1.1088 | 0.08639 | 21.5209 |
| 0.50 | 1.1087 | 0.08411 | 21.2983 |
| 1.00 | 1.1086 | 0.07828 | 20.7213 |
| 2.00 | **1.1076** | **0.05597** | **18.7104** |

`lambda=2.0` 是按验证集选择的实验候选值。必须同时保留负结果：它在 11 张冻结
测试 FDJYP-3 上把违例率从 `0.14242` 降到 `0.09761`，光度误差从 `32.2294`
降到 `27.8953`，但参考 EPE 从 `2.2735` 升到 `2.3699 px`。因此当前结论是软约束
明显提高 Mask/光度一致性，却没有跨采集组的绝对视差误差保证；在取得人工右目标签
和独立视差真值前，生产配置应保留 `baseline`，不能默认打开 `lambda=2.0`。

全 46 张冻结测试的无真值诊断得到相同方向：软引导把平均右目 Mask 违例率从
`0.10389` 降到 `0.07942`，光度误差从 `22.7402` 降到 `20.5048`。其中只有
FDJYP-3 的 11 张具有 Foundation Stereo 工程参考，不能把全 46 张诊断量解释成
真实 EPE 提升。

全 46 张首轮消融中，ROI 只有 5 张实际启用，平均计算面积仍为 `97.55%`；当前
工件大多占满画面，所以 ROI 路线尚未证明有速度收益。采集包含更多小目标/远距离工件
的双目对后再评价这一项。

### 人工右目 Mask 的跨区域匹配评价

人工右目标签使“右目 Mask 违例率”不再依赖模型自己预测的右目 Mask。这里用人工左目
前景选出有效视差，再检查 `x_right = x_left - disparity` 是否落入独立人工右目
前景；推理本身仍只使用 RGB 模型概率，人工 Mask 只参与评分。

| 划分/方法 | 人工右目违例率 | 灰度光度误差 | FDJYP-3 工程参考 EPE |
|---|---:|---:|---:|
| 验证 15 张 baseline | 0.07662 | 21.8505 | **1.8198 px** |
| 验证 15 张 guided (`lambda=2`) | **0.05733** | **19.8264** | 1.8231 px |
| 冻结测试 17 张 baseline | 0.17396 | 25.5947 | **4.5986 px** |
| 冻结测试 17 张 guided (`lambda=2`) | **0.14769** | **22.8757** | 4.8152 px |

引导在验证 15/15、测试 17/17 场景都降低了人工右目违例率；测试绝对下降
`0.02627`，相对下降约 `15.1%`。与此同时，测试中 3 个有 Foundation Stereo
工程参考的 FDJYP-3 场景 EPE 变差 `0.2165 px`。人工标注由此强化而没有推翻原结论：
语义软引导稳定减少目标/背景跨界和光度不一致，但不能保证绝对视差更准，因此默认部署
仍为 `post_mask`，`guided` 保留为有明确取舍的实验选项。

复现人工 Mask 消融：

```bash
python prepare_ablation_manifest.py --split test \
  --right-ground-truth-dataset ../../datasets/evaluation/workpiece-right-manual-isat-v1 \
  --only-with-right-ground-truth --output inputs/right_manual_test.csv
python evaluate_ablation.py --manifest inputs/right_manual_test.csv \
  --segmenter results/rgb_segmenter_grouped_v3/best.pt \
  --guidance-weight 2.0 --method baseline --method guided \
  --output results/right_manual_stereo_test_w2
```

## 4. 输出与检查

- `per_scene.csv`：逐场、逐方法质量与耗时；
- `summary.json`：方法聚合；
- `scenes/<name>/`：左右 Mask、完整视差和主体视差可视化；
- `--save-arrays`：另外保存浮点 `.npy`，默认关闭以节省空间。

当前推荐用已验证的原始匹配加目标 Mask 导出主体点云：

```bash
python guided_stereo.py \
  --left /path/to/im0.png --right /path/to/im1.png \
  --method post_mask --q-matrix /path/to/stereo_calibration.yml \
  --output results/single_pair_post_mask
```

该命令保存左右概率/Mask、完整和主体视差，并在提供 OpenCV `Q` 矩阵时写出
`subject_cloud.ply`。需要复现实验软引导时改为 `--method guided --guidance-weight 2.0`。
本地 FDJYP-0 场景 `202506261657-0011` 的端到端回归已成功导出 375,558 个主体点。

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

未提供 `right_gt_mask` 时，“右目 Mask 违例率”和“光度误差”只是无真值诊断量；
提供人工右目 Mask 后，跨区域违例率成为独立语义标签上的指标，但仍不能替代人工稠密
视差真值 EPE。
