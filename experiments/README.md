# 实验总览

十一个一级目录表示当前实验的先后顺序。实验 8 V3 Base 是现阶段推荐的 RGB-D
分割部署候选；实验 9 保留为负结果。实验 10 新增独立于视差的 RGB 分割和 Mask
引导双目，实验 11 接续主体点云做带反馈的多视角融合。

| 阶段 | 目的 | 数据分支 | 主要入口 |
|---|---|---|---|
| `01_stereo_comparison` | 引入 LiteAnyStereo，与 IGEV 对比 | rec_img_set、Jop1 | [README](01_stereo_comparison/README.md) |
| `02_initial_segmentation` | 直接引入 BiSeNetV2，建立初始分割基线 | FDJYP-3 | [README](02_initial_segmentation/README.md) |
| `03_manual_segmentation` | 人工标注后重新训练和测试 | FDJYP-3、Jop1、rec_img_set | [README](03_manual_segmentation/README.md) |
| `04_mask_refinement` | 修复孔洞、孤岛和连通域异常后复评 | FDJYP-3 | [README](04_mask_refinement/README.md) |
| `05_disparity_guided_segmentation` | 视差/语义软融合，输出主体视差和点云 | FDJYP-0、跨域130张 | [README](05_disparity_guided_segmentation/README.md) |
| `06_multidomain_segmentation` | 五类人工外轮廓训练、验证和类别路由 | FDJYP3、螺纹、general、scale、Jop1 | [README](06_multidomain_segmentation/README.md) |
| `07_rgbd_fusion` | 双 ResNet18 RGB-D 四尺度门控与连续实体约束 | 同上 | [README](07_rgbd_fusion/README.md) |
| `08_lightweight_rgbd` | MobileNetV4 + 浅层几何 + EMCAD，面向 RK3588 | 同上 | [README](08_lightweight_rgbd/README.md) |
| `09_jit_mask_projection` | JiT 启发的单步干净 Mask 投影消融 | 同上 | [README](09_jit_mask_projection/README.md) |
| `10_segmentation_guided_stereo` | RGB-only 左右目分割、ROI 与代价体软引导 | grouped-v3、双目右目伪标签 | [README](10_segmentation_guided_stereo/README.md) |
| `11_multiview_feedback_reconstruction` | 粗/精配准、位姿图、融合、质量反馈与下一视角 | 主体点云序列、合成回归 | [README](11_multiview_feedback_reconstruction/README.md) |

```text
01 双目模型对比
   ├── rec_img_set：203 场 IGEV/LAS 视差、点云与代表案例
   ├── jop1：9 场 IGEV/LAS/PLY 参考六宫格
   └── las_training：LAS 训练记录
02 初始分割
03 人工标注分割
04 Mask 规则修复
05 视差引导主体提取
   ├── FDJYP-0：18 张留出评价，输出主体视差/点云
   └── 跨域130张：五类主体视差总览
06 多域 BiSeNetV2 基线
07 大型 RGB-D 教师网络与连续实体/溢出救援
08 轻量 RGB-D 学生网络（当前部署候选）
09 JiT 启发 Mask 投影（负结果，不部署）
10 RGB-only 目标分割引导 LiteAnyStereo
11 多视角位姿图融合、质量评价与下一视角反馈
```

## 实验 10/11 新链路

实验 10 已在不读取视差的前提下训练 `1.434M` 参数 RGB 分割器，并把左右概率以
软先验加入 LiteAnyStereo 代价体。新增 32 张人工右目评价后，左目模型在 17 张冻结
右目测试上的 IoU 为 `0.8586`，略高于右目伪标签增强模型的 `0.8560`，因此不替换
默认模型。`lambda=2.0` 在 17/17 场都减少人工 Mask 跨界，平均违例率从 `0.17396`
降至 `0.14769`；但 3 张 FDJYP-3 工程参考 EPE 从 `4.5986` 升至 `4.8152 px`。
因此结论仍是“匹配一致性改善、绝对误差不保证改善”，详见
[实验 10 报告](10_segmentation_guided_stereo/README.md)。

实验 11 已完成 5 视角合成闭环：5/5 条配准边通过，`90.76%` 融合体素至少由两个
视角支持，相对合成参考面的 F-score 为 `1.0`（阈值 `1.6 mm`）。当前数据目录缺少
带可靠位姿/独立参考的同一静止工件真实多视角序列，真实精度仍待按
[采集要求](11_multiview_feedback_reconstruction/README.md)补测。

后续预检已审计现有真实主体点云：FDJYP-0 唯一高重叠候选仍缺少同一静止工件和单位
确认，并且固定机位不具备视角多样性；Jop1 九场是不同工件。工具链会把这类输入标为
blocker，避免“配准能够运行”被误写成“多视角重建有效”。

## 多域分割主线对比

下表使用同一批 21 张固定开发比较图，适合工程回归比较。它们参与过多轮模型选择和
难例分析，尤其实验 7.2 的溢出规则是在 `luowen 0031/0033` 问题暴露后形成的，
因此不能当作完全无偏的一次性最终测试集。

| 方法 | IoU | Precision | Recall | Boundary F1 | 参数量 | FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| 实验6 V4.1 BiSeNetV2 | 0.8829 | 0.8903 | 0.9908 | 0.3411 | — | — |
| 实验7 RGB-D + 实验4 | 0.8885 | 0.8900 | 0.9982 | 0.4400 | — | — |
| 实验7.1 连续实体 | 0.8915 | 0.8932 | 0.9980 | 0.4812 | — | — |
| **实验7.2 大型教师** | **0.9456** | **0.9489** | 0.9963 | **0.5590** | 40.060M | 192.000G |
| **实验8 Base（当前部署候选）** | 0.9415 | 0.9437 | **0.9976** | 0.4471 | **2.398M** | **9.048G** |
| 实验8 硬 Mask 蒸馏 | 0.9303 | 0.9355 | 0.9940 | 0.4777 | 2.398M | 9.048G |
| 实验9 JiT Mask 投影 | 0.9400 | 0.9420 | **0.9978** | 0.4421 | 2.425M | 10.195G |

结论：

- 实验 7.2 仍是这批开发图上的质量上限；
- 实验 8 Base 相对实验 7.2 只降低约 `0.0040` IoU，但参数量减少约 `94%`、
  FLOPs 减少约 `95%`，并取得更高 Recall，因此是当前 RK3588 部署候选；
- 实验 8 硬 Mask 蒸馏提高了 Base 之外的边界约束，但整体 IoU 明显下降，不作为主模型；
- 实验 9 相对实验 8 Base 的 IoU 下降约 `0.0015`，Boundary F1 也下降，证明单纯
  把输出 Mask 投影回“连续几何体流形”不能修复跨域语义误判，因此不加入部署链。

实验 8 完整表和难例见 [comparison.csv](08_lightweight_rgbd/results/comparison/comparison.csv)
与 [representative_6.jpg](08_lightweight_rgbd/results/comparison/representative_6.jpg)；
实验 9 对齐结果见 [comparison.csv](09_jit_mask_projection/results/comparison/comparison.csv)。

### 分组 V3 冻结测试（当前推荐口径）

为避免相邻连续帧跨集合泄漏，317 张标注图已经按工件/采集组重新划分为
218 张训练、53 张验证、46 张冻结测试。三个轻量模型在相同 46 张测试图上的结果为：

| 方法 | IoU | Precision | Recall | Boundary F1 | 类别宏平均 IoU | 参数量 | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| **实验 8 V3 Base** | **0.9304** | **0.9483** | 0.9801 | **0.7802** | **0.9420** | **2.398M** | **9.048G** |
| 实验 8 V3 蒸馏版 | 0.9234 | 0.9376 | **0.9839** | 0.7327 | 0.9396 | 2.398M | 9.048G |
| 实验 9 V3 Mask 投影 | 0.9275 | 0.9437 | 0.9817 | 0.7704 | 0.9394 | 2.425M | 10.194G |

实验 9 V3 虽在 26/46 张图上小幅提升，并把 Recall 提高 `0.0016`，但一张 Luowen
难例出现 `-0.2047` IoU 的严重退化，最终 IoU 比 Base 低 `0.0030`。46/46 输出均为
单连通域，说明“连续”不等于“语义正确”。详细结果和全量图片见
[实验 9 V3 报告](09_jit_mask_projection/results/comparison_grouped_v3/REPORT.md)与
[46 张总览](09_jit_mask_projection/results/comparison_grouped_v3/test_contact_sheet.jpg)。

## 视差图对比现状

### 1. 原始视差模型对比：已有

- `rec_img_set` 共 203 场，每场都有 IGEV++ RT 与 LiteAnyStereo 的完整/裁剪浮点
  视差、统一色标 PNG、点云和 `comparison.png`。其中 FDJYP-3 的 73 场有
  Foundation Stereo 工程参考：LAS EPE `2.0762 px`，IGEV EPE `4.6739 px`；
  详情见[正式报告](01_stereo_comparison/rec_img_set/reports/comparison_report.md)和
  [代表案例总览](01_stereo_comparison/rec_img_set/results/final_203/representative_gallery/gallery_overview.jpg)。
- Jop1 共 9 场，有左图、IGEV、LAS、PLY 稀疏参考和绝对误差六宫格；LAS/IGEV
  参考 EPE 分别为 `12.508/32.077 px`。直接查看
  [9场总览](01_stereo_comparison/jop1/results/final_9/comparison/overview.jpg)。
- 其余无正式参考的数据仍可做定性比较，但模型间差异不能称作真实误差。螺纹和
  `rectified_images/general` 还存在较明显极线风险，应先复核标定。

### 2. 掩膜后的主体视差：已有实验 5 结果

- FDJYP-0 的 18 张留出图已经输出 `subject_disparity.npy/png` 和主体 PLY，查看
  [18张总览](05_disparity_guided_segmentation/results/fdjyp0_holdout/contact_sheet.jpg)。
- FDJYP-3、luowen、general、scale、Jop1 共 130 张已有跨域主体视差总览：
  [FDJYP-3](05_disparity_guided_segmentation/results/cross_dataset_130_recall_v2/fdjyp3_contact_sheet.jpg)、
  [luowen](05_disparity_guided_segmentation/results/cross_dataset_130_recall_v2/luowen_contact_sheet.jpg)、
  [general](05_disparity_guided_segmentation/results/cross_dataset_130_recall_v2/general_1221_contact_sheet.jpg)、
  [scale](05_disparity_guided_segmentation/results/cross_dataset_130_recall_v2/scale_1221_contact_sheet.jpg)、
  [Jop1](05_disparity_guided_segmentation/results/cross_dataset_130_recall_v2/jop1_contact_sheet.jpg)。

### 3. 实验7.2/8/9主体视差统一对比：已生成

实验 8/9 的正式结果目前比较的是 `RGB / GT / Exp7.2 Mask / Exp8 Mask / Exp9 Mask`，
现已补充同一批21图的 `原始 LAS / GT主体 / Exp7.2 / Exp8 Base / Exp8 Distilled /
Exp9` 主体视差横向对比。查看
[21张总览](09_jit_mask_projection/results/subject_disparity_comparison/overview_21.jpg)或
[分组与逐图说明](09_jit_mask_projection/results/subject_disparity_comparison/README.md)。
该产物只复用已有浮点视差和 Mask，没有重新运行 LiteAnyStereo。

## 评价边界

编号表示实验脉络，不表示不同数据集上的指标可以直接横向比较。实验 5 的 FDJYP-0
18 张留出结果与实验 6～9 的 21 张多域开发比较结果不是同一数据，不能把数值高低直接
解释为模型提升。130 张人工标签只表达主体外轮廓，没有显式 `__background__` 孔洞
多边形，真实通孔仍需依靠视差连续性和后续几何规则处理。

## `datasets` 全目录去重评测（2026-08-24）

现已按唯一工业采集场景审计 `JieTai/datasets`：排除标注/训练格式副本、原始/矫正
副本、参考结果和归档后，共 353 场、11 组；其中 317 场有人工主体 Mask。统一应用
拓扑修复后，实验 8 Base 在 317 场上的逐图宏平均 IoU 为 `0.8846`，高于实验 7.2
的 `0.8431` 和蒸馏版的 `0.8089`；在完全未参与实验 8 训练的 FDJYP-0/2 共 187
场上仍以 `0.8306` 最高。完整结果见
[全量报告](08_lightweight_rgbd/results/all_datasets_353_20260824/REPORT.md)和
[11 组总览](08_lightweight_rgbd/results/all_datasets_353_20260824/overviews/)。

需要同时保留负面结论：DE0548、JXP、other_test 的盲测出现大量背景溢出，
`gongjian_test/8` 也失败，因此当前只能说“实验 8 Base 是综合最好的单模型”，不能说
“已经有模型适配 `datasets` 下全部环境”。`training/ETH3D` 是外部自然场景立体基准，
不作为工业工件主体分割集重复统计。
