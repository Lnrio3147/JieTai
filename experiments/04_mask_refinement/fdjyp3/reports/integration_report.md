# BiSeNetV2 接入 LiteAnyStereo 的 FDJYP-3 试验记录

**试验日期：** 2026-08-20  
**分割模型：** JMP 人工 ISAT 标注训练的 BiSeNetV2  
**双目模型：** LiteAnyStereo LAS1 官方权重  
**场景：** FDJYP-3 全部 73 场  
**结论：** 两阶段接入已跑通。初始硬掩码平均保留固定 ROI 中 91.29% 的参考有效像素；加入“单一前景连通域 + 视差连续暗斑修复”后为 91.36%，73/73 张掩码都严格只有一个前景连通域。改进后主体区域宏平均 EPE 为 1.4822 px、D1 为 4.64%。后置掩码只选择输出像素，不改变 LiteAnyStereo 的原始视差值，因此区域指标变化不能表述为视差网络精度提升。

## 1. 接入设计

采用如下流程：

```text
校正左图 -> BiSeNetV2 -> 左侧主体掩码 -----------------------┐
                                                             ├-> 主体视差
校正左图 + 校正右图 -> LiteAnyStereo -> 原始稠密视差 --------┘
```

具体原则：

1. BiSeNetV2 先输出左图主体掩码；
2. LiteAnyStereo 仍接收未涂黑的原始校正左右 RGB；
3. 主体掩码从 `288×512` 用最近邻插值恢复到 `720×1280`；
4. LiteAnyStereo 先完成全图匹配，再用左掩码过滤视差；
5. 掩码外的 `disp_subject.npy` 写为 NaN，掩码内视差值完全不变。

没有直接把左右图背景涂黑后送入双目网络。那样会制造两侧不一致的人工边界，并破坏主体边缘附近的匹配上下文。

### 1.1 环境解耦

两个上游项目依赖不同：

```text
BiSeNetV2:     dsrl_pi0 环境，TensorFlow 2.19 / TF1 compatibility
LiteAnyStereo: liteanystereo 环境，PyTorch 2.6.0+cu126
```

因此实验接口采用文件解耦：BiSeNetV2 输出单通道 `0/255` PNG，LiteAnyStereo 读取该掩码。这样不需要在同一个 Python 进程同时安装 TensorFlow 和 PyTorch，也便于后续将分割端替换为 ONNX/TensorRT。

## 2. 模型与数据追踪

BiSeNetV2 冻结模型：

```text
../bisenetv2-tensorflow/runs/jmp_workpiece/
  bisenetv2_manual_isat_v1_e20_retry1/
  bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb

SHA256:
b1f34a8caf2b4cab9be7a997a979dd1bb058152f771aa39c25bd37eef2bc4bee
```

LAS1 权重：

```text
./checkpoints/LiteAnyStereo.pth

SHA256:
ee0c3a0dc1d4b49cbd67edf00079b9993c0fa21f6c19a0eb812fa32f7ec1b9b1
```

数据清单：

```text
./data/datasets/JMP-LF6020-ETH3D/manifest.csv

SHA256:
2b1e7df91df8a7ef6d00840fc9e6973cbb7bd039b055eb9a56cf7aa7478628de
```

评价沿用此前统一复评口径：

| 项目 | 设置 |
| --- | --- |
| 场景 | FDJYP-3 全 73 场 |
| 左右输入 | `JMP-LF6020-ETH3D/<name>/im0.png, im1.png` |
| 原始图像尺寸 | 高 1280、宽 720 |
| 固定 ROI | `[y=234:1052, x=126:638]`，高 818、宽 512 |
| 参考视差 | `../tradition_stereo/datasets/FDJYP-3/<scene>/disp_cropped.npy` |
| 参考有效像素 | 有限且大于 0 |
| 汇总 | 先逐场计算，再做场景宏平均 |
| 最大视差 | 192 px |

FDJYP-3 没有人工主体分割 GT；本次可以评价掩码选中区域的视差误差，但不能计算主体分割的真实 IoU/Dice。

## 3. 新增接入工具

新增：

```text
tools/evaluate_bisenet_subject_stereo.py
```

功能：

- 将数据集文件名映射到传统评价场景名；
- 最近邻恢复 BiSeNetV2 掩码并裁到统一 ROI；
- `live` 模式真实加载 LAS 权重并执行前向；
- `saved` 模式读取经过验证的 LAS 浮点视差，适合批量复评；
- 同时计算全区域、主体区域和掩码外区域的 EPE/D1/Bad1/Bad2/Bad3；
- 保存每场主体视差、ROI 掩码、六宫格、逐场 CSV、汇总 JSON 和联系表；
- 拒绝覆盖已有输出目录。

对应测试：

```text
tests/test_bisenet_subject_stereo.py
```

## 4. 单场真实组合前向

先对 `fdjyp_3_1_202506281603_0001` 执行一次 full-resolution `live` 推理：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/LiteAnyStereo

/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python \
tools/evaluate_bisenet_subject_stereo.py \
  --manifest ./data/datasets/JMP-LF6020-ETH3D/manifest.csv \
  --split val --scene-prefix fdjyp_3_ \
  --mask-dir ../bisenetv2-tensorflow/runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/fdjyp_3_predictions/masks \
  --bisenet-model ../bisenetv2-tensorflow/runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --tradition-reference-root ../tradition_stereo/datasets/FDJYP-3 \
  --output-dir ./runs/inference/bisenet_las1_fdjyp3_live_smoke_v2 \
  --stereo-mode live --version las1 \
  --restore-ckpt ./checkpoints/LiteAnyStereo.pth \
  --device cuda --max-disp 192 --limit 1 --contact-sheet-samples 1
```

第一次尝试写入 `bisenet_las1_fdjyp3_live_smoke_v1`，在加载模型前因 `tools/` 入口未把仓库根目录放入 `sys.path` 而报 `ModuleNotFoundError: core`。该次没有执行网络前向或生成视差，空的失败目录已在 2026-08-20 整理时清理。修复入口后使用 `v2` 新目录重试成功并保留结果。

单场结果：

| 区域 | 有效像素 | EPE (px) | D1 (%) | Bad1 (%) | Bad2 (%) | Bad3 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全参考有效区 | 418,816 | 0.7680 | 5.93 | 19.70 | 10.78 | 5.93 |
| BiSeNetV2 主体区 | 331,519 | 0.4510 | 2.17 | 5.01 | 2.58 | 2.17 |
| 掩码外区域 | 87,297 | 1.9718 | 20.23 | 75.48 | 41.90 | 20.23 |

单次核心前向计时为 7.92 秒，包含 CUDA 首次加载和模型冷启动，不能当作稳态速度。此前 73 场正式复评的稳态 LAS1 FP32 速度为约 28.73 ms/对。

将本次 live 主体像素与此前同权重保存的浮点视差逐像素比较：

```text
主体像素数: 331,519
最大绝对差: 1.52587890625e-05 px
平均绝对差: 6.584089078387478e-07 px
np.allclose(atol=1e-5, rtol=1e-5): true
```

这验证了保存视差可以用于后续 73 场批量接入评价。

## 5. 全 73 场批量试验

执行命令：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/LiteAnyStereo

/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python \
tools/evaluate_bisenet_subject_stereo.py \
  --manifest ./data/datasets/JMP-LF6020-ETH3D/manifest.csv \
  --split val --scene-prefix fdjyp_3_ \
  --mask-dir ../bisenetv2-tensorflow/runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/fdjyp_3_predictions/masks \
  --bisenet-model ../bisenetv2-tensorflow/runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --tradition-reference-root ../tradition_stereo/datasets/FDJYP-3 \
  --output-dir ./runs/evaluation/bisenet_las1_fdjyp3_postmask_v2 \
  --stereo-mode saved \
  --las-output-root ./runs/evaluation/jmp_unified_rerun_73/liteanystereo \
  --version las1 --restore-ckpt ./checkpoints/LiteAnyStereo.pth \
  --max-disp 192 --contact-sheet-samples 18
```

### 5.1 掩码覆盖

```text
ROI 主体覆盖率平均值:       91.29%
ROI 主体覆盖率中位数:       95.33%
参考有效像素保留率最小值:   63.07%
参考有效像素保留率最大值:  100.00%
掩码外无参考有效像素的场景: 20/73
```

20 场在固定 ROI 内被预测为全部或几乎全部主体。FDJYP-3 包含大量工件贴近镜头的特写，因此这并不必然是错误，但在没有人工分割 GT 的情况下也不能确认全部正确。

### 5.2 全 73 场指标

| 评价区域 | 场景数 | 有效像素 | EPE↓ (px) | D1↓ (%) | Bad1↓ (%) | Bad2↓ (%) | Bad3↓ (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全参考有效区 | 73 | 30,573,568 | 2.0762 | 7.47 | 40.11 | 17.44 | 9.89 |
| BiSeNetV2 主体区 | 73 | 27,910,373 | **1.4851** | **4.66** | **38.66** | **14.87** | **7.15** |
| 掩码外区域 | 53 | 2,663,195 | 10.5860 | 49.19 | 68.37 | 57.84 | 50.50 |

主体区 EPE 数值比全区域低 28.47%，D1 低 37.60%；这是主体筛选后的区域差异，不是网络精度提升。程序核查 `post_mask_max_abs_change_inside_subject = 0.0`，说明后置掩码没有改变任何主体像素的视差值。

掩码外区域误差显著更高，说明 BiSeNetV2 能隔离大量黑色箱体、边缘和其他不需要的高误差区域，对后续只统计工件点云有实际价值。

### 5.3 固定 69 场指标

沿用旧协议排除 `0012`、`0019`、`0020`、`0053`：

| 评价区域 | 场景数 | EPE↓ (px) | D1↓ (%) | Bad1↓ (%) | Bad2↓ (%) | Bad3↓ (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全参考有效区 | 69 | 1.9457 | 7.03 | 38.86 | 16.02 | 8.77 |
| BiSeNetV2 主体区 | 69 | **1.3210** | **4.14** | **37.32** | **13.34** | **5.94** |
| 掩码外区域 | 52 | 10.6829 | 49.15 | 68.22 | 57.79 | 50.49 |

## 6. 目视复核

每场六宫格含：

| 左上 | 上中 | 右上 |
| --- | --- | --- |
| 校正左图 ROI | BiSeNetV2 红色叠加 | LAS1 原始视差 |
| 后置掩码主体视差 | 参考视差 | 主体绝对误差 |

代表结果：

- `202506281603-0001`：黑色箱体被排除，工件主体保留完整，主体 EPE 0.4510 px；
- `202506281615-0039`：参考有效像素保留率最低，为 63.07%，目视可见被去除部分主要是工件右侧黑色箱体；
- `202506281608-0018`：初始硬掩码把工件内部暗色、氧化表面误判为背景，主体 EPE 为 12.4907 px；这暴露出二值后处理会删除有效主体视差，后续第 9 节已针对该缺陷修订；
- 深凹槽、孔洞、强反光和靠边轮廓仍可能出现分割孔洞或边缘偏差。

因此，当前组合适合“从稠密视差中提取主体点”，不应表述为“分割后重新计算使主体视差变准”。

## 7. 结果目录与校验

正式输出：

```text
runs/evaluation/bisenet_las1_fdjyp3_postmask_v2/
  summary.json
  scene_metrics.csv
  contact_sheet.jpg
  README.md
  scenes/<scene>/foreground_mask.png
  scenes/<scene>/disp_subject.npy
  scenes/<scene>/comparison.jpg
```

共生成 73 份 ROI 掩码、73 份主体浮点视差和 73 份六宫格，目录约 126 MB。

```text
summary.json SHA256:
96e389c616120290bf621e59b96892324a15074c41a47d6096ef295d8e1b46aa

scene_metrics.csv SHA256:
506b652d91e05b21690db4e433a48a9c933f1d3c9863a43c3974570f6994e435

contact_sheet.jpg SHA256:
6d9861bd0d3cd0026f0a491588be34a2b113b287359110c8eb0bad127a6d5d71
```

## 8. 下一步建议

1. 人工修订 FDJYP-3 中 30～50 张主体掩码，测量真正的主体召回率、边界误差和被误删视差比例；
2. 对 LiteAnyStereo 的最终主体掩码做 1～3 像素安全膨胀，比较主体边缘召回与背景污染的权衡；
3. 从 `disp_subject.npy` 生成主体点云时，同时使用相机有效区、有限视差和深度范围过滤；
4. 若部署需要单进程，后续将 BiSeNetV2 转 ONNX/TensorRT，再与 LAS 推理引擎统一；本次文件接口已固定输入输出语义，可作为转换前基线。

## 9. 单连通域和内部暗斑修订

### 9.1 缺陷定位

初始后处理是：

```text
288×512 argmax 二值掩码 -> 最近邻放大 -> np.where(mask, disparity, NaN)
```

它有三个局限：

1. 没有利用 `final_probability`，低分辨率边界呈阶梯状；
2. 没有连通域约束，少量前景孤岛可能进入主体；
3. 暗色、氧化表面一旦被判为背景，会在主体视差中形成黑洞。

FDJYP-3 的平均最大类别概率为 `0.98917`。示例 `0018` 的主暗斑前景概率中位数只有约 `0.129`，属于高置信错误，不能靠把阈值从 0.5 略微降低来稳定补回。

根据业务先验“当前工件是一个连通主体”，新增：

```text
tools/refine_bisenet_subject_masks.py
tests/test_refine_bisenet_subject_masks.py
```

修订流程：

```text
前景概率双线性恢复到 720×1280
  -> 0.5 阈值和 3 px 半径闭运算
  -> 只保留最大 8 连通前景，删除孤岛
  -> 检查被主体包围的背景区域
  -> 面积受控且与周边前景视差中位数连续时补回
  -> 再次执行最大连通域检查，保证最终恰好一个前景连通域
```

内部候选区的判断只读取 LiteAnyStereo 原始预测视差，不读取传统视差参考或人工分割 GT，避免评价标签泄漏。参数如下：

| 参数 | 数值 | 作用 |
| --- | ---: | --- |
| 分割阈值 | 0.5 | 初始前景 |
| 闭运算半径 | 3 px | 平滑小裂缝和阶梯边界 |
| 邻域环半径 | 7 px | 统计候选区周边主体视差 |
| 绝对视差容差 | 1.5 px | 周边 MAD 很小时的下限 |
| MAD 倍数 | 1.0 | 自适应允许视差变化 |
| 最大补洞面积 | 全图 2.5% | 避免吞入大面积遮挡或真实背景 |
| ROI 外无视差小洞上限 | 1,000 px | 仅处理极小孤立孔 |

### 9.2 执行命令和一次环境重试

BiSeNetV2 概率导出命令记录在人工训练文档第 13 节。随后执行：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/LiteAnyStereo

python3 tools/refine_bisenet_subject_masks.py \
  --manifest ./data/datasets/JMP-LF6020-ETH3D/manifest.csv \
  --split val --scene-prefix fdjyp_3_ \
  --probability-dir ../bisenetv2-tensorflow/runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/fdjyp_3_predictions_probability_v2/probabilities \
  --las-output-root ./runs/evaluation/jmp_unified_rerun_73/liteanystereo \
  --output-dir ./runs/evaluation/bisenet_fdjyp3_mask_refinement_v1 \
  --threshold 0.5 --closing-radius 3 \
  --hole-ring-radius 7 --hole-absolute-tolerance 1.5 \
  --hole-mad-scale 1.0 --max-fill-hole-fraction 0.025 \
  --small-hole-area 1000 --contact-sheet-samples 24
```

复评继续复用同一批已经逐像素验证的 LAS1 浮点视差，唯一变化是掩码。第一次误用系统旧版 `python3` 执行评估，因不支持脚本中的字典合并语法而在首场退出，曾留下不完整目录 `bisenet_las1_fdjyp3_refined_postmask_v1`（整理时已清理）；没有修改模型或原视差。改用此前验证过的 Python 3.11 环境并写入新目录：

```bash
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
tools/evaluate_bisenet_subject_stereo.py \
  --manifest ./data/datasets/JMP-LF6020-ETH3D/manifest.csv \
  --split val --scene-prefix fdjyp_3_ \
  --mask-dir ./runs/evaluation/bisenet_fdjyp3_mask_refinement_v1/masks \
  --bisenet-model ../bisenetv2-tensorflow/runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --tradition-reference-root ../tradition_stereo/datasets/FDJYP-3 \
  --output-dir ./runs/evaluation/bisenet_las1_fdjyp3_refined_postmask_v2 \
  --stereo-mode saved \
  --las-output-root ./runs/evaluation/jmp_unified_rerun_73/liteanystereo \
  --version las1 --restore-ckpt ./checkpoints/LiteAnyStereo.pth \
  --max-disp 192 --contact-sheet-samples 24
```

### 9.3 连通性和修订统计

| 项目 | 结果 |
| --- | ---: |
| 场景 | 73 |
| 闭运算后的前景连通域总数 | 83 |
| 删除的前景孤岛像素 | 9,979 |
| 检查的内部候选暗区 | 19 |
| 按视差/面积规则补回 | 14 个，27,794 px |
| 保留为背景 | 5 个 |
| 最终恰好一个前景连通域 | **73/73** |

`0018` 的主暗斑为 15,388 px，候选区/周边主体的视差中位数分别为 `173.07/173.57 px`，差 `0.49 px`，小于周边 MAD `7.32 px`，因此补回。`0056/0057` 的长条背景分别约为 27,813/39,638 px，超过全图 2.5% 的面积上限，因此没有被无条件填充。

### 9.4 初始硬掩码与改进掩码对比

全 73 场使用相同原始视差和相同参考：

| 指标 | 初始硬掩码 | 改进掩码 | 变化 |
| --- | ---: | ---: | ---: |
| 平均参考有效像素保留率 | 91.2892% | **91.3590%** | +0.0698 pp |
| 中位参考有效像素保留率 | 95.3287% | **95.6093%** | +0.2806 pp |
| 主体有效像素 | 27,910,373 | **27,931,721** | +21,348 |
| 主体 EPE | 1.48513 px | **1.48216 px** | -0.00297 px |
| 主体 D1 | 4.65913% | **4.63913%** | -0.02000 pp |
| 主体 Bad1 | **38.65809%** | 38.68146% | +0.02337 pp |
| 主体 Bad2 | **14.87426%** | 14.89340% | +0.01915 pp |
| 主体 Bad3 | **7.15171%** | 7.16886% | +0.01715 pp |

固定 69 场中，主体 EPE 从 `1.32098` 降到 `1.31785 px`，D1 从 `4.13694%` 降到 `4.11615%`。

指标没有单向全面改善是合理的：补回此前被删掉的困难主体像素会改变评价像素集合。关键结果是主体完整性提高的同时，整体 EPE/D1 没有恶化；Bad1/2/3 的轻微上升小于 0.024 个百分点。所有保留位置的 LAS 视差仍逐值不变，`post_mask_max_abs_change_inside_subject = 0.0`。

缺陷场景 `202506281608-0018`：

| 指标 | 初始硬掩码 | 改进掩码 | 变化 |
| --- | ---: | ---: | ---: |
| 参考有效像素保留率 | 92.6216% | **96.3191%** | +3.6976 pp |
| 主体有效像素 | 387,914 | **403,400** | +15,486 |
| 主体 EPE | 12.4907 px | **12.2093 px** | -0.2814 px |
| 主体 D1 | 42.6195% | **41.0032%** | -1.6163 pp |

直接目视对比：

![0018 原掩码与改进掩码诊断](../results/mask_refinement/diagnostics/fdjyp_3_1_202506281608_0018.jpg)

初始和改进后的主体视差六宫格分别保存在：

```text
runs/evaluation/bisenet_las1_fdjyp3_postmask_v2/scenes/202506281608-0018/comparison.jpg
runs/evaluation/bisenet_las1_fdjyp3_refined_postmask_v2/scenes/202506281608-0018/comparison.jpg
```

### 9.5 结果目录与校验

```text
runs/evaluation/bisenet_fdjyp3_mask_refinement_v1/
  masks/                       # 73 张全分辨率单连通域掩码
  diagnostics/                 # 原/改掩码和原始视差逐图诊断
  refinement.csv
  hole_decisions.json
  metadata.json
  refinement_contact_sheet.jpg

runs/evaluation/bisenet_las1_fdjyp3_refined_postmask_v2/
  summary.json
  scene_metrics.csv
  contact_sheet.jpg
  scenes/<scene>/{foreground_mask.png,disp_subject.npy,comparison.jpg}
```

```text
mask refinement metadata.json SHA256:
bec0fa28dd63672ee31264d99b5c2ec7dd83b0d9036b42813cd79579052d92e9

refined post-mask summary.json SHA256:
f54e8f593da7ae653becf41c21db17048cf93bca8ab89f9fb8c09e314d681ce8

refined post-mask scene_metrics.csv SHA256:
d1cb12879057145f02200e6b1f52909ffc486fa4933c613d2968f8b47fb6c9e1

refined post-mask contact_sheet.jpg SHA256:
e6b7286d519c245adf82dca65bff42158f214ad98dda1e7bd083cbac5fe77020
```

当前规则已解决示例暗斑并强制单主体连通，但它仍是工程先验，不是分割 GT。下一步最有价值的工作仍是给 FDJYP-3 建立冻结的人工测试掩码，分别统计前景召回、边界误差、误删主体视差和误纳背景视差，再决定面积上限和 MAD 阈值是否需要调整。
