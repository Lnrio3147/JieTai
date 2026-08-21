# JMP 工件分割与 LiteAnyStereo 接入阶段汇报

**汇报范围：** 从 BiSeNetV2 源码落地，到数据准备、训练、FDJYP-3 预测、LiteAnyStereo 接入、缺陷分析和单连通域修订的完整过程。  
**工作日期：** 2026-08-18 ～ 2026-08-20  
**当前状态：** 训练、冻结导出、73 场预测、端到端接入、前后对比和第一轮掩码改进均已完成。  
**核心原则：** LiteAnyStereo 始终读取原始校正左右图；BiSeNetV2 只在视差生成后筛选主体，不通过涂黑输入背景改变双目匹配。

## 一、汇报摘要

本阶段已经打通以下链路：

```text
人工标注左图
  -> BiSeNetV2 训练与冻结模型
  -> FDJYP-3 左图主体概率/掩码
  -> 单一前景连通域和内部暗斑修订
  -> LiteAnyStereo 原图双目视差
  -> 后置主体掩码
  -> 主体视差、逐场指标和可视化
```

主要结果：

| 项目 | 结果 |
| --- | ---: |
| 人工标注输入 | 187 张图、187 个 ISAT JSON、188 个 `jinshu` 多边形对象 |
| 人工训练/验证 | 145/42，按完整采集组隔离 |
| BiSeNetV2 人工验证前景 IoU | **0.97706** |
| 人工验证 Dice | **0.98840** |
| 冻结模型 | 约 9.7 MiB，SHA256 `b1f34a8...4bee` |
| FDJYP-3 独立预测 | 73/73 完成 |
| 初始主体 EPE / D1 | 1.48513 px / 4.65913% |
| 改进主体 EPE / D1 | **1.48216 px / 4.63913%** |
| 改进后前景连通性 | **73/73 恰好一个 8 连通域** |
| 补回内部候选暗区 | 14 个、27,794 px |
| 缺陷场景 `0018` 保留率 | 92.62% -> **96.32%** |

结论分为三层：

1. **工程链路结论：** BiSeNetV2 与 LiteAnyStereo 已可靠串联，模型、数据、命令、日志和输出均按版本保留；
2. **掩码结论：** 单连通域和视差连续性规则解决了示例中的暗斑误删，同时没有吞入已识别的大面积长条背景；
3. **精度结论：** 主体区指标低于全区域，说明掩码隔离了大量高误差背景，但后置掩码没有修改 LAS 的像素预测值，不能把区域差值宣称为“视差模型精度提升”。

## 二、阶段总览

| 阶段 | 日期 | 目标 | 状态 | 主要产物 |
| --- | --- | --- | --- | --- |
| 0. 源码和环境落地 | 08-18 | 获取 BiSeNetV2、确认 LiteAnyStereo 版本和运行环境 | 完成 | 两个可追踪源码仓库 |
| 1. 数据审计与伪标签基线 | 08-18 | 在没有人工标签时先验证训练链路 | 完成 | 187/73 伪标签、5 epoch 基线 |
| 2. 人工修订规范 | 08-18～19 | 固定标注口径、版本和复核方法 | 完成 | 人工掩码操作规范和 QA 流程 |
| 3. 人工 ISAT 数据转换 | 08-19 | 将现有 187 份标注转换为可训练数据 | 完成 | `manual-isat-v1` 数据集 |
| 4. 正式训练和冻结 | 08-19 | 得到可部署 BiSeNetV2 | 完成 | 20 epoch 最佳 checkpoint 和 PB |
| 5. FDJYP-3 独立预测 | 08-19～20 | 检查新场景泛化并输出概率/掩码 | 完成 | 73 张掩码、概率和联系表 |
| 6. LiteAnyStereo 接入 | 08-20 | 生成主体视差并与不筛选结果对比 | 完成 | 单场 live 验证、73 场正式复评 |
| 7. 缺陷分析和掩码改进 | 08-20 | 消除前景孤岛，补回工件内部暗斑 | 完成 | 单连通域掩码和改进复评 |
| 8. 汇报与复现固化 | 08-20 | 汇总路径、命令、异常、指标和哈希 | 完成 | 本文和三份详细技术文档 |

## 三、阶段 0：源码和环境落地

### 3.1 源码位置和版本

工作目录：

```text
/home/uestc/mount_2T/uestc/lnrio/JieTai/projects
```

BiSeNetV2：

```text
仓库:   https://github.com/MaybeShewill-CV/bisenetv2-tensorflow.git
目录:   bisenetv2-tensorflow/
提交:   54075548018f113bf21ca2cc4e78cee63523d7a9
状态:   shallow + partial + sparse checkout
```

最终源码获取方式等价于：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/MaybeShewill-CV/bisenetv2-tensorflow.git \
  /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow

git -C /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow \
  sparse-checkout set bisenet_model config data_provider local_utils scripts tools trainner
```

LiteAnyStereo：

```text
仓库:   https://github.com/TomTomTommi/LiteAnyStereo.git
目录:   LiteAnyStereo/
提交:   8c97bd4c4da3712c2ac60003a23201dfdb5935f4
```

### 3.2 环境解耦

| 模块 | 环境 | 主要依赖 |
| --- | --- | --- |
| BiSeNetV2 | `dsrl_pi0` | Python 3.11、TensorFlow 2.19、`tf_keras` 2.19 |
| LiteAnyStereo | `liteanystereo` | PyTorch 2.6.0 + CUDA 12.6 |

两个模型没有强行安装到同一个 Python 环境。接口固定为 PNG 掩码、NumPy 概率和 NumPy 视差，便于单独复现，也方便后续替换为 ONNX/TensorRT。

### 3.3 TensorFlow 兼容修改

上游项目面向 TensorFlow 1.15，现有环境是 2.19，完成以下最小兼容：

- `bisenet_model/cnn_basenet.py`、`bisenet_model/bisenet_v2.py` 使用 `tensorflow.compat.v1`；
- 关闭 TensorFlow v2 behavior；
- 通道拆分的 `/` 改为整数除法 `//`；
- 用 `TF_USE_LEGACY_KERAS=1` 保留旧版 `tf.layers` 行为；
- 新训练流程直接用 NumPy/OpenCV 读取 JMP 数据，不依赖原 Cityscapes TFRecord 管线。

## 四、阶段 1：数据审计与伪标签基线

### 4.1 原始数据审计

统一清单：

```text
LiteAnyStereo/data/datasets/JMP-LF6020-ETH3D/manifest.csv
SHA256: 2b1e7df91df8a7ef6d00840fc9e6973cbb7bd039b055eb9a56cf7aa7478628de
```

| 场景 | 数量 | 用途 |
| --- | ---: | --- |
| FDJYP-0 | 82 | 训练候选 |
| FDJYP-2 | 105 | 训练候选 |
| FDJYP-3 | 73 | 独立验证/接入试验 |
| DE0548 | 6 | 域差异过大，排除 |

确认 `mask0nocc.png` 只是视差有效矩形，不是工件语义标签；`disp0GT.pfm` 也包含背景和夹具，不能直接转成主体掩码。

### 4.2 伪标签目的和方法

当时尚无可用人工掩码，先用 Otsu + GrabCut 生成 `187/73` 张弱监督标签，目的是验证数据读取、模型前后向、验证、checkpoint 和冻结导出，不把伪标签指标当作真实分割精度。

新增：

```text
config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml
tools/jmp_workpiece/prepare_pseudo_dataset.py
tools/jmp_workpiece/train_bisenetv2_jmp.py
tools/jmp_workpiece/export_bisenetv2_jmp.py
```

数据输出：

```text
LiteAnyStereo/data/datasets/JMP-workpiece-seg-pseudo-v1/
```

### 4.3 异常和处理

| 异常 | 定位 | 处理 |
| --- | --- | --- |
| 首次伪标签生成被外部时限中断 | 输出目录不完整 | 移到 `/tmp/JMP-workpiece-seg-pseudo-v1-incomplete-20260818`，新目录完整重跑 |
| 首次 5 epoch 验证全为 NaN | 187 张、batch 2 的最后一批只有 1 张；全局池化后 `1×1` BatchNorm 方差污染 | 每轮丢弃不完整尾批、禁止 batch 小于 2、增加非有限 loss 检查 |

失败目录原名为 `bisenetv2_pseudo_v1_e5`，不用于导出，其结论和原因保留在本文中，运行产物已于 2026-08-20 清理；修复后正式目录 `bisenetv2_pseudo_v1_e5_bnfix` 完整保留。

### 4.4 伪标签基线结果

5 epoch 最佳结果：

| 指标 | 数值 |
| --- | ---: |
| 前景 IoU（相对伪标签） | 0.88297 |
| mIoU（相对伪标签） | 0.79730 |
| Dice（相对伪标签） | 0.93785 |
| 像素准确率 | 0.90919 |

该阶段证明训练工程可用，同时暴露亮度伪标签会漏掉暗色/氧化工件，不能作为最终模型监督。

## 五、阶段 2：人工修订规范

在正式使用人工标签前，先固定以下规则：

- `0=背景`，`255=当前待测工件可见实体表面`；
- 暗色、氧化、高光和亮度变化不改变类别；
- 箱体、夹具、阴影和通孔内可见背景不属于工件；
- 原伪标签只作起始参考，必须版本化保存，不覆盖旧数据；
- 最终 PNG 必须为 `288×512`、单通道 `uint8`、只含 `0/255`；
- 每张标注记录标注人、复核人、状态和备注；
- 验证图不能回流训练集，避免数据泄漏；
- 目视检查原图/叠加/纯掩码，并由第二人复核困难边界。

详细操作、CVAT/GIMP/Krita 流程、自动格式检查和返工规则见：

```text
experiments/02_initial_segmentation/fdjyp3/reports/training_report.md
第 11 节《人工修订掩码操作规范》
```

## 六、阶段 3：人工 ISAT 数据转换

### 6.1 输入审计

用户已在以下目录完成图像分割标注：

```text
LiteAnyStereo/data/datasets/JMP-workpiece-seg-pseudo-v1/images/train/
```

审计结果：

| 项目 | 结果 |
| --- | ---: |
| PNG / 同名 JSON | 187 / 187 |
| 缺失文件 | 0 |
| 类别 | 全部 `jinshu` |
| 标注对象 | 188 |
| 空标注/非法多边形 | 0 / 0 |

### 6.2 转换和划分

新增：

```text
tools/jmp_workpiece/prepare_isat_manual_dataset.py
```

人工数据版本：

```text
LiteAnyStereo/data/datasets/JMP-workpiece-seg-manual-isat-v1/
```

没有随机逐帧拆分，而是按完整采集子序列隔离：

```text
验证: fdjyp_0_2（18）+ fdjyp_2_3（24）= 42
训练: 其他 8 个采集组                  = 145
```

这样避免相邻帧分别进入训练和验证造成过于乐观的指标。全部 187 张通过尺寸、类型、取值、文件名和联系表目视检查。

## 七、阶段 4：正式训练和冻结模型

### 7.1 训练设置

```text
输入尺寸:       512×288
类别数:         2
batch size:     2
epoch:          20
优化器:         Adam
学习率:         0.001 -> 0.00001，polynomial decay
随机种子:       20260819
初始化:         随机初始化，不加载伪标签 checkpoint
最佳模型选择:   人工验证集 foreground IoU
```

训练集 145 张为奇数；为避免前述 BatchNorm 问题，每轮使用 144 张完整 batch，并在每轮重新打乱。

第一次正式任务在首轮指标前收到外部 `SIGTERM`，退出码 143，无 OOM、NaN 或 Python 异常。失败目录原名为 `bisenetv2_manual_isat_v1_e20`，运行产物已于 2026-08-20 清理；同参数使用新目录 `bisenetv2_manual_isat_v1_e20_retry1` 重跑成功并完整保留。

### 7.2 最终指标

最佳 checkpoint 为第 20 轮：

| 指标 | 数值 |
| --- | ---: |
| train loss | 0.69191 |
| val loss | 0.65302 |
| 前景 IoU | **0.9770608** |
| mIoU | **0.9806120** |
| Dice | **0.9883973** |
| 像素准确率 | **0.9905426** |

对 651 个 checkpoint 数值张量逐一检查，全部有限。

### 7.3 冻结模型

```text
路径:
bisenetv2-tensorflow/runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/
  bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb

SHA256:
b1f34a8caf2b4cab9be7a997a979dd1bb058152f771aa39c25bd37eef2bc4bee
```

接口：

```text
input_tensor:0       float32 [1,512,288,3]
final_probability:0  float32 [1,512,288,2]
final_output:0       int64   [512,288]
```

## 八、阶段 5：FDJYP-3 独立预测

新增：

```text
tools/jmp_workpiece/predict_bisenetv2_jmp.py
```

对 73 张 `fdjyp_3_*.png` 全部推理，输出硬掩码、叠加图、CSV、元数据和联系表。统计：

```text
预测前景比例: min 0.61599, median 0.84743, mean 0.83567, max 1.00000
平均最大类别概率: mean 0.98917
低于 0.75 的像素比例: mean 0.01380
```

FDJYP-3 没有人工分割 GT，因此不能计算可信的 IoU/Dice。与旧伪标签的一致度只作诊断，不作为真实精度。

在缺陷分析阶段为脚本补充 `--save_probabilities`，生成 73 张 `[512,288] float32` 前景概率图：

```text
bisenetv2_manual_isat_v1_e20_retry1/fdjyp_3_predictions_probability_v2/
```

高平均置信度说明示例暗斑并非普通阈值抖动，而是跨域后的高置信误分类，需要拓扑先验或补充标注。

## 九、阶段 6：接入 LiteAnyStereo

### 9.1 为什么采用后置掩码

最终方案：

```text
左图 -> BiSeNetV2 -> 主体掩码 -----------------------┐
                                                       ├-> 主体视差
原始校正左图 + 右图 -> LiteAnyStereo -> 稠密视差 ----┘
```

不把背景先涂黑再送入 LiteAnyStereo，原因是左右掩码边界不完全一致会制造人工匹配边缘，主体边缘也会失去必要上下文。当前方案保证：

- LiteAnyStereo 输入和“不接入 BiSeNetV2”完全相同；
- 主体内视差值逐像素不变；
- 掩码外写为 NaN，只改变输出范围。

### 9.2 工具和验证

新增：

```text
tools/evaluate_bisenet_subject_stereo.py
tests/test_bisenet_subject_stereo.py
```

首次单场 live 运行因工具入口缺少仓库根目录而报 `ModuleNotFoundError: core`，未加载网络。修复 `sys.path` 后使用新目录 `bisenet_las1_fdjyp3_live_smoke_v2` 成功。

场景 `0001` 的 live 结果：

| 区域 | EPE | D1 |
| --- | ---: | ---: |
| 全参考有效区 | 0.7680 px | 5.93% |
| 主体区 | 0.4510 px | 2.17% |
| 掩码外 | 1.9718 px | 20.23% |

live 视差与此前同权重保存结果在主体内最大绝对差 `1.53e-5 px`、平均差 `6.58e-7 px`，证明可以安全复用保存视差完成全量复评。

### 9.3 与不接入 BiSeNetV2 的初始对比

全 73 场：

| 输出区域 | 有效像素 | EPE | D1 | Bad1 | Bad2 | Bad3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 不筛选：全参考有效区 | 30,573,568 | 2.0762 | 7.47% | 40.11% | 17.44% | 9.89% |
| 初始 BiSeNetV2 主体区 | 27,910,373 | 1.4851 | 4.66% | 38.66% | 14.87% | 7.15% |
| 掩码外区域 | 2,663,195 | 10.5860 | 49.19% | 68.37% | 57.84% | 50.50% |

主体区 EPE 数值比全区低 28.47%，D1 低 37.60%，表明大量高误差背景被排除。两行使用不同像素集合，不能把差值解释为分割令 LAS 在同一主体像素上变准。

初始结果：

```text
LiteAnyStereo/runs/evaluation/bisenet_las1_fdjyp3_postmask_v2/
```

## 十、阶段 7：缺陷分析和单连通域修订

### 10.1 发现的问题

场景 `202506281608-0018` 中，暗色/氧化表面被硬掩码误删，导致主体视差出现黑洞。初始处理没有概率平滑、连通域过滤或内部孔洞判断。

### 10.2 改进方法

新增：

```text
tools/refine_bisenet_subject_masks.py
tests/test_refine_bisenet_subject_masks.py
```

流程：

1. 前景概率双线性恢复到原图；
2. 0.5 阈值和 3 px 半径闭运算；
3. 只保留最大 8 连通前景，删除孤岛；
4. 查找被主体包围的背景候选区；
5. 面积不超过全图 2.5%，且候选区视差中位数与周边前景差值不超过 `max(1.5 px, 周边 MAD)` 时补回；
6. 最终再次验证只有一个前景连通域。

规则不使用人工 GT 或参考视差。大面积遮挡不会只因“被包围”而被无条件填充。

### 10.3 连通性结果

| 项目 | 结果 |
| --- | ---: |
| 原始候选前景连通域总数 | 83 |
| 删除孤岛像素 | 9,979 |
| 内部背景候选 | 19 |
| 补回 | 14 个、27,794 px |
| 保留背景 | 5 个 |
| 最终单连通掩码 | **73/73** |

### 10.4 全量前后对比

| 指标 | 初始硬掩码 | 改进掩码 | 变化 |
| --- | ---: | ---: | ---: |
| 平均参考有效像素保留率 | 91.2892% | **91.3590%** | +0.0698 pp |
| 中位保留率 | 95.3287% | **95.6093%** | +0.2806 pp |
| 主体有效像素 | 27,910,373 | **27,931,721** | +21,348 |
| 主体 EPE | 1.48513 px | **1.48216 px** | -0.00297 px |
| 主体 D1 | 4.65913% | **4.63913%** | -0.02000 pp |
| 主体 Bad1 | **38.65809%** | 38.68146% | +0.02337 pp |
| 主体 Bad2 | **14.87426%** | 14.89340% | +0.01915 pp |
| 主体 Bad3 | **7.15171%** | 7.16886% | +0.01715 pp |

总体 EPE/D1 小幅下降，Bad 指标小幅上升，说明补回区域包含困难主体像素。变化幅度很小，重点应放在掩码完整性，而不是把它包装成精度提升。

### 10.5 缺陷场景 `0018`

| 指标 | 初始 | 改进 | 变化 |
| --- | ---: | ---: | ---: |
| 保留率 | 92.6216% | **96.3191%** | +3.6976 pp |
| 主体像素 | 387,914 | **403,400** | +15,486 |
| EPE | 12.4907 px | **12.2093 px** | -0.2814 px |
| D1 | 42.6195% | **41.0032%** | -1.6163 pp |

![0018 原掩码与改进掩码](../results/mask_refinement/diagnostics/fdjyp_3_1_202506281608_0018.jpg)

改进结果：

```text
LiteAnyStereo/runs/evaluation/bisenet_fdjyp3_mask_refinement_v1/
LiteAnyStereo/runs/evaluation/bisenet_las1_fdjyp3_refined_postmask_v2/
```

复评第一次误用系统旧版 Python，在首场因字典合并语法不兼容退出；不完整 `v1` 目录保留。使用 Python 3.11 写入 `v2` 后 73 场全部完成。

## 十一、当前产物清单

### 11.1 核心脚本

| 仓库 | 文件 | 用途 |
| --- | --- | --- |
| BiSeNetV2 | `tools/jmp_workpiece/prepare_pseudo_dataset.py` | 生成伪标签基线 |
| BiSeNetV2 | `tools/jmp_workpiece/prepare_isat_manual_dataset.py` | 转换和质检 ISAT 人工标注 |
| BiSeNetV2 | `tools/jmp_workpiece/train_bisenetv2_jmp.py` | 训练与验证 |
| BiSeNetV2 | `tools/jmp_workpiece/export_bisenetv2_jmp.py` | 冻结 PB |
| BiSeNetV2 | `tools/jmp_workpiece/predict_bisenetv2_jmp.py` | 输出掩码、置信度和概率 |
| LiteAnyStereo | `tools/evaluate_bisenet_subject_stereo.py` | 后置掩码、指标和可视化 |
| LiteAnyStereo | `tools/refine_bisenet_subject_masks.py` | 单连通域和暗斑修订 |

### 11.2 详细文档

| 文档 | 内容 |
| --- | --- |
| `experiments/02_initial_segmentation/fdjyp3/reports/training_report.md` | 伪标签、兼容修改、失败定位、人工修订规范 |
| `experiments/03_manual_segmentation/fdjyp3/reports/training_report.md` | 人工数据、20 epoch 训练、冻结和 FDJYP-3 预测 |
| `experiments/04_mask_refinement/fdjyp3/reports/integration_report.md` | 接入设计、73 场初始结果、缺陷改进和完整命令 |
| 本文 | 面向汇报的阶段总结和关键结论 |

### 11.3 关键哈希

```text
数据 manifest:
2b1e7df91df8a7ef6d00840fc9e6973cbb7bd039b055eb9a56cf7aa7478628de

BiSeNetV2 frozen PB:
b1f34a8caf2b4cab9be7a997a979dd1bb058152f771aa39c25bd37eef2bc4bee

LiteAnyStereo checkpoint:
ee0c3a0dc1d4b49cbd67edf00079b9993c0fa21f6c19a0eb812fa32f7ec1b9b1

初始复评 summary.json:
96e389c616120290bf621e59b96892324a15074c41a47d6096ef295d8e1b46aa

改进规则 metadata.json:
bec0fa28dd63672ee31264d99b5c2ec7dd83b0d9036b42813cd79579052d92e9

改进复评 summary.json:
f54e8f593da7ae653becf41c21db17048cf93bca8ab89f9fb8c09e314d681ce8
```

## 十二、验证状态

已经完成：

- BiSeNetV2 TF2.19 图构建、训练、验证、冻结和 PB 实际加载；
- checkpoint 651 个张量有限值检查；
- FDJYP-3 73 张概率和掩码实际推理；
- 单场 LAS1 live 推理与保存结果逐像素比对；
- 73 场初始/改进掩码复评；
- 73 张最终掩码前景连通域检查；
- 新增掩码修订逻辑 3 个合成单元测试；
- 旧接入逻辑 3 个单元测试和既有训练测试；
- 各版本输出目录拒绝覆盖，失败目录和成功目录分开保留。

## 十三、风险和下一阶段建议

当前仍有三项边界：

1. **缺少 FDJYP-3 人工分割 GT。** 当前只能证明示例缺陷被修复、拓扑约束满足，不能给出 FDJYP-3 的真实前景召回和 IoU；
2. **拓扑规则是业务先验。** 如果未来工件本身存在真实通孔、被遮挡后分成多个可见区域，必须调整“单连通域/补洞”定义；
3. **后置掩码不修复 LAS 本身。** 场景 `0018` 即使补齐掩码，主体 EPE 仍高，说明还存在双目匹配困难。

建议按优先级推进：

1. 固定 FDJYP-3 中 30～50 张，人工标注并冻结为独立测试集；
2. 增加四个直接业务指标：主体召回率、边界 F-score、误删有效视差比例、误纳背景视差比例；
3. 根据人工 GT 调整 2.5% 面积上限、MAD 倍数和闭运算半径，只在开发集调参；
4. 对 `0018` 等 LAS 高误差场景单独分析曝光、纹理重复、反光和视差范围，不与分割问题混为一谈；
5. 部署阶段将 BiSeNetV2 转 ONNX/TensorRT，并保持当前“原图双目 + 后置掩码”的输入输出语义。

## 十四、汇报时建议使用的表述

推荐：

> 已完成基于人工标注的 BiSeNetV2 工件分割训练，并以文件解耦方式接入 LiteAnyStereo。通过单连通域和视差连续性后处理，73 张 FDJYP-3 最终掩码全部满足单主体约束，示例暗斑保留率提高 3.70 个百分点；在不改变原始视差值的前提下，整体主体 EPE/D1 保持稳定并略有下降。

不推荐：

> 接入 BiSeNetV2 后 LiteAnyStereo 精度提高 28%。

原因是 28% 来自评价区域从全图变为主体，并非同一批像素的模型预测改善。
