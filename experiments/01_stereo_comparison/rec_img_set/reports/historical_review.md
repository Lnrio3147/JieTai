# LiteAnyStereo 与一期 RT-IGEV 统一基准复评报告

**报告日期：** 2026-08-13  
**当前模型：** LiteAnyStereo LAS1（官方权重，本次重新推理）  
**一期模型：** RT-IGEV（使用 `tradition_stereo/igev_output` 保存的原始浮点视差）  
**核心结论：** 统一评价代码后，LiteAnyStereo 在全 73 场的五项指标全部更好；固定 69 场中 EPE、D1、Bad2、Bad3 更好，Bad1 与 RT-IGEV 基本持平但高 0.18 个百分点。

## 1. 为什么需要重新评价

上一版直接读取了 `IGEV_metrics.csv`，但进一步核查发现：

1. `IGEV_metrics.csv` 只有 72 场，缺少高误差场景 `202506281608-0018`；
2. CSV 中的部分场景数值与 `igev_output/<scene>/disp.npy` 按固定 ROI 重新计算的数值不一致，说明它不是当前 `igev_output` 原始结果的严格配套汇总；
3. 旧 CSV 已经应用过自身的场景过滤，而 LiteAnyStereo 使用的是未过滤场景，直接平均会造成评价集合不一致。

因此本次不再采用旧 CSV 的平均值，而是从两种算法的浮点视差重新计算全部指标。

## 2. 本次实际执行内容

本机没有找到一期 RT-IGEV 的代码和 checkpoint，无法重新执行其网络前向；但 `tradition_stereo/igev_output` 保存了完整的 73 场原始浮点 `disp.npy`，足以按新口径重新评价。

LiteAnyStereo 已使用官方权重重新推理全部 73 场，并额外保存浮点视差：

```text
runs/evaluation/jmp_unified_rerun_73/liteanystereo/<scene>/disp.npy
```

两种算法统一使用以下配置：

| 项目 | 统一设置 |
| :--- | :--- |
| RT-IGEV 预测 | `../tradition_stereo/igev_output/<scene>/disp.npy` |
| RT-IGEV ROI | 固定裁剪 `[234:1052, 126:638]` |
| LiteAnyStereo 预测 | 本次重新推理保存的 `<scene>/disp.npy` |
| 评价参考 | `../tradition_stereo/datasets/FDJYP-3/<scene>/disp_cropped.npy` |
| 评价尺寸 | 818×512 |
| 有效掩码 | 参考视差有限且大于 0 |
| 指标实现 | 两种算法调用同一个指标函数 |
| 汇总方式 | 先逐场景计算，再做场景宏平均 |
| EPE 场景过滤 | 关闭 |
| 最大预测视差 | 192 px |

本报告同时给出：

- 全部 73 场结果，不排除任何场景；
- 沿用旧工程固定排除 4 场后的 69 场结果。

## 3. 全部 73 场统一结果

### 表 1  全 73 场宏平均（三线表）

| 算法 | EPE↓ (px) | D1↓ (%) | Bad1↓ (%) | Bad2↓ (%) | Bad3↓ (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| RT-IGEV 保存结果 | 4.6745 | 10.64 | 40.65 | 19.78 | 12.65 |
| LiteAnyStereo LAS1 | **2.0762** | **7.47** | **40.11** | **17.44** | **9.89** |

### 表 2  LiteAnyStereo 相对 RT-IGEV 的改善（三线表）

| 指标 | EPE | D1 | Bad1 | Bad2 | Bad3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 绝对降低 | 2.5983 px | 3.17 个百分点 | 0.54 个百分点 | 2.34 个百分点 | 2.77 个百分点 |
| 相对降低 | **55.59%** | **29.81%** | **1.34%** | **11.85%** | **21.87%** |

![全 73 场统一指标](../results/baseline_73/unified_comparison_all73.png)

全量评价中 LiteAnyStereo 五项指标全部更低，说明它对旧方案异常场景的鲁棒性明显更好。

## 4. 固定 69 场统一结果

固定排除场景为 `0012`、`0019`、`0020`、`0053`，没有进行额外 EPE 过滤。

### 表 3  固定 69 场宏平均（三线表）

| 算法 | EPE↓ (px) | D1↓ (%) | Bad1↓ (%) | Bad2↓ (%) | Bad3↓ (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| RT-IGEV 保存结果 | 3.4483 | 8.90 | **38.68** | 17.35 | 10.29 |
| LiteAnyStereo LAS1 | **1.9457** | **7.03** | 38.86 | **16.02** | **8.77** |

### 表 4  LiteAnyStereo 相对 RT-IGEV 的变化（三线表）

| 指标 | EPE | D1 | Bad1 | Bad2 | Bad3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 绝对变化 | 降低 1.5026 px | 降低 1.87 个百分点 | 增加 0.18 个百分点 | 降低 1.33 个百分点 | 降低 1.52 个百分点 |
| 相对变化 | **改善 43.58%** | **改善 21.05%** | 变差 0.47% | **改善 7.67%** | **改善 14.79%** |

![固定 69 场统一指标](../results/baseline_73/unified_comparison_fixed69.png)

固定 69 场中 LiteAnyStereo 在四项指标上更好。Bad1 为 38.86%，RT-IGEV 为 38.68%，相差仅 0.18 个百分点，可以判断为基本持平，而不能写成五项全面领先。

## 5. 场景分布与结果解释

| 统计项 | RT-IGEV | LiteAnyStereo |
| :--- | ---: | ---: |
| 固定 69 场平均 EPE | 3.4483 | **1.9457** |
| 固定 69 场 EPE 中位数 | 1.9370 | **1.5728** |
| 单场 EPE 更低的场景数 | 44 | 25 |

虽然 RT-IGEV 在 44/69 场取得更低的单场 EPE，但其中许多优势很小；LiteAnyStereo 在部分困难场景的改善幅度很大，因此平均 EPE 和中位数均更低。例如：

| 场景 | RT-IGEV EPE | LiteAnyStereo EPE | 说明 |
| :--- | ---: | ---: | :--- |
| `0018` | 108.8157 | **12.3173** | LiteAnyStereo 显著避免灾难性失效 |
| `0019` | 55.5348 | **4.6715** | 固定排除场景，全量评价中保留 |
| `0012` | 18.8266 | **5.2625** | 固定排除场景，全量评价中保留 |
| `0001` | 2.0975 | **0.7680** | LiteAnyStereo 更优 |
| `0004` | 2.3936 | **0.9329** | LiteAnyStereo 更优 |
| `0038` | **7.4392** | 8.8100 | RT-IGEV 更优反例 |
| `0040` | **2.9463** | 3.5838 | RT-IGEV 更优反例 |

因此，更准确的表述是：LiteAnyStereo 的总体误差和困难场景鲁棒性更好，但并不是每个场景都比 RT-IGEV 更准。

## 6. 金属表面高反光专项评价

本数据没有“高反光”人工标签。为避免根据模型结果挑选样本，本报告只依据左图亮度筛选：在 818×512 ROI 内统计通道截断高光、极亮像素和近中性亮斑，并按以下分数从 73 场中选取最高的 15 场（约前 20%），之后人工确认这些图均包含明显金属亮斑。

```text
高反光分数 = 通道截断比例 + 0.5 × 极亮比例 + 0.25 × 近中性亮斑比例
通道截断：max(R,G,B) >= 250
极亮：mean(R,G,B) >= 220
近中性亮斑：mean(R,G,B) >= 200 且 max-min <= 35
```

这是一种可复现的高反光代理定义，不等同于人工材质分割。筛选过程完全不读取模型预测或误差，因此不会偏向任一算法。

![高反光场景联系表](../results/baseline_73/high_reflection_scene_contact_sheet.png)

15 场包括 `0035`、`0040`、`0036`、`0008`、`0003`、`0034`、`0041`、`0031`、`0053`、`0020`、`0032`、`0009`、`0043`、`0061`、`0045`。其中 `0053` 和 `0020` 属于固定 69 场协议排除项，因此同时报告全 15 场和排除这两场后的固定协议 13 场。

### 表 5  高反光 15 场全 ROI 指标（三线表）

| 算法 | EPE↓ (px) | D1↓ (%) | Bad1↓ (%) | Bad2↓ (%) | Bad3↓ (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| RT-IGEV | 4.1342 | 13.83 | 45.16 | 22.93 | 16.73 |
| LiteAnyStereo | **2.6720** | **10.95** | **43.60** | **20.01** | **13.71** |

全 15 个高反光场景中，LiteAnyStereo 五项指标全部更好，EPE 相对降低 35.37%。但该结果包含 `0053` 和 `0020` 两个旧协议排除场景，应结合下面的固定协议结果一起看。

### 表 6  固定协议高反光 13 场全 ROI 指标（三线表）

| 算法 | EPE↓ (px) | D1↓ (%) | Bad1↓ (%) | Bad2↓ (%) | Bad3↓ (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| RT-IGEV | 2.5439 | 10.70 | 41.30 | 17.53 | 11.50 |
| LiteAnyStereo | **2.5159** | **10.14** | **41.28** | **16.45** | **10.79** |

排除 `0053` 和 `0020` 后，LiteAnyStereo 仍在五项指标上略优：EPE 降低 1.10%，D1 降低 5.28%，Bad2/Bad3 分别降低 6.19%/6.23%，Bad1 基本持平。

![高反光 13 场全 ROI 指标](../results/baseline_73/high_reflection_fixed13_scene_comparison.png)

### 表 7  固定协议 13 场仅高光像素指标（三线表）

15 场中高光掩码平均覆盖 ROI 的 9.81%；固定 13 场中平均覆盖 9.67%，共有 526,605 个有效高光像素。下表只在这些亮斑位置计算指标，再按场景宏平均。

| 算法 | EPE↓ (px) | D1↓ (%) | Bad1↓ (%) | Bad2↓ (%) | Bad3↓ (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| RT-IGEV | 0.8804 | **0.35** | 33.90 | 8.66 | 1.32 |
| LiteAnyStereo | **0.8044** | 0.47 | **30.76** | **5.62** | **0.83** |

仅看高光像素，LiteAnyStereo 的 EPE 降低 8.64%，Bad1 降低 9.26%，Bad2 降低 35.09%，Bad3 降低 37.23%；D1 增加 0.12 个百分点。D1 同时受绝对误差和相对误差阈值影响，因此与 Bad3 不完全相同。

![高反光像素指标](../results/baseline_73/high_reflection_fixed13_pixel_comparison.png)

高反光专项结论：LiteAnyStereo 在金属亮斑位置的平均误差及 1/2/3 px 坏点率总体更好，对强高光有一定优势；但优势在排除异常场景后并不夸张，且 D1 略高，不能表述为已经完全解决金属反光问题。

高反光代表场景：

| `0035` | `0036` |
| :---: | :---: |
| ![0035 高反光](../results/baseline_73/comparisons/202506281614-0035/traditional_comparison.png) | ![0036 高反光](../results/baseline_73/comparisons/202506281614-0036/traditional_comparison.png) |

| `0034` | `0009` |
| :---: | :---: |
| ![0034 高反光](../results/baseline_73/comparisons/202506281614-0034/traditional_comparison.png) | ![0009 高反光](../results/baseline_73/comparisons/202506281605-0009/traditional_comparison.png) |

## 7. 统一色标视差图和误差图

以下图片均由两种算法的原始浮点视差重新生成，使用完全相同的固定色标：视差 0–192 px，绝对误差 0–20 px。

六宫格对应关系：

| 左上 | 上中 | 右上 |
| :--- | :--- | :--- |
| 左图 ROI | RT-IGEV 视差 | LiteAnyStereo 视差 |
| 参考视差 | RT-IGEV 绝对误差 | LiteAnyStereo 绝对误差 |

### 7.1 场景 `202506281603-0001`

![0001 统一对比](../results/baseline_73/comparisons/202506281603-0001/traditional_comparison.png)

LiteAnyStereo EPE 由 2.0975 px 降至 0.7680 px，主体区域误差更低。

### 7.2 场景 `202506281604-0004`

![0004 统一对比](../results/baseline_73/comparisons/202506281604-0004/traditional_comparison.png)

LiteAnyStereo EPE 由 2.3936 px 降至 0.9329 px。

### 7.3 场景 `202506281608-0018`

![0018 统一对比](../results/baseline_73/comparisons/202506281608-0018/traditional_comparison.png)

RT-IGEV 出现大范围错误，LiteAnyStereo 将 EPE 从 108.8157 px 降至 12.3173 px；当前模型仍有误差，但没有发生同等级别的崩溃。

### 7.4 场景 `202506281615-0038`

![0038 统一对比](../results/baseline_73/comparisons/202506281615-0038/traditional_comparison.png)

该场景是反例：RT-IGEV EPE 为 7.4392 px，LiteAnyStereo 为 8.8100 px。

### 7.5 场景 `202506281616-0040`

![0040 统一对比](../results/baseline_73/comparisons/202506281616-0040/traditional_comparison.png)

RT-IGEV EPE 为 2.9463 px，LiteAnyStereo 为 3.5838 px。

### 7.6 场景 `202506281613-0030`

![0030 统一对比](../results/baseline_73/comparisons/202506281613-0030/traditional_comparison.png)

两者差距较小，RT-IGEV EPE 2.4885 px，LiteAnyStereo 2.6387 px。

## 8. 推理时间

| 方法 | 设备 | 输入 | 核心推理时间 | FPS |
| :--- | :--- | :--- | ---: | ---: |
| RT-IGEV（一期） | A6000 服务器 | 1280×720、文档记录 12 次迭代 | 未保存 | 无法计算 |
| LiteAnyStereo FP32 | RTX 4090 | 1280×720，填充至 1280×736 | 28.73 ms/对 | 34.81 |
| LiteAnyStereo FP16 AMP | RTX 4090 | 1280×720，填充至 1280×736 | 22.01 ms/对 | 45.43 |

一期 RT-IGEV 没有留下推理时间日志，因此仍不能给出严格速度倍数。

## 9. 最终结论

1. 上一版异常结果来自评价源和场景过滤不一致，不能作为最终结论；
2. 统一读取浮点视差并使用同一个评价函数后，LiteAnyStereo 在全 73 场五项指标全部优于 RT-IGEV；
3. 固定 69 场中，LiteAnyStereo 的 EPE 降低 43.58%，D1 降低 21.05%，Bad2 降低 7.67%，Bad3 降低 14.79%；Bad1 高 0.18 个百分点，基本持平；
4. LiteAnyStereo 对 RT-IGEV 的灾难性失效场景明显更稳，是总体指标改善的主要来源；
5. 新模型整体效果更好，但 RT-IGEV 仍在部分常规场景占优，后续可针对 Bad1 和这些场景继续优化。
6. 高反光固定 13 场中 LiteAnyStereo 五项全 ROI 指标均略优；只看高光像素时 EPE 和 Bad1/2/3 更好，但 D1 高 0.12 个百分点。

## 10. 结果路径

- 统一汇总：[unified_summary.json](../results/baseline_73/metrics/unified_summary.json)
- 73 场逐场指标：[unified_scene_metrics.csv](../results/baseline_73/metrics/unified_scene_metrics.csv)
- 推理时间记录：[runtime_benchmark.json](../results/baseline_73/metrics/runtime_benchmark.json)
- LiteAnyStereo 重跑浮点视差：`runs/evaluation/jmp_unified_rerun_73/liteanystereo/`
- 73 场统一六宫格：`runs/evaluation/jmp_unified_rerun_73/comparisons/`
- 固定 69 场柱状图：[unified_comparison_fixed69.png](../results/baseline_73/unified_comparison_fixed69.png)
- 全 73 场柱状图：[unified_comparison_all73.png](../results/baseline_73/unified_comparison_all73.png)
- 高反光逐场指标：[high_reflection_scene_metrics.csv](../results/baseline_73/metrics/high_reflection_scene_metrics.csv)
- 高反光场景联系表：[high_reflection_scene_contact_sheet.png](../results/baseline_73/high_reflection_scene_contact_sheet.png)
- 高反光固定 13 场图表：[high_reflection_fixed13_scene_comparison.png](../results/baseline_73/high_reflection_fixed13_scene_comparison.png)
- 高反光像素图表：[high_reflection_fixed13_pixel_comparison.png](../results/baseline_73/high_reflection_fixed13_pixel_comparison.png)

> 建议汇报表述：在统一参考视差、固定 ROI、有效掩码、指标实现和场景集合后，LiteAnyStereo 在全 73 场五项指标全部更好；固定 69 场的 EPE、D1、Bad2、Bad3 更好，Bad1 基本持平。在独立筛选的金属高反光固定 13 场中，两种算法全 ROI 指标接近但 LiteAnyStereo 略优；只看亮斑像素时，LiteAnyStereo 的 EPE 和 Bad1/2/3 更低，说明其高反光区域总体更稳，但 D1 仍略高。
