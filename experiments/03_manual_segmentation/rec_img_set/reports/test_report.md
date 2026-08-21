# rec_img_set 其它数据集 BiSeNetV2 + LiteAnyStereo 测试报告

**测试日期：** 2026-08-20  
**测试范围：** `rec_img_set` 中除 FDJYP-3 外的 130 个唯一双目场景  
**结论：** 冻结方案在 FDJYP-0 上保持稳定，在刻度板组上双目结果也较一致；但螺纹组表现混合，普通 1221 组明显失效。当前模型不能作为整个 `rec_img_set` 的通用主体分割器。

## 1. 范围与方法

本次覆盖以下四组：

| 组 | 场景数 | 说明 |
|---|---:|---|
| `FDJYP-0-rectified_images` | 82 | 有人工掩码，但参与过训练或模型选择 |
| `luowen_rectified_images` | 37 | 无人工分割真值 |
| `rectified_images` | 6 | 无人工分割真值，记为 `general_1221` |
| `rectified_images_刻度` | 5 | 无人工分割真值，记为 `scale_1221` |

没有重复测试 FDJYP-3，因为它已有独立的 73 场评估；`kedu` 与 `rectified_images_刻度` 的 5 场逐文件相同，也只保留一份。因此总数为 `82 + 37 + 6 + 5 = 130`。

模型和后处理参数保持冻结，没有根据本批结果调参：

- BiSeNetV2 冻结图 SHA-256：`b1f34a8caf2b4cab9be7a997a979dd1bb058152f771aa39c25bd37eef2bc4bee`；
- 网络输入：`288 × 512`，阈值 `0.5`；
- 概率双线性上采样后执行半径 3 的闭运算、最大 8 连通域和视差连续性孔洞修订；
- LiteAnyStereo 与 IGEV++ RT 视差复用 `experiments/01_stereo_comparison/rec_img_set/results/final_203/` 中已保存的全分辨率浮点结果，没有重新推理或修改主体内部视差。

## 2. 汇总结果

| 组 | 最终单连通域 | 平均整图前景占比 | 平均 ROI 前景占比 | LAS/IGEV 主体差异 MAE：均值 / 中位数 | 几何状态 | 判断 |
|---|---:|---:|---:|---:|---|---|
| FDJYP-0 | 82/82 | 38.09% | 59.00% | 0.669 / 0.386 px | 56 good，26 warning | 稳定，但属于训练域回归检查 |
| luowen | 37/37 | 62.71% | 80.37% | 15.064 / 4.109 px | 1 good，32 warning，4 high-risk | 混合且不可靠，存在明显过分割和双目异常 |
| general_1221 | 6/6 | 83.90% | 91.80% | 12.980 / 9.473 px | 6 high-risk | 明显失效，不建议使用 |
| scale_1221 | 5/5 | 85.42% | 97.84% | 0.616 / 0.515 px | 4 good，1 warning | 视差较一致，但需先明确“大刻度板”是否就是目标 |

这里的 LAS/IGEV 数值是两套模型的绝对差异，不是真实误差。两者接近只能说明结果相互一致，不能证明两者都正确；差异很大则足以标记需要复核的场景。

所有最终掩码都满足一个前景连通域，但这只是拓扑约束。错误地把大块背景连进来以后仍可得到一个连通域，因此 `130/130` 不能单独作为分割成功率。

## 3. FDJYP-0 人工标注回归结果

FDJYP-0 的 82 张人工掩码中，64 张属于训练划分，18 张属于验证划分。它们用于检查已有能力是否退化，不是独立测试集指标。

| 范围 | 原始前景 IoU | 修订后前景 IoU | 原始 Dice | 修订后 Dice |
|---|---:|---:|---:|---:|
| 全部 82 张 | 0.97336 | 0.97270 | 0.98642 | 0.98608 |
| 训练 64 张 | 0.97618 | 0.97549 | 0.98790 | 0.98755 |
| 验证 18 张 | 0.96331 | 0.96279 | 0.98113 | 0.98086 |

单连通域修订令 29 张 IoU 上升、53 张下降，平均变化为 `-0.00065`。因此它的价值主要是统一输出拓扑，不是提高像素 IoU。最弱场景是 `202506261704-0028`，修订后 IoU 为 `0.8773`，同时也是 LAS/IGEV 主体差异最大的 FDJYP-0 场景（`10.308 px`）。

![FDJYP-0 抽样总览](../results/result_130/report_assets/overview_fdjyp0.jpg)

## 4. 跨域组目视复核

### 4.1 luowen

孤立、轮廓清楚的螺纹件有部分结果可用；在多个金属零件、灰色台面、强反光或物体贴边时，掩码经常把邻近背景和非目标零件并入主体。37 场中 22 场的 LAS/IGEV 主体差异 MAE 超过 3 px，15 场超过 10 px。最大差异场景为：

| 场景 | 主体差异 MAE | 几何状态 |
|---|---:|---|
| `656565-0002` | 86.861 px | high-risk |
| `656565-0024` | 64.923 px | high-risk |
| `656565-0019` | 58.644 px | warning |
| `656565-0033` | 40.819 px | high-risk |
| `656565-0022` | 37.396 px | warning |

![luowen 抽样总览](../results/result_130/report_assets/overview_luowen.jpg)

### 4.2 general_1221

6 场的极线几何审计均为 high-risk；其中 3 场最终前景超过整图 90%，有的接近 100%。6/6 场 LAS/IGEV 主体差异 MAE 都超过 3 px。分割与双目两端都存在风险，不能把当前主体视差用于定量测量。

![general_1221 全部场景](../results/result_130/report_assets/overview_general_1221.jpg)

### 4.3 scale_1221

5 场 LAS/IGEV 主体差异均值为 `0.616 px`，未出现超过 3 px 的场景，双目一致性是四组中较好的。分割结果连续，但主体覆盖 ROI 的 `97.84%`：模型实际保留的是大面积刻度板。如果任务目标就是整块刻度板，这批结果可继续人工核验；如果目标是刻度线、凸点或局部工件，就需要重新定义类别和标注，当前工件模型并没有完成该语义任务。

![scale_1221 全部场景](../results/result_130/report_assets/overview_scale_1221.jpg)

## 5. 结论与下一步

1. FDJYP-0 证明冻结模型在原训练域没有链路退化，但其高 IoU 不能代表跨域泛化能力。
2. `luowen` 应先抽取包含简单件、强反光、多零件和贴边目标的分层样本做人工标注，再决定微调还是改变后处理。
3. `general_1221` 应先解决标定/极线几何问题，再评估分割或视差；当前直接调分割阈值意义不大。
4. `scale_1221` 要先明确目标语义；“整块刻度板”和“刻度局部”是两种不同任务。

如果根据本次 130 张结果挑样、改阈值或训练模型，这批数据就已经是开发/验证数据，不应再作为最终测试集。更稳妥的做法是：从 `luowen` 和 1221 数据中标注一批分层开发集用于改进，同时另外冻结一批从未看过、从未调参的人工测试集，最后只评一次。

## 6. 产物与复现

结果目录：

```text
results/result_130/
  bisenet_raw/                  # 原始预测、概率和联系表
  outputs/<group>/<scene>/
    raw_mask.png
    foreground_mask.png        # 全分辨率单连通域掩码
    disp_subject_full.npy       # 掩码外为 NaN
    disp_subject_crop.npy
    disp_subject_crop_color.png
    comparison.jpg
  metrics/per_scene.csv
  metrics/summary.json
  metrics/hole_decisions.json
  report_assets/overview_*.jpg
  README.md
```

复用已保存的 BiSeNet 概率和双目视差执行评估：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai

/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python \
  experiments/03_manual_segmentation/rec_img_set/scripts/run_test.py \
  --bisenet-raw-root experiments/03_manual_segmentation/rec_img_set/results/result_130/bisenet_raw \
  --output-root experiments/03_manual_segmentation/rec_img_set/results/result_130
```

脚本使用新版本目录并拒绝覆盖已有完整结果。逐场原始数值以 `metrics/per_scene.csv` 为准，完整汇总以 `metrics/summary.json` 为准。
