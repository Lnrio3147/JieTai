# Jop(1) 双目推理与性能对比报告

## 1. 结论

本次有效数据共 9 组。所有场景均使用同一套预处理：原始 RGBA 图像转 BGR、逆时针旋转 90°、使用 `projects/tradition_stereo/config/stereo.yml` 进行双目矫正，最终推理尺寸为 720×1280（宽×高），最大视差为 192。

在 RTX 4090、FP32、batch size 1 条件下，LiteAnyStereo LAS1 的核心推理平均耗时为 33.61 ms（29.75 FPS），IGEV++ RT（8 次更新）为 64.29 ms（15.55 FPS）。LAS1 在本次测试中约快 1.91 倍，延迟降低 47.72%。

以压缩包附带 PLY 投影得到的参考视差进行稀疏一致性评估，LAS1 的场景宏平均 EPE 为 12.508 px，IGEV++ RT 为 32.077 px；LAS1 的 EPE 降低 19.569 px（61.01%）。9 个场景中，LAS1 的 EPE 均低于当前 IGEV++ RT 结果。

当前结果说明：在这批 Jop 数据和当前权重条件下，LAS1 同时具有更低延迟和更好的 PLY 一致性。IGEV++ RT 输出在多个场景中出现较大区域接近最大视差，表现出明显的目标域泛化问题。

## 2. 模型与权重

| 模型 | 参数量 | checkpoint | 推理设置 |
|---|---:|---:|---|
| IGEV++ RT | 4,165,162 | 16.17 MiB | 官方 `core_rt`，SceneFlow 权重，8 次更新，FP32 |
| LiteAnyStereo LAS1 | 7,603,626 | 29.35 MiB | 项目现有官方 LAS1 权重，FP32 |

`tradition_stereo` 目录包含 IGEV 历史结果、指标计算和视差后处理脚本，但没有网络定义、可运行推理入口或 checkpoint。交接文档明确记录模型推理代码原位于 `/home/uestc/cyq/IGEV-plusplus`，该旧路径当前不存在。因此本次使用补充到工作区的官方 IGEV++ RT 代码和官方 RT SceneFlow 权重，不等同于一期可能使用过的私有微调权重。

为兼容当前环境的 `timm 1.0.28`，对官方特征提取器做了等价适配：新版 `timm` 已将 MobileNetV2 stem 的 ReLU6 合并进 `BatchNormAct2d`，因此缺失的独立 `act1` 使用恒等映射；完整 IGEV checkpoint 已包含 backbone 参数，初始化时不再额外下载 ImageNet 权重。

## 3. 汇总结果

| 指标 | IGEV++ RT | LiteAnyStereo LAS1 | 更优 |
|---|---:|---:|---|
| 平均核心推理时间 | 64.29 ms | 33.61 ms | LAS1 |
| 平均吞吐率 | 15.55 FPS | 29.75 FPS | LAS1 |
| PLY 参考 EPE | 32.077 px | 12.508 px | LAS1 |
| PLY 参考 D1 | 94.327% | 85.516% | LAS1 |
| PLY 参考 Bad-1 | 98.370% | 96.356% | LAS1 |
| PLY 参考 Bad-2 | 96.730% | 92.666% | LAS1 |
| PLY 参考 Bad-3 | 95.059% | 88.673% | LAS1 |
| PLY 参考预测覆盖率 | 99.9997% | 100.0000% | 接近 |

计时只包含 CUDA 上的视差网络前向过程，开始和结束均调用 CUDA 同步；不包含读图、旋转、矫正、张量构造、PLY 投影、可视化和文件写入。两个模型在同一 GPU、同一输入尺寸、同一进程中顺序测试，并在正式计时前各预热一次。

## 4. 逐场景结果

| 场景 | IGEV++ RT (ms) | LAS1 (ms) | IGEV EPE | LAS1 EPE |
|---|---:|---:|---:|---:|
| camera-202412091814-0104 | 62.48 | 32.77 | 45.312 | 16.594 |
| camera-202412091814-0105 | 65.57 | 34.17 | 27.186 | 12.956 |
| camera-202412091815-0106 | 66.57 | 34.94 | 32.293 | 12.751 |
| camera-202412091815-0107 | 64.09 | 33.19 | 18.956 | 11.575 |
| camera-202412091816-0108 | 63.12 | 32.76 | 25.033 | 11.148 |
| camera-202412091816-0109 | 62.95 | 33.27 | 23.393 | 15.013 |
| camera-202412091818-0110 | 63.14 | 32.46 | 23.018 | 13.032 |
| camera-202412091822-0111 | 63.69 | 34.19 | 48.160 | 11.998 |
| camera-202412091822-0112 | 67.00 | 34.73 | 45.340 | 7.506 |

## 5. 输出说明

- `igev_rt/<scene>/disp.npy`：IGEV++ RT 原始 float32 视差。
- `liteanystereo/<scene>/disp.npy`：LAS1 原始 float32 视差。
- 两个模型目录中的 `disparity_color.png`、`vis.png`、`vis_fixed.png`：固定/自动色标视差可视化。
- 两个模型目录中的 `cloud.ply`：由预测视差和标定 Q 矩阵重投影得到的点云。
- `reference/<scene>/disp.npy`：附带 PLY 投影得到的参考视差。
- `comparison/<scene>/comparison.png`：左图、两模型视差、PLY 参考和绝对误差六宫格。
- `comparison/overview.jpg`：全部 9 个场景的总览。
- `metrics/per_scene.csv`：逐场景原始指标。
- `metrics/summary.json`：机器可读的完整配置与汇总指标。

## 6. 评估限制

附带 PLY 只覆盖左图的一部分，虽然已通过 PLY RGB 反投影验证其与矫正左图空间对齐，但它不是人工标注的全图稠密视差真值。因此这里的 EPE、D1 和 Bad-N 应理解为“与附带 PLY 的一致性指标”，不能直接等同于标准公开数据集上的模型精度。

此外，当前 IGEV++ RT 使用官方 SceneFlow 权重。如果补回一期实际使用的目标域 checkpoint，应使用本脚本替换 `--igev-checkpoint` 后重跑；该结果才适合用于严格复现一期最终版本。

## 7. 复现命令

```bash
/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python \
  /home/uestc/mount_2T/uestc/lnrio/JieTai/experiments/01_stereo_comparison/jop1/scripts/run_comparison.py \
  --overwrite
```
