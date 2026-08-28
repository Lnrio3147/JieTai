# 实验 1–4：LiteAnyStereo + BiSeNetV2 的 RK3588 移植

完整的处理流程、已执行记录、目标板操作步骤和结果回填表见
[`DEPLOYMENT_OPERATION_REPORT.md`](DEPLOYMENT_OPERATION_REPORT.md)。

本目录把实验 1–4 的最终推理链路整理成可复现的 RK3588 部署流程：

```text
校正左右 RGB
  ├─ LiteAnyStereo LAS1（RKNN）─────────────> 全图视差 ─┐
  └─ 实验 3 BiSeNetV2（RKNN）─> 前景概率 ─> 实验 4 CPU 修复 ├─> 主体视差
                                                       ┘
```

实验 2 的伪标签模型和实验 3 的人工标注模型是前后两代训练结果，不会在部署时串行
运行。最终部署只使用实验 3 的人工模型。实验 4 没有神经网络权重，继续使用
NumPy/OpenCV 完成形态学、最大连通域和视差连续性补洞。

## 0. 固定的模型和输入

| 模块 | 正式权重 | 固定输入 | 输出 |
|---|---|---|---|
| LiteAnyStereo LAS1 | `projects/LiteAnyStereo/checkpoints/LiteAnyStereo.pth` | ONNX 左右各 `1×3×1280×736` NCHW；板端 API 传 NHWC RGB；原图 `1280×720` 左右各复制填充 8 px | `1×1×1280×736` 视差 |
| BiSeNetV2 | `experiments/03_manual_segmentation/fdjyp3/results/model_manual/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb` | `1×512×288×3` NHWC RGB | `1×512×288×2` 概率 |
| 实验 4 | 无权重 | 原图尺寸概率、LAS 视差 | refined mask、`818×512` 主体视差 |

不要把 LAS 左右图先按 Mask 涂黑。Mask 只用于视差生成后的主体筛选，这和实验 1–4
正式结果的处理顺序一致。

Rockchip 的标准工作流是在 x86_64 PC 上用 RKNN-Toolkit2 转换模型，在 RK3588 上用
RKNN-Toolkit-Lite2 或 C API 推理。官方仓库当前列出的最新版本是 `2.3.2`，并明确
区分 PC 转换工具和板端 Lite2 runtime：

- <https://github.com/airockchip/rknn-toolkit2>
- <https://github.com/airockchip/rknn_model_zoo>
- TensorFlow PB 直接转换示例：<https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/tensorflow/ssd_mobilenet_v1>
- 多输入与 `dataset.txt` 格式：<https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/multi_input>

建议 PC 的 Toolkit2、板端 Lite2、板端 `librknnrt.so` 使用同一发行版本。至少要保证
板端 runtime 不旧于生成 `.rknn` 的 compiler；版本不匹配时不要继续记录性能数据。

## 1. 在 PC 上导出 LAS1 ONNX

仓库现有 `projects/LiteAnyStereo/export_onnx.py` 已补齐 LAS1 的 correlation volume 和
context upsample 静态改写，并支持 NCHW/NHWC 双输入。正式 RKNN 源模型固定使用 NCHW；
板端 `RKNNLite.inference` 通过 `data_format="nhwc"` 接收相机侧 NHWC 数据。

```bash
cd /home/uestc/lnrio/JieTai/experiments/04_mask_refinement/rk3588

LAS_PY=/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python
"$LAS_PY" export_las_onnx.py \
  --checkpoint ../../../projects/LiteAnyStereo/checkpoints/LiteAnyStereo.pth \
  --output artifacts/liteanystereo_las1_1280x736.onnx
```

脚本会执行 ONNX checker，并生成同名 `.onnx.json`，记录输入形状、padding、模型和
checkpoint 的 SHA-256。LAS1 的正式 checkpoint 固定使用 `max_disp=192`，不能只改
命令行数值把它降到 96；其 aggregation 层固定需要 48 个四分之一分辨率视差通道。

可以额外运行 PyTorch/ONNX 数值校验。全尺寸 CPU 校验较慢，但正式转换前至少应做
一次：

```bash
cd /home/uestc/lnrio/JieTai/projects/LiteAnyStereo
"$LAS_PY" verify_onnx.py \
  --version las1 \
  --ckpt checkpoints/LiteAnyStereo.pth \
  --onnx_file ../../experiments/04_mask_refinement/rk3588/artifacts/liteanystereo_las1_1280x736.onnx \
  --height 1280 --width 736 --max_disp 192 --input_layout nchw
```

代码回归用的 `64×96` 静态图上，PyTorch 与 ONNX 的 mean/max absolute difference
分别为 `3.64e-5/1.26e-4 px`。

## 2. 准备 INT8 校准数据

FP16 不需要校准集。INT8 必须使用真实工业双目图，且 LAS 每一行必须同时提供左、
右两个输入。脚本为 LAS 保存 NCHW uint8 `.npy`，为 BiSeNet 保存 NHWC uint8 `.npy`，
避免图片解码器对 RGB/BGR 和布局的隐式处理。

```bash
cd /home/uestc/lnrio/JieTai/experiments/04_mask_refinement/rk3588
python3 prepare_calibration.py \
  --dataset-root ../../../datasets/training/JMP-LF6020-ETH3D \
  --split train \
  --samples 20 \
  --output-dir build/calibration
```

输出：

```text
build/calibration/dataset_las.txt       # 每行 left.npy right.npy
build/calibration/dataset_bisenet.txt   # 每行一个 left.npy
build/calibration/metadata.json
```

20 张适合先验证转换；正式 INT8 精度实验建议固定 50–100 张、覆盖不同曝光和工件，且
不要使用冻结测试图。所有精度方案必须复用同一份校准列表。

## 3. 在 x86_64 PC 上转 RKNN

从 Rockchip 官方 `rknn-toolkit2` release 或 SDK 中安装与你的 Python 版本匹配的
`rknn_toolkit2-2.3.2-...-linux_x86_64.whl`。不要在 PC 上装 Lite2 来代替 Toolkit2，
也不要在 RK3588 板上执行模型转换。

先构建两个 FP16 基线：

```bash
python convert_rknn.py \
  --model las \
  --source artifacts/liteanystereo_las1_1280x736.onnx \
  --dtype fp16 \
  --output artifacts/liteanystereo_las1_fp16_rk3588.rknn

python convert_rknn.py \
  --model bisenet \
  --source ../../03_manual_segmentation/fdjyp3/results/model_manual/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --dtype fp16 \
  --output artifacts/bisenetv2_manual_fp16_rk3588.rknn
```

本次已实际生成：

- `liteanystereo_las1_fp16_rk3588.rknn`：66 MB，SHA-256
  `e1f1bffbe66fec5e621a8ccdbafbc38ded5471ea8a811d78f4236c466f21059e`；
- `bisenetv2_manual_fp16_rk3588.rknn`：13 MB，SHA-256
  `dc0337470c30fb14fe76aed81f59eb47e007082566503193997c37412e077aad`。

BiSeNet FP16 的 PC simulator 同输入检查得到概率 MAE `7.08e-5`、阈值 Mask IoU
`0.999991`。PC simulator 不能代替 RK3588 实机加载与计时。

BiSeNet PB 只截取 `final_probability`，不把 `ArgMax/final_output` 编入 NPU 图；板端用
前景概率阈值 0.5，和实验 3/4 一致。

FP16 能正确运行后，再构建 INT8：

```bash
python convert_rknn.py \
  --model bisenet \
  --dtype int8 \
  --dataset build/calibration/dataset_bisenet.txt \
  --output artifacts/bisenetv2_manual_int8_rk3588.rknn

python convert_rknn.py \
  --model las \
  --source artifacts/liteanystereo_las1_1280x736.onnx \
  --dtype int8 \
  --dataset build/calibration/dataset_las.txt \
  --output artifacts/liteanystereo_las1_int8_rk3588.rknn
```

建议实验顺序是：

1. LAS FP16 + BiSeNet FP16；
2. LAS FP16 + BiSeNet INT8；
3. LAS INT8 + BiSeNet INT8。

视差回归通常比分割阈值更怕全 INT8 量化误差，所以第 3 组不能因为 FPS 更高就直接
替换第 1 组。每个 `.rknn` 都有同名 JSON，记录源模型/产物哈希和 Toolkit2 版本。

## 4. 在 RK3588 上安装 runtime

已经生成可直接复制的 FP16 部署包：

```text
artifacts/jietai_rk3588_fp16_bundle.tar.gz
SHA-256: ca9329e8c5f0a3acd5683d8780d88a56bfb2970b014e40ba90d045a9c9fa1d9f
```

在 PC 上复制，在板端解压：

```bash
scp artifacts/jietai_rk3588_fp16_bundle.tar.gz <USER>@<RK3588_IP>:/tmp/
ssh <USER>@<RK3588_IP>
mkdir -p ~/jietai_rk3588
tar -xzf /tmp/jietai_rk3588_fp16_bundle.tar.gz -C ~/jietai_rk3588
cd ~/jietai_rk3588
```

部署包包含两个 FP16 `.rknn`、板端推理/后处理脚本和 `BOARD_QUICKSTART.md`，不包含
Lite2 wheel 和测试数据；Lite2 wheel 必须按板端 Python 版本选择，测试图应从原数据集
复制或使用相机输入。

另行把官方 aarch64 Lite2 wheel 和测试数据复制到板端。安装示例：

```bash
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install numpy opencv-python
pip install /path/to/rknn_toolkit_lite2-2.3.2-*-linux_aarch64.whl
```

板端镜像如果已经提供 OpenCV，优先使用系统包，避免 `opencv-python` wheel 与系统
GL/GUI 库冲突。首次运行时检查日志中的 Lite2、runtime、driver 和模型 compiler
版本；看到 model/runtime version mismatch 必须先升级板端 runtime。

为减少 DVFS 造成的漂移，在正式测速前固定散热条件，并把可用的 CPU/NPU governor
设为 `performance`。不同板卡的 sysfs 路径不同，先用下面的只读命令确认路径，不能
照抄不存在的节点：

```bash
find /sys/class/devfreq -maxdepth 2 -type f \( -name governor -o -name cur_freq \)
```

记录板卡型号、系统镜像、NPU driver/runtime、环境温度、散热方式和频率设置。

## 5. 单场正确性检查和测速

先用已有正式参考的 `202506281604-0004`：

```bash
python board_benchmark.py \
  --las-model artifacts/liteanystereo_las1_fp16_rk3588.rknn \
  --bisenet-model artifacts/bisenetv2_manual_fp16_rk3588.rknn \
  --left ../../../datasets/training/JMP-LF6020-ETH3D/fdjyp_3_1_202506281604_0004/im0.png \
  --right ../../../datasets/training/JMP-LF6020-ETH3D/fdjyp_3_1_202506281604_0004/im1.png \
  --warmup 10 --runs 50 \
  --las-core 0 --bisenet-core 0 \
  --output-dir artifacts/board_fp16_0004
```

默认使用单个 NPU core，便于和 Rockchip Model Zoo 的 single-core 口径对齐。完成后可
把两个 `--core` 改成 `0_1_2` 单独测试三核模式；不要把不同 core 设置的结果混在同
一张表里。

输出包括：

- `benchmark_report.json`：mean/median/P95/min/max/FPS、模型哈希、core 和系统信息；
- `disparity.npy`：去掉左右各 8 px padding 的 `1280×720` 视差；
- `foreground_probability.npy`：`512×288` 前景概率；
- `raw_mask.png`、`refined_mask.png`；
- `subject_disparity.npy`：固定 ROI 内 `818×512` 主体视差。

报告有三种计时：

| 字段 | 统计范围 |
|---|---|
| `model_only.liteanystereo/bisenetv2` | 已准备输入上的单模型阻塞式 RKNNLite 调用，包含输出回传 |
| `model_only.sequential_both` | 一次 LAS + 一次 BiSeNet 的实测串行时间，不是两个 P95 相加 |
| `pipeline.end_to_end` | 内存中 BGR→RGB、padding/resize、两个模型、输出整理、实验 4 修复和主体视差；不含磁盘 I/O/模型加载 |

## 6. 多场正式时间实验

实验 4 后处理耗时随 Mask 孔洞和连通域数量变化，正式报告不能只重复一张简单图。
在板端数据目录生成 pairs 文件：

```bash
python prepare_pairs_file.py \
  --dataset-root /data/JMP-LF6020-ETH3D \
  --scene-prefix fdjyp_3_ \
  --output build/fdjyp3_pairs.txt
```

然后让完整流水线轮流处理各场。`--runs` 至少等于 pairs 数量；模型单项测速仍固定用
第一场，完整流水线会按列表循环：

```bash
python board_benchmark.py \
  --las-model artifacts/liteanystereo_las1_fp16_rk3588.rknn \
  --bisenet-model artifacts/bisenetv2_manual_fp16_rk3588.rknn \
  --pairs-file build/fdjyp3_pairs.txt \
  --warmup 10 --runs 73 \
  --las-core 0 --bisenet-core 0 \
  --output-dir artifacts/board_fp16_fdjyp3_73
```

同样运行 FP16/INT8 三组组合。每组至少重复三轮，保留每轮 JSON；报告 mean、median、
P95 和 FPS，并单列冷启动/模型加载时间（本脚本故意不把它算入稳定态单帧时间）。

## 7. 和 FP32 正式结果做数值检查

比较 RKNN 时，FP32 参考必须使用与板端逐像素相同的输入文件。仓库中训练副本和
`rec_img_set` 副本有极少数 1–3 灰度级的像素差，因此不能把一个副本的 RKNN 输出
和另一个副本的历史视差直接作量化误差结论。

先用第 1 步生成的 FP32 ONNX 和原 TensorFlow PB，对板端选定的同一输入生成参考。
两个模型环境分开执行：

```bash
LAS_PY=/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python
TF_PY=/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python
SAMPLE=../../../datasets/training/JMP-LF6020-ETH3D/fdjyp_3_1_202506281604_0004

"$LAS_PY" host_reference.py las \
  --onnx artifacts/liteanystereo_las1_1280x736.onnx \
  --left "$SAMPLE/im0.png" --right "$SAMPLE/im1.png" \
  --output-dir artifacts/fp32_reference_0004

"$TF_PY" host_reference.py bisenet \
  --pb ../../03_manual_segmentation/fdjyp3/results/model_manual/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --left "$SAMPLE/im0.png" \
  --output-dir artifacts/fp32_reference_0004

python host_reference.py postprocess \
  --output-dir artifacts/fp32_reference_0004
```

把板端单场产物复制回仓库后运行：

```bash
python compare_accuracy.py \
  --reference-disparity artifacts/fp32_reference_0004/disparity.npy \
  --candidate-disparity artifacts/board_fp16_0004/disparity.npy \
  --reference-probability artifacts/fp32_reference_0004/foreground_probability.npy \
  --candidate-probability artifacts/board_fp16_0004/foreground_probability.npy \
  --reference-mask artifacts/fp32_reference_0004/refined_mask.png \
  --candidate-mask artifacts/board_fp16_0004/refined_mask.png \
  --output artifacts/board_fp16_0004/accuracy.json
```

至少检查：LAS disparity MAE/P95/bad-1px、BiSeNet 概率 MAE 和 0.5 阈值 Mask IoU、最终
refined Mask IoU。正式量化结论还要在 FDJYP-3 全集重新计算实验 1–4 原有指标，不能
只凭一张图的误差决定是否接受 INT8。

建议把以下门槛作为最初的工程回归线，再按项目容差冻结：

- ONNX 对 PyTorch：mean absolute disparity difference `< 1e-3 px`；
- RKNN FP16/INT8 对 FP32：73 场参考 EPE 不恶化超过 5%；
- BiSeNet 阈值 Mask 与 FP32 Mask 的宏平均 IoU `>= 0.99`；
- 最终 refined Mask 宏平均 IoU `>= 0.99`。

## 8. 常见失败与处理顺序

1. **ONNX 导出报 `Unfold` 不支持**：必须使用本仓库已修正的 exporter，不要调用未
   修改的上游脚本。
2. **PB 转换在 ArgMax/Squeeze 报错**：确认 `convert_rknn.py` 的输出只写
   `final_probability`，不要改成 `final_output`。
3. **板端提示 model/runtime version mismatch**：升级 Lite2 和 `librknnrt.so` 后重测，
   不能把该次数据写入正式报告。
4. **LAS full-resolution RKNN build 内存不足或出现不支持算子**：先保留完整日志和
   Toolkit2 版本，不要静默换输入分辨率。降低分辨率会同时改变有效视差尺度、实验 4
   ROI 和精度口径，必须作为新的部署变体重新导出、校准和评估。
5. **INT8 更快但视差明显漂移**：保留 LAS FP16，只量化 BiSeNet；这是本任务优先的
   混合方案。

## 9. 2026-08-27 RK3588 实机结果

两个 FP16 `.rknn` 已部署到 Firefly ITX-3588J 的
`/home/firefly/gq/rk3588_fp16`。实机环境为 Ubuntu 22.04.5、Linux 5.10.198、
Lite2 2.3.2、runtime 2.3.0、RKNPU driver 0.9.3；模型均能正常加载。固定单场、
warmup 3、runs 20 的稳定态结果如下：

| core | LAS mean/P95 | BiSeNet mean/P95 | 双模型串行 mean/P95 | 完整链路 mean/P95 | FPS |
|---|---:|---:|---:|---:|---:|
| 0 | 1922.15 / 1948.58 ms | 69.97 / 78.13 ms | 2011.61 / 2040.60 ms | 2030.15 / 2064.38 ms | 0.493 |
| 0_1_2 | 1837.33 / 1866.71 ms | 50.17 / 78.84 ms | 1900.85 / 1938.78 ms | 1942.95 / 1988.35 ms | 0.515 |

两组输出逐元素一致。完整环境、校验值、处理流程、限制和后续 73 场实验命令见
`DEPLOYMENT_OPERATION_REPORT.md`。这里的数据是 RK3588 实测，不是 PC simulator 时间。

板端结果回传后，与同输入 PC FP32 reference 的精确比较为：LAS disparity MAE
`0.09076 px`、P95 `0.14266 px`、bad-1px `0.02550%`；BiSeNet 0.5 阈值 Mask IoU
`0.999973`；最终 refined Mask IoU `0.999986`。两种 core 配置指标完全相同。

项目“采集双目图到点云完成 `<4 s`”是更大的端到端口径。当前三核
`pipeline.end_to_end` mean `1.943 s` 结束于主体视差，不包含相机采集、校正和点云；
它给其余阶段留下约 `2.057 s` 的 mean 预算。加入 `--include-pointcloud` 后，点云
重投影 mean/P95 为 `49.66/54.02 ms`，到内存 XYZRGB 点云的完整 mean/P95 为
`1961.72/1997.38 ms`；343,992 点的 5.16 MB 二进制 PLY 单次写盘为 `12.01 ms`。
这说明计算侧具备达到 4 秒的空间，但最终验收仍必须接入真实相机 API，并把采集、
校正及所需写盘纳入同一个端到端计时区间。

补充的 73 场不同图片正式测试（warmup 10、runs 73、core 0_1_2）结果为：到内存
XYZRGB 点云 mean `1958.00 ms`、median `1964.02 ms`、P95 `2000.24 ms`、范围
`1878.09–2034.01 ms`。73/73 场均低于 4 秒，68/73 场低于 2 秒。最慢场景为
`fdjyp_3_1_202506281614_0033`，总计 `2034.01 ms`；逐场耗时保存在多场结果 JSON 的
`pipeline_samples` 中。

73 场完整 LAS 视差也已在 RK3588 上独立导出并回传到
`artifacts/board_fp16_core012_disparities_73`。每个场景包含权威数值输出
`disparity.npy`、定点 `disparity_x256.png`（像素视差为 PNG 值除以 256）和仅供查看的
`disparity_preview.png`。共 73 组、219 个文件；所有 `.npy` 均为 `(1280, 720)`
`float32`、无 NaN/Inf，回传后的逐文件 SHA-256 校验错误数为 0。完整命令和板端目录见
部署操作报告第 5.4 节。

实验 1–4 的最终主体视差另保存在
`artifacts/board_fp16_core012_subject_disparities_73`，不要和上一段的 LAS 全图中间结果
混用。每个场景包含 `subject_disparity.npy`、`subject_disparity_x256.png`、黑底
`subject_disparity_preview.png` 和 `subject_mask.png`。共 73 组、292 个文件；最终
数组为 `(818, 512)` `float32`，主体全部有限、背景全部为 NaN，定点 PNG 与数组按
`×256` 转换逐像素一致，回传后的输入和输出 SHA-256 校验错误数为 0。完整处理流程与
复现命令见部署操作报告第 5.5 节。

包含摄像头原图的处理前后对比位于
`artifacts/board_fp16_core012_camera_subject_comparisons_73`。每张图按“原始左目输入
（红框标出 ROI）/ LAS ROI 全视差 / 最终主体视差 / 最终主体 Mask”从左到右排列，前后
视差共享同一色标；`all_subject_before_after.jpg` 是 73 场总览。原三栏对比仍保留在
`artifacts/board_fp16_core012_subject_comparisons_73`。生成和校验方法见部署操作报告第
5.6 节。
