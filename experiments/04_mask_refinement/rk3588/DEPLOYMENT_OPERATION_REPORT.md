# 实验 1–4 RK3588 处理流程与部署操作报告

**报告日期：** 2026-08-27
**部署对象：** LiteAnyStereo LAS1 + 人工标注 BiSeNetV2 + 实验 4 Mask 修复
**目标平台：** Rockchip RK3588
**当前状态：** PC 侧模型整理、LAS ONNX 导出、数值回归、RKNN-Toolkit2 2.3.2
FP16 转换、RK3588 实机加载、单场精度比较及 73 场多图点云计时均已完成。两模型已
在目标板连续推理并通过单场 FP16 精度门槛；真实相机采集/校正纳入同一计时区间的
最终四秒验收仍待完成。

## 一、任务结论

实验 1–4 最终不是四个模型串联，而是两个神经网络和一个 CPU 后处理阶段：

```text
校正左图 RGB ───────────────────┬─> BiSeNetV2 ─> 前景概率 ─┐
                                │                           │
校正左图 RGB + 校正右图 RGB ────┴─> LiteAnyStereo ─> 视差 ├─> 实验4 Mask修复
                                                            │
                                                            └─> 主体视差
```

- 实验 1：部署 LiteAnyStereo LAS1，生成完整双目视差；
- 实验 2：伪标签 BiSeNetV2 是早期训练基线，不进入最终部署；
- 实验 3：部署人工 ISAT 标注训练的 BiSeNetV2，替代实验 2 模型；
- 实验 4：用 OpenCV/NumPy 做闭运算、最大连通域、视差连续性补洞和主体视差生成。

最终部署原则保持不变：**不在 LAS 推理前涂黑背景**。BiSeNetV2 Mask 仅在 LAS 已经
得到完整视差后用于主体筛选，避免人为破坏双目匹配边缘。

## 二、模型、输入输出与版本基线

| 模块 | 源模型 | SHA-256 | 部署输入 | 部署输出 |
|---|---|---|---|---|
| LAS1 | `projects/LiteAnyStereo/checkpoints/LiteAnyStereo.pth` | `ee0c3a0...ec1b9b1` | ONNX 左右各 `1×3×1280×736` NCHW；板端传 NHWC RGB | `1×1×1280×736` disparity |
| BiSeNetV2 | `experiments/03_manual_segmentation/fdjyp3/results/model_manual/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb` | `b1f34a8c...c4bee` | `1×512×288×3`，NHWC RGB | `1×512×288×2` probability |
| 实验 4 | 无网络权重 | — | 全图概率和 ROI 视差 | refined mask、`818×512` 主体视差 |

正式原图为高 `1280`、宽 `720`。LAS 输入采用和原 PyTorch `InputPadder` 相同的复制
填充：左 8 px、右 8 px、上下 0 px。BiSeNet 输入采用 `INTER_AREA` 缩放到宽 288、
高 512。

建议固定工具链：

| 环境 | 建议版本 |
|---|---|
| PC LAS 导出 | Python 3.10、PyTorch 2.6、ONNX 1.22、ONNX Runtime 1.23 |
| PC RKNN 转换 | Python 3.8、RKNN-Toolkit2 2.3.2；PB 直转使用 TensorFlow 2.13.1 |
| RK3588 板端 | RKNN-Toolkit-Lite2 2.3.2；本次实机 `librknnrt.so` 2.3.0 |
| CPU 后处理 | NumPy、OpenCV |

模型 compiler、Lite2、runtime 和 driver 版本必须记录。本次 compiler/Lite2 2.3.2
模型已被 runtime 2.3.0 正常接受，未出现 mismatch；这只证明当前 model version 6
产物可兼容，不能推广到其他模型或版本。出现 model/runtime mismatch 时，该轮输出
不能进入正式性能报告。

## 三、已经完成的迁移处理

### 3.1 LiteAnyStereo ONNX 兼容处理

原 LAS1 导出路径存在两个问题：

1. LAS1 模块内部持有的 `context_upsample` 和 `build_correlation_volume` 没有被实际替换，
   导出时报 `Unfold` 不支持；
2. 原 ONNX 校验脚本对 PyTorch 输出多取了一次 `[0]`，会隐藏 batch 维不一致。

现已完成：

- 用固定卷积替换 context upsample 的 `F.unfold`；
- 用静态 Slice/Stack 构造 correlation volume；
- 修正 LAS1 monkey patch；
- 修正 ONNX 校验输出；
- 同时支持 NCHW/NHWC 导出；正式 RKNN 源模型固定为 NCHW，板端通过
  `data_format="nhwc"` 接收相机侧 NHWC RGB。

正式 ONNX 已生成：

```text
experiments/04_mask_refinement/rk3588/artifacts/
  liteanystereo_las1_1280x736.onnx
  liteanystereo_las1_1280x736.onnx.json
```

ONNX SHA-256：

```text
119ae71c23115bc516f486cf77ae27c7a43d483a13ca7c675e4a4b5bcb3a22dd
```

全尺寸随机输入数值回归：

| 指标 | 结果 |
|---|---:|
| 输入 | `1×3×1280×736` NCHW，左右双输入 |
| PyTorch/ONNX mean absolute difference | `1.0565e-5 px` |
| PyTorch/ONNX max absolute difference | `8.2970e-5 px` |
| `allclose(atol=1e-4, rtol=1e-3)` | True |

### 3.2 BiSeNetV2 frozen PB 检查

已在 TensorFlow 2.19 compatibility 模式加载人工模型并实际运行：

| Tensor | 形状 | 类型 |
|---|---|---|
| `input_tensor:0` | `1×512×288×3` | float32 |
| `final_probability:0` | `1×512×288×2` | float32 |

输出概率和误差检查：

- 概率范围：`7.22e-8` 至 `0.99999988`；
- 双类别概率和最大误差：`1.19e-7`；
- RKNN 转换只截取 `final_probability`，不把 ArgMax 类别图加入 NPU 图。

### 3.3 实验 4 后处理回归

部署版 `postprocess.py` 已使用正式 FDJYP-3 概率和视差进行回归：

| 检查项 | 结果 |
|---|---|
| raw mask | 与原实验脚本逐像素一致 |
| refined mask | 与原实验脚本逐像素一致 |
| 孔洞和连通域统计 | 全部字段一致 |

部署版继续使用实验 4 冻结参数：

| 参数 | 数值 |
|---|---:|
| 前景概率阈值 | 0.5 |
| 闭运算半径 | 3 px |
| 孔洞环半径 | 7 px |
| 绝对视差容差 | 1.5 px |
| 最大补洞面积 | 全图 2.5% |
| 小孔面积阈值 | 1,000 px |
| 视差 ROI | `(y0,y1,x0,x1)=(234,1052,126,638)` |

### 3.4 部署程序检查

已完成以下本地检查：

- 所有新增 Python 文件通过 `py_compile`；
- 所有命令入口的 `--help` 可正常执行；
- 校准数据可生成 LAS 双输入 NCHW uint8 `.npy` 和 BiSeNet NHWC uint8 `.npy`；
- FDJYP-3 pairs 文件共生成 73 对；
- 用模拟 RKNNLite 接口跑通完整板端计时、输出保存和 JSON 报告；
- 同输入 FP32 reference 的 LAS、BiSeNet、实验 4 三阶段全部跑通；
- 精度比较器用相同输入自检时 disparity 误差为 0、Mask IoU 为 1。

### 3.5 FP16 RKNN 实际转换结果

已在本机 x86_64 环境使用 RKNN-Toolkit2 2.3.2 完成两个 FP16 模型的真实转换：

| 产物 | 大小 | SHA-256 | PC 模拟检查 |
|---|---:|---|---|
| `liteanystereo_las1_fp16_rk3588.rknn` | 66 MB | `e1f1bffb...f21059e` | build/export 成功；全尺寸 simulator 会话创建成功，单帧过慢而中止 |
| `bisenetv2_manual_fp16_rk3588.rknn` | 13 MB | `dc033747...e077aad` | 概率 MAE `7.08e-5`，0.5 阈值 Mask IoU `0.999991` |

BiSeNet 编译日志显示一个 `AveragePool(count_include_pad=0)` 回退 CPU；LAS 编译日志有
两条 `Unknown op target: 0`，但 `build` 和 `export_rknn` 均成功返回。两项都必须保留在
板端验收记录中，不能只依据 PC 编译成功判定最终可用。

## 四、PC 侧部署操作

以下命令从部署目录执行：

```bash
cd /home/uestc/lnrio/JieTai/experiments/04_mask_refinement/rk3588
```

### 4.1 重新导出 LAS ONNX

仓库已经保留本次生成的 ONNX。需要换输入尺寸或重新核对权重时执行：

```bash
LAS_PY=/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python

"$LAS_PY" export_las_onnx.py \
  --checkpoint ../../../projects/LiteAnyStereo/checkpoints/LiteAnyStereo.pth \
  --source-height 1280 --source-width 720 \
  --input-layout nchw \
  --output artifacts/liteanystereo_las1_1280x736.onnx \
  --force
```

### 4.2 生成 INT8 校准输入

只从 manifest 的 `train` split 取图，避免使用验证集做量化校准：

```bash
python3 prepare_calibration.py \
  --dataset-root ../../../datasets/training/JMP-LF6020-ETH3D \
  --split train \
  --samples 50 \
  --output-dir build/calibration
```

校准输出：

```text
build/calibration/dataset_las.txt
build/calibration/dataset_bisenet.txt
build/calibration/metadata.json
```

### 4.3 先转换 FP16

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

转换成功的判定条件：

1. `config`、`load_onnx/load_tensorflow`、`build`、`export_rknn` 均返回 0；
2. 生成同名 `.rknn.json`；
3. JSON 中 Toolkit2、源模型 SHA-256、目标平台和 dtype 正确；
4. 转换日志没有导致 build/export 失败的 unsupported op、invalid model 或内存溢出；
   CPU fallback 和编译器 warning 必须记录并在板端复核。

### 4.4 再转换 INT8

先量化 BiSeNet，LAS 保持 FP16：

```bash
python convert_rknn.py \
  --model bisenet \
  --dtype int8 \
  --dataset build/calibration/dataset_bisenet.txt \
  --output artifacts/bisenetv2_manual_int8_rk3588.rknn
```

BiSeNet INT8 通过精度门槛后，再尝试 LAS INT8：

```bash
python convert_rknn.py \
  --model las \
  --source artifacts/liteanystereo_las1_1280x736.onnx \
  --dtype int8 \
  --dataset build/calibration/dataset_las.txt \
  --output artifacts/liteanystereo_las1_int8_rk3588.rknn
```

## 五、RK3588 板端部署操作

### 5.1 安装和版本检查

本次已生成板端部署包：

```text
artifacts/jietai_rk3588_fp16_bundle.tar.gz
SHA-256: ca9329e8c5f0a3acd5683d8780d88a56bfb2970b014e40ba90d045a9c9fa1d9f
```

完成点云计时后另生成不覆盖原包的完整版本，增加 `pointcloud.py`、精度/部署文档和
点云参数说明：

```text
artifacts/jietai_rk3588_fp16_pointcloud_bundle.tar.gz
SHA-256: 见同目录 `jietai_rk3588_fp16_pointcloud_bundle.tar.gz.sha256`
```

本次通过反向 SSH 隧道把文件传到目标板，并统一放在用户指定目录：

```bash
mkdir -p /home/firefly/gq/rk3588_fp16
cd /home/firefly/gq/rk3588_fp16
tar -xzf jietai_rk3588_fp16_bundle.tar.gz
```

板端系统 Python 缺少 `python3.10-venv`，因此没有修改系统包，而是使用板上已有的
Miniconda Python 3.10.9 创建项目内虚拟环境：

```bash
python3 -m venv .venv-rknn
.venv-rknn/bin/python -m pip install \
  numpy==1.26.4 opencv-python-headless==4.10.0.84 psutil ruamel.yaml \
  ./rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

部署包和官方 Lite2 wheel 在板端校验通过：

```text
jietai_rk3588_fp16_bundle.tar.gz
  SHA-256 ca9329e8c5f0a3acd5683d8780d88a56bfb2970b014e40ba90d045a9c9fa1d9f
rknn_toolkit_lite2-2.3.2-...-aarch64.whl
  MD5 010dc8d577d91ee779f456ccf9997c7e
```

实机环境为 Firefly ITX-3588J MIPI、16 GiB 内存、Ubuntu 22.04.5、Linux
5.10.198 aarch64。加载日志记录到 Lite2 2.3.2、`librknnrt.so` 2.3.0、RKNPU
driver 0.9.3、模型 compiler 2.3.2。CPU governor 为 `schedutil`，NPU governor
为 `rknpu_ondemand`，测速时 NPU 读数为 1 GHz。未使用 sudo 修改 governor；测试后
NPU thermal 约 28.7 °C。`rknn_server` 已由系统启动，版本 2.3.0。

### 5.2 单场功能和精度检查

```bash
.venv-rknn/bin/python board_benchmark.py \
  --las-model artifacts/liteanystereo_las1_fp16_rk3588.rknn \
  --bisenet-model artifacts/bisenetv2_manual_fp16_rk3588.rknn \
  --left data/sample/im0.png \
  --right data/sample/im1.png \
  --warmup 3 --runs 20 \
  --las-core 0 --bisenet-core 0 \
  --output-dir artifacts/board_fp16_core0_w3_r20
```

必须检查这些输出存在且数值有限：

```text
benchmark_report.json
disparity.npy
foreground_probability.npy
raw_mask.png
refined_mask.png
subject_disparity.npy
```

随后把两个 core 参数都改为 `0_1_2`，输出到
`artifacts/board_fp16_core012_w3_r20`。两组的三个 `.npy` 输出逐元素比较最大绝对差
均为 0，说明 core 配置变化没有改变数值结果。样本文件哈希为：

```text
im0.png  SHA-256 145f8cea81a4266ae7c3b2607baeb4a9030cd2c41e03a544720b7dba0cc31ca4
im1.png  SHA-256 1af6567c38a041991f5916b39235d2179ddb54fadfa0f2dd4d22287ed9624119
```

### 5.3 73 场正式计时

```bash
python prepare_pairs_file.py \
  --dataset-root /data/JMP-LF6020-ETH3D \
  --scene-prefix fdjyp_3_ \
  --output build/fdjyp3_pairs.txt

python board_benchmark.py \
  --las-model artifacts/liteanystereo_las1_fp16_rk3588.rknn \
  --bisenet-model artifacts/bisenetv2_manual_fp16_rk3588.rknn \
  --pairs-file build/fdjyp3_pairs.txt \
  --warmup 10 --runs 73 \
  --las-core 0 --bisenet-core 0 \
  --output-dir artifacts/board_fp16_fdjyp3_73
```

单 core 是正式基准。完成后可把两个 core 参数同时改为 `0_1_2`，作为三核实验单独
报告。不同 core、dtype、频率和散热条件的结果不能混为同一组。

### 5.4 导出 73 场全部视差

`board_benchmark.py` 用于计时和抽样保存。需要保留每个场景的完整 LAS 视差时，在板端
单独执行批量导出脚本，使磁盘写入不混入第 5.3 节的端到端计时：

```bash
cd /home/firefly/gq/rk3588_fp16
.venv-rknn/bin/python board_export_disparities.py \
  --model artifacts/liteanystereo_las1_fp16_rk3588.rknn \
  --pairs-file data/JMP-LF6020-ETH3D/fdjyp3_pairs.txt \
  --core 0_1_2 \
  --output-dir artifacts/board_fp16_core012_disparities_73
```

每个场景目录包含三种文件：

```text
disparity.npy          float32 像素视差，数值分析的权威输出
disparity_x256.png     uint16 定点视差，disparity_px = 像素值 / 256
disparity_preview.png  每场景独立归一化的 TURBO 彩图，仅用于查看
```

根目录的 `export_report.json` 保存模型哈希、逐场统计和每个输出文件的 SHA-256；
`export_manifest.csv` 便于表格分析。实机导出和回传检查结果为：73 个场景、219 个视差
文件，精确数组均为 `(1280, 720)`、`float32` 且全部有限，逐文件 SHA-256 错误数为 0。
目录总大小约 329 MiB。板端目录为：

```text
/home/firefly/gq/rk3588_fp16/artifacts/board_fp16_core012_disparities_73
```

PC 回传目录为：

```text
experiments/04_mask_refinement/rk3588/artifacts/board_fp16_core012_disparities_73
```

这次独立导出中的纯 LAS 阻塞式 RKNN 调用 mean/P95 为
`1735.25/1779.34 ms`，范围 `1650.99–1786.76 ms`；三种格式的保存 mean/P95 为
`176.25/189.78 ms`。该轮没有 warmup，且只计 LAS 调用，不能替代第 8.2 节的正式完整
链路数据。

### 5.5 导出 73 场最终主体视差

第 5.4 节是 LAS 全图中间结果；实验 1–4 的最终视差是经过 BiSeNetV2、实验 4 Mask
修复和固定 ROI 裁取后的 `subject_disparity`。为避免重复运行 LAS，下面的脚本读取第
5.4 节已校验的全图视差，在 RK3588 上运行剩余链路并逐场保存最终结果：

```bash
cd /home/firefly/gq/rk3588_fp16
.venv-rknn/bin/python board_export_subject_disparities.py \
  --bisenet-model artifacts/bisenetv2_manual_fp16_rk3588.rknn \
  --pairs-file data/JMP-LF6020-ETH3D/fdjyp3_pairs.txt \
  --full-disparity-dir artifacts/board_fp16_core012_disparities_73 \
  --core 0_1_2 \
  --output-dir artifacts/board_fp16_core012_subject_disparities_73
```

每个场景的最终输出为：

```text
subject_disparity.npy          (818,512) float32；主体为像素视差，背景为 NaN
subject_disparity_x256.png     uint16；主体视差 = PNG值/256，背景为 0
subject_disparity_preview.png  主体彩色视差图，背景为黑色，仅用于查看
subject_mask.png               uint8；主体 255，背景 0
```

板端输出目录：

```text
/home/firefly/gq/rk3588_fp16/artifacts/board_fp16_core012_subject_disparities_73
```

PC 回传目录：

```text
experiments/04_mask_refinement/rk3588/artifacts/board_fp16_core012_subject_disparities_73
```

本次生成 73 组、292 个场景文件，总大小约 148 MiB。板端与 PC 端均复核：73 个数组
形状全部为 `(818, 512)`、dtype 为 `float32`，主体全部有限、背景全部为 NaN；16 位
PNG 与 `.npy` 按 `×256` 转换逐像素一致；输入全图视差和全部输出 SHA-256 错误数均为
0。主体像素数为 `263927–418816`，主体有效视差总范围为 `8.671875–188.0 px`。

这轮复用已生成的 LAS 视差，新增 BiSeNet mean/P95 为 `29.91/36.19 ms`，实验 4
后处理为 `26.41/35.46 ms`，四种文件保存为 `42.01/46.83 ms`。这些是独立导出耗时，
正式的 LAS + BiSeNet + 实验 4 + 点云端到端结果仍以第 8.2 节为准。

### 5.6 处理前后对比图

PC 端使用已经回传并通过哈希检查的数值结果生成对比图：

```bash
python make_subject_comparisons.py \
  --dataset-root ../../../datasets/training/JMP-LF6020-ETH3D \
  --full-disparity-dir artifacts/board_fp16_core012_disparities_73 \
  --subject-disparity-dir artifacts/board_fp16_core012_subject_disparities_73 \
  --output-dir artifacts/board_fp16_core012_camera_subject_comparisons_73 \
  --contact-sheet-columns 3
```

每张 `before_after_subject.png` 从左到右分别是：原始左目摄像头输入（红框标出实验 4
ROI）、ROI 内的处理前 LAS 视差、处理后最终主体视差、最终主体 Mask。前后两幅视差
使用同一场景 ROI 的真实最小/最大视差作为共享色标，所以相同颜色代表相同视差；处理
后黑色区域是被 Mask 剔除的背景。另生成 `all_subject_before_after.jpg` 汇总 73 个场景，
`comparison_report.json` 和 `comparison_manifest.csv` 记录摄像头原图哈希、色标、主体
比例、视差输入哈希和对比图哈希。原先不带摄像头栏的三栏对比仍保留在
`artifacts/board_fp16_core012_subject_comparisons_73`。

PC 输出目录为：

```text
experiments/04_mask_refinement/rk3588/artifacts/board_fp16_core012_camera_subject_comparisons_73
```

共生成 73 张独立对比图和 1 张总览图，独立图尺寸均为 `2060×943`，目录约 88 MiB；
摄像头原图、对比图和总览图哈希以及图像尺寸复核错误数均为 0。

## 六、FP32/RKNN 精度核对流程

历史结果目录存在内容几乎相同但并非逐字节相同的图像副本。量化误差比较必须在同一
输入文件上重新生成 FP32 reference。

### 6.1 生成同输入 FP32 reference

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

### 6.2 比较 RKNN 输出

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

初始工程门槛：

| 检查项 | 建议门槛 |
|---|---:|
| ONNX 对 PyTorch mean absolute disparity difference | `< 1e-3 px` |
| RKNN 对 FP32 的 73 场参考 EPE 恶化 | `< 5%` |
| BiSeNet 阈值 Mask 对 FP32 Mask 宏平均 IoU | `>= 0.99` |
| refined Mask 对 FP32 refined Mask 宏平均 IoU | `>= 0.99` |

## 七、推理时间统计口径

板端 JSON 同时报告三类时间：

| 统计字段 | 范围 |
|---|---|
| `model_only.liteanystereo` | 已准备输入上的 LAS 阻塞调用和输出回传 |
| `model_only.bisenetv2` | 已准备输入上的 BiSeNet 阻塞调用和输出回传 |
| `model_only.sequential_both` | 同一轮串行执行 LAS 和 BiSeNet；不是两个独立 P95 相加 |
| `pipeline.preprocess` | BGR→RGB、LAS padding、BiSeNet resize |
| `pipeline.postprocess` | 输出布局整理、实验 4 修复和主体视差生成 |
| `pipeline.end_to_end` | 内存中完整稳定态链路，不含模型加载和磁盘 I/O |

每组报告 mean、median、P95、min、max 和由 mean 换算的 FPS。模型加载/冷启动时间另行
记录，不能混入稳定态单帧时间。

## 八、正式结果记录表

### 8.1 环境表

| 项目 | 实测值 |
|---|---|
| RK3588 板卡/内存 | Firefly ITX-3588J MIPI / 15 GiB 可见内存 |
| 系统镜像/内核 | Ubuntu 22.04.5 LTS / Linux 5.10.198 aarch64 |
| Toolkit2 compiler | 2.3.2 (`e045de294f`, 2025-04-07) |
| Lite2/runtime/driver | 2.3.2 / 2.3.0 (`c949ad889d`) / 0.9.3 |
| NPU core | Core 0 与 Core 0/1/2 两组独立测试 |
| CPU/NPU governor | `schedutil` / `rknpu_ondemand`；NPU 读数 1 GHz |
| 测试后温度 | NPU 约 28.7 °C；未改变板卡原散热配置 |
| Python/NumPy/OpenCV | 3.10.9 / 1.26.4 / 4.10.0 headless |
| warmup/runs | 单场 `3/20`；多场 `10/73`；模型加载和磁盘 I/O 不计时 |

### 8.2 延迟表

| 模型组合 | NPU core | LAS mean/P95 | BiSeNet mean/P95 | 双模型串行 mean/P95 | 完整链路 mean/P95 | FPS |
|---|---|---:|---:|---:|---:|---:|
| LAS FP16 + BiSeNet FP16 | 0 | 1922.15 / 1948.58 ms | 69.97 / 78.13 ms | 2011.61 / 2040.60 ms | 2030.15 / 2064.38 ms | 0.493 |
| LAS FP16 + BiSeNet INT8 | 0 | 待测 | 待测 | 待测 | 待测 | 待测 |
| LAS INT8 + BiSeNet INT8 | 0 | 待测 | 待测 | 待测 | 待测 | 待测 |
| LAS FP16 + BiSeNet FP16 | 0_1_2 | 1837.33 / 1866.71 ms | 50.17 / 78.84 ms | 1900.85 / 1938.78 ms | 1942.95 / 1988.35 ms | 0.515 |

这里的完整链路是单个样本重复 20 次的稳定态结果，不等同于 73 场正式实验。三核组相对
Core 0 的完整链路 mean 缩短 4.30%，FPS 提高约 4.49%；同步单帧不会获得三倍加速。

73 个不同场景各执行一次的正式三核 FP16 + 点云结果为：

| 阶段 | mean | median | P95 | min/max |
|---|---:|---:|---:|---:|
| LAS 单模型（固定首场稳定态） | 1825.42 ms | 1831.39 ms | 1867.85 ms | 1741.14 / 1879.65 ms |
| BiSeNet 单模型（固定首场稳定态） | 46.55 ms | 45.83 ms | 62.64 ms | 36.05 / 65.29 ms |
| 73 场到主体视差 | 1907.88 ms | 1910.72 ms | 1952.73 ms | 1830.16 / 1987.13 ms |
| 73 场点云重投影 | 50.12 ms | 50.72 ms | 55.53 ms | 39.71 / 59.84 ms |
| 73 场到内存点云 | 1958.00 ms | 1964.02 ms | 2000.24 ms | 1878.09 / 2034.01 ms |

73/73 场计算链路均低于 4 秒，68/73 场低于 2 秒，标准差 `32.58 ms`、变异系数
`1.66%`。LAS 平均占到内存点云时间约 `89.55%`，是后续进一步加速的主要目标。

### 8.3 精度表

| 模型组合 | disparity MAE/P95 | bad-1px | BiSeNet Mask IoU | refined Mask IoU | 是否通过 |
|---|---:|---:|---:|---:|---|
| LAS FP16 + BiSeNet FP16 | 0.09076 / 0.14266 px | 0.02550% | 0.999973 | 0.999986 | 单场 FP16 通过 |
| LAS FP16 + BiSeNet INT8 | 待测 | 待测 | 待测 | 待测 | 待定 |
| LAS INT8 + BiSeNet INT8 | 待测 | 待测 | 待测 | 待测 | 待定 |

本次板端 FP16 输出全部有限：disparity 为 `1280×720`，范围
`3.7715–117.6875 px`、均值 `39.4970 px`；BiSeNet 前景概率为 `512×288`，范围
`6.11e-5–1.0`、均值 `0.7626533`。对应 PC FP32 汇总为 disparity 均值
`39.4126 px`、BiSeNet 概率均值 `0.7626499`。这些汇总和前景像素检查用于发现明显
错误。板端文件回传后完成了逐元素比较：disparity RMSE `0.10383 px`、最大误差
`4.60952 px`、bad-3px `0.00347%`；BiSeNet 概率 MAE `1.0946e-4`、P95
`3.8689e-4`，0.5 阈值类别变化 `0.00203%`；最终 refined Mask 仅 10 个像素变化。
Core 0 和 Core 0/1/2 的 `accuracy.json` 内容一致。

### 8.4 四秒完整链路目标

项目目标口径是 `1280×720` 双目采集开始，到点云重建完成不超过 `4 s`，历史时间约
`8 s`，即要求完整链路至少缩短 50%。本报告第 8.2 节的 `pipeline.end_to_end` 还不是
这一完整口径：它从两张已在内存中的图像开始，结束于主体视差，仅覆盖预处理、两个
RKNN、Mask 修复和主体视差生成。

三核组该部分 mean 为 `1942.95 ms`、P95 为 `1988.35 ms`，因此按 mean 计算还给
相机采集、校正/裁剪、点云重投影和必要输出留下 `2057.05 ms`；按 P95 计算留下
`2011.65 ms`。是否最终达到 4 秒，必须使用同一块 RK3588、真实相机和最终点云输出
格式，从采集 API 前打点到点云可交付后打点，不能把当前 1.94 秒直接当作完整结果。

为补齐点云计算口径，板端程序已增加 `--include-pointcloud`，使用 FDJYP-3/JXP 的 Q
矩阵，并按固定 ROI `[234:1052,126:638]` 修正光心。Core 0/1/2、warmup 3、runs 20
的新增实测为：

| 阶段 | mean | P95 | 说明 |
|---|---:|---:|---|
| 到主体视差 `end_to_end` | 1912.05 ms | 1946.78 ms | 同一轮点云测试中的前半链路 |
| 点云重投影 `pointcloud` | 49.66 ms | 54.02 ms | ROI disparity→内存 XYZRGB |
| 到内存点云 `end_to_end_pointcloud` | 1961.72 ms | 1997.38 ms | 不含相机和磁盘 I/O |
| 二进制 PLY 写盘 | 12.01 ms | 单次值 | 343,992 点，5,160,060 bytes，计时范围外 |

因此，若“点云完成”定义为内存中得到 XYZRGB，按本次 mean/P95 分别还给相机采集与
校正留下约 `2038.28/2002.62 ms`。若还要求二进制 PLY 已落盘，本次观察到的组合时间
约 `1973.72 ms`，但写盘只有单次值，正式验收仍应把写盘纳入每轮端到端统计。生成的
PLY SHA-256 为
`e813ee48b960e3d7d62956f2f80c6f513c700d67b0653825027acd0f8d7f7e93`。

73 场正式结果进一步确认：到内存点云 mean `1958.00 ms`、P95 `2000.24 ms`、最大
`2034.01 ms`。因此相对 4 秒目标，按 mean/P95/最慢场景分别仍有
`2042.00/1999.76/1965.99 ms` 可用于相机采集、校正和必要输出。最慢场景是
`fdjyp_3_1_202506281614_0033`，总计 `2034.01 ms`；其中实验 4 后处理
`110.56 ms`。第二慢场景 `...0058` 为 `2025.46 ms`，后处理达到全组最大
`124.98 ms`。这些逐场记录已写入 `benchmark_report.json/pipeline_samples`，并另行
导出为同目录 `timing_per_scene.csv`。

工程优先级如下：

1. 主线保持 RKNN/NPU，不改为纯 ONNX/CPU；当前 LAS 已占双模型串行时间约 96.7%，
   CPU 分块会争用相机、校正和点云阶段的资源；
2. 相机帧直接进入预分配内存，校正映射预计算，避免 PNG/JPEG 落盘再读取；
3. 点云只生成业务需要的一种表示，优先内存 XYZRGB 或二进制 PLY，避免同时写
   ASCII PLY、`.npy` 和 `.bin`；
4. 若实测完整 P95 仍超过 4 秒，再评估 LAS 降分辨率/剪枝/轻量化；这类改动必须
   重新训练或至少重新做 73 场几何精度验证，不能只替换算子后沿用现有结论。

## 九、异常处理

| 现象 | 处理 |
|---|---|
| ONNX 导出报 `Unfold` | 使用本目录 exporter 和已修正的 `projects/LiteAnyStereo/export_onnx.py` |
| PB 在 ArgMax/Squeeze 失败 | RKNN 输出保持为 `final_probability` |
| model/runtime mismatch | 升级 Lite2 与 `librknnrt.so`，作废该轮数据 |
| LAS build 不支持某算子 | 保留完整 Toolkit2 日志和算子名，不静默改分辨率 |
| LAS full resolution build 内存不足 | 在更大内存 x86_64 转换机重试；降分辨率必须建立新实验分支 |
| BiSeNet INT8 通过但 LAS INT8 失真 | 采用 LAS FP16 + BiSeNet INT8 |
| 后处理耗时波动大 | 使用 73 场列表，不只重复单张简单图 |

## 十、当前边界与下一步

当前已经完成并有证据支持的是：

- LAS 正式 ONNX 可导出、可运行且与 PyTorch 数值一致；
- BiSeNet PB 输入输出和概率正确；
- 实验 4 部署后处理与正式脚本一致；
- LAS 和 BiSeNet 两个 FP16 `.rknn` 已生成并记录 SHA-256；
- BiSeNet FP16 已通过 PC simulator 同输入数值检查；
- 两模型已复制到 `/home/firefly/gq/rk3588_fp16`，能被 RK3588 实机加载和连续推理；
- Core 0 和 Core 0/1/2 的单场 20 次稳定态数据已生成，输出逐元素一致；
- 73 对正式图片已经复制到板端并完成三核 FP16 + 点云推理；
- 73/73 场计算链路低于 4 秒，逐场耗时、JSON、PLY、视差、概率和 Mask 已归档。
- 73 场完整 LAS 视差已按 `.npy`、16 位 PNG 和预览 PNG 三种格式导出并回传，文件数、
  数组类型、有限值和 SHA-256 均已复核。
- 73 场最终主体视差已按 `.npy`、16 位 PNG、黑底彩色预览和主体 Mask 导出并回传，
  数值、Mask、定点缩放和逐文件 SHA-256 均已复核。
- 73 场摄像头原图/处理前/处理后/主体 Mask 共享色标对比图及一张总览图已生成，尺寸和
  哈希已复核。

仍需在目标环境完成：

1. 接入真实相机 API，在采集前和点云可交付后打点，验收完整链路 P95 `<4 s`；
2. 如需进一步加速，先转换 BiSeNet INT8 并按精度门槛验证，再决定是否尝试 LAS INT8；
3. 根据精度门槛确定最终采用“全 FP16”还是
   “LAS FP16 + BiSeNet INT8”。

当前第八节毫秒数均来自 RK3588 实机，不包含 RTX 4090 或 PC simulator 时间。
