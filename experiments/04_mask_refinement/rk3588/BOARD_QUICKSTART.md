# RK3588 FP16 板端快速启动

本部署包已经包含 LiteAnyStereo LAS1、BiSeNetV2 的 FP16 RKNN 模型，以及实验 4
后处理和计时脚本。RK3588 只负责加载 `.rknn` 和推理，不在板端转换 ONNX/PB。

## 1. 准备 runtime

先确认板端架构和 Python 版本：

```bash
uname -m
python3 -c 'import platform, sys; print(platform.machine(), sys.version)'
```

安装 RKNN-Toolkit-Lite2 2.3.2 aarch64 wheel。Firefly 本次系统 Python 缺少
`python3.10-venv`，使用板上已有 Miniconda Python 建立项目内环境即可，无需改系统包：

```bash
python3 -m venv .venv-rknn
.venv-rknn/bin/python -m pip install \
  numpy==1.26.4 opencv-python-headless==4.10.0.84 psutil ruamel.yaml \
  ./rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

板端镜像已经提供 OpenCV 时，可以用 `--system-site-packages` 创建虚拟环境并省略
`opencv-python`，避免 GUI/GL 依赖冲突。

## 2. 单场验证和计时

把同一场景的校正左、右图放到板端，然后执行：

本次实机目录为 `/home/firefly/gq/rk3588_fp16`，样本放在 `data/sample`：

```bash
cd /home/firefly/gq/rk3588_fp16
.venv-rknn/bin/python board_benchmark.py \
  --las-model artifacts/liteanystereo_las1_fp16_rk3588.rknn \
  --bisenet-model artifacts/bisenetv2_manual_fp16_rk3588.rknn \
  --left data/sample/im0.png \
  --right data/sample/im1.png \
  --warmup 10 --runs 50 \
  --las-core 0 --bisenet-core 0 \
  --output-dir artifacts/board_fp16_sample
```

成功后应生成 `benchmark_report.json`、`disparity.npy`、
`foreground_probability.npy`、`raw_mask.png`、`refined_mask.png` 和
`subject_disparity.npy`。先检查单核 `core 0`，再把两个 core 参数都改成 `0_1_2`
单独测试三核模式。

如果验收口径是“图像已在内存到点云完成”，在命令中加
`--include-pointcloud`。报告会新增 `pipeline.pointcloud` 和
`pipeline.end_to_end_pointcloud`，并在计时结束后保存二进制
`pointcloud_xyzrgb_binary.ply`。该选项使用 FDJYP-3/JXP 固定 Q 矩阵与
`[234:1052,126:638]` ROI，换相机或标定文件时不能直接复用。

不要把 PC simulator 时间写成板端时间。正式报告使用 RK3588 输出 JSON 中的
mean、median、P95 和 FPS，并记录 Lite2、`librknnrt.so`、RKNPU driver、频率、温度
和散热条件。

本次已验证 compiler/Lite2 2.3.2 模型可在目标板 runtime 2.3.0、driver 0.9.3
加载运行；如果其他镜像出现明确的 model/runtime mismatch，才升级 runtime 后重测，
不要仅因小版本不同直接覆盖系统库。
