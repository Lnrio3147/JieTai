# LiteAnyStereo：JMP-LF6020 训练与测试使用说明

本说明对应当前整理后的目录。JMP 的模型输入始终是极线校正后的左右图；点云只在数据准备阶段
生成训练伪视差，推理时不会输入点云。

## 目录与保留结果

| 位置 | 内容 |
| --- | --- |
| `JMP-LF6020.zip` | 原始数据，保留用于重新转换 |
| `data/datasets/JMP-LF6020-ETH3D/` | 唯一的 JMP 训练数据根目录，266 场（训练 193、验证 73） |
| `checkpoints/LiteAnyStereo.pth` | LAS1 官方权重 |
| `runs/training/jmp_lf6020_las1/` | 已完成的 20 轮 JMP 微调运行，可用 `latest.pth` 断点恢复 |
| `runs/evaluation/jmp_unified_rerun_73/` | 最终统一复评：73 场 LAS 重跑浮点视差、RT-IGEV 对比指标和图片 |
| `runs/inference/tradition_extra/official/` | 额外 78 场推理；其中 64 场含历史算法对比图 |

## 当前模型状态

- `checkpoints/LiteAnyStereo.pth` 是已经训练完成的 LAS1 官方权重，可直接用于测试和新图片推理；
- `runs/training/jmp_lf6020_las1/best.pth` 和 `latest.pth` 是已经完成 20 轮、使用 193 场 JMP
  伪标签微调得到的权重；
- 当前统一口径验证中，官方权重优于这次 JMP 微调权重。因此现阶段部署和新图推理默认使用
  `checkpoints/LiteAnyStereo.pth`，无需先重新训练。

## 1. 环境

```bash
cd /path/to/JieTai/projects/LiteAnyStereo
conda env create -f environment-training.yml
conda activate liteanystereo
```

已有环境时只需 `conda activate liteanystereo`。训练和完整 JMP 测试默认使用 CUDA；没有 GPU 时在
命令末尾加 `--device cpu`，但速度会明显下降。

代码与数据检查：

```bash
python -m unittest discover -s tests -v
python tools/check_stereo_dataset.py \
  --manifest ./data/datasets/JMP-LF6020-ETH3D/manifest.csv \
  --max_disp 192
```

## 2. 从原始 ZIP 重新整理 JMP 数据（通常不需要重复执行）

```bash
python tools/prepare_jmp_lf6020.py \
  --archive ./JMP-LF6020.zip \
  --output ./data/datasets/JMP-LF6020-ETH3D

python tools/check_stereo_dataset.py \
  --manifest ./data/datasets/JMP-LF6020-ETH3D/manifest.csv \
  --max_disp 192 \
  --report ./data/datasets/JMP-LF6020-ETH3D/metadata/dataset_check.json
```

每个场景使用 ETH3D 兼容格式：`im0.png`、`im1.png`、`disp0GT.pfm`、`mask0nocc.png` 和
`calib.txt`。`disp0GT.pfm` 是由增强点云投影的训练伪标签，不是人工 GT；`mask0nocc.png`
表示伪视差是否有效，不是工件分割掩码。

## 3. 官方权重正式测试（与 tradition_stereo 统一口径）

```bash
python evaluate_stereo.py \
  --version las1 \
  --dataset jmp \
  --data_root ./data/datasets/JMP-LF6020-ETH3D \
  --evaluation_protocol tradition \
  --tradition_eval_root ../tradition_stereo/datasets/FDJYP-3 \
  --restore_ckpt ./checkpoints/LiteAnyStereo.pth \
  --max_disp 192 \
  --no_epe_filter \
  --save_vis \
  --vis_dir ./runs/evaluation/my_official_evaluation
```

数值在固定排除 4 个异常场景后的相同 69 场上做场景宏平均；指标 EPE、D1、Bad1、Bad2、Bad3
均是越小越好。已完成的正式结果为 EPE 1.9457 px、D1 7.03%、Bad1/2/3 为
38.86%/16.02%/8.77%。

每个场景的输出含义：

- `disp.npy`：模型实际输出的 float32 视差，用于重新计算指标或生成点云；
- `vis.png`：当前场景自适应色标的 LiteAnyStereo 视差图；
- `vis_fixed.png`：固定 0–192 px 色标的视差图，适合跨场景比较；
- `comparison.png`：左图、预测、Foundation Stereo 参考与预测绝对误差；
- `traditional_comparison.png`：若 `tradition_eval_root` 中保存了旧视差，则给出旧视差、
  LiteAnyStereo、参考及误差的六宫格。正式汇报若要求直接使用一期原图，应读取
  `../tradition_stereo/igev_output/<scene>/vis.png`，不要用六宫格中的重新着色图替代原始
  RT-IGEV `vis.png`。

### 3.1 与 RT-IGEV 保存结果统一复评（最终汇报口径）

先重新推理 LiteAnyStereo 全部 73 场，同时保存浮点视差：

```bash
python evaluate_stereo.py \
  --version las1 \
  --dataset jmp \
  --data_root ./data/datasets/JMP-LF6020-ETH3D \
  --evaluation_protocol tradition \
  --tradition_eval_root ../tradition_stereo/datasets/FDJYP-3 \
  --restore_ckpt ./checkpoints/LiteAnyStereo.pth \
  --max_disp 192 \
  --workers 4 \
  --no_epe_filter \
  --include_excluded \
  --save_vis \
  --vis_dir ./runs/evaluation/jmp_unified_rerun_73/liteanystereo
```

再用同一个评价函数重新计算 RT-IGEV 保存的原始 `disp.npy` 与 LiteAnyStereo：

```bash
python tools/compare_unified_saved_igev_las.py --save-comparisons
```

脚本统一参考视差、固定 ROI、有效掩码、指标函数和场景集合，并同时输出全 73 场与固定
69 场结果。最终汇报结果位于 `runs/evaluation/jmp_unified_rerun_73/`。固定 69 场中
LiteAnyStereo 的 EPE 为 1.9457 px，RT-IGEV 为 3.4483 px；全 73 场中分别为
2.0762 px 和 4.6745 px。脚本还会按图像亮斑特征独立筛选高反光 15 场，输出全 ROI 和
仅高光像素的专项指标、候选场景联系表及柱状图；定义与阈值见最终报告第 6 节。

### 3.2 新双目图片推理前的准备

新图片推理只需要左右两张图，不需要输入点云、参考视差或 mask。但输入必须满足：

1. 左右图由同一双目相机在同一时刻采集，不能把不同时间或不同场景的图片配成一对；
2. 左右图分辨率必须相同，且顺序固定为左图 `im0`、右图 `im1`；
3. 推理前必须完成去畸变和极线校正，使同一物点在左右图中的纵坐标基本一致；
4. 若更换相机、镜头、焦距、双目基线或采集分辨率，需要重新标定，不能继续使用旧校正映射；
5. PNG、JPG 均可，建议保存无损 PNG。当前 JMP 图像尺寸为高 1280、宽 720，但模型也能接受其他尺寸。

`tradition_stereo/config/map/` 下保存了已有相机的校正映射。只有确认新图片来自对应相机及对应
分辨率时才能复用，例如：

```python
import cv2
import numpy as np

left = cv2.imread("raw/im0.png")
right = cv2.imread("raw/im1.png")
left_rect = cv2.remap(
    left,
    np.load("../tradition_stereo/config/map/1221/left_map1.npy"),
    np.load("../tradition_stereo/config/map/1221/left_map2.npy"),
    cv2.INTER_LINEAR,
)
right_rect = cv2.remap(
    right,
    np.load("../tradition_stereo/config/map/1221/right_map1.npy"),
    np.load("../tradition_stereo/config/map/1221/right_map2.npy"),
    cv2.INTER_LINEAR,
)
cv2.imwrite("new_scene/im0.png", left_rect)
cv2.imwrite("new_scene/im1.png", right_rect)
```

这里的 `1221` 只是示例，必须替换为实际相机对应的映射目录。如果没有该相机的标定参数，应先完成
双目标定，再参考 `../tradition_stereo/read_stereo.py` 生成校正映射。

### 3.3 分开的左右图直接推理（推荐）

```bash
cd /path/to/JieTai/projects/LiteAnyStereo
conda activate liteanystereo

python demo.py \
  --version las1 \
  --restore_ckpt ./checkpoints/LiteAnyStereo.pth \
  --left ./new_scene/im0.png \
  --right ./new_scene/im1.png \
  --out_dir ./runs/inference/new_scene \
  --max_disp 192 \
  --device cuda \
  --get_pc 0
```

对于一般新图片建议使用 `--get_pc 0`：视差网络本身不需要点云，而 `demo.py` 的默认点云内参和
基线属于示例相机。如果要生成具有真实尺度的深度或点云，必须换成新相机的校正后焦距、主点和
双目基线，深度关系为 `Z = fx × baseline / disparity`。

### 3.4 左右并排图片推理

旧的调用方式仍然保留。输入文件必须是等宽的左图和右图水平拼接，左图在前、右图在后：

```bash
python demo.py \
  --version las1 \
  --restore_ckpt ./checkpoints/LiteAnyStereo.pth \
  --stereo_file ./assets/Explorer_HD2K_SN28883284_20-42-06.png \
  --out_dir ./runs/inference/demo \
  --get_pc 0
```

### 3.5 输出文件说明

- `disp.npy`：模型直接输出的浮点视差矩阵，单位为像素；这是后续测量和点云计算应读取的结果；
- `vis.png`：左图与彩色视差图的并排预览，仅用于观察，不能从颜色反推出精确视差；
- `img.gif`：左右图交替显示，用于快速检查极线校正和左右图差异；
- `cloud.ply`、`cloud.glb`：仅在 `--get_pc 1` 时生成，真实尺度依赖正确的相机内参和基线。

读取直接输出并建立有效区域：

```python
import numpy as np

disp = np.load("runs/inference/new_scene/disp.npy")
valid = np.isfinite(disp) & (disp > 0) & (disp < 192)
print(disp.shape, disp[valid].min(), disp[valid].max())
```

没有参考视差 GT 的新图片只能进行定性观察，不能计算 EPE、D1、Bad1/2/3。若预期实际视差可能
超过 192 px，应根据相机焦距、基线和最近工作距离估算后提高 `--max_disp`，否则大视差区域可能
预测不完整。

## 4. 微调训练与断点恢复

训练使用 193 场 JMP 伪标签，验证按第 3 节的统一口径执行。请使用新的输出目录，避免覆盖已有
`runs/training/jmp_lf6020_las1/` 的历史运行。

```bash
python train_stereo.py \
  --version las1 \
  --dataset jmp \
  --data_root ./data/datasets/JMP-LF6020-ETH3D \
  --evaluation_protocol tradition \
  --tradition_eval_root ../tradition_stereo/datasets/FDJYP-3 \
  --restore_ckpt ./checkpoints/LiteAnyStereo.pth \
  --output_dir ./runs/training/experiments/jmp_las1_lr1e-5 \
  --epochs 20 \
  --batch_size 2 \
  --workers 4 \
  --crop_height 256 \
  --crop_width 512 \
  --max_disp 192 \
  --lr 1e-5 \
  --save_every 5 \
  --no-eval_epe_filter \
  --deterministic
```

恢复中断的训练时，移除 `--restore_ckpt`，改为：

```bash
--resume ./runs/training/experiments/jmp_las1_lr1e-5/latest.pth \
--output_dir ./runs/training/experiments/jmp_las1_lr1e-5
```

训练会写入 `config.json`、`metrics.jsonl`、`train.log`、`best.pth`、`latest.pth`。
当前 JMP 的伪标签与 Foundation Stereo 参考存在系统差异，已有微调权重没有优于官方权重；
这不是推理阶段缺少点云，而是伪监督质量限制。因此建议以官方权重作为当前部署与汇报基线。

## 5. 额外 tradition_stereo 图像推理与对比

```bash
python tools/infer_tradition_extra.py \
  --tradition_root ../tradition_stereo \
  --current_data_root ./data/datasets/JMP-LF6020-ETH3D \
  --restore_ckpt ./checkpoints/LiteAnyStereo.pth \
  --output_dir ./runs/inference/tradition_extra/official \
  --version las1 \
  --max_disp 192 \
  --device cuda \
  --overwrite

python tools/compare_tradition_extra.py \
  --tradition_root ../tradition_stereo \
  --inference_dir ./runs/inference/tradition_extra/official \
  --max_disp 192 \
  --overwrite
```

该流程共推理 78 场，其中 JXP 15、工件测试 6、其他测试 6、螺纹件 37 场可找到传统算法结果并
生成六宫格。没有统一 GT，最后一格仅是两种输出的绝对差异，不能当作精度误差。JXP 的垂直极线
残差约 32–38 px，结果只可观察，不应纳入正式性能结论。

## 6. 重新生成汇报 PDF

```bash
python tools/generate_jmp_test_report_pdf.py
```

输出为 `docs/reports/JMP_LITEANYSTEREO_TEST_REPORT_zh-CN.pdf`。该脚本读取
`runs/evaluation/jmp_unified_rerun_73/` 的最终统一复评资产。
