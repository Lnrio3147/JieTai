# 技术流程文档

本文档将当前项目整理为两个主流程：数据处理和计算指标。所有命令默认在仓库根目录执行：

```bash
cd D:\Desktop\stereo_project\tradition_stereo
```

## 1. 数据处理

说明：本文档中的 `GT` 指评估用参考视差图，由 Foundation Stereo 生成，并统一裁剪为 `818x512` 后保存为 `disp_cropped.npy`。它不是人工标注真值，也不是甲方原始点云本身。

### 1.1 原始左右图整理

用途：将采集目录中的 `*_L.png` / `*_R.png` 配对，并整理为每个场景一个文件夹：

```text
output/
  202506281603-0001/
    im0.png
    im1.png
```

当前脚本：

```bash
python data_process/save_rawimg.py
```

注意：该脚本目前在文件内硬编码 `source_dir`，运行前需要修改为实际原始图片目录。后续建议改为：

```bash
python data_process/save_rawimg.py --source-dir <raw_image_dir> --output-dir <paired_output_dir>
```

### 1.2 生成矫正图像

推荐使用直接读取标定文件的版本。它会从 `config/stereo.yml` 读取相机参数，计算矫正映射，并批量生成矫正后的 `im0.png` / `im1.png`。

当前脚本：

```bash
python data_process/save_rectified_direct.py
```

运行前检查脚本内配置：

```python
config_path = r"D:\Desktop\stereo_project\tradition_stereo\config\stereo.yml"
input_root = r"D:\Desktop\20260205\20260205\output"
output_root = r"D:\Desktop\20260205\20260205\rectified_images"
```

如已有预计算 map，也可以使用：

```bash
python data_process/save_rectified.py
```

该脚本默认读取 `config/map/1221/` 下的 `left_map1.npy`、`left_map2.npy`、`right_map1.npy`、`right_map2.npy`。使用前需要确认 map 目录和输入输出路径。

### 1.3 检查矫正效果

用途：抽样显示左右矫正图，并绘制水平参考线，检查极线是否对齐。

```bash
python read_stereo.py
```

注意：该脚本主要用于调试。运行前需要修改 `im_path` 和标定文件选择。

### 1.4 Foundation Stereo 生成 GT 视差

用途：将 Foundation Stereo 输出的参考视差整理为评估用 GT，并保存为 `disp_cropped.npy`。当前验证后采用固定裁剪区域：

```python
disp_cropped = disp[234:1052, 126:638]  # H=818, W=512
```

命令：

```bash
python data_process/save_old_disp.py
```

运行前检查：

```python
root_directory = "../datasets/FDJYP-3"
```

输出示例：

```text
datasets/FDJYP-3/<scene>/
  disp_cropped.npy
  disp_color.png
```

注意：如果输入仍是甲方提供的 `_old.ply` 点云，需要先反投影生成视差图；但最终指标中的 `disp_cropped.npy` 应明确作为 Foundation Stereo 参考结果使用。

### 1.5 预测视差裁剪到统一尺寸

用途：将待评估模型预测的原始 `disp.npy` 裁剪到与 Foundation Stereo GT 相同的 `818x512` 尺寸。

当前脚本：

```bash
python data_process/save_model_dis.py
```

重要：该脚本注释写的是固定裁剪 `[234:1052, 126:638]`，但当前实际代码使用甲方动态裁剪逻辑。整理时应统一改为固定裁剪后再作为主线使用。

建议整理后输出：

```text
datasets/FDJYP-3/<scene>/
  disp_rknn.npy
  disp_rknn.png
```

### 1.6 由裁剪视差生成点云和 Pointmap

如果预测视差已经是 `512x818`，推荐使用：

```bash
python batch_process_512x818.py
```

该脚本会输出：

```text
<output_root>/<scene>/
  out.ply
  pointmap.npy
  pointmap.bin
  disp.npy
  im0.png
  im1.png
```

运行前检查脚本中的：

```python
input_root = r"D:\Desktop\test_lite\luowen-disparity"
image_root = r"D:\Desktop\stereo_project\tradition_stereo\rec_img_set\luowen_rectified_images"
output_root = r"D:\Desktop\test_lite\luowen-disparity_ply"
```

旧流程 `batch_process_igev.py` 适合输入仍是原始尺寸视差图的情况，但里面使用动态裁剪，整理后建议降级为 legacy。

## 2. 计算指标

指标计算时，`--gt-file` 默认指向 Foundation Stereo 生成并裁剪后的 `disp_cropped.npy`；`--pred-file` 指向待评估模型或点云反投影得到的预测视差。

### 2.1 核心指标函数

核心指标定义在：

```text
metric/cal_metric.py
```

包含：

```python
epe_metric()
d1_metric()
threshold_metric()
```

指标含义：

- `EPE`：平均端点误差，单位为像素。
- `D1`：误差大于 3px 且相对误差大于 5% 的像素比例。
- `Bad1/Bad2/Bad3`：误差分别大于 1/2/3px 的像素比例。

### 2.2 计算裁剪视差指标

推荐入口：

```bash
python tools/cal_metric_512x818.py ^
  --pred-dir D:\Desktop\scene_demo-imgs_512x832_crop_first ^
  --gt-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3 ^
  --pred-file disp_512x818_crop.npy ^
  --gt-file disp_cropped.npy ^
  --results-dir D:\Desktop\scene_demo-imgs_512x832_crop_first\evaluation_results ^
  --epe-threshold 20
```

常用参数：

```bash
--no-epe-filter       # 不过滤高 EPE 场景
--include-excluded    # 包含默认排除场景
```

输出：

```text
evaluation_results/
  disparity_evaluation_<timestamp>.txt
  disparity_metrics.csv
```

### 2.3 当前旧指标脚本

旧入口仍可运行，但路径硬编码较多：

```bash
python tools/cal_metric.py
python tools/cal_metric_igev.py
```

整理时建议用 `tools/cal_metric_512x818.py` 替代它们，避免维护多套重复逻辑。

### 2.4 计算过滤后点云指标

用途：将点云反投影回视差图，再与 `disp_cropped.npy` 计算指标。

普通多方法点云：

```bash
python tools/cal_metric_ply.py
```

RKNN 点云：

```bash
python tools/cal_metric_ply_rknn.py
```

运行前检查：

```python
PLY_ROOT = r"D:\Desktop\过滤后\原始ply点云_rknn"
GT_ROOT = r"D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3"
```

重要：这两个脚本当前仍使用动态裁剪参数调整 Q 矩阵。如果 GT 由固定裁剪 `[234:1052, 126:638]` 生成，则点云反投影也必须改成同一固定裁剪偏移，否则指标会有系统性偏差。

### 2.5 推荐整理后的指标主流程

先生成或整理 Foundation Stereo GT：

```bash
python data_process/save_old_disp.py
```

再生成预测裁剪视差：

```bash
python data_process/save_model_dis.py
```

最后计算指标：

```bash
python tools/cal_metric_512x818.py ^
  --pred-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3 ^
  --gt-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3 ^
  --pred-file disp_rknn.npy ^
  --gt-file disp_cropped.npy ^
  --results-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3\evaluation_results
```

如果评估点云输出，则先确认反投影裁剪逻辑已与 GT 对齐，再运行：

```bash
python tools/cal_metric_ply_rknn.py
```

### 2.6 使用 LiteAnyStereo 模型并保持本评价口径

LiteAnyStereo 仓库已经提供专用入口。模型读取 `JMP-LF6020-ETH3D` 中正确极线校正后的
完整 `im0.png/im1.png`，参考视差仍读取本仓库 `datasets/FDJYP-3/disp_cropped.npy`；模型
先做完整图推理，再固定裁剪 `[234:1052, 126:638]`。指标、有效掩码、
排除名单、EPE 过滤和场景宏平均均与 `tools/cal_metric_512x818.py` 对齐，不需要先导出预测
视差再运行第二套统计脚本。

不要把本目录历史整理的 `im0/im1` 直接作为 LAS 输入：它们仍存在明显垂直极线残差，且
不满足 LAS 的水平极线输入要求。统一的是评价参考和指标，不是强行让不同模型读取不满足
各自几何要求的输入。

从 `LiteAnyStereo` 根目录评价单个权重：

```bash
python evaluate_stereo.py \
  --version las1 \
  --dataset jmp \
  --data_root ./data/datasets/JMP-LF6020-ETH3D \
  --evaluation_protocol tradition \
  --tradition_eval_root ../tradition_stereo/datasets/FDJYP-3 \
  --restore_ckpt ./checkpoints/LiteAnyStereo.pth \
  --max_disp 192
```

如需保存 LiteAnyStereo 的彩色预测视差与对照图，在命令后增加：

```bash
  --save_vis \
  --vis_dir ./output-vis/jmp_evaluation
```

输出为 `<vis_dir>/<scene>/vis.png` 和 `comparison.png`；前者是纯预测视差彩色图，后者
为左图、预测、参考及绝对误差四联图。

比较多个训练权重：

```bash
python tools/compare_tradition_checkpoints.py \
  --checkpoint official=./checkpoints/LiteAnyStereo.pth \
  --checkpoint epoch05=./runs/jmp_lf6020_las1/epoch_005.pth \
  --checkpoint epoch20=./runs/jmp_lf6020_las1/epoch_020.pth \
  --data_root ../tradition_stereo/datasets/FDJYP-3 \
  --image_root ./data/datasets/JMP-LF6020-ETH3D \
  --output_dir ./runs/jmp_lf6020_las1/tradition_comparison
```

默认命令复现本仓库原策略：固定排除 4 个场景并对每个 checkpoint 独立过滤 EPE 大于
20 px 的场景。由于这可能让不同模型保留不同场景，正式模型对比应额外加
`--no_epe_filter`，报告固定场景集合结果。LiteAnyStereo 的训练流程也会在每轮直接输出
EPE、D1、Bad1、Bad2 和 Bad3，并以 tradition EPE 选择 `best.pth`。

## 3. 整理建议

后续建议把裁剪逻辑抽成公共函数，例如：

```text
utils/crop_utils.py
```

统一提供：

```python
FIXED_CROP = (234, 1052, 126, 638)
crop_to_gt(disparity)
adjust_q_for_fixed_crop(Q)
```

这样 `save_old_disp.py`、`save_model_dis.py`、`cal_metric_ply.py`、`cal_metric_ply_rknn.py` 都使用同一套裁剪和 Q 偏移，避免不同脚本之间产生对齐误差。
