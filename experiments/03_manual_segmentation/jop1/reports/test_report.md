# Jop1 BiSeNetV2 + LiteAnyStereo 测试报告

**测试日期：** 2026-08-20  
**测试数据：** Jop1 全部 9 组双目图像  
**分割模型：** FDJYP-0/2 人工标注训练的冻结 BiSeNetV2  
**双目模型：** LiteAnyStereo LAS1 官方权重  
**结论：** 工程链路运行正常，9/9 张最终掩码均为一个前景连通域；但 Jop1 上没有复现 FDJYP-3 的主体区域数值优势，当前模型存在明显跨域风险，不能在没有 Jop1 人工分割真值的情况下直接作为生产主体提取器。

## 1. 测试目的与流程

本次测试沿用已冻结的 FDJYP-3 方案，不根据 Jop1 指标修改阈值或模型：

```text
Jop1 原始左右图
  -> 逆时针旋转 90°、相机标定矫正
  -> 原始校正左右 RGB 输入 LAS1，得到完整视差

校正左图
  -> 冻结 BiSeNetV2 前景概率
  -> 双线性恢复到 720×1280
  -> 0.5 阈值、3 px 闭运算
  -> 保留最大 8 连通前景、删除孤岛
  -> 用 LAS 原始视差连续性判断内部暗区是否补回
  -> 后置筛选完整 LAS 视差，生成主体视差和主体点云
```

BiSeNetV2 不修改或涂黑 LAS 的输入图像。主体掩码只作用在 LAS 完成推理之后，掩码内视差值最大改变量为 `0.0 px`。补洞过程不读取 PLY，也不使用分割 GT。

Jop1 的预处理图和 LAS 浮点视差来自此前已经验证的 `experiments/01_stereo_comparison/jop1/results/final_9/`；模型权重和输入完全相同，本次复用保存结果，避免重复前向造成新的变量。

## 2. 模型与数据追踪

```text
BiSeNetV2 frozen PB SHA256:
b1f34a8caf2b4cab9be7a997a979dd1bb058152f771aa39c25bd37eef2bc4bee

LiteAnyStereo LAS1 checkpoint SHA256:
ee0c3a0dc1d4b49cbd67edf00079b9993c0fa21f6c19a0eb812fa32f7ec1b9b1

LAS1 来源 summary.json SHA256:
fdf713209a23102226d58af24305bd312b9f57c2d785ab871b6cce20de8a72ae
```

9 张校正左图均为宽 720、高 1280，与模型训练图的宽高比一致；BiSeNetV2 实际输入为宽 288、高 512。

## 3. 分割与连通域结果

| 项目 | 结果 |
| --- | ---: |
| 场景数 | 9 |
| 原始前景覆盖率宏平均 | 63.84% |
| 修订后前景覆盖率宏平均 | 63.42% |
| 原始预测平均最大类别置信度 | 0.97074 |
| 闭运算后前景连通域总数 | 30 |
| 删除孤岛像素 | 44,720 |
| 检查内部候选暗区 | 12 |
| 补回内部暗区 | 12 个、9,137 px |
| 最终恰好一个前景连通域 | **9/9** |

高平均置信度只说明模型输出确定，并不能证明跨域预测正确。目视可见模型通常能覆盖主要金属表面，但在标尺、反光区域、白色台面、复杂边缘和圆孔附近存在误纳或误删风险。

![Jop1 全 9 场分割与主体视差总览](../results/result/overview.jpg)

每个场景的六宫格依次包含：校正左图、原始分割叠加、单连通域修订叠加、LAS 完整视差、LAS 主体视差和稀疏 PLY 投影。

## 4. 稀疏 PLY 一致性结果

Jop1 没有人工主体分割 GT。压缩包附带的 PLY 投影不是稠密人工视差真值，也不等价于工件语义掩码，因此以下指标只能用于诊断。

| 评价区域 | PLY 投影点宏平均 | EPE↓ | D1↓ | Bad3↓ |
| --- | ---: | ---: | ---: | ---: |
| LAS 完整区域 | 338,719 | **12.508 px** | **85.52%** | **88.67%** |
| BiSeNetV2 主体区域 | 280,618 | 13.703 px | 87.94% | 90.67% |

主体掩码平均保留 `82.83%` 的 PLY 投影点。主体 EPE 比完整区域高 `1.195 px`，相对差约 `9.56%`。这不是 LAS 本身退化，因为主体内预测值未改变；两行使用不同像素集合，结果表明当前分割倾向于保留 PLY 一致性较差的困难区域，同时删除了一些误差较低区域。

逐场结果：

| 场景 | 前景覆盖率 | PLY 点保留率 | 全区域 EPE | 主体 EPE | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `0104` | 75.03% | 98.41% | 16.594 | 16.494 | 基本持平 |
| `0105` | 49.95% | 61.16% | 12.956 | 17.726 | 明显变差，主要风险场景 |
| `0106` | 65.98% | 98.16% | 12.751 | 12.689 | 基本持平 |
| `0107` | 65.89% | 67.08% | 11.575 | 14.344 | 明显变差，主要风险场景 |
| `0108` | 39.06% | 56.44% | 11.148 | 14.255 | 保留率最低，主要风险场景 |
| `0109` | 53.63% | 82.99% | 15.013 | 15.396 | 小幅变差 |
| `0110` | 58.66% | 84.67% | 13.032 | 13.029 | 基本持平 |
| `0111` | 78.94% | 96.56% | 11.998 | 11.891 | 基本持平 |
| `0112` | 83.64% | 100.00% | 7.506 | 7.506 | 相同 |

9 场中 4 场主体 EPE略低、4 场更高、1 场相同；差异主要由 `0105/0107/0108` 决定。由于 PLY 覆盖并不是主体分割真值，不能仅凭该表判断每个掩码像素的语义对错，但它足以否定“当前模型在 Jop1 上已经稳定改善输出”的说法。

## 5. 结论与建议

本次可以确认：

1. 冻结 BiSeNetV2、单连通域修订、LAS 后置筛选和主体点云链路在 Jop1 9 场全部运行成功；
2. 单连通域约束满足 9/9，且不会修改主体内部 LAS 视差；
3. Jop1 与训练数据在工件形状、标尺、反光、背景和孔洞结构上存在域差异；
4. 当前 Jop1 数值和目视结果不足以支持直接部署该分割模型。

Jop1 只有 9 张左图，最有价值的下一步不是继续盲调阈值，而是人工标注全部 9 张主体掩码。得到真值后可直接计算 IoU、主体召回率、边界 F-score、误删有效视差比例和误纳背景比例；再决定是只调整后处理，还是把 Jop1 标注加入训练/微调。若这些 9 张参与调参，它们应作为验证/开发集，最终测试应使用新采集且未参与调参的数据。

## 6. 产物与复现

结果目录：

```text
results/result/
  bisenet_raw/                 # 288×512 原始预测、概率、叠加和联系表
  scenes/<scene>/
    raw_mask.png
    foreground_mask.png       # 720×1280 单连通域掩码
    disp_subject.npy          # 掩码外为 NaN
    disparity_subject_color.png
    cloud_subject.ply
    comparison.jpg
  metrics/per_scene.csv
  metrics/summary.json
  metrics/hole_decisions.json
  overview.jpg
  README.md
```

BiSeNetV2 推理：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai

mkdir -p experiments/03_manual_segmentation/jop1/results/result/inputs
for source_path in experiments/01_stereo_comparison/jop1/results/final_9/preprocessed/*/left.png; do
  scene_dir=${source_path%/left.png}
  scene_name=${scene_dir##*/}
  ln -s "$(realpath "$source_path")" \
    "experiments/03_manual_segmentation/jop1/results/result/inputs/${scene_name}.png"
done

cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow

TF_USE_LEGACY_KERAS=1 \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
tools/jmp_workpiece/predict_bisenetv2_jmp.py \
  --model_pb runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --input_dir ../../experiments/03_manual_segmentation/jop1/results/result/inputs \
  --output_dir ../../experiments/03_manual_segmentation/jop1/results/result/bisenet_raw \
  --input_glob '*.png' --width 288 --height 512 --device cpu \
  --uncertain_threshold 0.75 --contact_sheet_samples 9 --save_probabilities
```

单连通域修订、主体视差和点云：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai

/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python \
experiments/03_manual_segmentation/jop1/scripts/run_test.py \
  --probability-dir experiments/03_manual_segmentation/jop1/results/result/bisenet_raw/probabilities \
  --bisenet-metadata experiments/03_manual_segmentation/jop1/results/result/bisenet_raw/metadata.json \
  --output-root experiments/03_manual_segmentation/jop1/results/result
```

本次新增评价脚本的 3 项单元测试通过。最终文件校验：

```text
metrics/summary.json:
89e93155d011f635807c653813ef202e355f3e0a15c022031cf4e8c4a47b8370

metrics/per_scene.csv:
f124dfbb4a9a7dfd93e2de3d383c4a4510587c4a58689ee3a3fb3990056f491a

overview.jpg:
f60e3c9e97019782bccde524d00b9013138a43409da36d9eeccdf534eb004901
```
