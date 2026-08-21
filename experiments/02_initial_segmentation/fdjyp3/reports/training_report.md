# JMP 工件 BiSeNetV2 伪标签训练复现记录

> 2026-08-19 已使用 187 份 ISAT 人工标注重新训练。最新结果见 [人工标注训练报告](../../../03_manual_segmentation/fdjyp3/reports/training_report.md)。本文继续保留作为早期伪标签基线审计。

本文记录 2026-08-18 在现有 JMP-LF6020 数据上进行工件二分类分割试训的全部关键操作。目标是先得到一个可接入 LiteAnyStereo 的基线模型，并保留可复查的数据、参数、命令、日志和模型产物。

> 重要限制：本次没有人工工件分割标注。训练标签由图像亮度经过 Otsu 和 GrabCut 自动生成，只是弱监督伪标签，不能当作真实分割 GT。验证指标衡量模型拟合伪标签的能力，不代表真实人工标注精度。

## 1. 路径和版本

工作区：

```text
/home/uestc/mount_2T/uestc/lnrio/JieTai/projects
```

相关目录：

```text
LiteAnyStereo/
  data/datasets/JMP-LF6020-ETH3D/          # 原始整理后的双目数据，只读使用
  data/datasets/JMP-workpiece-seg-pseudo-v1/ # 本次生成的分割伪标签数据

bisenetv2-tensorflow/
  config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml
  tools/jmp_workpiece/prepare_pseudo_dataset.py
  tools/jmp_workpiece/train_bisenetv2_jmp.py
  tools/jmp_workpiece/export_bisenetv2_jmp.py
  runs/jmp_workpiece/bisenetv2_pseudo_smoke/       # smoke 产物已清理，命令和结论保留在本文
  runs/jmp_workpiece/bisenetv2_pseudo_v1_e5/       # 首次全量失败运行，产物已清理
  runs/jmp_workpiece/bisenetv2_pseudo_v1_e5_bnfix/ # 修复后的正式五轮基线
```

上游源码当前提交：

```text
5407554
```

实际训练环境：

```text
Python       3.11（dsrl_pi0 环境）
TensorFlow   2.19.0，使用 tensorflow.compat.v1
tf_keras     2.19.0
GPU          NVIDIA GeForce RTX 4090，24 GB
训练可见显存 约 6.9 GB（同时存在其他 GPU 进程）
```

TensorFlow 2.19 必须在导入前设置 `TF_USE_LEGACY_KERAS=1`，否则原项目使用的 `tf.layers.batch_normalization` 在 Keras 3 下不可用。

## 2. 数据审计

原始清单：

```text
../LiteAnyStereo/data/datasets/JMP-LF6020-ETH3D/manifest.csv
SHA256: 2b1e7df91df8a7ef6d00840fc9e6973cbb7bd039b055eb9a56cf7aa7478628de
```

清单共 266 场：

| 场景 | 原划分 | 数量 | 本次用途 |
| --- | --- | ---: | --- |
| `FDJYP-0` | train | 82 | 训练 |
| `FDJYP-2` | train | 105 | 训练 |
| `FDJYP-3` | val | 73 | 验证 |
| `DE0548` | train | 6 | 排除 |

最终训练/验证数量为 `187/73`。`DE0548` 六张图包含圆球、标尺、圆环等完全不同的近距离场景，与黑色箱体中的金属工件域差异过大，试验基线中排除。

原数据中的 `mask0nocc.png` 是点云伪视差的矩形有效区域，不沿工件轮廓，不能用于语义分割训练。原始 `disp0GT.pfm` 也同时包含工件、背景箱体和夹具，不能直接视为主体标签。

## 3. 伪标签方案

对每张已校正左图独立执行以下确定性流程：

1. 从 `720×1280` 等比例缩放为 `288×512`；
2. 转灰度并做 `7×7` 高斯模糊；
3. Otsu 自动阈值获得亮前景种子；
4. `7×7` 闭运算连接亮区域；
5. 膨胀区域设为 GrabCut 可能前景，腐蚀区域设为确定前景；
6. 灰度低于 20 且不属于确定前景的区域设为确定背景；
7. GrabCut 迭代 4 次；
8. `7×7` 闭运算，删除面积小于 500 像素的连通域；
9. 输出 `0=背景，255=工件` 的 PNG 掩码。

该方法在暗背景、亮金属工件上能生成多数主体轮廓，但会有三类系统误差：

- 暗色、氧化和弱反光工件区域漏标；
- 与主体相连的亮夹具可能被并入前景；
- 强反光造成局部孔洞或轮廓收缩。

## 4. 生成伪标签数据

在 BiSeNetV2 仓库根目录运行：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow

/usr/bin/python tools/jmp_workpiece/prepare_pseudo_dataset.py \
  --manifest ../LiteAnyStereo/data/datasets/JMP-LF6020-ETH3D/manifest.csv \
  --output_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-pseudo-v1 \
  --width 288 \
  --height 512 \
  --source_prefix fdjyp_ \
  --seed 20260818 \
  --grabcut_iters 4 \
  --min_component_area 500 \
  --qa_samples 24
```

生成结果：

```text
train: 187
val:    73
前景比例: min 0.1054, median 0.3742, mean 0.4281, max 0.8970
Otsu 阈值: min 57, median 95, mean 97.1154, max 150
```

首次执行因外部执行通道时限被中断，未完成目录被移动到 `/tmp/JMP-workpiece-seg-pseudo-v1-incomplete-20260818`，随后以相同参数重新完整生成。该临时目录不参与训练；正式数据只使用上述版本化输出目录。

主要产物：

```text
JMP-workpiece-seg-pseudo-v1/
  images/{train,val}/*.png
  masks/{train,val}/*.png
  index/{train,val}.csv
  qa/{train,val}_contact_sheet.jpg
  metadata.json
```

`metadata.json` 记录源清单哈希、排除样本、生成参数、计数和统计量。两张联系表分别均匀抽取 24 场，按“原图 / 红色掩码覆盖 / 二值标签”排列，是复查标签质量的首要入口。

## 5. TensorFlow 2.19 兼容修改

原项目针对 TensorFlow 1.15。为了在现有环境运行，做了以下最小修改：

1. `bisenet_model/cnn_basenet.py` 和 `bisenet_model/bisenet_v2.py` 改用 `tensorflow.compat.v1` 并关闭 v2 behavior；
2. 卷积核 shape 中的 `in_channel / split` 改为整数除法 `in_channel // split`，避免 Python 3 产生浮点维度；
3. 使用已安装的 `tf_keras==2.19.0` 提供 legacy `tf.layers`；
4. 新训练脚本使用 NumPy/OpenCV 读取数据，不依赖原 Cityscapes TFRecord 管线。

图构建和前向冒烟检查结果：

```text
输入: [1, 64, 64, 3]
输出: [1, 64, 64]
全局变量数: 292
随机初始化测试 loss: 4.457919
```

## 6. 小样本冒烟训练

执行命令：

```bash
TF_USE_LEGACY_KERAS=1 \
MPLCONFIGDIR=/tmp/mpl-bisenet-jmp \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
  tools/jmp_workpiece/train_bisenetv2_jmp.py \
  --dataset_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-pseudo-v1 \
  --config ./config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml \
  --output_dir ./runs/jmp_workpiece/bisenetv2_pseudo_smoke \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --end_learning_rate 0.00001 \
  --seed 20260818 \
  --device cuda \
  --max_train_samples 8 \
  --max_val_samples 4 \
  --preview_samples 4
```

结果：

```text
train_loss       4.15280
val_loss         4.55566
foreground IoU  0.18113
mean IoU        0.17429
foreground Dice 0.30670
```

该结果仅证明图构建、GPU 反向传播、验证、checkpoint 保存和预测预览均可运行。命令和指标保留在本文中，约 174 MB 的 smoke 检查点与预测已于 2026-08-20 清理。

## 7. 第一次全量运行及 BatchNorm 问题

第一次全量运行写入：

```text
./runs/jmp_workpiece/bisenetv2_pseudo_v1_e5
```

训练损失从 `2.20365` 降至 `1.33903`，但五轮 `val_loss` 全部为 `NaN`，预测全部为背景。checkpoint 检查发现以下两个 moving variance 全为 NaN：

```text
BiseNetV2/semantic_branch/stage_5/ce_block_3_repeat_1/context_embedding_block/bn/moving_variance
BiseNetV2/semantic_branch/stage_5/ce_block_3_repeat_1/context_embedding_block/conv_block_1/bn/moving_variance
```

根因是训练集 187 张、batch size 2 时最后一个 batch 只有 1 张。context embedding 先做全局池化，特征空间为 `1×1`；单样本 BatchNorm 的 moving variance 更新产生 NaN。

修复方式：

1. 训练阶段丢弃不足一个完整 batch 的尾批；
2. 每轮先确定性打乱，所以不同 epoch 被丢弃的不是固定样本；
3. 脚本禁止 `batch_size < 2`；
4. 训练过程中发现非有限 loss 时立即报错。

失败原因、命令和指标保留在本文中；该目录不能用于部署或导出，运行产物已于 2026-08-20 清理。

## 8. 修复后的全量五轮基线训练

执行命令：

```bash
TF_USE_LEGACY_KERAS=1 \
MPLCONFIGDIR=/tmp/mpl-bisenet-jmp \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
  tools/jmp_workpiece/train_bisenetv2_jmp.py \
  --dataset_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-pseudo-v1 \
  --config ./config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml \
  --output_dir ./runs/jmp_workpiece/bisenetv2_pseudo_v1_e5_bnfix \
  --epochs 5 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --end_learning_rate 0.00001 \
  --seed 20260818 \
  --device cuda \
  --preview_samples 12
```

选择 batch size 2 是因为运行时 GPU 还有两个其他任务，各占约 7.35 GB，TensorFlow 本次实际可见空闲显存约 6.9 GB。

每轮从 187 张训练图中使用 186 张，合计 93 个完整 batch。由于每轮重新打乱，五轮训练会覆盖全部训练样本。

训练结果：

| epoch | train loss | val loss | foreground IoU | mean IoU | foreground Dice |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.20029 | 5.00683 | 0.80216 | 0.73423 | 0.89022 |
| 2 | 1.71163 | 3.87405 | 0.87812 | 0.79130 | 0.93510 |
| 3 | 1.51685 | 2.09974 | 0.87688 | 0.78797 | 0.93440 |
| 4 | 1.39985 | 1.91091 | 0.86447 | 0.75679 | 0.92731 |
| 5 | **1.34586** | **1.82044** | **0.88297** | **0.79730** | **0.93785** |

第 5 轮按前景 IoU 选为最佳 checkpoint；像素准确率为 `0.90919`。运行目录大小约 186 MB，伪标签数据目录约 61 MB。

对最佳 checkpoint 的 648 个数值张量逐一执行有限值检查，未发现 NaN 或 Inf。

完整运行目录会包含：

```text
best.ckpt.*
latest.ckpt.*
best_metrics.json
metrics.jsonl
run_config.json
train.log
predictions/*.png
val_predictions_contact_sheet.jpg
```

`best.ckpt` 按验证集前景 IoU 选择。`run_config.json` 记录数据 metadata 哈希、模型配置哈希、环境版本、命令和随机种子。

## 9. 冻结模型

训练完成后运行：

```bash
TF_USE_LEGACY_KERAS=1 \
MPLCONFIGDIR=/tmp/mpl-bisenet-jmp \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
  tools/jmp_workpiece/export_bisenetv2_jmp.py \
  --checkpoint ./runs/jmp_workpiece/bisenetv2_pseudo_v1_e5_bnfix/best.ckpt \
  --config ./config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml \
  --output_pb ./runs/jmp_workpiece/bisenetv2_pseudo_v1_e5_bnfix/bisenetv2_jmp_workpiece_frozen.pb \
  --width 288 \
  --height 512 \
  --device cpu
```

冻结图接口：

```text
input_tensor:0       float32 [1,512,288,3]
final_probability:0  float32 [1,512,288,2]
final_output:0       int64   [512,288]
```

预处理：BGR 转 RGB，除以 255，再执行 `(x - 0.5) / 0.5`。

TensorFlow 2.19 中旧的内部 `tensorflow.python.framework.graph_util.convert_variables_to_constants` 已不存在，导出脚本使用 `tf.compat.v1.graph_util.convert_variables_to_constants`。

冻结产物：

```text
文件大小: 10,119,892 bytes（约 9.7 MiB）
SHA256: ce552245a7c0bea535f8da1478d6859a15080bab115ab796337df5b45592db0a
```

使用一张验证图加载 frozen PB 实测：

```text
input shape:       [1,512,288,3]
probability shape: [1,512,288,2]
mask shape:        [512,288]
输出类别:          [0,1]
概率全部有限:      true
概率和最大误差:    1.1920929e-07
```

## 10. 如何复查与正确解释结果

复查顺序：

1. 先看伪标签 `qa/train_contact_sheet.jpg` 和 `qa/val_contact_sheet.jpg`；
2. 再看训练输出 `val_predictions_contact_sheet.jpg`；
3. 对照 `best_metrics.json` 和逐轮 `metrics.jsonl`；
4. 查看 `train.log` 是否有中断、OOM 或数值异常；
5. 检查 `run_config.json` 中的数据和配置 SHA256 是否与本文一致。

本次 IoU/Dice 的参照物是自动生成的伪标签。模型即使取得很高指标，也可能只是学会亮度分割。要得到可用于严肃测量的主体掩码，下一阶段应人工修订一批验证图，至少形成独立的人工测试集，再判断是否需要继续训练或更换标签策略。

## 11. 人工修订掩码操作规范

本节是下一阶段的执行说明，当前基线实验尚未使用人工修订标签。目标是把伪标签作为初始轮廓，由标注人员补全漏标区域、删除误标背景，并得到可独立衡量真实性能的人工验证集。

### 11.1 标注范围与数据隔离

优先修订 `FDJYP-3` 的全部 73 张验证图。若时间有限，至少固定抽取 30～50 张并记录名单；未完成人工修订的图不能混入人工指标。人工验证图只能用于评估和选定最终阈值，不参与训练或微调。

如果后续需要提升模型，应另外从 `FDJYP-0`、`FDJYP-2` 训练集选择图像进行人工修订。禁止把人工验证集复制到训练集，否则会产生数据泄漏。

不要直接修改 `JMP-workpiece-seg-pseudo-v1`。先建立版本化副本，并同时保留自动伪标签：

```bash
set -euo pipefail
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/LiteAnyStereo/data/datasets

test ! -e JMP-workpiece-seg-manual-val-v1
cp -a JMP-workpiece-seg-pseudo-v1 JMP-workpiece-seg-manual-val-v1
mv JMP-workpiece-seg-manual-val-v1/masks \
   JMP-workpiece-seg-manual-val-v1/masks_pseudo
cp -a JMP-workpiece-seg-manual-val-v1/masks_pseudo \
      JMP-workpiece-seg-manual-val-v1/masks
mkdir -p JMP-workpiece-seg-manual-val-v1/annotations
```

后续只编辑新目录中的 `masks/val/*.png`。`masks_pseudo/val/*.png` 是修改前基线，必须保持只读。正式开始前，应将本节使用的目录名、标注人员和开始日期填入 `annotations/review.csv`。

### 11.2 前景定义

本任务只有两个类别：

| 像素值 | 类别 | 处理规则 |
| ---: | --- | --- |
| `0` | 背景 | 箱体、夹具/支架、阴影、孔内可见背景以及其他非测量目标 |
| `255` | 工件 | 当前待测工件所有可见实体表面，包括暗色、氧化、反光不足和高光区域 |

统一按以下边界原则修订：

1. 只标当前需要计算视差的工件；即使夹具与工件相连，也要沿真实接触边界分开；
2. 工件上的颜色、亮度、锈蚀和高光变化不改变类别，均为前景；
3. 通孔中能看到箱体或夹具的部分标为背景，凹槽内仍属于工件表面的部分标为前景；
4. 阴影不是工件，不能因为与轮廓相连而标入；
5. 只标图像中实际可见的像素，不推测遮挡区域；
6. 边界不确定时以可见实体边缘为准，并在复核表中记录有争议的文件名。

如果实际业务希望把夹具或多个零件一起测量，应先改变类别定义并创建 `manual-v2`，不能在同一个版本中混用两套口径。

### 11.3 推荐标注方式

多人协作或需要审核流时优先使用 CVAT；几十张以内也可用 GIMP、Krita 等支持图层和画笔的图像编辑器。无论使用什么软件，操作逻辑相同：

1. 打开 `images/val/<name>.png` 作为底图；
2. 导入对应的 `masks_pseudo/val/<name>.png`，以红色、约 40% 透明度叠加；
3. 用前景画笔补齐伪标签漏掉的暗色或弱反光工件；
4. 用背景画笔删除箱体、夹具、阴影和孔内背景；
5. 放大检查外轮廓、孔洞、工件与夹具接触处；
6. 关闭叠加后再看一次纯二值掩码，排除孤立点和内部误洞；
7. 覆盖保存到 `masks/val/<name>.png`，文件名必须与图像完全一致。

当前模型输入就是 `288×512`，建议直接在这一分辨率修订，以避免再次缩放造成边界偏移。原始 `720×1280` 左图可并排作为细节参考，但不要直接覆盖准备后的训练图。若确实在原始分辨率标注，缩放掩码时只能使用最近邻插值，禁止双线性或双三次插值。

最终文件必须满足：

- PNG 格式、单通道、`uint8`；
- 尺寸严格为宽 288、高 512；
- 只含 `0` 和 `255`，不能有灰度过渡、调色板类别编号或透明通道；
- 不得为空掩码或整幅前景，除非复核人员确认图像确实如此；
- 图像、掩码和 CSV 中的样本名一一对应。

如果工具导出的是多边形 JSON，应把类别 `workpiece` 映射为 255、背景映射为 0，再栅格化为上述 PNG；JSON 原件放在 `annotations/source/` 中保留，不应只留下转换后的图片。

### 11.4 修订记录

在 `annotations/review.csv` 中逐张记录，推荐字段如下：

```csv
name,annotator,reviewer,annotated_at,status,source_mask_version,notes
fdjyp_3_1_202506281603_0001,张三,李四,2026-08-19,approved,otsu_grabcut_v1,删除夹具并补齐暗色边缘
```

`status` 只使用 `draft`、`needs_fix`、`approved`。只有 `approved` 样本可以进入人工评估。每次修改标注规则或批量返工都应创建新的数据版本，而不是覆盖已用于出报告的版本。

### 11.5 自动检查

全部修订完成后，在数据集目录运行以下只读检查：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/LiteAnyStereo/data/datasets/JMP-workpiece-seg-manual-val-v1

/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python - <<'PY'
from pathlib import Path
import cv2
import numpy as np

root = Path(".")
images = {p.name for p in (root / "images/val").glob("*.png")}
masks = {p.name for p in (root / "masks/val").glob("*.png")}
assert images == masks, f"文件名不匹配: 缺掩码={images-masks}, 多掩码={masks-images}"

for name in sorted(images):
    mask = cv2.imread(str(root / "masks/val" / name), cv2.IMREAD_UNCHANGED)
    assert mask is not None, name
    assert mask.ndim == 2 and mask.shape == (512, 288), (name, mask.shape)
    assert mask.dtype == np.uint8, (name, mask.dtype)
    values = set(np.unique(mask).tolist())
    assert values <= {0, 255}, (name, values)
    fraction = float(np.mean(mask == 255))
    assert 0.001 < fraction < 0.999, (name, fraction)

print(f"通过: {len(images)} 张掩码，尺寸/类型/取值/文件名均符合要求")
PY
```

该脚本只能发现格式和极端前景比例问题，不能判断轮廓是否正确。还必须对全部修订结果生成半透明叠加图并逐张目视检查；至少随机抽取 20% 由第二人复核，工件与夹具接触边界复杂的图应全部复核。

### 11.6 重新评估与后续训练

人工验证集准备完成后，先保持当前 `best.ckpt` 不变，在这些人工标签上重新计算 IoU、Dice、像素准确率，并同时报告：人工验证样本数、数据版本、审核通过数和模型 SHA256。此结果才可用于判断分割是否满足 LiteAnyStereo 前处理要求。

建议把人工结果与本次伪标签结果并列，不要覆盖旧指标。重点观察前景召回率和主体边缘：漏掉的主体像素会直接删除有效视差，夹具误入前景则会污染主体视差统计。

如果人工评估不达标，再单独修订训练集中的 50～100 张代表性图像进行微调。训练集应覆盖亮/暗工件、氧化、高反光、孔洞和夹具接触等困难情况；人工验证集继续冻结，微调过程中不得使用其标签。每次训练仍创建新的 `runs/jmp_workpiece/<run_name>` 并保存配置、日志、指标和标签版本哈希。

## 12. 接入 LiteAnyStereo 时的使用原则

分割模型用于先定位主体，但 LiteAnyStereo 仍应接收未涂黑的原始校正 RGB：

```text
已校正左右图 -> BiSeNetV2 掩码 -> 可选共享 ROI
已校正左右原图/ROI -> LiteAnyStereo -> 左图视差 -> 左掩码过滤主体视差
```

不要直接把背景置零后再做双目匹配，否则人工边界和纹理丢失可能降低主体边缘视差。左右 ROI 必须使用相同裁剪坐标，并在左侧保留至少一个 `max_disp` 的匹配搜索余量。
