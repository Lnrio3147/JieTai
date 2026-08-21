# JMP 工件 BiSeNetV2 人工 ISAT 标注训练复现记录

本文记录 2026-08-19 使用 `JMP-workpiece-seg-pseudo-v1/images/train` 中新增的 ISAT 人工多边形标注，重新训练 BiSeNetV2 的全部关键操作、数据划分、命令、异常、指标和模型哈希。

本次模型只使用人工标签训练和验证，没有加载 2026-08-18 的伪标签 checkpoint。原伪标签数据、人工 JSON 和以前的训练产物均未被覆盖。

## 1. 输入数据审计

输入目录：

```text
/home/uestc/mount_2T/uestc/lnrio/JieTai/projects/LiteAnyStereo/data/datasets/
  JMP-workpiece-seg-pseudo-v1/images/train/
```

审计结果：

| 项目 | 结果 |
| --- | ---: |
| PNG 图像 | 187 |
| 同名 ISAT JSON | 187 |
| 缺失图像或 JSON | 0 |
| 标注对象 | 188 |
| 类别 | 全部为 `jinshu` |
| 每图对象数 | 186 张为 1 个，1 张为 2 个 |
| 每个多边形点数 | 7～78 |
| 空标注 | 0 |
| 非法多边形、尺寸或文件名 | 0 |
| 图像/标注尺寸 | 宽 288，高 512 |

`isat.yaml` 中类别定义为 `__background__` 和 `jinshu`。训练时统一映射为：

```text
0   = background
255 = workpiece（由 jinshu 多边形栅格化）
```

## 2. 人工训练数据的划分原则

不使用随机逐帧划分，因为相邻帧高度相似，随机拆分容易让近邻图像同时进入训练和验证，造成过于乐观的指标。本次按完整采集子序列留出验证集：

```text
验证组: fdjyp_0_2（18 张）+ fdjyp_2_3（24 张）= 42 张
训练组: 其他 8 个采集组                    = 145 张
```

训练/验证总计 187 张，全部使用人工 ISAT 标注。验证组中的图像和标签从未参与梯度更新。

该验证集仍来自与训练集相同的设备和采集环境，只能衡量当前数据域内的泛化。`FDJYP-3` 的 73 张原验证图没有人工标注，因此未混入本次人工验证指标。

## 3. ISAT JSON 转换脚本

新增脚本：

```text
tools/jmp_workpiece/prepare_isat_manual_dataset.py
```

转换规则：

1. 要求每张 PNG 存在同名 JSON；
2. 检查 JSON 中的图像名、宽高、对象和多边形点；
3. 仅接受类别 `jinshu`；
4. 将浮点坐标裁剪到图像范围，四舍五入为整数；
5. 使用 OpenCV `fillPoly` 将同图的一个或多个对象合并为前景；
6. 输出只含 `0/255` 的单通道 `uint8` PNG；
7. 复制原图和 JSON 到新版本目录，保存每个输入、标注和掩码的 SHA256；
8. 生成训练/验证 CSV、元数据以及抽样联系表；
9. 输出目录已存在时拒绝运行，防止覆盖旧版本。

语法检查：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow

/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python -m py_compile \
  tools/jmp_workpiece/prepare_isat_manual_dataset.py \
  tools/jmp_workpiece/train_bisenetv2_jmp.py
```

## 4. 生成版本化人工数据集

执行命令：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow

/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
  tools/jmp_workpiece/prepare_isat_manual_dataset.py \
  --source_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-pseudo-v1/images/train \
  --output_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-manual-isat-v1 \
  --foreground_category jinshu \
  --val_groups fdjyp_0_2 fdjyp_2_3 \
  --width 288 \
  --height 512 \
  --qa_samples 24
```

输出结构：

```text
JMP-workpiece-seg-manual-isat-v1/
  images/{train,val}/
  masks/{train,val}/
  annotations/{train,val}/
  index/train.csv
  index/val.csv
  index/annotations.sha256
  qa/train_contact_sheet.jpg
  qa/val_contact_sheet.jpg
  metadata.json
```

结果：

```text
train: 145
val:    42
对象数: 188
数据目录大小: 约 42 MB
前景比例: min 0.14199, median 0.41690, mean 0.42964, max 0.95188
```

数据追踪哈希：

```text
metadata.json SHA256:
c056fb7c8c8218d26d21d46f6e2d5a10af1c2e6b58ae84abaa955fe1d0963d38

index/annotations.sha256 SHA256:
ca8fe60025b2d1510e17e4dcccb08b7b82ac0c2b8ed7741428b4b6340cc6bc8a

index/train.csv SHA256:
a08bd3b4aebd032e0a146c42306d4b723bbd120faa153d36cf2da85b1d55405b

index/val.csv SHA256:
d5221914c7551a3d56a63a20ba62644a1442a5f5aa98e131b341ae14ca8222ab
```

## 5. 数据质检

对 145/42 张训练和验证样本逐张执行以下自动检查：

- 图像必须是 `(512, 288, 3)`；
- 掩码必须是 `(512, 288)` 单通道 `uint8`；
- 掩码像素值只能是 `0/255`；
- CSV 中图像和掩码均可读取；
- 训练和验证数量必须分别为 145 和 42。

全部检查通过。随后目视检查了：

```text
qa/train_contact_sheet.jpg
qa/val_contact_sheet.jpg
```

联系表每项从左到右为原图、红色人工标签叠加、纯二值标签。抽样结果未发现坐标颠倒、尺寸错位、空掩码或明显栅格化异常。最大前景比例样本 `fdjyp_2_3_202506281544_0068` 约为 0.95188；复查原图后确认是工件近距离占据画面绝大部分，并非转换错误。

## 6. 训练环境和参数

```text
Python:       3.11.11
TensorFlow:   2.19.0，tensorflow.compat.v1
tf_keras:     2.19.0
GPU:          NVIDIA GeForce RTX 4090 24 GB
本进程显存:   约 6.8 GB
模型输入:     1/2 × 512 × 288 × 3，NHWC
类别数:       2
初始化:       随机初始化，不加载伪标签模型
优化器:       Adam
学习率:       0.001 -> 0.00001，polynomial decay，power 0.9
batch size:   2
epoch:        20
随机种子:     20260819
数据增强:     50% 水平翻转、轻微亮度/对比度变化
最佳模型:     按人工验证集 foreground IoU 选择
```

训练集为奇数 145 张。为避免 context embedding 的 `1×1` 特征在最后单样本 batch 上污染 BatchNorm，每轮丢弃最后一个不完整 batch，实际每轮参与 144 张；每轮重新随机打乱，因此不固定遗漏同一张。

TensorFlow 2.19 运行原 TF1 模型时必须在导入前设置 `TF_USE_LEGACY_KERAS=1`。

## 7. 冒烟训练

先用 8/4 张图验证数据读取、前后向、评估、checkpoint 和预测图保存：

```bash
TF_USE_LEGACY_KERAS=1 MPLCONFIGDIR=/tmp/mpl-bisenet-jmp \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
tools/jmp_workpiece/train_bisenetv2_jmp.py \
  --dataset_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-manual-isat-v1 \
  --config ./config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml \
  --output_dir ./runs/jmp_workpiece/bisenetv2_manual_isat_v1_smoke \
  --epochs 1 --batch_size 2 \
  --learning_rate 0.001 --end_learning_rate 0.00001 \
  --seed 20260819 --device cuda \
  --max_train_samples 8 --max_val_samples 4 --preview_samples 4
```

结果：

```text
train loss:     4.17077
val loss:       4.38880
foreground IoU: 0.33176
```

该任务的目的只是验证训练链路，不能把指标当作模型效果。命令和指标保留在本文中，约 173 MB 的 smoke 检查点与预测已于 2026-08-20 清理。

## 8. 正式训练与一次外部中止

第一次正式运行写入：

```text
runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20/
```

进程在首轮指标写出前收到外部 `SIGTERM`，退出码为 143。`metrics.jsonl` 为空，未生成 checkpoint；日志中没有 Python 异常、OOM 或 NaN。执行审计保留在本文中，空的失败运行目录已于 2026-08-20 清理。

随后以相同数据和超参数重新运行，只更换输出目录，并使用 PTY 保持长任务执行通道：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow

TF_USE_LEGACY_KERAS=1 MPLCONFIGDIR=/tmp/mpl-bisenet-jmp \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
tools/jmp_workpiece/train_bisenetv2_jmp.py \
  --dataset_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-manual-isat-v1 \
  --config ./config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml \
  --output_dir ./runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1 \
  --epochs 20 \
  --batch_size 2 \
  --learning_rate 0.001 \
  --end_learning_rate 0.00001 \
  --seed 20260819 \
  --device cuda \
  --preview_samples 18
```

正式运行目录约 186 MB，包含 `run_config.json`、逐轮指标、日志、best/latest checkpoint、验证预测图和冻结 PB。

## 9. 完整训练指标

| epoch | train loss | val loss | foreground IoU | mIoU | Dice | pixel accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.43315 | 4.02890 | 0.69184 | 0.75463 | 0.81786 | 0.87050 |
| 2 | 1.98145 | 3.76519 | 0.30487 | 0.48950 | 0.46728 | 0.71488 |
| 3 | 1.76675 | 2.77422 | 0.83174 | 0.86315 | 0.90814 | 0.93068 |
| 4 | 1.56341 | 1.67447 | 0.82362 | 0.85741 | 0.90328 | 0.92786 |
| 5 | 1.43485 | 1.30234 | 0.89925 | 0.91656 | 0.94695 | 0.95841 |
| 6 | 1.27475 | 1.11077 | 0.91164 | 0.92669 | 0.95378 | 0.96361 |
| 7 | 1.23005 | 1.08177 | 0.93120 | 0.94269 | 0.96437 | 0.97172 |
| 8 | 1.06646 | 0.93340 | 0.94397 | 0.95277 | 0.97118 | 0.97668 |
| 9 | 1.02186 | 1.48830 | 0.65367 | 0.65519 | 0.79057 | 0.79168 |
| 10 | 1.04973 | 0.98489 | 0.94927 | 0.95723 | 0.97398 | 0.97892 |
| 11 | 0.93544 | 0.87742 | 0.93018 | 0.94188 | 0.96383 | 0.97132 |
| 12 | 0.86584 | 0.78516 | 0.96017 | 0.96631 | 0.97968 | 0.98345 |
| 13 | 0.84981 | 0.77297 | 0.95447 | 0.96191 | 0.97671 | 0.98134 |
| 14 | 0.81203 | 0.76715 | 0.94896 | 0.95735 | 0.97381 | 0.97907 |
| 15 | 0.85934 | 0.78045 | 0.96547 | 0.97074 | 0.98243 | 0.98564 |
| 16 | 0.78653 | 0.72433 | 0.96990 | 0.97454 | 0.98472 | 0.98754 |
| 17 | 0.73458 | 0.68921 | 0.96910 | 0.97389 | 0.98431 | 0.98722 |
| 18 | 0.69237 | 0.66862 | 0.97454 | 0.97850 | 0.98710 | 0.98950 |
| 19 | 0.68562 | 0.66663 | 0.97562 | 0.97940 | 0.98766 | 0.98994 |
| 20 | 0.69191 | 0.65302 | **0.97706** | **0.98061** | **0.98840** | **0.99054** |

第 20 轮是最佳 checkpoint。其混淆矩阵为：

```text
              预测背景   预测工件
真实背景       3639842      27890
真实工件         30681    2494739
```

第 2、9 轮出现验证波动，但之后恢复；全程 loss 有限。最终检查 `best.ckpt` 中 651 个数值张量，未发现 NaN 或 Inf。`val_predictions_contact_sheet.jpg` 中每项依次为原图、人工 GT、预测，抽样目视结果与指标一致。

## 10. 导出冻结模型

执行：

```bash
TF_USE_LEGACY_KERAS=1 MPLCONFIGDIR=/tmp/mpl-bisenet-jmp \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
tools/jmp_workpiece/export_bisenetv2_jmp.py \
  --checkpoint ./runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/best.ckpt \
  --config ./config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml \
  --output_pb ./runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --width 288 --height 512 --device cpu
```

接口：

```text
input_tensor:0       float32 [1,512,288,3]
final_probability:0  float32 [1,512,288,2]
final_output:0       int64   [512,288]
预处理: BGR -> RGB, /255, (x - 0.5) / 0.5
```

使用验证图 `fdjyp_0_2_202506261657_0011.png` 加载 PB 实测：

```text
输入:                  (1,512,288,3)
概率:                  (1,512,288,2)
掩码:                  (512,288)
输出类别:              [0,1]
概率全部有限:          true
概率和最大误差:        1.1920928955078125e-07
预测前景比例:          0.40245
```

产物哈希：

```text
frozen PB SHA256:
b1f34a8caf2b4cab9be7a997a979dd1bb058152f771aa39c25bd37eef2bc4bee

best.ckpt.index SHA256:
051c4426ea61dfc2499e4eef55e892096d11575597a4fea33b407e2469b6f986

best_metrics.json SHA256:
0d51a4e86bfa7647a40d49ff304f5b75d7102febc70fa831bd421f0ebece2752
```

冻结模型大小约 9.7 MiB。

## 11. 结果解释和下一步

本次 `0.97706` 前景 IoU 来自 42 张人工标注、按采集子序列隔离的验证图，可信度明显高于先前对伪标签计算的指标。但仍需注意：

1. 训练和验证来自相同设备、黑色箱体和两个采集域，不能代表新工件、新背景或新光照；
2. 相邻采集组可能仍包含相似工件姿态，因此不应把该指标等同于生产环境精度；
3. `FDJYP-3` 尚无人工标签，不能用其伪标签指标替代独立人工测试；
4. 接入 LiteAnyStereo 后应在未参与训练的新双目场景上同时复查分割边界和主体视差完整率。

推荐下一步人工标注 `FDJYP-3` 中至少 30～50 张作为完全独立测试集，只做一次最终评估，不再用于调参或训练。接入时仍应让 LiteAnyStereo 对原始校正 RGB 计算视差，再用本模型输出的左图掩码过滤视差，而不是先把 RGB 背景涂黑。

## 12. FDJYP-3 独立场景预测

2026-08-19 使用冻结模型对 `JMP-workpiece-seg-pseudo-v1/images/val` 中全部 73 张 `fdjyp_3_*.png` 进行预测。新增可复用推理脚本：

```text
tools/jmp_workpiece/predict_bisenetv2_jmp.py
```

脚本会输出 `0/255` 二值掩码、红色叠加图、逐图置信度 CSV、元数据和抽样联系表。输出目录已存在时拒绝覆盖。

执行命令：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow

TF_USE_LEGACY_KERAS=1 \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
tools/jmp_workpiece/predict_bisenetv2_jmp.py \
  --model_pb ./runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --input_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-pseudo-v1/images/val \
  --input_glob 'fdjyp_3_*.png' \
  --output_dir ./runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/fdjyp_3_predictions \
  --width 288 --height 512 \
  --device cpu \
  --uncertain_threshold 0.75 \
  --contact_sheet_samples 30
```

本次主动使用 CPU，避免影响机器上的其他 GPU 任务。73 张全部完成，未出现文件缺失、非法类别或非有限概率。

输出目录：

```text
fdjyp_3_predictions/
  prepared_inputs/               # 实际送入模型的 288×512 图像
  masks/                         # 73 张 0/255 预测掩码
  overlays/                      # 73 张红色预测叠加图
  predictions.csv                # 每张图的前景比例和置信度
  predictions_contact_sheet.jpg  # 原图/叠加/掩码联系表
  metadata.json
```

整体统计：

```text
图像数: 73
预测前景比例: min 0.61599, median 0.84743, mean 0.83567, max 1.00000
平均置信度:   min 0.95909, median 0.99070, mean 0.98917, max 0.99672
低于 0.75 的像素比例: min 0.00113, median 0.01145, mean 0.01380, max 0.05321
概率和最大误差: 1.1920928955078125e-07
输出目录大小: 约 25 MB
```

前景比例明显高于之前的伪标签，不应仅据此判断模型失败。FDJYP-3 多数图像是工件贴近镜头的局部特写，工件确实占据画面大部分；例如预测接近整幅前景的 `fdjyp_3_1_202506281608_0016`，原图几乎没有可见箱体背景。联系表目视结果显示：

- 大面积金属主体预测连续，暗色和氧化表面不再像亮度伪标签那样大量漏掉；
- 可见的黑色箱体背景通常能被保留为背景；
- 深凹槽、孔洞、强反光和画面边缘仍出现局部孔洞或边界误差；
- 个别近距离图被预测为几乎整幅前景，需要人工标签确认其边缘是否过扩张。

仅作诊断，将预测与旧 Otsu/GrabCut 伪标签对比得到：

```text
foreground IoU agreement: 0.82270
background IoU agreement: 0.52062
mean IoU agreement:       0.67166
预测平均前景比例:         0.83567
伪标签平均前景比例:       0.69272
```

这里的数值只是与伪标签的一致度，不是真实精度。较低的背景一致度同时包含两种可能：模型误把背景标成工件，或者旧亮度伪标签漏掉了暗色金属；从抽样目视看，两者都存在，但暗色金属漏标是主要差异之一。没有 FDJYP-3 人工 GT 时不能计算可信 IoU/Dice，也不能据此设定生产阈值。

产物哈希：

```text
fdjyp_3_predictions/metadata.json SHA256:
4ec112035ceafd69225ab99c2353ad778bc409d218e641a47cd5e0a648f84a21

fdjyp_3_predictions/predictions.csv SHA256:
d895b2e7f8f0c0ced8b44efe83dcada433259f64448e01dd234a165e00e49c0f

fdjyp_3_predictions/predictions_contact_sheet.jpg SHA256:
fb8bbe3071b12381cbf1e165c28262e3d15d865abae68bbd48b4c39c267fde0f
```

## 13. 为连通域修订导出前景概率

2026-08-20 复核 LiteAnyStereo 主体视差时发现，硬类别掩码会把工件内部的暗色、氧化区域误删。原推理目录只保存 `argmax` 后的二值掩码，无法区分边界低置信度和模型的高置信错误，因此在推理脚本中新增可选参数：

```text
--save_probabilities
```

启用后，在不改变原有二值输出的前提下，额外保存每张图的工件类别概率：

```text
probabilities/<name>.npy  # float32，[512,288]，取值 [0,1]
```

重跑命令：

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/projects/bisenetv2-tensorflow

TF_USE_LEGACY_KERAS=1 \
/home/uestc/mount_2T/uestc/.conda/envs/dsrl_pi0/bin/python \
tools/jmp_workpiece/predict_bisenetv2_jmp.py \
  --model_pb ./runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/bisenetv2_jmp_workpiece_manual_isat_v1_frozen.pb \
  --input_dir ../LiteAnyStereo/data/datasets/JMP-workpiece-seg-pseudo-v1/images/val \
  --input_glob 'fdjyp_3_*.png' \
  --output_dir ./runs/jmp_workpiece/bisenetv2_manual_isat_v1_e20_retry1/fdjyp_3_predictions_probability_v2 \
  --width 288 --height 512 --device cpu \
  --uncertain_threshold 0.75 --contact_sheet_samples 30 \
  --save_probabilities
```

73 张全部完成，类别、概率和有限值检查通过。整体最大类别概率平均值仍为 `0.98917`，说明示例暗斑不是简单降低 0.5 阈值即可可靠解决：错误区域可能也是高置信背景。概率输出主要用于全分辨率双线性恢复和边界平滑；内部暗斑是否补回还必须结合工件拓扑和原始视差连续性。

新目录约 66 MB，校验如下：

```text
metadata.json SHA256:
8c384ac65f6f673d57efcecff92e7e8a1121710aa3d095da421462e700e490da

predictions.csv SHA256:
05e518cc1053756a9ba947c65c406134fe7b2275ba71dddce473797c86076bc1
```

后续连通域修订和 LiteAnyStereo 复评见：

```text
../../../04_mask_refinement/fdjyp3/reports/integration_report.md
```
