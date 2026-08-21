# 杰泰二期项目交接摸底材料

## 0. 材料说明

本文面向张玉圻老师及后续研一同学接手二期项目使用，重点梳理一期已经完成和验证过的工作。材料覆盖三部分：项目背景与现状、数据全貌、模型技术方案。当前代码仓库主要承担数据处理、视差与点云结果整理、指标计算和部分传统算法/工程验证工作；模型训练与推理主代码位于 A6000 服务器：

```text
../IGEV-plusplus
```

本项目一期最终采用的是 IGEV 的 RT 版本，核心考虑是轻量化、推理效率和跨工业零件数据的泛化效果。

## 1. 项目背景与现状总结

### 1.1 业务目标

项目全称可概括为“基于传统算法融合神经网络模型的双目三维重建及工程化”。业务目标不是单纯做一个双目匹配模型，而是围绕工业内窥镜/工业零件测量场景，形成一条能落地的三维重建链路：

1. 输入双目图像，完成图像配对、相机矫正和视差估计。
2. 将视差图转换为点云或 pointmap，供后续测量、可视化和点云增强使用。
3. 以 Foundation Stereo 生成的视差图作为评估参考，定量比较研发算法与甲方仪器算法、竞品算法之间的误差。
4. 最终面向 RK3588 等国产化边缘平台部署，兼顾精度、速度和资源占用。

一期验收场景主要围绕甲方提供的工业零件数据集，图像分辨率为 `1280x720`。由于缺少真实密集视差标注，验收口径采用英伟达 Foundation Stereo 输出的视差图和点云作为参考标准，计算研发算法的 EPE、BPR-1.0、BPR-2.0、BPR-3.0 等指标。

### 1.2 为什么不能直接用传统算法

一期最早验证过传统 SGBM。传统算法的优势是实现简单、推理速度快、部署门槛低，但在当前场景存在明显瓶颈：

- 工业内窥镜图像常有弱纹理、重复纹理、高反光和局部阴影，SGBM 容易误匹配。
- 零件表面局部纹理单一，传统匹配依赖局部块相似性，空洞和毛刺较多。
- 大视差场景下，传统参数需要针对不同工况反复调节，稳定性不足。

因此一期没有把 SGBM 作为最终主方案，而是保留在仓库中作为传统基线和对照工具。对应脚本：

```bash
python SGBM.py
```

### 1.3 为什么选择 IGEV RT

深度学习模型在精度和泛化上优于传统算法，但工程落地时存在算力和部署压力。一期在方案选择上主要考虑三点：

- `轻量级`：二期目标包含边缘部署，模型不能只看 PC 端精度。
- `泛化效果`：工业零件数据量有限，模型需要在少量目标域数据下保持可用。
- `迭代精化能力`：双目匹配误差往往集中在边缘、反光和弱纹理区域，迭代式模型比一次性回归更容易逐步修正。

IGEV 系列模型本身采用迭代更新思路，RT 版本在速度和资源占用上更适合工程化尝试。因此一期最终使用 IGEV RT 作为主模型，而不是继续沿用传统 SGBM 或重型大模型。

### 1.4 一期已解决的核心问题

一期已经完成或基本跑通以下关键环节：

- 双目原始图整理：将左右图整理为 `im0.png` / `im1.png` 的场景目录。
- 图像矫正：基于标定 YAML 或预计算 map 批量生成矫正图像。
- Foundation Stereo 参考视差整理：将参考视差统一裁剪为 `818x512`，保存为 `disp_cropped.npy`。
- 模型预测结果对齐：将 IGEV/RKNN 等预测视差裁剪到同一尺寸后参与评估。
- 指标计算：实现 EPE、D1、Bad1/Bad2/Bad3 等指标。
- 点云反投影评估：支持将 PLY 点云反投影为视差图，再与参考视差计算指标。
- pointmap 格式输出：保存 `pointmap.npy` 和自定义二进制 `pointmap.bin`，便于 C++ 或后续算法读取。

当前数据处理和指标计算文档见：

```text
TECHNICAL_WORKFLOW.md
```

### 1.5 当前遗留风险与待优化点

当前最大风险不是某个单独模型，而是数据和评估口径容易不一致。

第一，裁剪逻辑存在历史混用。一期曾对比过甲方动态裁剪和非零视差边界裁剪。甲方动态裁剪参数大致为：

```text
x=104, y=230, w=512, h=818
```

但根据 Foundation Stereo/点云反投影后的非零视差边界，实际更对齐的区域是：

```text
x=126..637, y=234..1051
```

因此当前建议统一采用：

```python
disp_cropped = disp[234:1052, 126:638]  # H=818, W=512
```

如果不同脚本继续混用动态裁剪和固定裁剪，指标会出现系统性偏差。这是二期最先应收敛的问题。

第二，部分脚本路径硬编码较多，例如数据根目录、输出目录、标定文件路径等。短期可以手动修改，长期应改成命令行参数。

第三，`CMakeLists.txt` 引用了一些当前仓库不存在的 C++ 文件，说明 C++ 工程配置有历史残留，需要重新整理后再作为正式构建入口。

第四，部分测试脚本与当前 `metric/cal_metric.py` 不完全匹配，例如 `test_metric_improvements.py` 引用了已经不存在的函数。二期需要先清理测试资产，否则容易误判代码状态。

## 2. 数据全貌说明

### 2.1 当前数据来源

一期主要使用三类数据：

1. 甲方提供的工业零件双目图像数据，分辨率主要为 `1280x720`。
2. Foundation Stereo 生成的参考视差图，作为当前指标计算中的 `GT`。
3. 模型或仪器输出的预测视差图、点云和过滤后点云，用于和 Foundation Stereo 参考结果对比。

需要强调：当前文档中的 `GT` 不是人工标注真值，也不是甲方原始点云本身，而是 Foundation Stereo 输出后经过统一裁剪的参考视差。

### 2.2 当前仓库中的数据量级

以本地仓库 `datasets/FDJYP-3` 为统计口径，当前可识别到 `73` 个场景目录，命名格式类似：

```text
202506281603-0001
202506281603-0002
202506281603-0003
...
```

本地文件类型统计大致为：

```text
.png  593 个
.npy  305 个
.ply  146 个
.txt  105 个
.csv   38 个
.xlsx   1 个
```

典型单场景样例如：

```text
datasets/FDJYP-3/202506281603-0001/
  im0.png
  im1.png
  disp.npy
  disp_right.npy
  disp_cropped.npy
  disp_igev.npy
  0001_old.ply
  0001.ply
  0001_disp_cropped.npy
  0001_disp_color.png
```

其中：

- `im0.png` / `im1.png`：左、右矫正图或待处理图像。
- `disp.npy`：原始尺寸视差或模型输出视差，具体含义需结合生成脚本确认。
- `disp_cropped.npy`：Foundation Stereo 参考视差裁剪结果，是指标计算默认 GT。
- `disp_igev.npy`：IGEV 模型预测视差结果。
- `*_old.ply`：甲方或历史流程中的点云文件。
- `*_disp_color.png`：视差可视化结果。

### 2.3 数据处理主流程

数据处理分为五步。

第一步，原始左右图配对。当前脚本会把原始 `*_L.png` / `*_R.png` 整理成每个场景一个文件夹：

```bash
python data_process/save_rawimg.py
```

当前脚本内硬编码了 `source_dir`，后续建议改为：

```bash
python data_process/save_rawimg.py --source-dir <raw_image_dir> --output-dir <paired_output_dir>
```

第二步，生成矫正图像。推荐主脚本为：

```bash
python data_process/save_rectified_direct.py
```

该脚本直接读取标定文件，例如：

```python
config_path = r"D:\Desktop\stereo_project\tradition_stereo\config\stereo.yml"
```

然后使用 OpenCV `initUndistortRectifyMap()` 和 `remap()` 批量生成矫正后的 `im0.png` / `im1.png`。

如果已有预计算 map，可以使用：

```bash
python data_process/save_rectified.py
```

当前已有 map 目录包括：

```text
config/map/1221
config/map/gongjian_map
config/map/JXP_map
config/map/luowen_map
config/map/new_map
config/map/others_map
config/map/test
```

第三步，检查矫正效果。调试脚本会显示左右图并绘制水平线，观察极线是否对齐：

```bash
python read_stereo.py
```

第四步，生成或整理 Foundation Stereo GT。当前统一口径是裁剪为 `818x512`：

```bash
python data_process/save_old_disp.py
```

关键裁剪逻辑：

```python
disp_cropped = disp[234:1052, 126:638]
```

第五步，裁剪待评估模型输出，使预测视差与 GT 同尺寸：

```bash
python data_process/save_model_dis.py
```

注意：该脚本当前仍需整理，注释写的是固定裁剪，但实际代码存在动态裁剪逻辑。二期应优先将其改为固定裁剪，并改名为 `save_model_disp.py`，避免文件名和功能不一致。

### 2.4 数据更新频率

一期不是持续在线采集模式，数据更新主要来自甲方阶段性提供的新批次采集数据，或研发过程中为了验证特定型号、特定标定参数而补充的数据。实际更新频率取决于甲方采集和验收节点，不是每天固定更新。

二期建议每次新增数据时同步记录：

- 采集日期和批次编号。
- 对应产品或工件类型。
- 图像分辨率。
- 使用的标定文件。
- 是否经过矫正。
- Foundation Stereo GT 的生成日期和版本。
- 是否进入正式评估集。

建议建立一个 `DATASET_MANIFEST.md` 或 `datasets/manifest.csv`，否则后续很难追溯每个 `disp.npy` 到底来自哪版模型或哪次处理。

### 2.5 质量清洗规则

当前清洗规则主要集中在视差、点云和评估场景三个层面。

视差层面：

- 统一裁剪到 `818x512`。
- 无效视差通常以 `0` 表示。
- 指标计算时默认只在 `gt_disp > 0` 区域计算。
- 点云反投影指标中，应使用 GT 与预测都有值的联合有效区域。

点云层面：

- 过滤 `Z <= 0` 的点。
- 过滤 NaN、无穷大等非法坐标。
- 根据视差范围过滤过小或过大的异常点。
- 基于图像颜色过滤黑色背景区域。
- 后续增强流程可基于连通域过滤小面积噪声区域。

评估场景层面：

- 当前部分脚本默认排除 4 个异常场景：

```text
202506281607-0012
202506281609-0019
202506281609-0020
202506281619-0053
```

- 部分评估脚本还会过滤 EPE 高于阈值的场景，例如 `10` 或 `20` 像素。

这些过滤规则必须在报告中显式说明。否则同一个模型在“包含异常场景”和“排除异常场景”下会得到不同指标，容易造成验收沟通风险。

## 3. 模型技术方案

### 3.1 一期采用的模型方案

一期采用 IGEV 的 RT 版本作为主模型，代码位于：

```text
../IGEV-plusplus
```

选择该方案的核心原因是：在工业内窥镜场景中，传统算法速度快但精度和稳定性不足；Foundation Stereo 等大模型效果强，但直接部署和实时推理压力大；IGEV RT 在速度、精度、模型复杂度和可改造性之间相对均衡。

IGEV 的关键思想是先构建立体匹配所需的代价信息，再通过迭代更新逐步优化视差。相比一步回归的模型，迭代式模型更适合处理边缘、弱纹理和局部歧义区域。

### 3.2 与项目原始研究方案的关系

项目原始研究方案中包含多范围分级代价体、3D CNN 正则化、ConvGRU 迭代精化、轻量级特征提取和 RK3588 部署改造等方向。实际一期推进中，模型侧优先选择了已有成熟模型 IGEV RT，而不是从零实现全新网络。

这样做的原因主要有三个：

第一，项目周期要求先跑通工程链路。双目三维重建不是单模型问题，还包括标定、矫正、裁剪、视差转点云、点云过滤、指标计算和验收对齐。如果先从零训练模型，容易长期卡在模型精度而无法验证完整链路。

第二，工业数据量有限。自研模型若没有足够目标域数据，容易过拟合少量样例，泛化不稳定。使用已有模型权重可以先获得一个稳定起点。

第三，部署目标要求轻量化。IGEV RT 本身比大模型更适合做速度和部署优化，也更适合后续尝试 RKNN 转换或结构替换。

### 3.3 训练与蒸馏尝试

一期曾尝试过将原模型 `12` 次迭代的输出，通过 SceneFlow 数据集蒸馏成 `1` 次迭代模型，目标是降低推理时间。这个方向的动机是明确的：迭代次数越少，推理速度越快，越接近边缘部署需求。

但实际结果不理想。蒸馏后模型输出出现明显过度平滑，细节和边缘变差。对于工业零件测量，边缘和局部结构往往是关键区域，过平滑会直接影响点云边界和测量精度。也就是说，蒸馏虽然降低了计算量，但牺牲了项目最关心的几何细节。

因此一期最终放弃该蒸馏模型，继续使用原本模型权重作为主评估模型。这个坑对二期有参考价值：如果继续做加速，不能只看平均 EPE，必须额外观察边缘区域、薄结构、反光区域和点云表面连续性。

### 3.4 评估指标

当前指标以 Foundation Stereo 输出作为参考。常用指标包括：

`EPE`：平均视差绝对误差。

```text
EPE = mean(|d_pred - d_gt|)
```

`BPR/BadX`：坏点率，即误差超过阈值的像素比例。常用阈值为 `1px`、`2px`、`3px`。

```text
BPR_tau = count(|d_pred - d_gt| > tau) / N
```

当前代码中指标函数位于：

```text
metric/cal_metric.py
```

推荐指标计算命令：

```bash
python tools/cal_metric_512x818.py ^
  --pred-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3 ^
  --gt-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3 ^
  --pred-file disp_rknn.npy ^
  --gt-file disp_cropped.npy ^
  --results-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3\evaluation_results
```

如果评估过滤后点云，则使用：

```bash
python tools/cal_metric_ply_rknn.py
```

但要先确认点云反投影使用的 Q 矩阵偏移与固定裁剪 `[234:1052, 126:638]` 对齐，否则指标会因为坐标系偏移而失真。

### 3.5 典型瓶颈和踩坑

第一类瓶颈是大视差和高分辨率。工业近距离场景视差可能很大，原始分辨率又达到 `1280x720`。如果构建单一高分辨率大范围 cost volume，显存和计算会快速增长。模型选型时必须同时考虑最大视差、输入尺寸和实际硬件可承受的推理时间。

第二类瓶颈是弱纹理和反光。传统算法在这些区域容易失败，深度模型虽然更稳，但仍会在反光边缘、遮挡边界和重复纹理处出现误差。因此不能只看整图平均指标，也要看点云边缘和局部测量区域。

第三类瓶颈是迭代次数与速度的冲突。减少迭代可以加速，但一期蒸馏实验表明，一次迭代模型容易过度平滑。后续如果继续加速，更可行的方向可能是局部裁剪、输入分辨率策略、模型结构轻量化或硬件算子适配，而不是简单把迭代次数压到 1。

第四类瓶颈是 RK3588 部署。原始研究方案中已经指出，RKNN 对 3D 卷积、3D 反卷积、BatchNorm3d、GridSample、Einsum、部分双线性插值和五维张量支持存在限制。二期如果继续推进板端部署，需要重点处理：

- PyTorch 到 ONNX 的导出一致性。
- ONNX 到 RKNN 的算子兼容性。
- 3D 算子的 2D 等价替换。
- 固定输入尺寸限制。
- 量化前后误差。
- RK3588 NPU 寄存器或图优化导致的隐式错误。

第五类瓶颈是评估基准本身。Foundation Stereo 是参考标准，但不是严格意义上的真实真值。它适合用于一期验收和相对比较，但如果二期要追求更高可信度，最好补充 3D 激光扫描点云或标准件几何尺寸作为外部验证。

## 4. 二期建议优先事项

第一，先统一数据和指标口径。建议把固定裁剪封装成公共函数：

```python
FIXED_CROP = (234, 1052, 126, 638)
```

并让 `save_old_disp.py`、`save_model_dis.py`、`cal_metric_ply.py`、`cal_metric_ply_rknn.py` 全部调用同一套函数。

第二，把数据处理脚本改成命令行参数。当前大量路径硬编码，不利于张老师带新同学复现实验。

第三，整理数据清单。建议为每批数据建立 manifest，记录采集时间、分辨率、标定文件、GT 生成方式、模型版本和是否进入正式评估。

第四，重新确认模型评估集。当前 73 个场景可以作为一期基线，但二期需要明确哪些是训练/调参数据，哪些是固定测试集，避免在同一批数据上反复调参导致指标虚高。

第五，模型侧优先复现一期结果，再考虑新优化。建议先在 A6000 上复现 IGEV RT 原权重推理结果和指标，再重新评估是否继续蒸馏、剪枝、量化或 RKNN 部署。

第六，板端部署不要过早承诺指标。RK3588 的算子限制会影响模型结构和数值一致性，建议先完成小模型/小输入的端到端验证，再扩大到正式输入尺寸。

## 5. 常用命令汇总

进入项目：

```bash
cd D:\Desktop\stereo_project\tradition_stereo
```

安装依赖：

```bash
pip install -r requirements.txt
```

整理原始左右图：

```bash
python data_process/save_rawimg.py
```

生成矫正图像：

```bash
python data_process/save_rectified_direct.py
```

检查矫正效果：

```bash
python read_stereo.py
```

整理 Foundation Stereo GT：

```bash
python data_process/save_old_disp.py
```

裁剪预测视差：

```bash
python data_process/save_model_dis.py
```

计算视差指标：

```bash
python tools/cal_metric_512x818.py ^
  --pred-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3 ^
  --gt-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3 ^
  --pred-file disp_rknn.npy ^
  --gt-file disp_cropped.npy ^
  --results-dir D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3\evaluation_results
```

计算点云反投影指标：

```bash
python tools/cal_metric_ply_rknn.py
```

生成点云和 pointmap：

```bash
python batch_process_512x818.py
```

## 6. 交接结论

一期已经把“图像整理、矫正、Foundation Stereo 参考视差、模型预测视差、点云输出、指标计算”这一整条评估链路基本跑通。当前真正需要二期优先解决的不是重新选一个模型，而是把一期的流程工程化、参数化和可复现化。

模型侧，IGEV RT 是一个合理的一期基线，原权重效果优于蒸馏后的一次迭代模型。二期如果继续优化，应围绕轻量化和部署适配逐步推进，同时始终用固定测试集和统一指标口径验证，不建议在裁剪、过滤、异常场景排除规则不统一的情况下比较模型优劣。
