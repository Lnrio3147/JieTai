# 数据集目录

这里是 JieTai 的唯一实体数据根目录。`projects/` 中同名数据路径仅为兼容旧命令的符号链接，不再保存第二份数据。

| 目录 | 内容 | 规模/用途 |
|---|---|---|
| `Jop_1/` | Jop1 原始压缩包及解压数据 | 9 组左右图、DAT、PLY |
| `rec_img_set/` | 已矫正双目图主集合 | 203 个唯一场景，另含 FDJYP-3 未矫正源图和标定文件 |
| `tradition_raw/` | 传统工程原始数据集合 | FDJYP-0/3、螺纹、JXP、DE0548 等 |
| `training/JMP-LF6020-ETH3D/` | LAS 训练/验证格式数据 | 266 场及 manifest |
| `training/workpiece-seg-isat-v2/` | 多域人工工件分割数据 | 130 张，分层 train/val/test 为 88/21/21 |
| `training/workpiece-seg-grouped-v3/` | 全部人工标注 RGB-D 分组划分 | 317 张，按采集组隔离为 train/val/test 218/53/46 |
| `training/workpiece-seg-stereo-v1/` | 实验 10 RGB 双视图增强分割集 | 381/53/46；训练含 163 张视差投影右目伪标签，val/test 仅人工左目 |
| `evaluation/workpiece-right-manual-isat-v1/` | 人工右目冻结评价集 | ISAT 修订 32 张，val/test 为 15/17，覆盖 7 类 |
| `training/ETH3D/` | ETH3D 数据 | 外部基准/训练辅助 |
| `annotations/` | 工件分割标注 | 人工 ISAT v1、伪标签 v1，以及已完成人工外轮廓修正的多域 ISAT v2 |
| `references/FDJYP-3_foundation_stereo/` | FDJYP-3 评价参考 | 73 场浮点视差及可视化 |
| `archives/JMP-LF6020.zip` | 原始训练压缩包 | 只读归档 |

## rec_img_set 去重说明

- `kedu` 已验证与 `rectified_images_刻度` 逐文件一致，现为指向后者的链接；
- `test/矫正图片` 已验证是 `FDJYP-3-rectified_images` 的相同文件子集，现为指向规范副本的链接；
- 两份被替换的实体重复目录已于 2026-08-20 移入系统回收站，可恢复；
- `test/未矫正图片` 与 `test/标定文件` 是唯一内容，继续保留。

实验脚本应优先使用本目录中的规范路径。需要复现旧项目命令时，`projects/tradition_stereo/datasets`、`projects/tradition_stereo/rec_img_set` 和 `projects/LiteAnyStereo/data/datasets/*` 仍可用。

## 分割数据划分说明

`workpiece-seg-grouped-v3` 汇总了当前 317 张人工标注图，固定随机种子为
`20260824`。FDJYP-0/2 使用文件名中已有的真实采集组，其余缺少工件 ID 的
数据使用分钟级采集段或连续 6 帧块作为代理。跨集合的采集组和原图 SHA-256
重复均为 0；详细规则、逐组分配和掩码总览见该目录的 `metadata.json` 与 `qa/`。
现有标签和旧模型输出都已参与过开发观察，因此其中的 test 是后续工程回归集，
不能表述为从未观察过的论文最终测试集。

`workpiece-seg-stereo-v1` 由实验 10 脚本生成，不新增人工标注。右目训练 Mask 来自
已有视差前向投影并经过有效率检查，只用于增强视角不变性；不能把它当作右目真值。

`annotations/workpiece-right-isat-v1` 是实验 10 生成的 32 张右目人工复核任务包，
包含 ISAT JSON 预标注、投影 Mask、叠加预览和 `tasks.csv`。`tasks.csv` 的
`manual_status=pending` 是任务创建时的历史状态。2026-08-25 标注者确认 32 张均已在
ISAT 中修订；实验 10 的导出器完成一一配对、类别、尺寸、有限坐标、有效多边形和
图层 QA 后，冻结到 `evaluation/workpiece-right-manual-isat-v1`。该评价集正确处理
`__background__` 擦除层，带 SHA-256 标注清单和全量叠加总览，只用于评价而不回流训练。
