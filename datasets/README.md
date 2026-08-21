# 数据集目录

这里是 JieTai 的唯一实体数据根目录。`projects/` 中同名数据路径仅为兼容旧命令的符号链接，不再保存第二份数据。

| 目录 | 内容 | 规模/用途 |
|---|---|---|
| `Jop_1/` | Jop1 原始压缩包及解压数据 | 9 组左右图、DAT、PLY |
| `rec_img_set/` | 已矫正双目图主集合 | 203 个唯一场景，另含 FDJYP-3 未矫正源图和标定文件 |
| `tradition_raw/` | 传统工程原始数据集合 | FDJYP-0/3、螺纹、JXP、DE0548 等 |
| `training/JMP-LF6020-ETH3D/` | LAS 训练/验证格式数据 | 266 场及 manifest |
| `training/workpiece-seg-isat-v2/` | 多域人工工件分割数据 | 130 张，分层 train/val/test 为 88/21/21 |
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
