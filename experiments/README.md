# 实验总览

七个一级目录就是当前实验的先后顺序：

| 阶段 | 目的 | 数据分支 | 主要入口 |
|---|---|---|---|
| `01_stereo_comparison` | 引入 LiteAnyStereo，与 IGEV 对比 | rec_img_set、Jop1 | `01_stereo_comparison/README.md` |
| `02_initial_segmentation` | 直接引入 BiSeNetV2，建立初始分割基线 | FDJYP-3 | `02_initial_segmentation/README.md` |
| `03_manual_segmentation` | 人工标注后重新训练和测试 | FDJYP-3、Jop1、rec_img_set | `03_manual_segmentation/README.md` |
| `04_mask_refinement` | 修复孔洞、孤岛和连通域异常后复评 | FDJYP-3 | `04_mask_refinement/README.md` |
| `05_disparity_guided_segmentation` | 用 LAS 视差提取主体并生成干净视差/点云 | FDJYP-0 | `05_disparity_guided_segmentation/README.md` |
| `06_multidomain_segmentation` | 使用五类人工外轮廓训练、验证并做类别路由 | FDJYP3、螺纹、general、scale、Jop1 | `06_multidomain_segmentation/README.md` |
| `07_rgbd_fusion` | 冻结 LAS，以 RGB+视差双流网络学习主体与边界 | FDJYP3、螺纹、general、scale、Jop1 | `07_rgbd_fusion/README.md` |

```text
01 模型对比
   ├── rec_img_set：203 场正式结果
   ├── jop1：9 场正式结果
   └── las_training：LAS 训练记录
02 初始分割
   └── fdjyp3：伪标签模型
03 人工标注
   ├── fdjyp3：人工模型和初始后掩膜测试
   ├── jop1：9 场跨域测试
   └── rec_img_set：130 场扩展测试
04 分割修复
   └── fdjyp3：掩码修订和最终 73 场复评
05 视差引导主体提取
   └── fdjyp0：64 张选参 + 18 张冻结测试，输出主体视差与点云
06 多域人工标注训练
   └── 130 张分层拆分，旧模型/V1/平衡V2/路由V3/Recall V4 对比
07 RGB-D 双流融合
   └── 冻结 LiteAnyStereo，四尺度门控融合，并以实验7.1连续实体约束清理碎片
```

编号表示实验脉络，不表示不同数据集的指标可以直接横向比较。已完成结果中的 JSON/CSV 可能保留整理前路径，这些字段不影响当前脚本和报告入口。

实验 6 保留 BiSeNetV2 基线。实验 7.1 的连续实体版本在同一开发比较集达到 IoU 0.8915、Precision 0.8932、Recall 0.9980、Boundary F1 0.4812；完整限制与结果见实验7报告。
