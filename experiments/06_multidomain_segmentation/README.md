# 实验 6：多域 BiSeNetV2 基线

本目录只保留人工多域标注训练得到的 BiSeNetV2 基线、两条最终方案及其验证/测试结果。RGB 与 LiteAnyStereo 视差的可训练融合已独立为[实验 7](../07_rgbd_fusion/README.md)。

## 当前保留结果

| 目录 | 用途 |
|---|---|
| `results/bisenetv2_recall_v4_1_jop_rescue/` | 主体保留优先的最终入口，模型和结果均使用符号链接 |
| `results/tune01_ce_light_b4_p025/` | V4.1 实际 BiSeNetV2 权重、冻结 PB、日志和训练配置 |
| `results/tune01_jop_reflective_rescue_val_v2/` | V4.1 验证集结果 |
| `results/tune01_jop_reflective_rescue_test_v2/` | V4.1 开发比较测试结果 |
| `results/bisenetv2_balanced_v2/` | 综合质量路线的 BiSeNetV2 权重和冻结 PB |
| `results/exp4_exp5_balanced_v2_test_v1/` | 综合质量路线的最终测试结果 |

V4.1 在21张开发比较图上的前景 IoU/Precision/Recall/Boundary F1 为 `0.88294/0.89025/0.99079/0.34108`。这些图片参与过多轮工程比较，不能再作为严格的一次性论文测试集。

## 代码与报告

- `scripts/prepare_dataset.py`：从 ISAT JSON 生成固定拆分；
- `scripts/evaluate_test.py`：单模型评测；
- `scripts/evaluate_exp4_exp5.py`：实验4/5和V4.1规则评测；
- `scripts/build_routed_result.py`：历史V3路由复现脚本；
- `reports/`：V1～V4.1的完整参数、指标与结论；
- `reports/cleanup_report.md`：本次目录清理范围。

## 标注范围

130张人工标签只表达主体外轮廓，没有显式 `__background__` 孔洞多边形。因此监督指标不能证明模型学会识别内部通孔；最终点云仍需使用视差连续性约束。
