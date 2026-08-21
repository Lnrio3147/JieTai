# rec_img_set 人工模型测试

冻结人工标注 BiSeNetV2，在 130 个非 FDJYP-3、去重场景上测试。

- `reports/test_report.md`：完整报告；
- `results/result_130/`：汇总、逐场指标和四组总览；
- `scripts/run_test.py`：评估入口。

FDJYP-0 是训练域回归检查；其余三组没有人工分割真值。
