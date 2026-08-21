# rec_img_set 模型对比

范围：203 个唯一场景；FDJYP-3 的 73 场用于定量评价。

结论：LiteAnyStereo LAS1 的全 73 场 EPE 为 `2.0762 px`，IGEV++ RT 为 `4.6739 px`；LAS1 的总体稳定性和尾部表现更好，IGEV 在部分普通单场仍有优势。

主要入口：

- `reports/comparison_report.md`：正式报告；
- `results/final_203/`：正式 203 场结果；
- `scripts/run_comparison.py`：正式运行入口；
- `slides/`：汇报材料。

辅助结果：`baseline_73` 是历史复评，`extra_78` 是额外场景观察，`igev_recheck_73` 是复现核验，`igev_legacy_73` 是一期保存结果。它们不替代 `final_203`。
