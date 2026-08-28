# Experiment 9: JiT-inspired clean-mask projection

This experiment transfers only the parts of JiT that fit the segmentation and
RK3588 constraints: direct clean-data (`x`) prediction, structured corruption,
large raw patches, and a low-rank manifold bottleneck. It does **not** use a
diffusion sampler or a large Vision Transformer.

The frozen Experiment 8 Base student first generates a coarse foreground
probability and boundary map. A small residual projector receives RGB, robust
disparity, disparity Sobel/validity, the coarse outputs, and a corruption-level
channel. During training, connected overflow blobs, missing boundary sections,
holes, erosion and dilation are added to the coarse mask. The projector predicts
the human GT clean mask in a single pass. Its correction head is zero-initialized,
so epoch zero exactly reproduces Experiment 8 Base and training cannot silently
discard the previous best checkpoint.

```bash
cd /home/uestc/mount_2T/uestc/lnrio/JieTai/experiments/09_jit_mask_projection
./run_grouped_v3.sh
```

`run_experiment9.sh` is retained only to reproduce the legacy V2 88/21/21 run.

Validation and comparison remain separated in code. The current grouped V3 run
selects its threshold on 53 validation images and reports it once on the frozen
46-image test split. Images from the same acquisition group are restricted to one
split, so adjacent frames cannot leak across training, validation and test.

## Current grouped V3 result (recommended comparison)

Training stopped at epoch 19; epoch 9 was selected by validation IoU. The
validation-selected deployment threshold was `0.05`.

| Method | IoU | Precision | Recall | Boundary F1 | Macro-category IoU | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **V3 Base** | **0.9304** | **0.9483** | 0.9801 | **0.7802** | **0.9420** | **2.398** | **9.048** |
| V3 Distilled | 0.9234 | 0.9376 | **0.9839** | 0.7327 | 0.9396 | 2.398 | 9.048 |
| Exp9 V3 projector | 0.9275 | 0.9437 | 0.9817 | 0.7704 | 0.9394 | 2.425 | 10.194 |

Experiment 9 preserves slightly more foreground than Base (`Recall +0.0016`),
but loses `0.0030` IoU and `0.0098` Boundary F1. It improves 26/46 test images,
yet most gains are very small; one Luowen image falls from `0.9261` to `0.7214`
IoU and dominates the aggregate loss. All 46 outputs remain a single connected
component. The result therefore supports using V3 Base directly, not adding the
projector to the deployment chain.

Detailed V3 artifacts:

- [full report](results/comparison_grouped_v3/REPORT.md)
- [comparison table](results/comparison_grouped_v3/comparison.csv)
- [all 46 visual comparisons](results/comparison_grouped_v3/test_contact_sheet.jpg)
- [per-image changes](results/comparison_grouped_v3/per_image.csv)
- [validation threshold sweep](results/comparison_grouped_v3/threshold_sweep.csv)
- [training summary](results/clean_mask_projector_grouped_v3/training_summary.json)

## Legacy V2 development result

The earlier 88/21/21 split was reused across several experiments and is retained
only for reproducibility, not as the current unbiased comparison.

| Method | IoU | Precision | Recall | Boundary F1 | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|---:|
| Exp8 Base | **0.9415** | **0.9437** | 0.9976 | **0.4471** | 2.398 | 9.048 |
| Exp9 clean-mask projector | 0.9400 | 0.9420 | **0.9978** | 0.4421 | 2.425 | 10.195 |

Experiment 9 is a negative result: the learned projection changes an already
strong Base mask too little to repair hard errors, yet enough to reduce test IoU
by 0.0015. Only 3/21 test scenes improve, while 17/21 degrade. The projector is
therefore retained for reproducibility but is not recommended for deployment.
