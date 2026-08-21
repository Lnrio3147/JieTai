<h1 align="center">Lite Any Stereo 系列</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2511.16555" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/LAS1-Paper-red" alt="LAS1 arXiv 论文"></a>
  <a href="https://tomtomtommi.github.io/LiteAnyStereo/"><img src="https://img.shields.io/badge/LAS1-Project%20Page-deepgreen" alt="LAS1 项目主页"></a>
  <a href="https://arxiv.org/abs/2606.24457" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/LAS2-Paper-red" alt="LAS2 arXiv 论文"></a>
  <a href="https://tomtomtommi.github.io/LiteAnyStereoV2/"><img src="https://img.shields.io/badge/LAS2-Project%20Page-blue" alt="LAS2 项目主页"></a>
</p>

<p align="center">
  <strong>Lite Any Stereo（LAS）系列的官方代码库。</strong><br>
  本仓库支持 LAS1 以及 LAS2 S/M/L/H 发布版模型。
</p>


## 概述

**Lite Any Stereo** 是一系列面向实际部署的高效零样本立体匹配模型。本仓库包含 **LAS1** 和 **LAS2** 的公开评测与推理代码。

## 本地训练与测试

本地 JMP-LF6020 工作流、训练、断点恢复、统一口径测试和可视化说明见
[使用说明](./docs/USAGE_zh-CN.md)。数据清洗、点云伪视差和 ETH3D 目录转换见
[JMP-LF6020 数据说明](./docs/JMP_LF6020_GUIDE_zh-CN.md)。

BiSeNetV2 主体分割接入 LAS1 的 FDJYP-3 全 73 场试验见
[BiSeNetV2 + LiteAnyStereo 接入报告](../../experiments/04_mask_refinement/fdjyp3/reports/integration_report.md)，
从源码落地到单连通域改进的汇报版总结见
[JMP 工件分割与 LiteAnyStereo 接入阶段汇报](../../experiments/04_mask_refinement/fdjyp3/reports/summary_report.md)。
初始和改进结果分别位于 `runs/evaluation/bisenet_las1_fdjyp3_postmask_v2/`、
`runs/evaluation/bisenet_las1_fdjyp3_refined_postmask_v2/`。

正式的“LiteAnyStereo 与 RT-IGEV”统一复评报告位于
[E01 实验报告](../../experiments/01_stereo_comparison/rec_img_set/reports/)，对应浮点视差、统一指标和 73 场对比图位于
`runs/evaluation/jmp_unified_rerun_73/`。额外 78 场传统工程图像的推理及其中 64 场的算法对比
位于 `runs/inference/tradition_extra/official/`。运行结果目录的分类说明见
[`runs/README_zh-CN.md`](./runs/README_zh-CN.md)。

| 版本 | 标题 | 资源 |
| --- | --- | --- |
| LAS1 | [CVPR 2026] Lite Any Stereo：高效零样本立体匹配 | [论文](https://arxiv.org/abs/2511.16555)、[项目主页](https://tomtomtommi.github.io/LiteAnyStereo/) |
| LAS2 | Lite Any Stereo V2：更快、更强的高效零样本立体匹配 | [论文](https://arxiv.org/abs/2606.24457)、[项目主页](https://tomtomtommi.github.io/LiteAnyStereoV2/) |

## 性能概览

<p align="center">
  <img src="./assets/fig2.png" alt="LAS 系列在零样本立体匹配基准上的性能对比" width="560">
</p>

<p align="center">
  <em>零样本性能与运行时间对比。运行时间在 H200 / Orin 8G 上测得。</em>
</p>

## 模型权重

预训练权重托管在 [Hugging Face](https://huggingface.co/tomtomtommi/LiteAnyStereoV2) 上，同时也提供了 [Google Drive 镜像](https://drive.google.com/drive/folders/1UvDx296pVk7pC2rozKIpQF_EXcOleZOB?usp=sharing)。请将下载的权重文件放入 `./checkpoints/`。各发布模型默认使用以下文件名：

| 模型 | 默认权重文件 |
| --- | --- |
| LAS1 | `./checkpoints/LiteAnyStereo.pth` |
| LAS2-S | `./checkpoints/LAS2_S.pth` |
| LAS2-M | `./checkpoints/LAS2_M.pth` |
| LAS2-L | `./checkpoints/LAS2_L.pth` |
| LAS2-H | `./checkpoints/LAS2_H.pth` |

未指定 `--model_size` 时，LAS2 默认使用 M 型号。你也可以随时通过 `--restore_ckpt` 显式指定权重文件。

## 演示

`./assets/` 中提供了若干组并排放置的双目图像。可通过 `--stereo_file` 使用其他图像对。运行 LAS1：

```bash
python demo.py --version las1 --restore_ckpt ./checkpoints/LiteAnyStereo.pth
```

运行 LAS2-M：

```bash
python demo.py --version las2 --model_size m --restore_ckpt ./checkpoints/LAS2_M.pth
```

修改 `--model_size` 即可运行其他 LAS2 发布版模型：

```bash
python demo.py --version las2 --model_size h --restore_ckpt ./checkpoints/LAS2_H.pth
```

演示程序会将视差可视化图、原始视差数组以及可选的点云输出保存到 `--out_dir` 指定的目录中。

## 评测

运行以下命令以复现基准评测：

```bash
VERSION=las1 sh evaluate.sh
VERSION=las2 MODEL_SIZE=s sh evaluate.sh
VERSION=las2 MODEL_SIZE=m sh evaluate.sh
VERSION=las2 MODEL_SIZE=l sh evaluate.sh
VERSION=las2 MODEL_SIZE=h sh evaluate.sh
```

也可以直接评测单个数据集：

```bash
python evaluate_stereo.py --version las2 --model_size h --restore_ckpt ./checkpoints/LAS2_H.pth --dataset middlebury_H
```

支持的数据集包括 `jmp`、`middlebury_F`、`middlebury_H`、`middlebury_Q`、`eth3d`、
`kitti` 和 `drivingstereo`。其中 `jmp` 默认采用 tradition 统一评价口径。

## MACs

计算模型复杂度：

```bash
python flops_count.py --version las1
python flops_count.py --version las2 --model_size m
python flops_count.py --version las2 --model_size h
```

## 运行时间

测量推理时间：

```bash
python profile_speed.py --version las1
python profile_speed.py --version las2 --model_size m
python profile_speed.py --version las2 --model_size h
```

在 GPU 上运行时，运行时间测试脚本会使用 CUDA 同步。

## 引用

如果本项目发布的代码对你有所帮助，请考虑引用：

```bibtex
@InProceedings{jing2026litestereo,
    author    = {Jing, Junpeng and Luo, Weixun and Mao, Ye and Mikolajczyk, Krystian},
    title     = {Lite Any Stereo: Efficient Zero-Shot Stereo Matching},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {21725-21735}
}

@article{jing2026litestereov2,
      title={Lite Any Stereo V2: Faster and Stronger Efficient Zero-Shot Stereo Matching}, 
      author={Junpeng Jing and Ronglai Zuo and Zhelun Shen and Shangchen Zhou and Rolandos Alexandros Potamias and Stefanos Zafeiriou and Krystian Mikolajczyk and Jiankang Deng},
      year={2026},
      eprint={2606.24457},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.24457}, 
}
```
