# 项目源码

| 目录 | 用途 |
|---|---|
| `tradition_stereo/` | 传统双目、标定、点云及历史工具 |
| `LiteAnyStereo/` | LAS 模型、训练和推理代码 |
| `IGEV-plusplus/` | IGEV++ RT 官方代码与权重 |
| `bisenetv2-tensorflow/` | BiSeNetV2 分割代码 |

本目录以代码、配置和模型权重为主。数据实体统一位于 `../datasets/`，实验运行实体统一位于 `../experiments/`。为兼容原工程相对路径，少数 `data/datasets`、`runs`、`igev_output` 路径保留为符号链接。
