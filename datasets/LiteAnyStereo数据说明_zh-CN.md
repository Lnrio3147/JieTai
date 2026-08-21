# 数据集目录

| 目录 | 用途 |
| --- | --- |
| `training/JMP-LF6020-ETH3D/` | 当前 JMP 唯一训练/测试数据根目录，266 场 ETH3D 兼容样本 |
| `training/ETH3D/` | 上游 ETH3D 基准数据，可用于通用训练链路验证 |

请勿在本目录保留从 JMP 生成的平铺中间副本；需要重新生成时，以仓库根目录的
`archives/JMP-LF6020.zip` 为输入运行 `projects/LiteAnyStereo/tools/prepare_jmp_lf6020.py`。JMP 的详细结构和处理方法见
[`JMP_LF6020_GUIDE_zh-CN.md`](../projects/LiteAnyStereo/docs/JMP_LF6020_GUIDE_zh-CN.md)。
