# JiT 思想在当前主体分割任务中的适用性

论文：Tianhong Li, Kaiming He, *Back to Basics: Let Denoising Generative
Models Denoise*，arXiv:2511.13720。官方 PyTorch 复现：
https://github.com/LTH14/JiT

## 可以迁移的部分

1. **直接预测干净目标（x-prediction）**：把人工 GT Mask 看成低维几何流形上的
   干净数据，把 Exp8 概率图的溢出、缺口和孔洞看成偏离流形的噪声，网络直接输出
   干净 Mask，而不是预测“错误噪声”。
2. **低容量瓶颈**：论文表明 x-prediction 不要求中间宽度覆盖高维原始 patch，低秩
   bottleneck 反而可能有益。Exp9 因此把 4x4 特征 patch 压到 16 通道后再恢复。
3. **大 patch/固定 token 数的计算思想**：只在投影器最深层使用大 patch 压缩，避免
   在 512x288 网格上做高成本全局注意力。

## 不应迁移的部分

1. **完整扩散生成过程**：JiT 默认需多步 ODE 求解；分割是确定性判别任务，加入多步
   采样会直接破坏 RK3588 的实时目标。
2. **大 Vision Transformer**：官方 JiT 面向 ImageNet 百万级数据，复现配置为 600
   epochs、8xH200；当前只有 88 张训练图，直接训练 ViT 极易过拟合。
3. **16/32 像素输出 patch**：工业螺纹边界恰好依赖细像素，大输出 patch 会损伤
   Boundary F1。
4. **“无需额外损失”照搬**：生成模型采用连续 RGB 回归，而当前二值分割仍需要
   BCE/Tversky 与距离变换边界监督来满足高召回和细边界目标。

## 本次验证结论

单步 x-pred Mask Projector 在验证集只增加约 0.00007 IoU，在固定开发比较集反而从
0.9415 降到 0.9400，Boundary F1 从 0.4471 降到 0.4421。说明 Exp8 Base 的错误并非
一般随机噪声，而是未见域下的语义误判；仅依赖输出 Mask 流形无法知道被连入的区域
究竟是主体还是背景。

因此保留 Exp8 Base 作为当前最佳部署模型。JiT 的核心启发更适合放在训练期的
“直接人工 GT 监督”和低维瓶颈设计中，不适合增加一个部署期去噪生成阶段。
