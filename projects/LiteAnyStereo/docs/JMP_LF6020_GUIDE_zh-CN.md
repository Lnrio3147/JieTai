# JMP-LF6020 数据整理与接入说明

## 数据来源与最终目录

原始数据保留在仓库根目录的 `JMP-LF6020.zip`，转换过程不会修改该压缩包。训练和测试只读取
整理后的唯一目录：

```text
data/datasets/JMP-LF6020-ETH3D/
├── <scene>/
│   ├── im0.png          # 左校正图
│   ├── im1.png          # 右校正图
│   ├── disp0GT.pfm      # 点云投影得到的 float32 伪视差
│   ├── mask0nocc.png    # 有效伪视差掩码
│   └── calib.txt        # ETH3D 兼容相机参数
├── splits/train.txt     # 193 场
├── splits/val.txt       # 73 场
├── manifest.csv
└── metadata/
```

共 266 个有效双目场景。`disp0GT.pfm` 的命名只为兼容 ETH3D 接口，并不代表人工真值。

## 转换过程

转换脚本是 `tools/prepare_jmp_lf6020.py`，主要步骤如下：

1. 读取 ZIP 中的左右图、相机内外参和增强点云；
2. 使用立体标定参数进行极线校正，得到水平对应的 `im0.png`、`im1.png`；
3. 将增强点云反投影到左相机，按近处优先处理投影碰撞，生成像素视差；
4. 将视差保存为 PFM，将有效像素保存为单通道 PNG 掩码；
5. 按采集组划分 193 个训练场景和 73 个验证场景，并输出清单、校验报告和重命名映射。

重新生成命令：

```bash
python tools/prepare_jmp_lf6020.py \
  --archive ./JMP-LF6020.zip \
  --output ./data/datasets/JMP-LF6020-ETH3D
```

转换后执行完整校验：

```bash
python tools/check_stereo_dataset.py \
  --manifest ./data/datasets/JMP-LF6020-ETH3D/manifest.csv \
  --max_disp 192 \
  --report ./data/datasets/JMP-LF6020-ETH3D/metadata/dataset_check.json
```

当前目录已验证 266/266 场可读取，左右图、视差与掩码尺寸一致。

## 为什么 mask 看起来像空白

`mask0nocc.png` 不是工件区域掩码，而是“增强点云可投影为有效伪视差”的位置。FDJYP 的点云
通常只覆盖图像中间一块固定区域，外围为 0；普通图片查看器缩小时会显得近似全黑或全白。判断它
是否有效应使用 `dataset_check.json` 的 `valid_pixels`，而不是凭缩略图观察。

## 训练标签与正式评价的区别

训练使用 `JMP-LF6020-ETH3D` 中由点云产生的伪标签。正式评价则使用
`tradition_stereo/datasets/FDJYP-3/<scene>/disp_cropped.npy` 的 Foundation Stereo 参考：

- 输入：当前目录内的校正后完整图像，尺寸 1280×720；
- 评价 ROI：`[234:1052, 126:638]`，尺寸 818×512；
- 有效像素：参考视差有限且大于 0；
- 汇总：逐场景计算后做宏平均；
- 固定排除 4 个异常场景；正式对比关闭按模型独立的 EPE 过滤。

因此，不能将 tradition_stereo 中未经正确校正的历史 `im0/im1` 直接输入 LiteAnyStereo；
垂直极线残差会使结果失真。详细训练和测试命令见 [使用说明](./USAGE_zh-CN.md)。
