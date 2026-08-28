# 实验 11：基于反馈的多视角三维融合重建

本实验实现了完整的粗配准、精配准、位姿图优化、带观测次数的体素融合、质量评价和
下一最佳视角建议。输入是每个视角已经去除背景的主体点云；实验 10 可作为它的上游。

```text
主体点云序列
  -> FPFH + RANSAC（无位姿先验时）
  -> 多尺度 GICP
  -> 边质量门控
  -> 全局位姿图优化
  -> 多视角体素融合与观测次数统计
  -> 低支持区域识别
  -> 质量/停止条件 + 下一视角建议
```

## 1. 准备序列

复制 [sequence_manifest.example.json](sequence_manifest.example.json)，至少填写两个
视角的主体点云路径。所有点云单位必须一致，`voxel_size` 也使用同一单位。

`pose_camera_to_world` 约定为把相机坐标点变换到世界坐标的 4×4 刚体矩阵。
有机器人/转台位姿时，程序用它初始化并归一到第一个相机坐标系；没有位姿时使用
FPFH + RANSAC 粗配准。候选视角需要显式位姿；未提供候选时，程序会围绕当前融合体
自动生成球面候选。

`reference_cloud` 的坐标约定也必须明确：全部视角有位姿时它应处在同一个 manifest
世界坐标系，程序会自动归一化；视角均无位姿时它必须已经处在第一相机坐标系。

真实采集需要满足：

- 同一静止工件，至少 3 个有重叠区域的视角，建议相邻方位差 `20°–40°`；
- 每个视角使用同一标定和尺度生成主体点云；
- 尽量记录机器人/转台位姿，它能降低局部对称结构的错误配准风险；
- 若要报告绝对精度，另采独立高精度扫描或标准件点云作为 `reference_cloud`。

当前 `datasets/` 中没有一组同时满足“同一静止工件、多视角主体点云、可靠相机位姿/
独立参考”的序列，因此本次没有把不同场景硬拼成所谓真实重建结果。

可由有序点云列表生成带采集声明的 manifest，并先执行资格预检：

```bash
python prepare_sequence_manifest.py \
  --sequence-id workpiece_001 --units mm --voxel-size 1.0 \
  --same-static-object-confirmed --common-calibration-confirmed \
  --calibration-id robot_cell_stereo_v1 \
  --cloud /path/view_000/subject_cloud.ply \
  --cloud /path/view_001/subject_cloud.ply \
  --output inputs/workpiece_001.json
python preflight_sequence.py \
  --manifest inputs/workpiece_001.json \
  --output results/workpiece_001_preflight.json
```

预检会检查静止对象/共同标定/单位声明、有限点比例、鲁棒尺度、点间距、顺序边配准和
视角多样性，并分别输出 `reconstruction_ready` 与 `coverage_claim_ready`。没有通过时
不能把后续探索性融合称作有效多视角重建。

## 2. 运行重建

```bash
cd JieTai/experiments/11_multiview_feedback_reconstruction
python reconstruct.py \
  --manifest /path/to/sequence_manifest.json \
  --preflight-report /path/to/preflight.json \
  --output results/static_workpiece_001
```

[run_experiment11.sh](run_experiment11.sh) 会自动先运行预检，并把报告传给重建程序；
若有 blocker，重建会在创建输出目录前停止。

主要参数在 [config_experiment11.py](config_experiment11.py)：默认配准边需满足
`fitness >= 0.20` 且 `inlier_rmse <= 2 * voxel_size`；连续两个视角新增覆盖率低于
`1%` 时建议停止采集。

## 3. 输出

| 文件 | 内容 |
|---|---|
| `fused_cloud.ply` | 优化后融合点云 |
| `low_support_cloud.ply` | 少于 2 个独立视角支持的红色区域 |
| `voxel_support.npy` | 每个融合体素的独立视角观测次数 |
| `optimized_poses.json` | 以第一相机为全局系的优化位姿 |
| `registration_edges.csv/json` | 每条边的 fitness、RMSE、接受/拒绝原因 |
| `quality.json` | 覆盖率、增量覆盖、参考点云精度和停止条件 |
| `next_view.json` | 下一视角得分与建议位姿 |

下一视角评分综合低支持区域可见比例、与已有视角的角度新颖性和运动代价。它当前只做
几何视锥近似，不包含机器人碰撞、关节限位或工件自遮挡验证；下发机器人前必须经过
真实运动规划器检查。

## 4. 可复现合成回归

先生成带已知真值位姿和参考表面的非对称工件，再运行完整链路：

```bash
python generate_synthetic_sequence.py --output inputs/synthetic_sequence
python reconstruct.py \
  --manifest inputs/synthetic_sequence/manifest.json \
  --output results/synthetic_regression_v2
```

5 视角回归结果：5/5 条配准边通过，平均 fitness `0.9631`，平均 RMSE
`0.1250 mm`；融合得到 7,654 个体素，其中 `90.76%` 至少由两个视角支持，
在 `1.6 mm` 阈值下相对参考表面的 precision/recall/F-score 均为 `1.0`。
该结果证明软件闭环与坐标变换可运行，不代表真实传感器精度。

还可以隐藏真值位姿，单独回归 FPFH/RANSAC 粗配准入口：

```bash
python generate_synthetic_sequence.py --views 3 --omit-poses \
  --output inputs/synthetic_sequence_no_pose
python reconstruct.py \
  --manifest inputs/synthetic_sequence_no_pose/manifest.json \
  --output results/synthetic_no_pose
```

该无位姿回归的 2/2 条顺序边均由 FPFH/RANSAC 初始化后通过 GICP 门控，平均
fitness `0.9635`、RMSE `0.1252 mm`，参考 F-score 同样为 `1.0`。这验证了无机器人
位姿的软件分支，但真实重复/对称工件仍应优先提供位姿先验。

## 5. 现有真实点云审计

对实验 5 的 18 个 FDJYP-0 主体点云做了固定相机重叠审计：连续对中只有
`202506261659-0014` 与 `202506261700-0015` 的初始重叠明显较高。两帧 GICP 边可
通过（fitness `0.4841`、RMSE `0.1520`，单位未确认），但融合后只有 `35.52%`
体素被两帧共同支持，第二帧新增体素高达 `55.72%`。预检正确给出三个限制：

- 无法从历史目录确认是同一静止工件；
- 点云长度单位没有元数据确认；
- 两帧都是固定相机，没有多视角多样性。

因此该结果仅保留为真实数据稳定性负试验，不能报告为多视角覆盖提升。Jop1 九场从
原图可确认是不同工件，同样不进入融合实验。

运行测试：

```bash
python -m unittest discover -s tests -v
```

真实实验完成条件是：配准边通过率、至少两视角支持率、新增覆盖率曲线、参考点云
双向距离/F-score（若有真值）均随原始 manifest 一起归档；不能只凭融合点云“看起来
完整”判断成功。
