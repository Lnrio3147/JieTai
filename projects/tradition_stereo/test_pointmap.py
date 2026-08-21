import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import cv2

def crop_disparity(disparity):
    """
    对视差图进行裁剪，参考 batch_process_igev.py 中的裁剪逻辑

    参数:
        disparity: 原始视差图 (H, W)

    返回:
        裁剪后的视差图
    """
    h_orig, w_orig = disparity.shape[:2]

    # 原始裁剪参数计算（精确匹配 batch_process_igev.py）
    minDisparity = -104
    numDisparities = 208
    edge = abs(minDisparity) // 2
    edgeL = minDisparity + numDisparities
    start_x = edgeL
    start_y = edge // 2

    # 计算动态调整的长宽比裁剪
    roi_width = w_orig - 2 * edgeL
    roi_height = h_orig - edge

    k = roi_width / roi_height
    if k > 1.8:
        h_new = (roi_height * 16 // 10) // 2 * 2
        offset = (roi_width - h_new) // 4 * 2
        start_x += offset
        roi_width = h_new
    elif 1 / k > 1.8:
        h_new = (roi_width * 16 // 10) // 2 * 2
        offset = (roi_height - h_new) // 4 * 2
        start_y += offset
        roi_height = h_new

    # 裁剪视差图
    disparity_cropped = disparity[start_y:start_y + roi_height, start_x:start_x + roi_width]

    print(f"  视差图裁剪:")
    print(f"    原始尺寸: {h_orig} × {w_orig}")
    print(f"    裁剪区域: start_x={start_x}, start_y={start_y}, width={roi_width}, height={roi_height}")
    print(f"    裁剪后尺寸: {disparity_cropped.shape[0]} × {disparity_cropped.shape[1]}")

    return disparity_cropped

def load_and_visualize_pointmap(pointmap_path):
    """
    从 pointmap.npy 加载并可视化点云

    参数:
        pointmap_path: pointmap.npy 文件路径
    """
    print(f"加载 pointmap: {pointmap_path}")

    # 1. 加载 pointmap
    pointmap = np.load(pointmap_path)
    print(f"Pointmap 形状: {pointmap.shape}")

    # 2. 提取有效点（Z > 0 表示有效点）
    mask = pointmap[..., 2] > 0
    valid_count = np.sum(mask)
    total_count = mask.size

    print(f"有效点数: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.2f}%)")

    # 3. 获取坐标和颜色
    points = pointmap[mask, :3]  # XYZ 坐标
    colors = pointmap[mask, 3:]  # RGB 颜色 (0-255)

    # 归一化颜色到 [0, 1]
    colors = colors / 255.0

    print(f"点云统计:")
    print(f"  X 范围: {points[:, 0].min():.2f} ~ {points[:, 0].max():.2f}")
    print(f"  Y 范围: {points[:, 1].min():.2f} ~ {points[:, 1].max():.2f}")
    print(f"  Z 范围: {points[:, 2].min():.2f} ~ {points[:, 2].max():.2f}")

    # 4. 构建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 5. 可视化
    print("\n启动点云可视化...")
    print("操作提示:")
    print("  - 鼠标左键拖动: 旋转")
    print("  - 鼠标右键拖动: 平移")
    print("  - 滚轮: 缩放")
    print("  - Q 键: 退出")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Pointmap 可视化",
        width=1280,
        height=720,
        left=100,
        top=100
    )

def analyze_depth_disparity_relationship(pointmap_path, disparity_path):
    """
    分析深度Z和视差倒数1/d的关系，验证 Z 是否正比于 1/d

    理论关系：Z = f * B / d
    其中 f 是焦距，B 是基线，d 是视差
    因此 Z 应该和 1/d 成正比

    参数:
        pointmap_path: pointmap.npy 文件路径
        disparity_path: disp.npy 文件路径（裁剪前的原始视差图）
    """
    print(f"\n{'='*60}")
    print("分析深度与视差的关系")
    print(f"{'='*60}")

    # 1. 加载 pointmap
    print(f"加载 pointmap: {pointmap_path}")
    pointmap = np.load(pointmap_path)
    Z = pointmap[..., 2]  # 深度值
    print(f"  Pointmap 形状: {pointmap.shape}")

    # 2. 加载视差图
    # 优先使用提供的路径，否则从同一文件夹自动查找
    if disparity_path and os.path.exists(disparity_path):
        disp_npy_path = disparity_path
        print(f"加载视差图: {disp_npy_path}")
    else:
        # 尝试从同一文件夹加载视差图
        folder = os.path.dirname(pointmap_path)
        disp_npy_path = os.path.join(folder, "disp.npy")
        print(f"加载视差图: 自动查找")

        if not os.path.exists(disp_npy_path):
            print(f"  错误: 未找到 disp.npy")
            print(f"  查找路径: {disp_npy_path}")
            print(f"\n请提供原始的 disp.npy 文件路径")
            return

    print(f"  找到 disp.npy: {disp_npy_path}")
    disparity = np.load(disp_npy_path).astype(np.float32)

    print(f"  视差图形状: {disparity.shape}")

    # 3. 检查尺寸是否匹配
    if disparity.shape[:2] != Z.shape[:2]:
        print(f"\n警告: 尺寸不匹配！")
        print(f"  Pointmap (Z): {Z.shape}")
        print(f"  Disparity: {disparity.shape}")
        print(f"\n检测到 pointmap 是裁剪后的，但 disp.npy 是原始尺寸")
        print(f"正在自动裁剪视差图以匹配 pointmap...")

        # 自动裁剪视差图
        disparity = crop_disparity(disparity)

        # 再次检查尺寸
        if disparity.shape[:2] != Z.shape[:2]:
            print(f"\n错误: 裁剪后尺寸仍然不匹配！")
            print(f"  Pointmap (Z): {Z.shape}")
            print(f"  Disparity (裁剪后): {disparity.shape}")
            return

        print(f"✅ 裁剪成功！尺寸已匹配: {disparity.shape}")

    # 4. 对视差图进行双边滤波（与 batch_process_igev.py 保持一致）
    print(f"\n对视差图进行双边滤波（与生成 pointmap 时保持一致）...")
    print(f"  滤波前范围: {disparity[disparity > 0].min():.2f} ~ {disparity[disparity > 0].max():.2f}")

    disparity_filtered = cv2.bilateralFilter(
        disparity.astype(np.float32),
        d=5,              # 滤波器直径（与 batch_process_igev.py 一致）
        sigmaColor=50,    # 颜色空间标准差
        sigmaSpace=50     # 坐标空间标准差
    )

    print(f"  滤波后范围: {disparity_filtered[disparity_filtered > 0].min():.2f} ~ {disparity_filtered[disparity_filtered > 0].max():.2f}")
    print(f"  滤波参数: d=5, sigmaColor=50, sigmaSpace=50")

    # 使用滤波后的视差图进行后续分析
    disparity = disparity_filtered

    # 5. 提取有效点（Z > 0 且视差 > 0）
    mask = (Z > 0) & (disparity > 0) & np.isfinite(Z) & np.isfinite(disparity)
    valid_count = np.sum(mask)

    print(f"\n有效点数: {valid_count:,} / {mask.size:,} ({valid_count/mask.size*100:.2f}%)")

    if valid_count == 0:
        print("错误: 没有有效点，无法分析")
        return

    # 6. 提取有效的深度和视差
    Z_valid = Z[mask]
    d_valid = disparity[mask]
    inv_d_valid = 1.0 / d_valid  # 视差的倒数

    # 7. 计算统计信息
    print(f"\n统计信息:")
    print(f"  深度 Z 范围: {Z_valid.min():.2f} ~ {Z_valid.max():.2f}")
    print(f"  视差 d 范围: {d_valid.min():.2f} ~ {d_valid.max():.2f}")
    print(f"  视差倒数 1/d 范围: {inv_d_valid.min():.6f} ~ {inv_d_valid.max():.6f}")

    # 8. 线性拟合 Z = k * (1/d) + b
    # 理论上 b 应该接近 0，k = f * B
    coeffs = np.polyfit(inv_d_valid, Z_valid, 1)
    k, b = coeffs
    print(f"\n线性拟合: Z = k * (1/d) + b")
    print(f"  斜率 k (f*B): {k:.2f}")
    print(f"  截距 b: {b:.2f}")

    # 9. 计算拟合优度 (R²)
    Z_pred = k * inv_d_valid + b
    ss_res = np.sum((Z_valid - Z_pred) ** 2)
    ss_tot = np.sum((Z_valid - np.mean(Z_valid)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  R² (拟合优度): {r_squared:.6f}")

    if r_squared > 0.99:
        print(f"  ✅ 优秀！Z 和 1/d 高度线性相关")
    elif r_squared > 0.95:
        print(f"  ✅ 良好！Z 和 1/d 线性相关")
    elif r_squared > 0.90:
        print(f"  ⚠️  一般，存在一定偏差")
    else:
        print(f"  ❌ 较差，线性关系不明显")

    # 10. 绘制散点图和拟合线
    print(f"\n正在生成可视化图表...")
    print(f"  绘制所有 {valid_count:,} 个有效点（可能需要一些时间）...")

    # 使用所有有效点绘图（不采样）
    inv_d_sample = inv_d_valid
    Z_sample = Z_valid
    d_sample = d_valid

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('深度 Z 与视差 d 的关系分析', fontsize=16, fontproperties='SimHei')

    # 子图1: Z vs 1/d (散点图 + 拟合线)
    ax1 = axes[0, 0]
    ax1.scatter(inv_d_sample, Z_sample, alpha=0.1, s=0.5, c='blue', label=f'数据点 (N={valid_count:,})')

    # 绘制拟合线
    inv_d_range = np.linspace(inv_d_valid.min(), inv_d_valid.max(), 100)
    Z_fit = k * inv_d_range + b
    ax1.plot(inv_d_range, Z_fit, 'r-', linewidth=2, label=f'拟合线: Z = {k:.2f} * (1/d) + {b:.2f}')

    ax1.set_xlabel('视差倒数 1/d (1/pixel)', fontproperties='SimHei')
    ax1.set_ylabel('深度 Z (mm)', fontproperties='SimHei')
    ax1.set_title(f'深度 vs 视差倒数 (R² = {r_squared:.6f})', fontproperties='SimHei')
    ax1.legend(prop={'family': 'SimHei'})
    ax1.grid(True, alpha=0.3)

    # 子图2: 残差分布
    ax2 = axes[0, 1]
    residuals = Z_valid - (k * inv_d_valid + b)
    ax2.hist(residuals, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('残差 (mm)', fontproperties='SimHei')
    ax2.set_ylabel('频数', fontproperties='SimHei')
    ax2.set_title(f'残差分布 (均值={residuals.mean():.2f}, 标准差={residuals.std():.2f})', fontproperties='SimHei')
    ax2.grid(True, alpha=0.3)

    # 子图3: Z vs d (直接关系)
    ax3 = axes[1, 0]
    ax3.scatter(d_sample, Z_sample, alpha=0.1, s=0.5, c='orange')
    ax3.set_xlabel('视差 d (pixel)', fontproperties='SimHei')
    ax3.set_ylabel('深度 Z (mm)', fontproperties='SimHei')
    ax3.set_title('深度 vs 视差 (应为双曲线)', fontproperties='SimHei')
    ax3.grid(True, alpha=0.3)

    # 子图4: 相对误差分布
    ax4 = axes[1, 1]
    relative_error = (residuals / Z_valid) * 100  # 百分比
    ax4.hist(relative_error, bins=100, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('相对误差 (%)', fontproperties='SimHei')
    ax4.set_ylabel('频数', fontproperties='SimHei')
    ax4.set_title(f'相对误差分布 (均值={relative_error.mean():.2f}%)', fontproperties='SimHei')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(os.path.dirname(pointmap_path), "depth_disparity_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")

    plt.show()

    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 指定数据路径
    folder_path = r"D:\Desktop\原始ply点云\202506281603-0002"
    pointmap_path = os.path.join(folder_path, "pointmap.npy")
    disparity_path = os.path.join(folder_path, "disp.npy")

    print("=" * 60)
    print("开始分析深度与视差的关系")
    print("=" * 60)
    print(f"数据文件夹: {folder_path}")
    print(f"Pointmap: {pointmap_path}")
    print(f"Disparity: {disparity_path}")
    print("=" * 60)

    # 分析深度与视差的关系
    # 注意：如果同一文件夹有 disp.npy，会自动加载
    # 如果尺寸不匹配，会自动裁剪原始视差图
    analyze_depth_disparity_relationship(pointmap_path, disparity_path)

    # 可视化点云（可选）
    # load_and_visualize_pointmap(pointmap_path)
