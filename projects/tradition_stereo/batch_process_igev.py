import numpy as np
import cv2
import open3d as o3d
import os
import glob
import shutil
import struct
from tqdm import tqdm

def save_pointmap_binary(pointmap, output_path):
    """
    将 pointmap 保存为自定义二进制格式，方便 C++ 读取

    文件格式：
    ========================================
    头部（64 bytes）：
    - 魔法数（4 bytes）: "PMAP" (ASCII)
    - 版本号（4 bytes, uint32）: 1
    - 高度（4 bytes, uint32）
    - 宽度（4 bytes, uint32）
    - 通道数（4 bytes, uint32）: 6
    - 数据类型（4 bytes, uint32）: 0 表示 float32
    - 通道顺序（24 bytes, ASCII）: "XYZRGB" + 填充
    - 保留字段（20 bytes）: 全0，用于未来扩展
    ========================================
    数据部分（H × W × 6 × 4 bytes）：
    - 按行优先顺序存储（row-major）
    - 每个像素包含 6 个 float32 值：[X, Y, Z, R, G, B]
    - 字节序：小端（little-endian），适配 x86 架构
    ========================================

    参数:
        pointmap: numpy array, shape=(H, W, 6), dtype=float32
        output_path: 输出文件路径
    """
    h, w, c = pointmap.shape
    assert c == 6, f"通道数必须为6，当前为{c}"

    with open(output_path, 'wb') as f:
        # 写入头部
        # 1. 魔法数 (4 bytes)
        f.write(b'PMAP')

        # 2. 版本号 (4 bytes, uint32, little-endian)
        f.write(struct.pack('<I', 1))

        # 3. 高度 (4 bytes, uint32)
        f.write(struct.pack('<I', h))

        # 4. 宽度 (4 bytes, uint32)
        f.write(struct.pack('<I', w))

        # 5. 通道数 (4 bytes, uint32)
        f.write(struct.pack('<I', c))

        # 6. 数据类型 (4 bytes, uint32): 0 = float32
        f.write(struct.pack('<I', 0))

        # 7. 通道顺序 (24 bytes, ASCII)
        channel_order = b'XYZRGB' + b'\x00' * 18  # 填充到 24 bytes
        f.write(channel_order)

        # 8. 保留字段 (20 bytes)
        f.write(b'\x00' * 20)

        # 头部总共 64 bytes

        # 写入数据部分（行优先顺序，小端字节序）
        # 确保数据是 float32 类型且使用小端字节序
        pointmap_float32 = pointmap.astype('<f4')  # little-endian float32
        f.write(pointmap_float32.tobytes())

def process_single_folder(input_folder, output_folder, left_map1, left_map2,
                         min_disparity=5.0, max_disparity=300.0,
                         black_threshold=15):
    """
    处理单个文件夹，生成点云并保存

    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        left_map1, left_map2: 左图校正映射表
        min_disparity: 最小视差值，过滤掉小于此值的点（默认5.0）
        max_disparity: 最大视差值，过滤掉大于此值的点（默认300.0）
        black_threshold: 黑色背景阈值，RGB三通道均小于此值则过滤（默认15）
    """
    folder_name = os.path.basename(input_folder)
    print(f"Processing: {folder_name}")

    # 检查必要文件是否存在
    disp_path = os.path.join(input_folder, "disp.npy")
    im0_path = os.path.join(input_folder, "im0.png")
    im1_path = os.path.join(input_folder, "im1.png")
    disp_igev_path = os.path.join(input_folder, "disp_igev.png")

    if not os.path.exists(disp_path) or not os.path.exists(im0_path):
        print(f"  ❌ 跳过 {folder_name}: 缺少必要文件 (disp.npy 或 im0.png)")
        return False

    try:
        # 1. 加载视差图和图像
        disparity = np.load(disp_path).astype(np.float32)
        imgL = cv2.imread(im0_path)

        # 2. 校正左图
        # imgL = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)

        # 3. 中心裁剪代码（参考save_IGEV.py）
        # 注意：disparity 和 imgL 尺寸相同（都是原始尺寸）
        h_orig, w_orig = disparity.shape[:2]

        print(f"  原始尺寸: {h_orig} × {w_orig}")

        # 原始裁剪参数计算（精确匹配）
        minDisparity = -104
        numDisparities = 208
        edge = abs(minDisparity) // 2
        edgeL = minDisparity + numDisparities
        start_x = edgeL
        start_y = edge // 2

        # 计算动态调整的长宽比裁剪
        # 基于原始图像尺寸计算ROI
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

        print(f"  裁剪区域: start_x={start_x}, start_y={start_y}, width={roi_width}, height={roi_height}")

        # 裁剪图像和视差图
        imgL = imgL[start_y:start_y + roi_height, start_x:start_x + roi_width]
        disparity = disparity[start_y:start_y + roi_height, start_x:start_x + roi_width]

        print(f"  裁剪后尺寸: {disparity.shape[0]} × {disparity.shape[1]}")

        # 4. 对视差图进行双边滤波，去除噪声同时保持边缘
        # 双边滤波参数说明:
        # - d=9: 滤波器直径(像素邻域大小)
        # - sigmaColor=75: 颜色空间的标准差,值越大,颜色差异越大的像素也会被混合
        # - sigmaSpace=75: 坐标空间的标准差,值越大,更远的像素也会相互影响
        #
        # 中值滤波 (已废弃,效果不佳):
        # disparity_filtered = cv2.medianBlur(disparity.astype(np.float32), 5)

        disparity_filtered = cv2.bilateralFilter(
            disparity.astype(np.float32),
            d=5,              # 滤波器直径 (可选: 5=轻度, 9=中度, 15=强力)  9
            sigmaColor=50,    # 颜色空间标准差 (可调: 50-150)   75
            sigmaSpace=50     # 坐标空间标准差 (可调: 50-150)   75
        )  

        print(f"  视差图双边滤波:")
        print(f"    滤波前范围: {disparity.min():.2f} ~ {disparity.max():.2f}")
        print(f"    滤波后范围: {disparity_filtered.min():.2f} ~ {disparity_filtered.max():.2f}")
        print(f"    滤波参数: d=9, sigmaColor=75, sigmaSpace=75")

        # 5. 3D重投影（使用JXP的Q矩阵，参考save_IGEV.py）
        # Q_jxp_original = np.array([[  1., 0., 0., -2.2227048492431641e+02],
        #                             [ 0., 1., 0.,-7.2185147857666016e+02],
        #                             [0., 0., 0., 9.4420949981369597e+02],
        #                             [0.,0., 3.9456987572407826e-01, 0.]])
        Q_jxp_original = np.array([[  1., 0., 0., -2.2227048492431641e+02],
                                    [ 0., 1., 0.,-7.2185147857666016e+02],
                                    [0., 0., 0., 9.4420949981369597e+02],
                                    [0.,0., 3.9456987572407826e-01, 0.]])

        # Q_jxp_original = np.array([ [  1., 0., 0., -2.5134127044677734e+02],
        #                             [ 0., 1., 0.,  -6.5667977905273438e+02],
        #                             [0., 0., 0., 8.8205398705187622e+02],
        #                             [0.,0., 3.8920665588077730e-01, 0.]])

        #########螺纹#########
        # Q_jxp_original = np.array([ [  1., 0., 0., -2.6481863403320312e+02],
        #                             [ 0., 1., 0.,  -7.1894045257568359e+02],
        #                             [0., 0., 0., 8.8171996613061117e+02],
        #                             [0.,0., 4.0724755146191277e-01, 0.]])


        # 调整Q矩阵以适应裁剪后的图像
        # 裁剪导致光心位置发生偏移，需要更新 cx 和 cy
        Q_jxp = Q_jxp_original.copy()
        Q_jxp[0, 3] += start_x  # cx' = cx - start_x，所以 Q[0,3] = -cx' = -cx + start_x
        Q_jxp[1, 3] += start_y  # cy' = cy - start_y，所以 Q[1,3] = -cy' = -cy + start_y

        print(f"  Q矩阵调整:")
        print(f"    裁剪偏移: start_x={start_x}, start_y={start_y}")
        print(f"    原始光心: cx={-Q_jxp_original[0,3]:.2f}, cy={-Q_jxp_original[1,3]:.2f}")
        print(f"    调整后光心: cx'={-Q_jxp[0,3]:.2f}, cy'={-Q_jxp[1,3]:.2f}")

        # 使用调整后的Q矩阵和滤波后的视差图进行3D重投影
        pts3d = cv2.reprojectImageTo3D(disparity_filtered, Q_jxp, handleMissingValues=True)

        # 6. 多重过滤条件
        X, Y, Z = pts3d[...,0], pts3d[...,1], pts3d[...,2]

        # 6.1 基础有效性过滤（视差>0 且 Z坐标有限且在合理范围内）
        # 注意：这里使用滤波后的视差图进行过滤
        mask_valid = (disparity_filtered > 0) & np.isfinite(Z) & (Z > 0) & (Z < 200)

        # 6.2 视差范围过滤
        # 过滤掉过小和过大的视差值
        mask_disparity = (disparity_filtered >= min_disparity) & (disparity_filtered <= max_disparity)

        # 6.3 黑色背景过滤（过滤纯黑或接近黑色的像素）
        # BGR格式，检查所有通道是否都小于阈值
        mask_not_black = (imgL[..., 0] > black_threshold) | \
                        (imgL[..., 1] > black_threshold) | \
                        (imgL[..., 2] > black_threshold)

        # 6.4 合并所有过滤条件
        mask = mask_valid & mask_disparity & mask_not_black

        print(f"  过滤统计:")
        print(f"    原始点数: {disparity.size:,}")
        print(f"    有效点数: {np.sum(mask_valid):,}")
        print(f"    视差过滤后: {np.sum(mask_valid & mask_disparity):,}")
        print(f"    黑色过滤后: {np.sum(mask):,}")
        print(f"    过滤率: {(1 - np.sum(mask)/disparity.size)*100:.2f}%")

        points = pts3d[mask]
        colors = imgL[mask] / 255.0

        # 转换BGR到RGB (OpenCV读取是BGR,Open3D需要RGB)
        colors = colors[:, ::-1]  # 反转最后一个维度: BGR -> RGB

        # 7. 构造 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 8. 创建输出目录并保存
        output_subfolder = os.path.join(output_folder, folder_name)
        os.makedirs(output_subfolder, exist_ok=True)

        # 保存点云
        ply_path = os.path.join(output_subfolder, "out.ply")
        o3d.io.write_point_cloud(ply_path, pcd)

        # 保存6通道pointmap（XYZ + RGB）为.npy格式
        # 将BGR转换为RGB
        imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

        # 构建6通道pointmap: (H, W, 6) -> [X, Y, Z, R, G, B]
        pointmap = np.zeros((imgL.shape[0], imgL.shape[1], 6), dtype=np.float32)

        # 将有效点（mask为True）的XYZ和RGB填入pointmap
        pointmap[mask, :3] = pts3d[mask]  # XYZ坐标
        pointmap[mask, 3:] = imgL_rgb[mask]  # RGB颜色

        # mask之外的点保持为0

        # 保存为 .npy 格式（Python 方便读取）
        pointmap_path = os.path.join(output_subfolder, "pointmap.npy")
        np.save(pointmap_path, pointmap)

        # 保存为自定义二进制格式（C++ 方便读取）
        pointmap_bin_path = os.path.join(output_subfolder, "pointmap.bin")
        save_pointmap_binary(pointmap, pointmap_bin_path)

        # 保存裁剪后的视差图（用于后续分析 Z 和 1/d 的关系）
        disp_cropped_path = os.path.join(output_subfolder, "disp.npy")
        np.save(disp_cropped_path, disparity_filtered)

        print(f"  Pointmap保存:")
        print(f"    .npy路径: {pointmap_path}")
        print(f"    .bin路径: {pointmap_bin_path}")
        print(f"    disp.npy路径: {disp_cropped_path}")
        print(f"    形状: {pointmap.shape} -> (H, W, 6) [X, Y, Z, R, G, B]")
        print(f"    有效点数: {np.sum(mask):,} / {mask.size:,}")
        print(f"    有效率: {np.sum(mask)/mask.size*100:.2f}%")
        valid_xyz = pointmap[mask, :3]
        print(f"    X范围: {valid_xyz[:, 0].min():.2f} ~ {valid_xyz[:, 0].max():.2f}")
        print(f"    Y范围: {valid_xyz[:, 1].min():.2f} ~ {valid_xyz[:, 1].max():.2f}")
        print(f"    Z范围: {valid_xyz[:, 2].min():.2f} ~ {valid_xyz[:, 2].max():.2f}")
        print(f"    RGB范围: [0-255] (无效点为0)")

        # 9. 复制原始图像文件
        # 复制原始im0.png（覆盖校正后的版本）
        if os.path.exists(im0_path):
            shutil.copy2(im0_path, os.path.join(output_subfolder, "im0.png"))

        # 复制im1.png
        if os.path.exists(im1_path):
            shutil.copy2(im1_path, os.path.join(output_subfolder, "im1.png"))

        # 复制并重命名disp_igev.png为disp.png
        if os.path.exists(disp_igev_path):
            shutil.copy2(disp_igev_path, os.path.join(output_subfolder, "disp.png"))
        else:
            # 如果没有disp_igev.png，则保存处理后的视差图
            disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disp_path_out = os.path.join(output_subfolder, "disp.png")
            cv2.imwrite(disp_path_out, disp_norm)

        print(f"  ✅ 完成 {folder_name}")
        return True

    except Exception as e:
        print(f"  ❌ 处理 {folder_name} 时出错: {str(e)}")
        return False

def main():
    # ==================== 配置参数 ====================
    # 路径配置
    # input_root = r"D:\Desktop\scene_output_luowen"
    # output_root = r"D:\Desktop\luowem_ply点云"
    # input_root = r"D:\Desktop\指标计算\scene_demo-imgs"
    # output_root = r"D:\Desktop\test_1230"
    input_root = r"D:\Desktop\scene_demo-imgs_last"
    output_root = r"D:\Desktop\scene_demo-imgs_last_ply"

    # 过滤参数配置
    MIN_DISPARITY = 5.0        # 最小视差值，过滤掉过小的视差（通常是暗纹理导致的异常点）
    MAX_DISPARITY = 300.0      # 最大视差值，过滤掉过大的视差
    BLACK_THRESHOLD = 50       # 黑色背景阈值 (0-255)，RGB三通道均小于此值则视为黑色背景 50
    # ================================================

    # 加载校正映射表（使用gongjian配置）
    left_map1_path = "./config/map/gongjian_map/left_map1.npy"
    left_map2_path = "./config/map/gongjian_map/left_map2.npy"

    if not os.path.exists(left_map1_path) or not os.path.exists(left_map2_path):
        print("❌ 错误: 找不到校正映射表文件")
        print(f"   请检查: {left_map1_path}")
        print(f"   请检查: {left_map2_path}")
        return

    left_map1 = np.load(left_map1_path)
    left_map2 = np.load(left_map2_path)

    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)

    # 获取所有子文件夹
    input_folders = [f for f in glob.glob(os.path.join(input_root, "*"))
                    if os.path.isdir(f)]

    if not input_folders:
        print(f"❌ 在 {input_root} 中未找到任何文件夹")
        return

    print(f"找到 {len(input_folders)} 个文件夹待处理")
    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"\n过滤参数:")
    print(f"  视差范围: {MIN_DISPARITY} ~ {MAX_DISPARITY}")
    print(f"  黑色阈值: {BLACK_THRESHOLD}")
    print("-" * 50)

    # 批量处理
    success_count = 0
    total_count = len(input_folders)

    for input_folder in tqdm(input_folders, desc="处理进度"):
        if process_single_folder(input_folder, output_root, left_map1, left_map2,
                                min_disparity=MIN_DISPARITY,
                                max_disparity=MAX_DISPARITY,
                                black_threshold=BLACK_THRESHOLD):
            success_count += 1

    print("-" * 50)
    print(f"处理完成! 成功: {success_count}/{total_count}")

    if success_count < total_count:
        print(f"失败: {total_count - success_count} 个文件夹")

if __name__ == "__main__":
    main()