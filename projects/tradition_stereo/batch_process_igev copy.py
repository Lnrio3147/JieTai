import numpy as np
import cv2
import open3d as o3d
import os
import glob
import shutil
from tqdm import tqdm

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
        h, w = disparity.shape[:2]

        # 原始裁剪参数计算（精确匹配）
        minDisparity = -104
        numDisparities = 208
        edge = abs(minDisparity) // 2
        edgeL = minDisparity + numDisparities
        start_x = edgeL
        start_y = edge // 2

        # 计算动态调整的长宽比裁剪
        disp = imgL
        roi_width = disp.shape[1] - 2 * edgeL
        roi_height = disp.shape[0] - edge

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

        # 裁剪图像和视差图
        imgL = imgL[start_y:start_y + roi_height, start_x:start_x + roi_width]
        disparity = disparity[start_y:start_y + roi_height, start_x:start_x + roi_width]

        # 4. 使用基于梯度的异常检测去除脉冲噪声
        # 原理：噪声斑点的视差值与周围区域差异很大，通过检测局部梯度异常来识别

        # 4.1 计算视差梯度（Sobel算子）
        # 归一化视差到0-1范围，避免大数值影响
        disp_norm = disparity / (np.max(disparity) + 1e-6)

        # 计算X和Y方向的梯度
        grad_x = cv2.Sobel(disp_norm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disp_norm, cv2.CV_32F, 0, 1, ksize=3)

        # 计算梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 4.2 识别异常梯度区域
        # 使用自适应阈值：基于梯度的统计分布
        valid_disp_mask = disparity > 0
        valid_gradients = gradient_magnitude[valid_disp_mask]

        if len(valid_gradients) > 0:
            # 计算梯度的均值和标准差
            grad_mean = np.mean(valid_gradients)
            grad_std = np.std(valid_gradients)

            # 设置阈值为均值 + 2倍标准差（可调整）
            gradient_threshold = grad_mean + 2.5 * grad_std

            # 创建异常梯度掩码（True表示正常，False表示异常）
            normal_gradient_mask = gradient_magnitude <= gradient_threshold

            print(f"  梯度分析:")
            print(f"    梯度均值: {grad_mean:.4f}")
            print(f"    梯度标准差: {grad_std:.4f}")
            print(f"    梯度阈值: {gradient_threshold:.4f}")
            print(f"    异常点数: {np.sum(~normal_gradient_mask):,}")
        else:
            normal_gradient_mask = np.ones_like(disparity, dtype=bool)

        # 4.3 对异常区域进行形态学处理，扩展异常区域以完全去除噪声斑点
        abnormal_mask = (~normal_gradient_mask).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        abnormal_mask_dilated = cv2.dilate(abnormal_mask, kernel, iterations=2)

        # 4.4 在膨胀后的异常区域中，使用连通组件分析只删除小的噪声块
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            abnormal_mask_dilated, connectivity=8)

        # 创建最终的噪声掩码（要删除的区域）
        noise_mask = np.zeros_like(abnormal_mask_dilated)
        min_noise_area = 50   # 只删除面积小于此值的异常区域   10
        max_noise_area = 400  # 只删除面积小于此值的异常区域（防止误删大块物体边缘） 500
        removed_patches = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_noise_area <= area <= max_noise_area:
                noise_mask[labels == i] = 1
                removed_patches += 1

        # 4.5 应用噪声掩码到视差图
        disparity_filtered = disparity.copy()
        disparity_filtered[noise_mask > 0] = 0

        print(f"  噪声过滤:")
        print(f"    检测到异常区域数: {num_labels - 1}")
        print(f"    删除噪声斑点数: {removed_patches}")
        print(f"    过滤前有效点: {np.sum(disparity > 0):,}")
        print(f"    过滤后有效点: {np.sum(disparity_filtered > 0):,}")
        print(f"    删除点数: {np.sum(noise_mask > 0):,}")

        # 5. 3D重投影（使用JXP的Q矩阵，参考save_IGEV.py）
        Q_jxp = np.array([[  1., 0., 0., -2.2227048492431641e+02],
                          [ 0., 1., 0.,-7.2185147857666016e+02],
                          [0., 0., 0., 9.4420949981369597e+02],
                          [0.,0., 3.9456987572407826e-01, 0.]])

        # 使用滤波后的视差图进行3D重投影
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
    input_root = r"D:\Desktop\rknn_scene_output"
    output_root = r"D:\Desktop\原始ply点云"

    # 过滤参数配置
    MIN_DISPARITY = 5.0        # 最小视差值，过滤掉过小的视差（通常是暗纹理导致的异常点）
    MAX_DISPARITY = 300.0      # 最大视差值，过滤掉过大的视差
    BLACK_THRESHOLD = 50       # 黑色背景阈值 (0-255)，RGB三通道均小于此值则视为黑色背景
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