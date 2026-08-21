"""
从PLY点云反投影生成视差图

原理：
    正向: disparity + Q -> 3D点云
        Z = f * B / d
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f

    反向: 3D点云 + Q -> disparity
        u = X * f / Z + cx
        v = Y * f / Z + cy
        d = f * B / Z = f / (Z * invB)
"""

import numpy as np
import cv2
import open3d as o3d
import os
import glob
from tqdm import tqdm


def ply_to_disparity(ply_path, Q_matrix, output_shape=None):
    """
    从PLY点云反投影生成视差图

    参数:
        ply_path: PLY点云文件路径
        Q_matrix: 4x4重投影矩阵 (应该是裁剪调整后的Q矩阵)
        output_shape: 输出视差图尺寸 (height, width)，如果为None则自动推断

    返回:
        disparity: 视差图 (H, W), float32
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        print(f"警告: {ply_path} 中没有点")
        return None

    # 从Q矩阵提取参数
    # Q矩阵格式:
    # [ 1  0  0  -cx ]
    # [ 0  1  0  -cy ]
    # [ 0  0  0   f  ]
    # [ 0  0 1/B  0  ]
    cx = -Q_matrix[0, 3]
    cy = -Q_matrix[1, 3]
    f = Q_matrix[2, 3]       # 焦距
    invB = Q_matrix[3, 2]    # 1/baseline

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    # 过滤无效点 (Z <= 0)
    valid = Z > 0
    X, Y, Z = X[valid], Y[valid], Z[valid]

    # 反投影到像素坐标
    u = np.round(X * f / Z + cx).astype(np.int32)
    v = np.round(Y * f / Z + cy).astype(np.int32)

    # 计算视差值: d = f * B / Z = f / (Z * invB)
    d = f / (Z * invB)

    # 确定输出尺寸
    if output_shape is None:
        # 自动推断：取最大坐标+边距
        h = int(np.max(v)) + 10
        w = int(np.max(u)) + 10
    else:
        h, w = output_shape

    # 过滤超出范围的点
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, d = u[mask], v[mask], d[mask]

    # 创建视差图
    disparity = np.zeros((h, w), dtype=np.float32)

    # 处理遮挡：多个点投影到同一像素时，保留视差最大的（距离最近的）
    # 方法：先按视差从小到大排序，后写入的会覆盖先写入的
    sort_idx = np.argsort(d)
    u, v, d = u[sort_idx], v[sort_idx], d[sort_idx]
    disparity[v, u] = d

    return disparity


def process_single_folder(input_folder, Q_original, output_disp=True):
    """
    处理单个文件夹，从out.ply生成视差图

    参数:
        input_folder: 输入文件夹路径 (包含out.ply)
        Q_original: 原始Q矩阵（未裁剪调整的）
        output_disp: 是否保存视差图

    返回:
        disparity: 视差图
    """
    folder_name = os.path.basename(input_folder)
    ply_path = os.path.join(input_folder, "out.ply")

    if not os.path.exists(ply_path):
        print(f"  跳过 {folder_name}: 找不到 out.ply")
        return None

    # 读取原始图像获取尺寸（如果有的话）
    im0_path = os.path.join(input_folder, "im0.png")
    if os.path.exists(im0_path):
        img = cv2.imread(im0_path)
        h_orig, w_orig = img.shape[:2]
    else:
        # 默认使用1280x720
        h_orig, w_orig = 1280, 720

    # ========== 重要：计算裁剪参数（与batch_process_igev.py一致）==========
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

    # 裁剪后的图像尺寸
    crop_h, crop_w = roi_height, roi_width

    # 调整Q矩阵（与batch_process_igev.py一致）
    Q_adjusted = Q_original.copy()
    Q_adjusted[0, 3] += start_x  # 调整cx
    Q_adjusted[1, 3] += start_y  # 调整cy

    print(f"  {folder_name}: 裁剪后尺寸 {crop_h}x{crop_w}, 偏移 ({start_x}, {start_y})")

    # 反投影生成视差图
    disparity = ply_to_disparity(ply_path, Q_adjusted, output_shape=(crop_h, crop_w))

    if disparity is None:
        return None

    # 统计信息
    valid_mask = disparity > 0
    valid_count = np.sum(valid_mask)
    if valid_count > 0:
        print(f"    有效像素: {valid_count:,} ({100*valid_count/(crop_h*crop_w):.1f}%)")
        print(f"    视差范围: {disparity[valid_mask].min():.2f} ~ {disparity[valid_mask].max():.2f}")

    # 保存视差图
    if output_disp:
        # 保存float32格式
        disp_npy_path = os.path.join(input_folder, "disp_from_ply.npy")
        np.save(disp_npy_path, disparity)

        # 保存可视化图像
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        disp_png_path = os.path.join(input_folder, "disp_from_ply.png")
        cv2.imwrite(disp_png_path, disp_color)

        print(f"    保存: {disp_npy_path}")
        print(f"    保存: {disp_png_path}")

    return disparity


def batch_process(input_root, Q_original):
    """
    批量处理所有子文件夹

    参数:
        input_root: 输入根目录
        Q_original: 原始Q矩阵
    """
    # 获取所有包含out.ply的子文件夹
    folders = []
    for item in os.listdir(input_root):
        folder_path = os.path.join(input_root, item)
        if os.path.isdir(folder_path):
            ply_path = os.path.join(folder_path, "morph_kernel7_gauss7.ply")
            if os.path.exists(ply_path):
                folders.append(folder_path)

    print(f"找到 {len(folders)} 个包含out.ply的文件夹")
    print("-" * 50)

    success_count = 0
    for folder in tqdm(folders, desc="反投影进度"):
        result = process_single_folder(folder, Q_original)
        if result is not None:
            success_count += 1

    print("-" * 50)
    print(f"处理完成! 成功: {success_count}/{len(folders)}")


def visualize_comparison(folder_path, Q_original):
    """
    可视化对比：原始视差图 vs 从PLY反投影的视差图

    参数:
        folder_path: 文件夹路径
        Q_original: 原始Q矩阵
    """
    # 处理生成反投影视差图
    disp_from_ply = process_single_folder(folder_path, Q_original, output_disp=False)

    if disp_from_ply is None:
        print("反投影失败")
        return

    # 读取原始视差图（如果有）
    disp_orig_path = os.path.join(folder_path, "disp.npy")
    if os.path.exists(disp_orig_path):
        disp_orig = np.load(disp_orig_path)

        # 确保尺寸一致
        if disp_orig.shape != disp_from_ply.shape:
            print(f"警告: 尺寸不一致 - 原始{disp_orig.shape} vs 反投影{disp_from_ply.shape}")
            # 裁剪到相同尺寸
            min_h = min(disp_orig.shape[0], disp_from_ply.shape[0])
            min_w = min(disp_orig.shape[1], disp_from_ply.shape[1])
            disp_orig = disp_orig[:min_h, :min_w]
            disp_from_ply = disp_from_ply[:min_h, :min_w]

        # 计算差异
        diff = np.abs(disp_orig - disp_from_ply)
        valid_mask = (disp_orig > 0) & (disp_from_ply > 0)

        if np.sum(valid_mask) > 0:
            print(f"有效区域差异统计:")
            print(f"  平均误差: {np.mean(diff[valid_mask]):.4f}")
            print(f"  最大误差: {np.max(diff[valid_mask]):.4f}")
            print(f"  标准差: {np.std(diff[valid_mask]):.4f}")

        # 可视化
        def to_color(d):
            d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_JET)

        vis_orig = to_color(disp_orig)
        vis_ply = to_color(disp_from_ply)
        vis_diff = to_color(diff)

        # 拼接显示
        canvas = np.hstack([vis_orig, vis_ply, vis_diff])

        cv2.namedWindow("Original | From PLY | Difference", cv2.WINDOW_NORMAL)
        cv2.imshow("Original | From PLY | Difference", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # 只显示反投影结果
        vis = cv2.normalize(disp_from_ply, None, 0, 255, cv2.NORM_MINMAX)
        vis = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.namedWindow("Disparity from PLY", cv2.WINDOW_NORMAL)
        cv2.imshow("Disparity from PLY", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # ==================== 配置参数 ====================
    # 输入目录（包含out.ply的文件夹）
    input_root = r"D:\Desktop\过滤后\原始ply点云"

    # 原始Q矩阵（与batch_process_igev.py中相同，未裁剪调整的）
    Q_original = np.array([
        [1., 0., 0., -2.5134127044677734e+02],
        [0., 1., 0., -6.5667977905273438e+02],
        [0., 0., 0.,  8.8205398705187622e+02],
        [0., 0., 3.8920665588077730e-01, 0.]
    ])
    # ==================================================

    # 选择运行模式
    MODE = "batch"  # "batch" 批量处理 | "single" 单个处理+可视化

    if MODE == "batch":
        # 批量处理所有文件夹
        batch_process(input_root, Q_original)

    elif MODE == "single":
        # 单个文件夹处理+可视化对比
        single_folder = r"D:\Desktop\过滤后\原始ply点云\202506281603-0002"
        visualize_comparison(single_folder, Q_original)
