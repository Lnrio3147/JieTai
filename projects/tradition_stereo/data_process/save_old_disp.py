import numpy as np
import cv2
import open3d as o3d
import os


def ply_to_disparity_and_visualize(ply_file, Q_new, save_visualization=False):
    """
    从PLY点云生成裁剪后的视差图并保存可视化结果
    参数:
        ply_file: PLY点云文件路径
        Q_new: 4x4重投影矩阵
        save_visualization: 是否保存可视化图像
    返回:
        disp_cropped: 裁剪后的视差图 [234:1052, 126:638]
        color_map: 彩色可视化图像 (如果保存)
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_file)
    pts = np.asarray(pcd.points)

    # 解析重投影矩阵参数
    f = Q_new[2, 3]
    cx, cy = -Q_new[0, 3], -Q_new[1, 3]
    invB = Q_new[3, 2]

    # 过滤无效点 (Z<=0)
    valid_mask = pts[:, 2] > 0
    X, Y, Z = pts[valid_mask, 0], pts[valid_mask, 1], pts[valid_mask, 2]

    # 计算投影坐标
    u = np.round(X * f / Z + cx).astype(int)
    v = np.round(Y * f / Z + cy).astype(int)

    # 创建视差图容器 (1280×720)
    disp = np.zeros((1280, 720), dtype=np.float32)

    # 生成视差值 (避免除零)
    valid_z = Z.copy()
    valid_z[valid_z == 0] = 1e-6  # 防止除以零
    disp_values = f / (valid_z * invB)

    # 投影点到视差图
    disp[v, u] = disp_values

    # 应用固定裁剪区域
    disp_cropped = disp[234:1052, 126:638]  # 原始高度1280→裁剪后818, 原始宽度720→裁剪后512

    # 准备可视化结果
    color_map = None
    if save_visualization:
        # 归一化并转为uint8
        disp_normalized = cv2.normalize(disp_cropped, None, 0, 255, cv2.NORM_MINMAX)
        disp_uint8 = disp_normalized.astype(np.uint8)

        # 应用色彩映射并保存
        color_map = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)
        # 填充到目标尺寸 (根据图片显示其他文件都是818×512)
        padded_image = cv2.copyMakeBorder(
            color_map,
            0, 0, 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    return disp_cropped, color_map


def process_all_datasets(root_dir, Q_matrix):
    """
    批量处理所有子文件夹中的_old.ply文件
    参数:
        root_dir: 根目录路径 (e.g., 'datasets/FDJYP-3')
        Q_matrix: 重投影矩阵
    """
    # 遍历所有子文件夹 (202506281603-0001 等格式)
    subdirs = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d)) and '-' in d]

    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        print(f"\n处理文件夹: {subdir}")

        # 查找当前文件夹内的_old.ply文件
        old_ply_files = [f for f in os.listdir(subdir_path)
                         if f.endswith("_old.ply")]

        if not old_ply_files:
            print(f"  未找到 *_old.ply 文件")
            continue

        for file in old_ply_files:
            ply_path = os.path.join(subdir_path, file)
            print(f"  处理文件: {file}")

            try:
                # 处理点云文件
                disp_cropped, color_image = ply_to_disparity_and_visualize(
                    ply_path, Q_matrix, save_visualization=True
                )

                # 提取基础文件名 (0001_old → 0001)
                base_name = file.split('_')[0]

                # 保存裁剪后的视差图 (npy格式)
                disp_save_path = os.path.join(subdir_path, f"{base_name}_disp_cropped.npy")
                np.save(disp_save_path, disp_cropped)
                print(f"  已保存视差图: {disp_save_path} ({disp_cropped.shape})")

                # 保存可视化图像 (png格式)
                if color_image is not None:
                    img_save_path = os.path.join(subdir_path, f"{base_name}_disp_color.png")
                    cv2.imwrite(img_save_path, color_image)
                    print(f"  已保存可视化: {img_save_path}")

            except Exception as e:
                print(f"  处理文件 {file} 时出错: {str(e)}")


if __name__ == "__main__":
    # 定义重投影矩阵 (根据原始代码)
    Q_gongjian = np.array([
        [1.0, 0.0, 0.0, -312.7411],
        [0.0, 1.0, 0.0, -663.5256],
        [0.0, 0.0, 0.0, 877.7027],
        [0.0, 0.0, 0.3976856, 0.0]
    ])

    # 设置根目录路径 (根据图片显示)
    root_directory = "../datasets/FDJYP-3"

    # 批量处理所有数据集
    process_all_datasets(root_directory, Q_gongjian)
    print("\n所有文件处理完成！")