import numpy as np
import cv2
import os
import glob
import tkinter as tk
from tkinter import filedialog

# 配置输出根目录
OUTPUT_ROOT = r"D:\Desktop\stereo_project\tradition_stereo\datasets\FDJYP-3"

def process_disparity_file(disp_file):
    """
    处理单个视差图文件：裁剪固定区域并保存结果

    参数:
        disp_file: .npy视差图文件路径
    """
    # 获取文件所在目录和文件名
    file_dir = os.path.dirname(disp_file)
    filename = os.path.basename(disp_file)
    subfolder = os.path.basename(file_dir)

    # 检查并创建目标目录
    target_dir = os.path.join(OUTPUT_ROOT, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    print(f"输出目录: {target_dir}")


    # 加载视差图
    disp = np.load(disp_file)
    print(f"成功加载视差图: {filename}，尺寸: {disp.shape[1]}×{disp.shape[0]}")
    # 执行固定区域裁剪 (1280×720 → 818×512)
    # disp_cropped = disp[234:1052, 126:638]

    #原始裁剪参数计算（精确匹配）
    minDisparity = -104
    numDisparities = 208
    edge = abs(minDisparity) // 2
    edgeL = minDisparity + numDisparities
    start_x = edgeL
    start_y = edge // 2

    # 计算动态调整的长宽比裁剪
    roi_width = disp.shape[1] - 2 * edgeL
    roi_height = disp.shape[0] - edge

    k = roi_width / roi_height
    if k > 1.8:
        h = (roi_height * 16 // 10) // 2 * 2
        offset = (roi_width - h) // 4 * 2  # //2再/2取整
        start_x += offset
        roi_width = h
    elif 1 / k > 1.8:
        h = (roi_width * 16 // 10) // 2 * 2
        offset = (roi_height - h) // 4 * 2
        start_y += offset
        roi_height = h

    disp_cropped = disp[start_y:start_y + roi_height, start_x:start_x + roi_width]

    print(f"裁剪后尺寸: {disp_cropped.shape[1]}×{disp_cropped.shape[0]}")

    # 保存裁剪后的npy文件
    cropped_filename = filename.replace(".npy", "_rknn.npy")
    cropped_path = os.path.join(target_dir, cropped_filename)
    np.save(cropped_path, disp_cropped)
    print(f"已保存裁剪视差图: {cropped_path}")

    # 可视化处理
    # 转换为灰度图 (0-255)
    disp_normalized = cv2.normalize(disp_cropped, None, 0, 255, cv2.NORM_MINMAX)
    disp_uint8 = disp_normalized.astype(np.uint8)

    # 创建灰度版本和彩色版本
    gray_image = cv2.cvtColor(disp_uint8, cv2.COLOR_GRAY2BGR)  # 灰度转为BGR
    color_image = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)  # 彩色映射

    # 水平拼接图像 (灰度和彩色)
    combined = cv2.hconcat([gray_image, color_image])

    # 保存可视化结果
    vis_filename = cropped_filename.replace(".npy", "_rknn.png")
    vis_path = os.path.join(target_dir, vis_filename)
    cv2.imwrite(vis_path, combined)
    print(f"已保存可视化图像: {vis_path}")

    # 显示结果
    win_title = f"Disparity View: {cropped_filename}"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1024, 512)
    cv2.imshow(win_title, combined)
    cv2.waitKey(1)  # 短暂显示但不阻塞

    return cropped_path


def batch_process_disparity_files(root_path):
    """
    批量处理文件夹中的所有视差图文件

    参数:
        root_path: 包含多个子文件夹的根目录路径
    """
    # 查找所有子文件夹
    subdirs = [d for d in glob.glob(os.path.join(root_path, "*"))
               if os.path.isdir(d) and len(os.path.basename(d).split('-')) > 1]

    print(f"找到 {len(subdirs)} 个子文件夹")

    all_files = []
    for subdir in subdirs:
        # 在子文件夹中查找disp.npy文件
        disp_file = os.path.join(subdir, "disp.npy")
        if os.path.exists(disp_file):
            all_files.append(disp_file)
        else:
            print(f"子文件夹 {os.path.basename(subdir)} 中未找到 disp.npy 文件")

    print(f"找到 {len(all_files)} 个视差图文件")

    for i, disp_file in enumerate(all_files):
        print(f"\n处理文件 {i + 1}/{len(all_files)}: {disp_file}")
        try:
            process_disparity_file(disp_file)
        except Exception as e:
            print(f"处理文件 {disp_file} 出错: {str(e)}")

    print("\n所有文件处理完成!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_directory():
    """使用GUI选择根目录"""
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(
        title='选择包含所有数据集的根目录 (如FDJYP-3_LR)'
    )
    return folder_selected


if __name__ == "__main__":
    print("视差图处理工具 - 裁剪与可视化")
    print("=" * 50)
    print("此工具将：")
    print("1. 加载1280×720的视差图(.npy)")
    print("2. 应用固定裁剪区域 [234:1052, 126:638] → 818×512")
    print("3. 保存裁剪后的视差图 (*_rknn.npy)")
    print("4. 生成并保存可视化图像 (*_rknn.png)")
    print("")

    root_directory = select_directory()

    if not root_directory:
        print("未选择目录，程序退出")
    else:
        print(f"\n开始处理目录: {root_directory}")
        batch_process_disparity_files(root_directory)