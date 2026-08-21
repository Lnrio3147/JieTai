import os
import cv2
import numpy as np


def load_stereo_config(config_path):
    """从YAML文件加载立体相机配置参数"""
    fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)

    # 读取相机内参
    M1 = fs.getNode('M1').mat()
    M2 = fs.getNode('M2').mat()

    # 读取畸变系数
    D1 = fs.getNode('D1').mat()
    D2 = fs.getNode('D2').mat()

    # 读取矫正旋转矩阵
    R1 = fs.getNode('R1').mat()
    R2 = fs.getNode('R2').mat()

    # 读取投影矩阵
    P1 = fs.getNode('P1').mat()
    P2 = fs.getNode('P2').mat()

    # 读取Q矩阵
    Q = fs.getNode('Q').mat()

    fs.release()

    return M1, D1, M2, D2, R1, R2, P1, P2, Q


# 配置文件路径
config_path = r"D:\Desktop\stereo_project\tradition_stereo\config\stereo.yml"

# 加载立体相机参数
print("[INFO] 正在加载立体相机配置参数...")
M1, D1, M2, D2, R1, R2, P1, P2, Q = load_stereo_config(config_path)
print("[INFO] 配置参数加载完成")
print(f"[INFO] Q矩阵:\n{Q}")

# 输入输出路径
input_root = r"D:\Desktop\20260205\20260205\output"
output_root = r"D:\Desktop\20260205\20260205\rectified_images"

# 创建输出根目录（如果不存在）
os.makedirs(output_root, exist_ok=True)

# 获取图像尺寸（从第一张图像）
# 遍历找到第一个有效的图像来确定尺寸
image_size = None
for folder in sorted(os.listdir(input_root)):
    input_folder = os.path.join(input_root, folder)
    if not os.path.isdir(input_folder):
        continue

    im0_path = os.path.join(input_folder, "im0.png")
    if os.path.exists(im0_path):
        test_img = cv2.imread(im0_path)
        if test_img is not None:
            image_size = (test_img.shape[1], test_img.shape[0])  # (width, height)
            print(f"[INFO] 检测到图像尺寸: {image_size}")
            break

if image_size is None:
    print("[ERROR] 无法找到有效图像来确定尺寸")
    exit(1)

# 使用OpenCV直接计算矫正映射
# 使用CV_16SC2与C++代码保持一致，节省内存并提升性能
print("[INFO] 正在计算立体矫正映射...")
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    M1, D1, R1, P1, image_size, cv2.CV_16SC2
)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    M2, D2, R2, P2, image_size, cv2.CV_16SC2
)
print("[INFO] 矫正映射计算完成")

# 遍历子文件夹处理图像
for folder in sorted(os.listdir(input_root)):
    input_folder = os.path.join(input_root, folder)
    if not os.path.isdir(input_folder):
        continue

    # 加载图像
    im0_path = os.path.join(input_folder, "im0.png")
    im1_path = os.path.join(input_folder, "im1.png")
    if not os.path.exists(im0_path) or not os.path.exists(im1_path):
        print(f"[WARNING] 缺少图像文件于 {input_folder}，跳过")
        continue

    frame1 = cv2.imread(im0_path)
    frame2 = cv2.imread(im1_path)

    if frame1 is None or frame2 is None:
        print(f"[WARNING] 无法读取图像于 {input_folder}，跳过")
        continue

    # 立体校正 - 直接使用计算的映射进行重映射
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
    # img1_rectified = cv2.remap(frame1, right_map1, right_map2, cv2.INTER_LINEAR)
    # img2_rectified = cv2.remap(frame2, left_map1, left_map2, cv2.INTER_LINEAR)

    # 保存矫正图像到新的目录
    output_folder = os.path.join(output_root, folder)
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, "im0.png"), img1_rectified)
    cv2.imwrite(os.path.join(output_folder, "im1.png"), img2_rectified)

    print(f"[INFO] 已保存矫正图像到 {output_folder}")

print("[INFO] 所有图像处理完成")
