import os
import cv2
import numpy as np

# 加载立体校正映射表
# left_map1 = np.load("new_map/left_map1.npy")
# left_map2 = np.load("new_map/left_map2.npy")
# right_map1 = np.load("new_map/right_map1.npy")
# right_map2 = np.load("new_map/right_map2.npy")


# left_map1 = np.load("JXP_map/left_map1.npy")
# left_map2 = np.load("JXP_map/left_map2.npy")
# right_map1 = np.load("JXP_map/right_map1.npy")
# right_map2 = np.load("JXP_map/right_map2.npy")

# left_map1 = np.load(r"..\config\map\gongjian_map\left_map1.npy")
# left_map2 = np.load(r"..\config\map\gongjian_map\left_map2.npy")
# right_map1 = np.load(r"..\config\map\gongjian_map\right_map1.npy")
# right_map2 = np.load(r"..\config\map\gongjian_map\right_map2.npy")

# left_map1 = np.load("../config/map/luowen_map/left_map1.npy")
# left_map2 = np.load("../config/map/luowen_map/left_map2.npy")
# right_map1 = np.load("../config/map/luowen_map/right_map1.npy")
# right_map2 = np.load("../config/map/luowen_map/right_map2.npy")

right_map1 = np.load("../config/map/1221/right_map1.npy")
right_map2 = np.load("../config/map/1221/right_map2.npy")
left_map1 = np.load("../config/map/1221/left_map1.npy")
left_map2 = np.load("../config/map/1221/left_map2.npy")

# 输入输出路径
# input_root = "./JXP"
# input_root = "../datasets/FDJYP-3"
input_root = "D:\Desktop\点云文件3\V2\output"
output_root = "../rectified_images"


# 创建输出根目录（如果不存在）
os.makedirs(output_root, exist_ok=True)

# 遍历子文件夹（1 到 8）
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

    # 立体校正
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)

    # 保存矫正图像到新的目录
    output_folder = os.path.join(output_root, folder)
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, "im0.png"), img1_rectified)
    cv2.imwrite(os.path.join(output_folder, "im1.png"), img2_rectified)

    print(f"[INFO] 已保存矫正图像到 {output_folder}")
