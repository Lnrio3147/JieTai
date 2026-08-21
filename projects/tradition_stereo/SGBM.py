import numpy as np
import cv2
import os
import open3d as o3d

WIN_NAME = 'Deep disp'
WINDOW_WIDTH, WINDOW_HEIGHT = 720, 1280
# WINDOW_WIDTH, WINDOW_HEIGHT = 800, 800
def write_ply(filename, verts, colors):
    """保存点云为 PLY 文件"""
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    header = '''ply
format ascii 1.0
element vertex {vert_num}
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''.format(vert_num=len(verts))
    with open(filename, 'w') as f:
        f.write(header)
        for (x, y, z), (b, g, r) in zip(verts, colors):
            f.write(f"{x} {y} {z} {int(b)} {int(g)} {int(r)}\n")


# ---------------- 加载校正映射表和图像 ----------------
gongjian_test_im_path = "./datasets/gongjian_test/1"
other_test_im_path    = "datasets/other_test/98"
new_test_im_path    = "./new_test/110"
# FDJYP = "./datasets/FDJYP-3/202506281603-0001"
FDJYP = r"rec_img_set\luowen_rectified_images\656565-0006"
# FDJYP = r"./datasets/test"

gongjian_left_map1 = "./config/map/gongjian_map/left_map1.npy"
gongjian_left_map2 = "./config/map/gongjian_map/left_map2.npy"
other_left_map1 = "./others_map/left_map1.npy"
other_left_map2 = "./others_map/left_map2.npy"
new_left_map1 = "./new_map/left_map1.npy"
new_left_map2 = "./new_map/left_map2.npy"

stereo_left_map1 = "config/map/1221/left_map1.npy"
stereo_left_map2 = ".config/map/1221/left_map2.npy"

gongjian_right_map1 = "./config/map/gongjian_map/right_map1.npy"
gongjian_right_map2 = "./config/map/gongjian_map/right_map2.npy"
other_right_map1 = "./others_map/right_map1.npy"
other_right_map2 = "./others_map/right_map2.npy"
new_right_map1 = "./new_map/right_map1.npy"
new_right_map2 = "./new_map/right_map2.npy"

stereo_right_map1 = "config/map/1221/right_map1.npy"
stereo_right_map2 = ".config/map/1221/right_map2.npy"

im_path = FDJYP       #图像位置
left_map1_path = stereo_left_map1  #左图矫正表1位置
left_map2_path = stereo_left_map2  #左图矫正表2位置
right_map1_path = stereo_right_map1  #右图矫正表1位置
right_map2_path = stereo_right_map2  #右图矫正表2位置


# 校正左图
left_map1 = np.load(left_map1_path)
left_map2 = np.load(left_map2_path)
right_map1 = np.load(right_map1_path)
right_map2 = np.load(right_map2_path)

frame1 = cv2.imread(os.path.join(im_path, "im0.png"))
frame2 = cv2.imread(os.path.join(im_path, "im1.png"))


img1_rectified = frame1
img2_rectified = frame2
grayL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
# img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)

# grayL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
# grayR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

# ---------------- SGBM视差计算 ----------------
window_size = 9
min_disp = 0
num_disp = 16 * 9
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=5,
    uniquenessRatio=10,
    speckleWindowSize=200,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_HH
)

disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# ---------------- 视差图滤波 ----------------
# disparity = cv2.medianBlur(disparity, 5)
# disparity = cv2.bilateralFilter(disparity, 9, 50, 75)

# ---------------- 三维重建 ----------------
Q_gongjian = np.array([[1., 0., 0., -312.7411],
              [0., 1., 0., -663.5256],
              [0., 0., 0., 877.7027],
              [0., 0., 0.3976856, 0.]])
Q_other = np.array([[1., 0., 0., -3.4427591323852539e+02],
              [0., 1., 0.,-6.7546062469482422e+02],
              [0., 0., 0., 8.7817168731454080e+02],
              [0.,0., 3.9373972336799606e-01, 0.]])

Q_new = np.array([[  1., 0., 0., -4.1124235534667969e+02],
              [ 0., 1., 0.,-3.9077036666870117e+02],
              [0., 0., 0., 4.0706681451380024e+02],
              [0.,0., 4.5771057047685543e-01, 0.]])
threeD = cv2.reprojectImageTo3D(disparity, Q_gongjian, handleMissingValues=True)

X, Y, Z = threeD[:, :, 0], threeD[:, :, 1], threeD[:, :, 2]
mask = (
    (disparity > min_disp)
    # np.isfinite(Z) &
    # (Z > 4) & (Z < 1000) &
    # (np.abs(X) < 40) &
    # (np.abs(Y) < 60)
)

points = threeD[mask]
colors = img1_rectified[mask]

# ---------------- Open3D离群点滤除 ----------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

# 使用索引重新选择点与颜色
points = np.asarray(pcd.points)[ind]
colors = np.asarray(pcd.colors)[ind] * 255  # 转为整数

# ---------------- 打印点云范围 ----------------
print(f"[INFO] 点云范围：")
print(f"  X: {points[:, 0].min():.2f} ~ {points[:, 0].max():.2f}")
print(f"  Y: {points[:, 1].min():.2f} ~ {points[:, 1].max():.2f}")
print(f"  Z: {points[:, 2].min():.2f} ~ {points[:, 2].max():.2f}")

# ---------------- 保存点云 ----------------
ply_filename = "output/pointcloud.ply"
write_ply(ply_filename, points, colors)
print(f"[INFO] 点云已保存为：{ply_filename}")

# ---------------- 可视化视差图和深度图 ----------------
disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

top_row = np.hstack((img1_rectified, img2_rectified))
bottom_row = np.hstack((cv2.cvtColor(disp_norm, cv2.COLOR_GRAY2BGR), disp_color))
combined = np.vstack((top_row, bottom_row))

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.imshow(WIN_NAME, combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
