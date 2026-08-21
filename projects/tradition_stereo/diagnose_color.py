import numpy as np
import cv2
import open3d as o3d
import os

"""
颜色诊断工具 - 检查点云色差问题
"""

# 加载数据
im_path = r"D:\Desktop\rknn_scene_output\202506281603-0002"
disparity = np.load(r"D:\Desktop\rknn_scene_output\202506281603-0002\disp.npy").astype(np.float32)

# 加载校正映射
left_map1 = np.load("./config/map/gongjian_map/left_map1.npy")
left_map2 = np.load("./config/map/gongjian_map/left_map2.npy")

# 读取原始图像
imgL_original = cv2.imread(os.path.join(im_path, "im0.png"))
print(f"原始图像shape: {imgL_original.shape}, dtype: {imgL_original.dtype}")
print(f"原始图像范围: min={imgL_original.min()}, max={imgL_original.max()}")

# 校正图像
imgL = cv2.remap(imgL_original, left_map1, left_map2, cv2.INTER_LINEAR)
print(f"校正后图像shape: {imgL.shape}, dtype: {imgL.dtype}")

# 裁剪
minDisparity = -104
numDisparities = 208
edge = abs(minDisparity) // 2
edgeL = minDisparity + numDisparities
start_x = edgeL
start_y = edge // 2

h, w = disparity.shape[:2]
roi_width = w - 2 * edgeL
roi_height = h - edge

k = roi_width / roi_height
if k > 1.8:
    h = (roi_height * 16 // 10) // 2 * 2
    offset = (roi_width - h) // 4 * 2
    start_x += offset
    roi_width = h
elif 1 / k > 1.8:
    h = (roi_width * 16 // 10) // 2 * 2
    offset = (roi_height - h) // 4 * 2
    start_y += offset
    roi_height = h

imgL = imgL[start_y:start_y + roi_height, start_x:start_x + roi_width]
disparity = disparity[start_y:start_y + roi_height, start_x:start_x + roi_width]

print(f"\n裁剪后图像shape: {imgL.shape}")
print(f"裁剪后视差shape: {disparity.shape}")

# 保存裁剪后的图像用于对比
cv2.imwrite("output/cropped_image_bgr.png", imgL)
imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
cv2.imwrite("output/cropped_image_rgb.png", cv2.cvtColor(imgL_rgb, cv2.COLOR_RGB2BGR))

# 3D重投影
Q_jxp = np.array([[  1., 0., 0., -2.2227048492431641e+02],
          [ 0., 1., 0.,-7.2185147857666016e+02],
          [0., 0., 0., 9.4420949981369597e+02],
          [0.,0., 3.9456987572407826e-01, 0.]])

pts3d = cv2.reprojectImageTo3D(disparity, Q_jxp, handleMissingValues=True)

# 过滤
X, Y, Z = pts3d[...,0], pts3d[...,1], pts3d[...,2]
mask = (disparity>0) & np.isfinite(Z) & (Z>0)&(Z<200)

points = pts3d[mask]
colors_bgr = imgL[mask]

print(f"\n点云数量: {len(points)}")
print(f"颜色数组shape: {colors_bgr.shape}")
print(f"颜色范围 (BGR): min={colors_bgr.min()}, max={colors_bgr.max()}")

# 检查前10个点的颜色
print("\n前10个点的BGR颜色值:")
for i in range(min(10, len(colors_bgr))):
    print(f"点{i}: B={colors_bgr[i,0]}, G={colors_bgr[i,1]}, R={colors_bgr[i,2]}")

# ===== 方案1: 使用cv2.cvtColor转换整个图像 =====
print("\n\n=== 方案1: cv2.cvtColor转换整个图像 ===")
imgL_rgb_full = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
colors_rgb_v1 = imgL_rgb_full[mask] / 255.0

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points)
pcd1.colors = o3d.utility.Vector3dVector(colors_rgb_v1)
o3d.io.write_point_cloud("output/test_v1_cvtColor.ply", pcd1)
print("已保存: output/test_v1_cvtColor.ply")
print(f"颜色范围 (归一化): min={colors_rgb_v1.min():.3f}, max={colors_rgb_v1.max():.3f}")

# ===== 方案2: 使用切片反转 =====
print("\n=== 方案2: 切片反转 ===")
colors_bgr_norm = colors_bgr / 255.0
colors_rgb_v2 = colors_bgr_norm[:, ::-1]  # BGR -> RGB

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points)
pcd2.colors = o3d.utility.Vector3dVector(colors_rgb_v2)
o3d.io.write_point_cloud("output/test_v2_slice.ply", pcd2)
print("已保存: output/test_v2_slice.ply")
print(f"颜色范围 (归一化): min={colors_rgb_v2.min():.3f}, max={colors_rgb_v2.max():.3f}")

# ===== 方案3: 使用numpy翻转通道顺序 =====
print("\n=== 方案3: numpy翻转 ===")
colors_rgb_v3 = colors_bgr[:, [2, 1, 0]] / 255.0  # BGR -> RGB

pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(points)
pcd3.colors = o3d.utility.Vector3dVector(colors_rgb_v3)
o3d.io.write_point_cloud("output/test_v3_numpy.ply", pcd3)
print("已保存: output/test_v3_numpy.ply")
print(f"颜色范围 (归一化): min={colors_rgb_v3.min():.3f}, max={colors_rgb_v3.max():.3f}")

# ===== 方案4: 检查是否需要gamma校正 =====
print("\n=== 方案4: 带gamma校正 ===")
colors_rgb_v4 = colors_bgr[:, [2, 1, 0]] / 255.0
# 应用gamma校正 (sRGB -> linear)
colors_rgb_v4 = np.power(colors_rgb_v4, 2.2)

pcd4 = o3d.geometry.PointCloud()
pcd4.points = o3d.utility.Vector3dVector(points)
pcd4.colors = o3d.utility.Vector3dVector(colors_rgb_v4)
o3d.io.write_point_cloud("output/test_v4_gamma.ply", pcd4)
print("已保存: output/test_v4_gamma.ply")
print(f"颜色范围 (gamma校正): min={colors_rgb_v4.min():.3f}, max={colors_rgb_v4.max():.3f}")

# 验证转换是否正确
print("\n=== 验证前10个点的RGB颜色 ===")
for i in range(min(10, len(colors_rgb_v3))):
    r, g, b = colors_rgb_v3[i]
    print(f"点{i}: R={r:.3f}, G={g:.3f}, B={b:.3f}")

print("\n\n=== 诊断完成 ===")
print("请用MeshLab或CloudCompare打开以下文件对比:")
print("  1. output/test_v1_cvtColor.ply  - cv2.cvtColor方法")
print("  2. output/test_v2_slice.ply     - 切片反转方法")
print("  3. output/test_v3_numpy.ply     - numpy索引方法")
print("  4. output/test_v4_gamma.ply     - 带gamma校正")
print("\n同时对比:")
print("  - output/cropped_image_bgr.png  - BGR格式图像")
print("  - output/cropped_image_rgb.png  - RGB格式图像")
