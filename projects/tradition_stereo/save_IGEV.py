import numpy as np
import cv2
import open3d as o3d
import os

# 1. 加载视差和校正图
gongjian_test_im_path = "datasets/gongjian_test/1"
other_test_im_path    = "datasets/other_test/98_2"
new_test_im_path    = "./new_test/111"
# jxp_test_im_path    = "datasets/JXP/0006"
# jyp_test_im_path = r"D:\Desktop\stereo_project\tradition_stereo\datasets/FDJYP-3/20250628160-0014"
jyp_test_im_path = r"D:\Desktop\rknn_scene_output\202506281603-0002"

gongjian_left_map1 = "./config/map/gongjian_map/left_map1.npy"
gongjian_left_map2 = "./config/map/gongjian_map/left_map2.npy"

other_left_map1 = "config/map/others_map/left_map1.npy"
other_left_map2 = "config/map/others_map/left_map2.npy"
new_left_map1 = "./new_map/left_map1.npy"
new_left_map2 = "./new_map/left_map2.npy"
jxp_left_map1 = "./JXP_map/left_map1.npy"
jxp_left_map2 = "./JXP_map/left_map2.npy"

im_path = jyp_test_im_path
left_map1_path = gongjian_left_map1
left_map2_path = gongjian_left_map2

# disparity = np.load(os.path.join(im_path, "disp.npy")).astype(np.float32)
# disparity = np.load(os.path.join(im_path, "disp.npy")).astype(np.float32)
disparity = np.load(r"D:\Desktop\rknn_scene_output\202506281603-0002\disp.npy").astype(np.float32)

# 对视差图进行180度旋转 (上下颠倒)
# disparity = cv2.rotate(disparity, cv2.ROTATE_180)

# 校正左图
left_map1 = np.load(left_map1_path)
left_map2 = np.load(left_map2_path)

imgL = cv2.imread(os.path.join(im_path, "im0.png"))
# imgL = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)


# ==== 新增中心裁剪代码 ====
h, w = disparity.shape[:2]
# target_w, target_h = 512, 818
# start_x = (w - target_w) // 2
# start_y = (h - target_h) // 2

#原始裁剪参数计算（精确匹配）
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
    h = (roi_height * 16 // 10) // 2 * 2
    offset = (roi_width - h) // 4 * 2  # //2再/2取整
    start_x += offset
    roi_width = h
elif 1 / k > 1.8:
    h = (roi_width * 16 // 10) // 2 * 2
    offset = (roi_height - h) // 4 * 2
    start_y += offset
    roi_height = h

# 在裁剪前对整个图像进行180度旋转
# imgL = cv2.rotate(imgL, cv2.ROTATE_180)

imgL = imgL[start_y:start_y + roi_height, start_x:start_x + roi_width]
disparity = disparity[start_y:start_y + roi_height, start_x:start_x + roi_width]

# 对视差图和校正左图进行同样的裁剪
# disparity = disparity[start_y:start_y+target_h, start_x:start_x+target_w]
# imgL = imgL[start_y:start_y+target_h, start_x:start_x+target_w, :]

# 2. 滤波
# disparity = cv2.medianBlur(disparity, 5)
# disparity = cv2.bilateralFilter(disparity, 9, 50, 75)

# 3. 重投影
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


Q_jxp = np.array([[  1., 0., 0., -2.2227048492431641e+02],
              [ 0., 1., 0.,-7.2185147857666016e+02],
              [0., 0., 0., 9.4420949981369597e+02],
              [0.,0., 3.9456987572407826e-01, 0.]])


pts3d = cv2.reprojectImageTo3D(disparity, Q_jxp, handleMissingValues=True)

# 4. 范围过滤
X, Y, Z = pts3d[...,0], pts3d[...,1], pts3d[...,2]
# mask = (disparity>0) & np.isfinite(Z) & (Z>50)&(Z<800)&(np.abs(X)<50)&(np.abs(Y)<60)  #工件范围

#1.RGB + 视差（原始） 2、过滤视差图函数（视差<(参数)  颜色暗色 < (参数)）, 3、转成过滤后的点云 4、给示例代码

mask = (disparity>0) & np.isfinite(Z) & (Z>0)&(Z<200)     #限制视差/   rgb+视差       点云   
# mask = (disparity>0)& np.isfinite(Z) & (Z>5)&(Z<50)    #标定板范围/非常近的范围

# mask = (disparity>0)& np.isfinite(Z) & (Z>5)&(Z<50)

points = pts3d[mask]
colors = imgL[mask] / 255.0

# 转换BGR到RGB (OpenCV读取是BGR,Open3D需要RGB)
colors = colors[:, ::-1]  # 反转最后一个维度: BGR -> RGB

# 5. 构造 Open3D 点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)



# 6. 保存并显示
o3d.io.write_point_cloud("output/cleaned_pc.ply", pcd)
print("Saved cleaned_pc.ply")

o3d.visualization.draw_geometries([pcd])
