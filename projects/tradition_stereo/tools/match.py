import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 设置相机参数
img_w, img_h = 720, 1280
w_gt, h_gt = 720, 1280
Q = np.array([
    [1., 0., 0., -411.2424],
    [0., 1., 0., -390.7704],
    [0., 0., 0., 407.0668],
    [0., 0., 0.4577106, 0.]
])

# 读取点云
ply_dense_path = r"F:\研究生\项目\数据集\FDJYP-0\叶片.ply\叶片.PLY"
ply_sparse_path = r"F:\研究生\项目\数据集\FDJYP-0\measure\1-202506261651-0001.ply"
pcd_dense = o3d.io.read_point_cloud(ply_dense_path)
pcd_sparse = o3d.io.read_point_cloud(ply_sparse_path)


# 1. 可视化原始位置（检查左右、上下、前后关系）
# o3d.visualization.draw_geometries([
#     pcd_dense.paint_uniform_color([1,0,0]),
#     pcd_sparse.paint_uniform_color([0,1,0])
# ], window_name="原始位置：红=稠密，绿=稀疏")

# 2. 试验性旋转：绕不同轴旋转 90° / 180°
#    你可以打开下面某一行，运行后再可视化，直到大致对齐方向
axis_angles = (np.pi/1.5, 0, 0)    # 绕 X 轴 90°
#axis_angles = (0, np.pi/2, 0)    # 绕 Y 轴 90°
# axis_angles = (0, 0, np.pi/2)    # 绕 Z 轴 90°
# axis_angles = (0, np.pi, 0)       # 绕 Y 轴 180°
R = pcd_dense.get_rotation_matrix_from_xyz(axis_angles)
pcd_dense.rotate(R, center=(0,0,0))

# 3. （可选）手动平移以大致重叠
#    定义平移向量，[dx, dy, dz]
translation = np.array([-70, -250, -80])  # 根据可视化结果调整
pcd_dense.translate(translation, relative=True)




# 4. 再次可视化，检查大致重叠
o3d.visualization.draw_geometries([
    pcd_dense.paint_uniform_color([1,0,0]),
    pcd_sparse.paint_uniform_color([0,1,0])
], window_name="初始旋转/平移后对齐检查")

# 5. 如果看上去有明显重叠，再执行 ICP
voxel_size = 1.0
dense_ds  = pcd_dense.voxel_down_sample(voxel_size)
sparse_ds = pcd_sparse.voxel_down_sample(voxel_size)
dense_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))
sparse_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))

reg = o3d.pipelines.registration.registration_icp(
    dense_ds, sparse_ds,
    max_correspondence_distance=voxel_size * 1.5,
    init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
)
print("ICP fitness:", reg.fitness)
print("ICP rmse   :", reg.inlier_rmse)




# 6. 应用 ICP 变换并可视化最终结果
pcd_dense.transform(reg.transformation)
o3d.visualization.draw_geometries([
    pcd_dense.paint_uniform_color([1,0,0]),
    pcd_sparse.paint_uniform_color([0,1,0])
], window_name="最终对齐：ICP 后")



# # ICP 配准（稠密配准到稀疏）
# voxel_size = 1.0
# dense_down = pcd_dense.voxel_down_sample(voxel_size)
# sparse_down = pcd_sparse.voxel_down_sample(voxel_size)
# dense_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))
# sparse_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))
#
# reg = o3d.pipelines.registration.registration_icp(
#     dense_down, sparse_down,
#     max_correspondence_distance=voxel_size * 1.5,
#     init=np.eye(4),
#     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
# )
# print("ICP 配准完成，fitness:", reg.fitness)
#
# # 变换稠密点云
# pcd_dense.transform(reg.transformation)
#
# # 投影为视差图
# def ply_to_disparity_from_pcd(pcd, Q, img_w, img_h, w_gt, h_gt):
#     pts = np.asarray(pcd.points)
#     f, invB = Q[2, 3], Q[3, 2]
#     cx, cy = -Q[0, 3], -Q[1, 3]
#     X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
#     valid = Z > 0
#     X, Y, Z = X[valid], Y[valid], Z[valid]
#     u = np.round(X * f / Z + cx).astype(int)
#     v = np.round(Y * f / Z + cy).astype(int)
#     mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
#     u, v, Z = u[mask], v[mask], Z[mask]
#     disp = np.zeros((img_h, img_w), np.float32)
#     disp[v, u] = f / (Z * invB)
#     start_x = (img_w - w_gt) // 2
#     start_y = (img_h - h_gt) // 2
#     return disp[start_y:start_y + h_gt, start_x:start_x + w_gt]
#
# # 生成并显示视差图
# disp_map = ply_to_disparity_from_pcd(pcd_dense, Q, img_w, img_h, w_gt, h_gt)
# plt.imshow(disp_map, cmap='jet')
# plt.colorbar()
# plt.title("Dense Point Cloud Projected Disparity")
# plt.axis('off')
# plt.show()
