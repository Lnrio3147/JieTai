import numpy as np
import cv2
import open3d as o3d
# import torch
import os


def ply_to_disparity_and_visualize(ply_file, Q_new):
    """
    从PLY点云生成视差图并可视化
    参数:
        ply_file: PLY点云文件路径
        Q_new: 4x4重投影矩阵
    """

    def to_gray_uint8(d):
        """归一化并转为 uint8 灰度图。"""
        g = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
        return g.astype(np.uint8)

    def pad_to_target(img, target_w, target_h, border_value=0):
        """在 img 四周填充，使其变为 target_w×target_h"""
        h0, w0 = img.shape[:2]
        pad_vert = max(target_h - h0, 0)
        pad_top = pad_vert // 2
        pad_bot = pad_vert - pad_top

        pad_horiz = max(target_w - w0, 0)
        pad_left = pad_horiz // 2
        pad_right = pad_horiz - pad_left

        return cv2.copyMakeBorder(
            img, pad_top, pad_bot, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=border_value
        )

    # 设置默认参数（可根据需要调整）
    img_w_recon, img_h_recon = 720, 1280  # 点云重投影分辨率
    w_gt, h_gt = 512, 818  # 期望裁剪尺寸（Middlebury标准）
    # w_gt, h_gt = 720, 1280

    # 从点云生成视差图
    pcd = o3d.io.read_point_cloud(ply_file)
    pts = np.asarray(pcd.points)

    # 获取点的个数 - 数组的第一维度
    num_points = pts.shape[0]
    print(f"点云中包含 {num_points} 个点")


    f, invB = Q_new[2, 3], Q_new[3, 2]
    cx, cy = -Q_new[0, 3], -Q_new[1, 3]
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = Z > 0
    X, Y, Z = X[valid], Y[valid], Z[valid]

    u = np.round(X * f / Z + cx).astype(int)
    v = np.round(Y * f / Z + cy).astype(int)
    # mask = (u >= 0) & (u < img_w_recon) & (v >= 0) & (v < img_h_recon)
    # u, v, Z = u[mask], v[mask], Z[mask]

    disp = np.zeros((img_h_recon, img_w_recon), np.float32)
    disp[v, u] = f / (Z * invB)

    # 获取所有非零视差值的坐标
    nonzero_points = np.argwhere(disp != 0)
    if len(nonzero_points) > 0:
        min_v, min_u = np.min(nonzero_points, axis=0)
        max_v, max_u = np.max(nonzero_points, axis=0)
        print(f"非零边界范围:")
        print(f"  u方向: {min_u} → {max_u} (宽度: {max_u - min_u}像素)")
        print(f"  v方向: {min_v} → {max_v} (高度: {max_v - min_v}像素)")
    else:
        print("警告: 未找到非零视差值")
    disp_cropped = disp[234:1052, 126:638]

    # 中心裁剪
    # start_x = (img_w_recon - w_gt) // 2
    # start_y = (img_h_recon - h_gt) // 2
    # disp_cropped = disp[start_y:start_y + h_gt, start_x:start_x + w_gt]
    # # disp_cropped = disp
    # print(f"成功生成视差图 尺寸: {disp_cropped.shape[1]}×{disp_cropped.shape[0]}")

    #######################原始裁剪参数计算（精确匹配）--->存在问题###################################
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





    # 可视化处理
    recon_gray = to_gray_uint8(disp_cropped)
    recon_gray_bgr = cv2.cvtColor(recon_gray, cv2.COLOR_GRAY2BGR)
    recon_gray_bgr = pad_to_target(recon_gray_bgr, 512, 818)

    recon_color = cv2.applyColorMap(recon_gray, cv2.COLORMAP_JET)
    #可视化
    # cv2.imshow("111", recon_color)
    recon_color = pad_to_target(recon_color, 512, 818)
    canvas = cv2.hconcat([recon_gray_bgr, recon_color])

    # 显示结果
    win = f"Point Cloud Disparity: {os.path.basename(ply_file)}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    show_w = img_w_recon
    show_h = img_h_recon
    cv2.resizeWindow(win,  512*2, 818)
    cv2.imshow(win, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return disp_cropped


# 示例用法
if __name__ == "__main__":
    # 定义Q矩阵（示例值，应根据相机标定设置）
    Q_gongjian = np.array([[1., 0., 0., -312.7411],
                           [0., 1., 0., -663.5256],
                           [0., 0., 0., 877.7027],
                           [0., 0., 0.3976856, 0.]])

    # ply_file = r"D:\Desktop\大模型结果\202506281603-0003\out.ply"  # 替换为实际PLY文件路径
    ply_file = r"D:\Desktop\项目\传统算法\datasets\FDJYP-3\202506281603-0003\0003_old.ply"  # 替换为实际PLY文件路径

    disparity = ply_to_disparity_and_visualize(ply_file, Q_gongjian)