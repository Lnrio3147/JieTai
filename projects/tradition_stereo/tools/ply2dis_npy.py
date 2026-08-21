import numpy as np
import cv2
import open3d as o3d
from metric.cal_metric import d1_metric,threshold_metric,epe_metric
import torch

def read_disp_npy(path):
    """读取 NPY 文件，返回形状为 (H, W) 的 float32 数组。"""
    disp = np.load(path)
    # 确保是二维数组
    if disp.ndim == 3:
        disp = disp.squeeze()
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

    return disp_cropped.astype(np.float32)

def ply_to_disparity(ply_filename, Q, img_w, img_h, w_gt, h_gt):
    """从 PLY 重建视差图 (float32)，先生成 img_w×img_h，再中心裁剪到 w_gt×h_gt。"""
    pcd = o3d.io.read_point_cloud(ply_filename)
    pts = np.asarray(pcd.points)

    f, invB = Q[2,3], Q[3,2]
    cx, cy  = -Q[0,3], -Q[1,3]
    X, Y, Z = pts[:,0], pts[:,1], pts[:,2]
    valid   = Z > 0
    X, Y, Z = X[valid], Y[valid], Z[valid]

    u = np.round(X * f / Z + cx).astype(int)
    v = np.round(Y * f / Z + cy).astype(int)
    mask = (u>=0)&(u<img_w)&(v>=0)&(v<img_h)
    u, v, Z = u[mask], v[mask], Z[mask]

    disp = np.zeros((img_h, img_w), np.float32)
    disp[v, u] = f / (Z * invB)

    # 中心裁剪
    # start_x = (img_w - w_gt) // 2
    # start_y = (img_h - h_gt) // 2
    # disp_cropped = disp[start_y:start_y + h_gt,
    #                    start_x:start_x + w_gt]
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



    return disp_cropped

def to_gray_uint8(d):
    """归一化并转为 uint8 灰度图。"""
    g = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    return g.astype(np.uint8)

def pad_to_target(img, target_w, target_h, border_value=0):
    """
    在 img 四周填充，使其变为 target_w×target_h。
    img 可以是 H×W 或 H×W×3。
    """
    h0, w0 = img.shape[:2]
    pad_vert = max(target_h - h0, 0)
    pad_top  = pad_vert // 2
    pad_bot  = pad_vert - pad_top

    pad_horiz = max(target_w - w0, 0)
    pad_left  = pad_horiz // 2
    pad_right = pad_horiz - pad_left

    if img.ndim == 2:
        return cv2.copyMakeBorder(img,
                                  pad_top, pad_bot,
                                  pad_left, pad_right,
                                  cv2.BORDER_CONSTANT,
                                  value=border_value)
    else:
        return cv2.copyMakeBorder(img,
                                  pad_top, pad_bot,
                                  pad_left, pad_right,
                                  cv2.BORDER_CONSTANT,
                                  value=[border_value]*3)

if __name__ == "__main__":
    ply_file    = r"D:\Desktop\项目\传统算法\datasets\FDJYP-3\202506281603-0003\0003_old.ply"
    gt_npy      = r"D:\Desktop\项目\传统算法\datasets\FDJYP-3\202506281603-0003\disp.npy"  # 修改为你的npy文件路径
    target_size = 800  # 最终拼接时每个子图大小

    # 1. 读入真值视差，并裁剪、clip
    gt_disp = read_disp_npy(gt_npy)  # 使用新的读取函数
    gt_disp = np.clip(gt_disp, 0, np.percentile(gt_disp, 99))
    h_gt, w_gt = gt_disp.shape

    # 2. 重建画布分辨率：先生成 800×800，再中心裁剪到 gt 大小
    img_w_recon, img_h_recon = 720, 1280
    Q_gongjian = np.array([[1., 0., 0., -312.7411],
                           [0., 1., 0., -663.5256],
                           [0., 0., 0., 877.7027],
                           [0., 0., 0.3976856, 0.]])
    recon_disp = ply_to_disparity(
        ply_file, Q_gongjian,
        img_w_recon, img_h_recon,
        w_gt, h_gt
    )

###################计算指标###################
    # 3. 转为 Torch Tensor，并添加 batch 维度 [B=1, H, W]
    disp_gt_t = torch.from_numpy(gt_disp).unsqueeze(0)  # shape [1, H, W]
    disp_pred_t = torch.from_numpy(recon_disp).unsqueeze(0)  # shape [1, H, W]

    # 4. 构造有效像素 mask：这里我们令 gt_disp>0 视为有效
    mask_t = (disp_gt_t > 0)

    # 5. 计算指标
    epe = epe_metric(disp_pred_t, disp_gt_t, mask_t)  # 平均 EPE
    d1 = d1_metric(disp_pred_t, disp_gt_t, mask_t)  # D1 错误率 %
    bad1 = threshold_metric(disp_pred_t, disp_gt_t, mask_t, threshold=1.0)  # Bad 1 (像素误差 >1)
    bad2 = threshold_metric(disp_pred_t, disp_gt_t, mask_t, threshold=2.0)  # Bad 2
    bad3 = threshold_metric(disp_pred_t, disp_gt_t, mask_t, threshold=3.0)  # Bad 3

    # 6. 打印结果
    print(f"EPE (mean endpoint error)        : {epe.item():.4f} pixels")
    print(f"D1 error rate (>3px & >5%)       : {d1.item():.2f} %")
    print(f"Bad1 error rate (>1px)           : {bad1.item():.2f} %")
    print(f"Bad2 error rate (>2px)           : {bad2.item():.2f} %")
    print(f"Bad3 error rate (>3px)           : {bad3.item():.2f} %")


#########################可视化#################################
    # 3. 灰度化
    gt_gray    = to_gray_uint8(gt_disp)
    recon_gray = to_gray_uint8(recon_disp)

    # 4. 转 BGR，再统一填充到 target_size
    gt_gray_bgr    = cv2.cvtColor(gt_gray,    cv2.COLOR_GRAY2BGR)
    recon_gray_bgr = cv2.cvtColor(recon_gray, cv2.COLOR_GRAY2BGR)
    gt_gray_bgr    = pad_to_target(gt_gray_bgr,    target_size, target_size, border_value=0)
    recon_gray_bgr = pad_to_target(recon_gray_bgr, target_size, target_size, border_value=0)

    # 5. 伪彩
    gt_color    = cv2.applyColorMap(gt_gray,    cv2.COLORMAP_JET)
    recon_color = cv2.applyColorMap(recon_gray, cv2.COLORMAP_JET)
    gt_color    = pad_to_target(gt_color,    target_size, target_size, border_value=0)
    recon_color = pad_to_target(recon_color, target_size, target_size, border_value=0)

    # 6. 2×2 拼接
    top    = cv2.hconcat([gt_gray_bgr, recon_gray_bgr])
    bottom = cv2.hconcat([gt_color,    recon_color])
    canvas = cv2.vconcat([top, bottom])

    # 7. 显示与保存
    win = "Disparity Comparison"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, target_size, target_size)
    cv2.imshow(win, canvas)
    # cv2.imwrite("compare_disparity.png", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()