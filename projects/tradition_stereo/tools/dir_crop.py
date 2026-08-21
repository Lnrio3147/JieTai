import numpy as np
import cv2
import os


def to_gray_uint8(d):
    """归一化并转为 uint8 灰度图"""
    if d.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    valid = d != 0  # 只处理非零值
    if not np.any(valid):
        return np.zeros_like(d, dtype=np.uint8)

    min_val = np.min(d[valid])
    max_val = np.max(d[valid])

    g = np.zeros_like(d, dtype=np.uint8)
    g[valid] = ((d[valid] - min_val) / (max_val - min_val + 1e-6) * 255).astype(np.uint8)
    return g


def resize_to_height(img, target_height):
    """按比例调整图像高度"""
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))


def visualize_disparity_direct_crop(disp_path):
    """
    使用固定裁剪坐标可视化视差图

    参数:
        disp_path: 视差图路径 (.npy文件)
    """
    # 1. 加载视差图
    if not os.path.exists(disp_path):
        print(f"错误：文件不存在 {disp_path}")
        return None

    try:
        disp = np.load(disp_path)
        print(f"成功加载视差图，原始尺寸: {disp.shape[1]}×{disp.shape[0]}")
    except Exception as e:
        print(f"加载视差图失败: {e}")
        return None

    # 2. 应用直接裁剪
    y_start, y_end = 234, 1051  # 行范围
    x_start, x_end = 126, 637  # 列范围

    # 执行固定坐标裁剪
    disp_cropped = disp[y_start:y_end + 1, x_start:x_end + 1]

    # 打印裁剪信息
    cropped_h, cropped_w = disp_cropped.shape
    print(f"直接裁剪参数: y_start={y_start}, y_end={y_end}, x_start={x_start}, x_end={x_end}")
    print(f"裁剪后尺寸: {cropped_w}×{cropped_h} (宽×高)")

    # 3. 可视化处理
    # 目标高度
    target_height = 720

    # 原始视差图处理
    disp_gray = to_gray_uint8(disp)
    disp_color = cv2.applyColorMap(disp_gray, cv2.COLORMAP_JET)

    # 裁剪后视差图处理
    cropped_gray = to_gray_uint8(disp_cropped)
    cropped_color = cv2.applyColorMap(cropped_gray, cv2.COLORMAP_JET)

    # 调整高度
    disp_gray_resized = resize_to_height(disp_gray, target_height)
    disp_color_resized = resize_to_height(disp_color, target_height)
    cropped_gray_resized = resize_to_height(cropped_gray, target_height)
    cropped_color_resized = resize_to_height(cropped_color, target_height)

    # 4. 创建并排比较图
    # 灰度图并排
    gray_comparison = cv2.hconcat([
        cv2.cvtColor(disp_gray_resized, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cropped_gray_resized, cv2.COLOR_GRAY2BGR)
    ])

    # 彩色图并排
    color_comparison = cv2.hconcat([
        disp_color_resized,
        cropped_color_resized
    ])

    # 垂直堆叠
    canvas = cv2.vconcat([gray_comparison, color_comparison])

    # 5. 添加文本标签
    title = f"Depth Map Direct Cropping: {os.path.basename(disp_path)}"
    orig_label = f"Original: {disp.shape[1]}×{disp.shape[0]}"
    cropped_label = f"Cropped: {cropped_w}×{cropped_h}"
    params_label = f"Crop Coordinates: y={y_start}-{y_end}, x={x_start}-{x_end}"

    # 标题
    cv2.putText(canvas, title, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(canvas, params_label, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 图像标签
    # 左侧标签
    cv2.putText(canvas, orig_label, (20, target_height + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # 右侧标签
    right_x_pos = disp_gray_resized.shape[1] + 20
    cv2.putText(canvas, cropped_label, (right_x_pos, target_height + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 底部标签
    cv2.putText(canvas, "Grayscale", (20, target_height * 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "Color Map", (20, target_height * 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 6. 显示结果
    win_name = "Depth Visualization: Direct Cropping"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # 调整窗口大小
    canvas_h, canvas_w = canvas.shape[:2]
    display_w = min(canvas_w, 1920)
    display_h = min(canvas_h, 1080)
    cv2.resizeWindow(win_name, display_w, display_h)

    cv2.imshow(win_name, canvas)

    # 7. 保存结果图像
    output_img_path = os.path.splitext(disp_path)[0] + "_direct_crop_comparison.png"
    cv2.imwrite(output_img_path, canvas)
    print(f"可视化结果已保存至: {output_img_path}")

    # 8. 等待用户操作
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return disp_cropped


if __name__ == "__main__":
    # 示例使用
    disp_path = r"D:\Desktop\项目\传统算法\datasets\FDJYP-3\202506281603-0003\disp.npy"

    # 调用直接裁剪函数 (不需要 minDisparity 和 numDisparities 参数)
    cropped_disp = visualize_disparity_direct_crop(disp_path)

    # 保存裁剪结果
    if cropped_disp is not None:
        output_path = os.path.splitext(disp_path)[0] + "_direct_cropped.npy"
        np.save(output_path, cropped_disp)
        print(f"裁剪结果已保存至: {output_path}")