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


def visualize_disparity_with_algorithm_crop(disp_path, minDisparity, numDisparities):
    """
    使用OpenCV的imshow可视化视差图及其裁剪版本（基于视差算法参数裁剪）

    参数:
        disp_path: 视差图路径 (.npy文件)
        minDisparity: 最小视差值 (通常为负数)
        numDisparities: 视差范围数
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

    h, w = disp.shape[:2]

    # 2. 应用算法裁剪逻辑
    edge = abs(minDisparity) // 2
    edgeL = minDisparity + numDisparities
    print(f"裁剪参数: minDisparity={minDisparity}, numDisparities={numDisparities}")
    print(f"计算edge={edge}, edgeL={edgeL}")

    # 计算初始ROI
    roi_x = edgeL
    roi_y = edge // 2
    roi_w = w - 2 * edgeL
    roi_h = h - edge

    print(f"初始裁剪区域: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

    # 调整ROI坐标确保在图像范围内
    roi_x = max(0, min(roi_x, w - 1))
    roi_y = max(0, min(roi_y, h - 1))
    roi_w = max(10, min(roi_w, w - roi_x))
    roi_h = max(10, min(roi_h, h - roi_y))

    # 计算宽高比
    k = roi_w / roi_h
    print(f"初始宽高比: {k:.3f}")

    # 长宽比调整
    if k > 1.8:  # 图像过宽
        new_w = (roi_h * 16 // 10) // 2 * 2
        offset = ((roi_w - new_w) // 2) // 2 * 2
        print(f"过宽处理: new_w={new_w}, offset={offset}")
        roi_x += offset
        roi_w = new_w
    elif (1 / k) > 1.8:  # 图像过高
        new_h = (roi_w * 16 // 10) // 2 * 2
        offset = ((roi_h - new_h) // 2) // 2 * 2
        print(f"过高处理: new_h={new_h}, offset={offset}")
        roi_y += offset
        roi_h = new_h

    # 最终调整确保在图像范围内
    roi_x = max(0, min(roi_x, w - 1))
    roi_y = max(0, min(roi_y, h - 1))
    roi_w = max(10, min(roi_w, w - roi_x))
    roi_h = max(10, min(roi_h, h - roi_y))

    print(f"最终裁剪区域: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

    # 执行裁剪
    disp_cropped = disp[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 3. 简单可视化：只显示裁剪前和裁剪后结果
    # 目标高度
    target_height = 720

    # 裁剪前处理
    disp_gray = to_gray_uint8(disp)
    disp_color = cv2.applyColorMap(disp_gray, cv2.COLORMAP_JET)

    # 裁剪后处理
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
    title = f"Depth Map Cropping: {os.path.basename(disp_path)}"
    orig_label = f"Original: {w}×{h}"
    cropped_label = f"Cropped: {roi_w}×{roi_h}"
    params_label = f"Params: minD={minDisparity}, numD={numDisparities}"

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
    cv2.putText(canvas, cropped_label, (disp_gray_resized.shape[1] + 20, target_height + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 底部标签
    cv2.putText(canvas, "Grayscale", (20, target_height * 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "Color Map", (20, target_height * 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 6. 显示结果
    win_name = "Depth Visualization: Cropping Comparison"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # 调整窗口大小以适应内容
    canvas_h, canvas_w = canvas.shape[:2]
    display_w = min(canvas_w, 1920)
    display_h = min(canvas_h, 1080)
    cv2.resizeWindow(win_name, display_w, display_h)

    cv2.imshow(win_name, canvas)

    # 7. 保存结果图像
    output_img_path = os.path.splitext(disp_path)[0] + "_comparison.png"
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
    minDisparity = -104
    numDisparities = 208

    cropped_disp = visualize_disparity_with_algorithm_crop(
        disp_path, minDisparity, numDisparities
    )

    # 可选：保存裁剪结果
    if cropped_disp is not None:
        output_path = os.path.splitext(disp_path)[0] + "_cropped.npy"
        np.save(output_path, cropped_disp)
        print(f"裁剪结果已保存至: {output_path}")