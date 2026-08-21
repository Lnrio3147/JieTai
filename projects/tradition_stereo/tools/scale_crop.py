import numpy as np
import cv2
import os


def to_gray_uint8(d):
    """归一化并转为 uint8 灰度图"""
    if d.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    min_val = np.min(d)
    max_val = np.max(d)

    if min_val == max_val:
        return np.zeros_like(d, dtype=np.uint8)

    g = (d - min_val) / (max_val - min_val) * 255
    return g.astype(np.uint8)


def resize_to_height(img, target_height):
    """按比例调整图像高度"""
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))


def visualize_disparity_with_imshow(disp_path, crop_ratio_h, crop_ratio_w):
    """
    使用OpenCV的imshow可视化视差图及其中心裁剪版本（按比例裁剪）

    参数:
        disp_path: 视差图路径 (.npy文件)
        crop_ratio_h: 高度裁剪比例 (0~1之间)
        crop_ratio_w: 宽度裁剪比例 (0~1之间)
    """
    # 验证裁剪比例是否在有效范围内
    if not (0 < crop_ratio_h <= 1):
        print(f"错误：高度裁剪比例 {crop_ratio_h} 必须在(0,1]范围内")
        return None
    if not (0 < crop_ratio_w <= 1):
        print(f"错误：宽度裁剪比例 {crop_ratio_w} 必须在(0,1]范围内")
        return None

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

    # 2. 验证尺寸
    h, w = disp.shape[:2]
    if (h, w) != (1280, 720):
        print(f"警告: 视差图尺寸 {w}×{h} 不是720×1280。将按实际尺寸处理")

    # 3. 按比例计算裁剪尺寸
    crop_h = int(h * crop_ratio_h)
    crop_w = int(w * crop_ratio_w)
    print(f"裁剪比例: {crop_ratio_h:.2f} (高), {crop_ratio_w:.2f} (宽)")
    print(f"裁剪像素尺寸: {crop_w}×{crop_h} (宽×高)")

    # 确保裁剪尺寸不超过原图
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    # 计算中心裁剪起始点
    start_y = max(0, (h - crop_h) // 2)
    start_x = max(0, (w - crop_w) // 2)

    disp_cropped = disp[start_y:start_y + crop_h, start_x:start_x + crop_w]
    print(f"裁剪后实际尺寸: {disp_cropped.shape[1]}×{disp_cropped.shape[0]} (宽×高)")

    # 4. 创建可视化组件
    # 原始图 - 灰度和彩色
    full_gray = to_gray_uint8(disp)
    full_color = cv2.applyColorMap(full_gray, cv2.COLORMAP_JET)

    # 裁剪图 - 灰度和彩色
    cropped_gray = to_gray_uint8(disp_cropped)
    cropped_color = cv2.applyColorMap(cropped_gray, cv2.COLORMAP_JET)

    # 5. 统一图像高度以便拼接
    target_height = max(full_gray.shape[0], cropped_gray.shape[0])

    # 调整原始图像高度
    if full_gray.shape[0] != target_height:
        full_gray = resize_to_height(full_gray, target_height)
        full_color = resize_to_height(full_color, target_height)

    # 调整裁剪图像高度
    if cropped_gray.shape[0] != target_height:
        cropped_gray = resize_to_height(cropped_gray, target_height)
        cropped_color = resize_to_height(cropped_color, target_height)

    # 6. 创建组合图像
    # 水平拼接灰度图
    gray_composite = cv2.hconcat([
        cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cropped_gray, cv2.COLOR_GRAY2BGR)
    ])

    # 水平拼接彩色图
    color_composite = cv2.hconcat([full_color, cropped_color])

    # 垂直拼接所有图像
    canvas = cv2.vconcat([gray_composite, color_composite])

    # 7. 添加文本标签
    full_label = f"Original: {w}×{h}"
    ratio_label = f"Crop Ratio: ({crop_ratio_h:.2f}, {crop_ratio_w:.2f})"
    cropped_label = f"Cropped Size: {disp_cropped.shape[1]}×{disp_cropped.shape[0]}"

    # 在图像上添加文本
    cv2.putText(canvas, full_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # 在右侧位置显示裁剪信息
    cv2.putText(canvas, ratio_label, (full_gray.shape[1] + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(canvas, cropped_label, (full_gray.shape[1] + 20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 添加区域标签
    cv2.putText(canvas, "Grayscale", (20, target_height + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "Color Map", (20, target_height * 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 添加文件名标题
    title = f"Depth Visualization: {os.path.basename(disp_path)}"
    cv2.putText(canvas, title, (w // 2 - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 8. 显示结果
    win_name = "Depth Visualization"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # 调整窗口大小以适应内容
    canvas_h, canvas_w = canvas.shape[:2]
    display_w = min(canvas_w, 1920)
    display_h = min(canvas_h, 1080)
    cv2.resizeWindow(win_name, display_w, display_h)

    cv2.imshow(win_name, canvas)

    # 9. 等待用户操作
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return disp_cropped


if __name__ == "__main__":
    # 示例使用
    disp_path = r"../rectified_images\202506281603-0001\disp.npy"

    # 裁剪比例示例（高度保留2/3，宽度保留3/4）
    crop_ratio_h = 2 / 3  # 高度裁剪比例
    crop_ratio_w = 2 / 3  # 宽度裁剪比例

    cropped_disp = visualize_disparity_with_imshow(disp_path, crop_ratio_h, crop_ratio_w)

    # 可选：保存裁剪结果
    if cropped_disp is not None:
        output_path = os.path.splitext(disp_path)[0] + "_cropped.npy"
        np.save(output_path, cropped_disp)
        print(f"裁剪结果已保存至: {output_path}")