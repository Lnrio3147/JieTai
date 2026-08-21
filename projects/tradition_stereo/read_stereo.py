import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
# 从YAML文件加载相机标定参数
def load_calibration(filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"无法打开文件: {filename}")

    calibration = {}
    # 读取左相机参数
    calibration['M1'] = fs.getNode("M1").mat()  # 左相机内参矩阵
    calibration['D1'] = fs.getNode("D1").mat()  # 左相机畸变系数

    # 读取右相机参数
    calibration['M2'] = fs.getNode("M2").mat()  # 右相机内参矩阵
    calibration['D2'] = fs.getNode("D2").mat()  # 右相机畸变系数

    # 读取外参
    calibration['R'] = fs.getNode("R").mat()  # 旋转矩阵
    calibration['T'] = fs.getNode("T").mat()  # 平移向量

    # 读取校正参数（直接从文件获取，不需要计算）
    calibration['R1'] = fs.getNode("R1").mat()  # 左相机校正旋转矩阵
    calibration['R2'] = fs.getNode("R2").mat()  # 右相机校正旋转矩阵
    calibration['P1'] = fs.getNode("P1").mat()  # 左相机投影矩阵
    calibration['P2'] = fs.getNode("P2").mat()  # 右相机投影矩阵
    calibration['Q'] = fs.getNode("Q").mat()  # 视差转深度矩阵

    fs.release()
    return calibration

def draw_horizontal_lines(img, interval=40, color=(0, 255, 0), thickness=1):
    """
    在 img 上每隔 interval 像素画一条水平直线
    """
    h, w = img.shape[:2]
    for y in range(0, h, interval):
        cv2.line(img, (0, y), (w, y), color, thickness)
    return img



# 主程序
if __name__ == "__main__":
    # 加载标定参数
    stereo_gongjian = "./config/stereo_gongjian.yml"
    stereo_new = "stereo_new.yml"
    stereo_other = "stereo_other.yml"
    stereo_jxp = "stereo_jxp.yml"
    stereo_luowen = "./config/stereo_luowen.yml"
    stereo_calib = "config\stereo_calib.yaml"
    stereo = "config\stereo.yml"

    calib = load_calibration(stereo)

    # 从标定数据中提取参数
    left_camera_matrix = calib['M1']
    left_distortion = calib['D1']
    # left_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


    right_camera_matrix = calib['M2']
    right_distortion = calib['D2']
    # right_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    R = calib['R']
    T = calib['T']


    # size = (800, 800)
    size = (720, 1280)
    #测试自己生成
    # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
    #                                                                   right_camera_matrix, right_distortion, size, R,
    #                                                                   T)

    # 直接从标定文件获取校正参数（不需要调用stereoRectify）
    R1 = calib['R1']
    R2 = calib['R2']
    P1 = calib['P1']
    P2 = calib['P2']
    Q = calib['Q']

    # # # 图像尺寸（从标定参数推断）
    # cx = int(left_camera_matrix[0, 2])  # 光心x坐标
    # cy = int(left_camera_matrix[1, 2])  # 光心y坐标
    # # 假设图像尺寸为光心坐标的2倍（常见设置）
    # width = int(2 * cx)
    # height = int(2 * cy)
    # # size = (720, 1280)

    # 创建校正映射表（使用文件中的校正参数）
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix,
        left_distortion,
        R1,  # 使用文件中的R1
        P1,  # 使用文件中的P1
        size,
        cv2.CV_16SC2
    )

    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix,
        right_distortion,
        R2,  # 使用文件中的R2
        P2,  # 使用文件中的P2
        size,
        cv2.CV_16SC2
    )

    # 读取图片
    # im_path = r"./datasets/FDJYP-3/202506281603-0001"
    # im_path = r"./datasets/gongjian_test/1"
    # im_path = r"rectified_images\202506281603-0001"
    im_path = r"D:\Desktop\output\202511201052-0016"
    frame1 = cv2.imread(os.path.join(im_path, "im0.png"))  # 左视图
    frame2 = cv2.imread(os.path.join(im_path, "im1.png"))  # 右视图

    # 使用映射表进行图像校正（直接使用BGR图像）
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)

    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)


    # 检查矫正图片
    # 设置显示尺寸
    display_size = (360, 640)
    # display_size = (400, 400)
    # 缩放图像到统一大小
    img1_resized = cv2.resize(img1_rectified, display_size)
    img2_resized = cv2.resize(img2_rectified, display_size)

    # 在每张图上画水平参考线
    img1_with_lines = draw_horizontal_lines(img1_resized.copy(), interval=40)
    img2_with_lines = draw_horizontal_lines(img2_resized.copy(), interval=40)

    # 拼接图像：水平堆叠
    concatenated = np.hstack((img1_with_lines, img2_with_lines))
    # 显示拼接后的图像
    cv2.imshow("Rectified Stereo Pair", concatenated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # # 打印Q矩阵（视差转深度矩阵）
    # print("Q矩阵（视差转深度）:")
    # print(Q)
    #
    # 保存映射表供后续使用（可选）
    np.save("left_map1.npy", left_map1)
    np.save("left_map2.npy", left_map2)
    np.save("right_map1.npy", right_map1)
    np.save("right_map2.npy", right_map2)



