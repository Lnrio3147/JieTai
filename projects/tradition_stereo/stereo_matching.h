#ifndef STEREO_MATCHING_H
#define STEREO_MATCHING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 点云结构体
typedef struct {
    double x, y, z;
    unsigned char r, g, b;
} Point3D;

// 视差图结构体
typedef struct {
    float* data;
    int width;
    int height;
} DisparityMap; //

// 立体匹配输出上下文（包含裁剪后的数据）
typedef struct {
    float* depth3d;          // 3D坐标数组 (width * height * 3)，存储顺序：x0,y0,z0, x1,y1,z1, ...
    unsigned char* undistort_image; // 裁剪后的左图像 (width * height * 3)
    float* disp;             // 裁剪后的视差图 (width * height)
    int width;               // 裁剪后的宽度
    int height;              // 裁剪后的高度
} StereoOutputContext;

// 立体匹配上下文
typedef struct StereoMatchingContext StereoMatchingContext;

/**
 * @brief 创建立体匹配上下文
 * @param model_path RKNN模型文件路径
 * @return 成功返回上下文指针，失败返回NULL
 */
StereoMatchingContext* stereo_matching_create(const char* model_path);

/**
 * @brief 执行立体匹配
 * @param ctx 立体匹配上下文
 * @param left_img 左图像数据 (RGB格式)
 * @param right_img 右图像数据 (RGB格式)
 * @param width 图像宽度
 * @param height 图像高度
 * @param inference_time_ms 输出推理时间(毫秒)
 * @return 成功返回视差图指针，失败返回NULL
 */
DisparityMap* stereo_matching_infer(StereoMatchingContext* ctx, 
                                   const unsigned char* left_img,
                                   const unsigned char* right_img,
                                   int width, int height,
                                   double* inference_time_ms);

/**
 * @brief 保存视差图为彩色PNG
 * @param disparity 视差图
 * @param output_path 输出文件路径
 * @return 成功返回0，失败返回-1
 */
int stereo_matching_save_disparity_png(const DisparityMap* disparity, const char* output_path);

/**
 * @brief 生成点云文件
 * @param disparity 视差图
 * @param left_img 左图像数据 (RGB格式)
 * @param output_path 输出PLY文件路径
 * @param min_disparity 最小视差阈值（推荐值：5.0）
 * @param max_disparity 最大视差阈值（推荐值：300.0）
 * @param max_z 最大深度阈值（推荐值：200.0，单位mm）
 * @param black_threshold 黑色背景阈值
 * @param Q 重投影矩阵 (4x4)
 * @param output_ctx 输出上下文（可选，传NULL则不输出）。包含裁剪后的深度图、图像和视差图
 * @param bin_output_path 输出二进制 pointmap 文件路径（可选，传NULL则不输出）。格式与 Python batch_process_igev.py 一致
 * @return 成功返回0，失败返回-1
 */
int stereo_matching_generate_point_cloud(const DisparityMap* disparity,
                                        const unsigned char* left_img,
                                        const char* output_path,
                                        float min_disparity,
                                        float max_disparity,
                                        float max_z,
                                        int black_threshold,
                                        double Q[4][4],
                                        StereoOutputContext* output_ctx,
                                        const char* bin_output_path
                                    );

/**
 * @brief 释放视差图内存
 * @param disparity 视差图指针
 */
void stereo_matching_free_disparity(DisparityMap* disparity);

/**
 * @brief 释放输出上下文内存
 * @param output_ctx 输出上下文指针
 */
void stereo_matching_free_output_context(StereoOutputContext* output_ctx);

/**
 * @brief 销毁立体匹配上下文
 * @param ctx 立体匹配上下文指针
 */
void stereo_matching_destroy(StereoMatchingContext* ctx);

/**
 * @brief 获取错误信息
 * @param ctx 立体匹配上下文
 * @return 错误信息字符串
 */
const char* stereo_matching_get_error(StereoMatchingContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // STEREO_MATCHING_H