#ifdef STEREO_MATCHING_EXPORTS
#define STEREO_MATCHING_API __declspec(dllexport)
#else
#define STEREO_MATCHING_API __declspec(dllimport)
#endif
#include "stereo_matching.h"



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "rknn_api.h"

// 内部上下文结构
struct StereoMatchingContext {
    rknn_context rknn_ctx;
    char error_msg[256];
    int divis_by;
};

// 计时函数
static double get_current_time() {
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1000.0 + tv.tv_nsec / 1000000.0;
}

// 图像填充函数
static unsigned char* pad_image_replicate(unsigned char* img, int h, int w, int c, 
                                         int pad_left, int pad_right, int pad_top, int pad_bottom, 
                                         int* out_h, int* out_w) {
    *out_h = h + pad_top + pad_bottom;
    *out_w = w + pad_left + pad_right;
    unsigned char* padded = (unsigned char*)malloc(*out_h * *out_w * c);
    if (!padded) return NULL;

    for (int y = 0; y < *out_h; y++) {
        for (int x = 0; x < *out_w; x++) {
            int src_y = y - pad_top;
            if (src_y < 0) src_y = 0;
            if (src_y >= h) src_y = h - 1;
            
            int src_x = x - pad_left;
            if (src_x < 0) src_x = 0;
            if (src_x >= w) src_x = w - 1;
            
            for (int i = 0; i < c; i++) {
                padded[(y * *out_w + x) * c + i] = img[(src_y * w + src_x) * c + i];
            }
        }
    }
    return padded;
}

// 计算填充量
static void calculate_padding(int height, int width, int divis_by, 
                             int* pad_top, int* pad_bottom, int* pad_left, int* pad_right) {
    int pad_ht = (((height / divis_by) + 1) * divis_by - height) % divis_by;
    int pad_wd = (((width / divis_by) + 1) * divis_by - width) % divis_by;
    
    *pad_left = pad_wd / 2;
    *pad_right = pad_wd - *pad_left;
    *pad_top = pad_ht / 2;
    *pad_bottom = pad_ht - *pad_top;
}

// 裁剪视差图
static float* crop_disparity(float* disparity, int src_h, int src_w, 
                            int pad_top, int pad_bottom, int pad_left, int pad_right, 
                            int* out_h, int* out_w) {
    *out_h = src_h - pad_top - pad_bottom;
    *out_w = src_w - pad_left - pad_right;
    
    float* cropped = (float*)malloc(*out_h * *out_w * sizeof(float));
    if (!cropped) return NULL;
    
    for (int y = 0; y < *out_h; y++) {
        for (int x = 0; x < *out_w; x++) {
            int src_y = y + pad_top;
            int src_x = x + pad_left;
            cropped[y * *out_w + x] = disparity[src_y * src_w + src_x];
        }
    }
    return cropped;
}

// 精确的重投影计算（带过滤）
static void precise_reprojection(float* disparity, unsigned char* left_img, int width, int height,
                                double Q[4][4], Point3D* point_cloud, int* valid_count,
                                int start_x, int start_y, int roi_width, int roi_height,
                                float min_disparity, float max_disparity, float max_z, int black_threshold) {
    *valid_count = 0;

    for (int y = 0; y < roi_height; y++) {
        for (int x = 0; x < roi_width; x++) {
            int src_y = y + start_y;
            int src_x = x + start_x;

            if (src_y >= height || src_x >= width) continue;

            float d = disparity[src_y * width + src_x];

            // 1. 基础有效性检查
            if (d <= 0) continue;

            // 2. 视差过滤 - 过滤掉过小和过大的视差值（与Python一致）
            if (d < min_disparity || d > max_disparity) continue;

            // 3. 黑色背景过滤 - RGB任一通道 > black_threshold 则保留
            int img_idx = (src_y * width + src_x) * 3;
            unsigned char r = left_img[img_idx + 0];
            unsigned char g = left_img[img_idx + 1];
            unsigned char b = left_img[img_idx + 2];

            if (r <= black_threshold && g <= black_threshold && b <= black_threshold) {
                continue;  // 过滤掉纯黑或接近黑色的像素
            }

            // 完整的重投影计算
            double w = Q[3][0] * src_x + Q[3][1] * src_y + Q[3][2] * d + Q[3][3];
            if (fabs(w) < 1e-12) continue;

            double X = (Q[0][0] * src_x + Q[0][1] * src_y + Q[0][2] * d + Q[0][3]) / w;
            double Y = (Q[1][0] * src_x + Q[1][1] * src_y + Q[1][2] * d + Q[1][3]) / w;
            double Z = (Q[2][0] * src_x + Q[2][1] * src_y + Q[2][2] * d + Q[2][3]) / w;

            if (!isfinite(X) || !isfinite(Y) || !isfinite(Z)) continue;
            // 4. Z范围过滤 - 只保留 0 < Z < max_z 的点（与Python一致）
            if (Z <= 0 || Z >= max_z) continue;

            // 保存点云数据
            point_cloud[*valid_count].x = X;
            point_cloud[*valid_count].y = Y;
            point_cloud[*valid_count].z = Z;
            point_cloud[*valid_count].r = r;
            point_cloud[*valid_count].g = g;
            point_cloud[*valid_count].b = b;

            (*valid_count)++;
        }
    }
}

// 创建立体匹配上下文
StereoMatchingContext* stereo_matching_create(const char* model_path) {
    StereoMatchingContext* ctx = (StereoMatchingContext*)malloc(sizeof(StereoMatchingContext));
    if (!ctx) {
        return NULL;
    }
    
    memset(ctx, 0, sizeof(StereoMatchingContext));
    ctx->divis_by = 32;  // 使用32的倍数填充
    
    // 加载RKNN模型
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Cannot open model file: %s", model_path);
        free(ctx);
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* model_data = malloc(model_size);
    if (!model_data) {
        fclose(fp);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Failed to allocate model buffer");
        free(ctx);
        return NULL;
    }
    
    if (fread(model_data, 1, model_size, fp) != model_size) {
        fclose(fp);
        free(model_data);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Failed to read model file");
        free(ctx);
        return NULL;
    }
    fclose(fp);
    
    // 初始化RKNN
    int ret = rknn_init(&ctx->rknn_ctx, model_data, model_size, 0, NULL);
    free(model_data);
    
    if (ret != RKNN_SUCC) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "RKNN init failed: %d", ret);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

// 执行立体匹配
DisparityMap* stereo_matching_infer(StereoMatchingContext* ctx, 
                                   const unsigned char* left_img,
                                   const unsigned char* right_img,
                                   int width, int height,
                                   double* inference_time_ms) {
    if (!ctx || !left_img || !right_img) {
        if (ctx) snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Invalid parameters");
        return NULL;
    }
    
    // 计算填充量
    int pad_top, pad_bottom, pad_left, pad_right;
    calculate_padding(height, width, ctx->divis_by, &pad_top, &pad_bottom, &pad_left, &pad_right);
    
    // 填充图像
    int padded_h, padded_w;
    unsigned char* left_padded = pad_image_replicate((unsigned char*)left_img, height, width, 3, 
                                                   pad_left, pad_right, pad_top, pad_bottom, 
                                                   &padded_h, &padded_w);
    unsigned char* right_padded = pad_image_replicate((unsigned char*)right_img, height, width, 3, 
                                                    pad_left, pad_right, pad_top, pad_bottom, 
                                                    &padded_h, &padded_w);
    
    if (!left_padded || !right_padded) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Failed to pad images");
        if (left_padded) free(left_padded);
        if (right_padded) free(right_padded);
        return NULL;
    }
    
    // 设置RKNN输入
    rknn_input inputs[2] = {0};
    inputs[0].index = 0;
    inputs[0].buf = left_padded;
    inputs[0].size = padded_w * padded_h * 3;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    
    inputs[1].index = 1;
    inputs[1].buf = right_padded;
    inputs[1].size = padded_w * padded_h * 3;
    inputs[1].type = RKNN_TENSOR_UINT8;
    inputs[1].fmt = RKNN_TENSOR_NHWC;
    
    
    int ret = rknn_inputs_set(ctx->rknn_ctx, 2, inputs);
    if (ret != RKNN_SUCC) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "RKNN inputs set failed: %d", ret);
        free(left_padded);
        free(right_padded);
        return NULL;
    }

    // 记录推理时间
    double start_time = get_current_time();
    
    // 执行推理
    ret = rknn_run(ctx->rknn_ctx, NULL);
    if (ret != RKNN_SUCC) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "RKNN run failed: %d", ret);
        free(left_padded);
        free(right_padded);
        return NULL;
    }
    
    double end_time = get_current_time();
    if (inference_time_ms) {
        *inference_time_ms = end_time - start_time;
    }
    
    // 获取输出
    rknn_output output = {0};
    output.index = 0;
    output.want_float = 1;
    output.is_prealloc = 0;
    
    ret = rknn_outputs_get(ctx->rknn_ctx, 1, &output, NULL);
    if (ret != RKNN_SUCC) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "RKNN outputs get failed: %d", ret);
        free(left_padded);
        free(right_padded);
        return NULL;
    }
    
    // 裁剪视差图
    int cropped_h, cropped_w;
    float* cropped_disparity = crop_disparity((float*)output.buf, padded_h, padded_w, 
                                            pad_top, pad_bottom, pad_left, pad_right, 
                                            &cropped_h, &cropped_w);
    
    // 释放临时资源
    free(left_padded);
    free(right_padded);
    rknn_outputs_release(ctx->rknn_ctx, 1, &output);
    
    if (!cropped_disparity) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Failed to crop disparity map");
        return NULL;
    }
    
    // 创建视差图结构
    DisparityMap* disparity = (DisparityMap*)malloc(sizeof(DisparityMap));
    if (!disparity) {
        free(cropped_disparity);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Failed to allocate disparity map");
        return NULL;
    }
    
    disparity->data = cropped_disparity;
    disparity->width = cropped_w;
    disparity->height = cropped_h;
    
    return disparity;
}

// 保存视差图为彩色PNG
#include <opencv2/opencv.hpp>

int stereo_matching_save_disparity_png(const DisparityMap* disparity, const char* output_path) {
    if (!disparity || !disparity->data || !output_path) {
        return -1;
    }
    
    // 计算视差范围
    float min_val = __FLT_MAX__, max_val = -__FLT_MAX__;
    int total_pixels = disparity->width * disparity->height;
    
    for (int i = 0; i < total_pixels; i++) {
        float d = disparity->data[i];
        if (d < min_val) min_val = d;
        if (d > max_val) max_val = d;
    }
    
    float range = max_val - min_val;
    if (range < 1e-6) {
        range = 1.0f;
        min_val = 0.0f;
    }
    
    // 创建归一化的视差图 (0-255)
    cv::Mat normalized_disparity(disparity->height, disparity->width, CV_8UC1);
    for (int y = 0; y < disparity->height; y++) {
        for (int x = 0; x < disparity->width; x++) {
            int idx = y * disparity->width + x;
            float d = disparity->data[idx];
            float normalized = (d - min_val) / range;
            normalized = (normalized < 0.0f) ? 0.0f : (normalized > 1.0f) ? 1.0f : normalized;
            normalized_disparity.at<unsigned char>(y, x) = (unsigned char)(normalized * 255.0f);
        }
    }
    
    // 应用 OpenCV 的 JET 色彩映射
    cv::Mat color_disparity;
    cv::applyColorMap(normalized_disparity, color_disparity, cv::COLORMAP_JET);
    
    // 使用 OpenCV 保存为 PNG
    bool success = cv::imwrite(output_path, color_disparity);
    return success ? 0 : -1;
}


// 使用与Python代码相同的Q矩阵
double Q[4][4] = {
    {1.0, 0.0, 0.0, -312.7411},
    {0.0, 1.0, 0.0, -663.5256},
    {0.0, 0.0, 0.0, 877.7027},
    {0.0, 0.0, 0.3976856, 0.0}
};

// 生成点云文件（带过滤参数）#ASCII格式
// int stereo_matching_generate_point_cloud(const DisparityMap* disparity,
//                                         const unsigned char* left_img,
//                                         const char* output_path,
//                                         float min_disparity,
//                                         int black_threshold,
//                                         double Q[4][4]
//                                     ) {
//     if (!disparity || !disparity->data || !left_img || !output_path) {
//         return -1;
//     }
    
    
    
//     // 应用与Python代码相同的裁剪参数
//     int minDisparity = -104;
//     int numDisparities = 208;
//     int edge = abs(minDisparity) / 2;
//     int edgeL = minDisparity + numDisparities;
    
//     int start_x = edgeL;
//     int start_y = edge / 2;
//     int roi_width = disparity->width - 2 * edgeL;
//     int roi_height = disparity->height - edge;
    
//     // 动态调整
//     double k = (double)roi_width / roi_height;
//     if (k > 1.8) {
//         int h = (roi_height * 16 / 10) / 2 * 2;
//         int offset = (roi_width - h) / 4 * 2;
//         start_x += offset;
//         roi_width = h;
//     } else if (1.0 / k > 1.8) {
//         int h = (roi_width * 16 / 10) / 2 * 2;
//         int offset = (roi_height - h) / 4 * 2;
//         start_y += offset;
//         roi_height = h;
//     }
    
//     // 边界检查
//     if (start_x < 0) start_x = 0;
//     if (start_y < 0) start_y = 0;
//     if (start_x + roi_width > disparity->width) roi_width = disparity->width - start_x;
//     if (start_y + roi_height > disparity->height) roi_height = disparity->height - start_y;
    
//     // 统计最大可能点数
//     int max_points = roi_width * roi_height;
//     Point3D* point_cloud = (Point3D*)malloc(max_points * sizeof(Point3D));
//     if (!point_cloud) {
//         return -1;
//     }
    
//     int valid_count = 0;
//     precise_reprojection(disparity->data, (unsigned char*)left_img, disparity->width, disparity->height,
//                         Q, point_cloud, &valid_count, start_x, start_y, roi_width, roi_height,
//                         min_disparity, black_threshold);

//     if (valid_count == 0) {
//         free(point_cloud);
//         return -1;
//     }
    
//     // 保存PLY文件
//     FILE* ply_file = fopen(output_path, "w");
//     if (!ply_file) {
//         free(point_cloud);
//         return -1;
//     }
    
//     // 写入PLY头部
//     fprintf(ply_file, "ply\n");
//     fprintf(ply_file, "format ascii 1.0\n");
//     fprintf(ply_file, "element vertex %d\n", valid_count);
//     fprintf(ply_file, "property float x\n");
//     fprintf(ply_file, "property float y\n");
//     fprintf(ply_file, "property float z\n");
//     fprintf(ply_file, "property uchar red\n");
//     fprintf(ply_file, "property uchar green\n");
//     fprintf(ply_file, "property uchar blue\n");
//     fprintf(ply_file, "end_header\n");
    
//     // 写入点云数据
//     for (int i = 0; i < valid_count; i++) {
//         fprintf(ply_file, "%.8f %.8f %.8f %d %d %d\n", 
//                 (float)point_cloud[i].x, (float)point_cloud[i].y, (float)point_cloud[i].z,
//                 point_cloud[i].r, point_cloud[i].g, point_cloud[i].b);
//     }
    
//     fclose(ply_file);
//     free(point_cloud);
    
//     return 0;
// }

// 保存 pointmap 为二进制格式（与 Python batch_process_igev.py 中的格式一致）
static int save_pointmap_binary(const StereoOutputContext* output_ctx, const char* bin_path) {
    if (!output_ctx || !bin_path) {
        return -1;
    }

    FILE* f = fopen(bin_path, "wb");
    if (!f) {
        return -1;
    }

    int h = output_ctx->height;
    int w = output_ctx->width;
    int c = 6;  // XYZRGB

    // 1. 魔法数 (4 bytes): "PMAP"
    fwrite("PMAP", 1, 4, f);

    // 2. 版本号 (4 bytes, uint32, little-endian)
    uint32_t version = 1;
    fwrite(&version, sizeof(uint32_t), 1, f);

    // 3. 高度 (4 bytes, uint32)
    uint32_t height_u32 = (uint32_t)h;
    fwrite(&height_u32, sizeof(uint32_t), 1, f);

    // 4. 宽度 (4 bytes, uint32)
    uint32_t width_u32 = (uint32_t)w;
    fwrite(&width_u32, sizeof(uint32_t), 1, f);

    // 5. 通道数 (4 bytes, uint32)
    uint32_t channels_u32 = (uint32_t)c;
    fwrite(&channels_u32, sizeof(uint32_t), 1, f);

    // 6. 数据类型 (4 bytes, uint32): 0 = float32
    uint32_t dtype = 0;
    fwrite(&dtype, sizeof(uint32_t), 1, f);

    // 7. 通道顺序 (24 bytes, ASCII): "XYZRGB" + 填充
    char channel_order[24] = "XYZRGB";
    memset(channel_order + 6, 0, 18);  // 填充剩余18字节
    fwrite(channel_order, 1, 24, f);

    // 8. 保留字段 (20 bytes)
    char reserved[20] = {0};
    fwrite(reserved, 1, 20, f);

    // 头部总共 64 bytes

    // 9. 写入数据部分 (H × W × 6 × 4 bytes)
    // 每个像素包含 6 个 float32: [X, Y, Z, R, G, B]
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            int depth_idx = idx * 3;
            int img_idx = idx * 3;

            // XYZ (float32)
            float xyz[3];
            xyz[0] = output_ctx->depth3d[depth_idx + 0];
            xyz[1] = output_ctx->depth3d[depth_idx + 1];
            xyz[2] = output_ctx->depth3d[depth_idx + 2];
            fwrite(xyz, sizeof(float), 3, f);

            // RGB (转换为 float32，范围 0-255)
            float rgb[3];
            rgb[0] = (float)output_ctx->undistort_image[img_idx + 0];
            rgb[1] = (float)output_ctx->undistort_image[img_idx + 1];
            rgb[2] = (float)output_ctx->undistort_image[img_idx + 2];
            fwrite(rgb, sizeof(float), 3, f);
        }
    }

    fclose(f);
    return 0;
}

// 生成点云文件（带过滤参数）- 二进制格式
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
                                    ) {
    if (!disparity || !disparity->data || !left_img || !output_path) {
        return -1;
    }
    
    // 应用与Python代码相同的裁剪参数
    int minDisparity = -104;
    int numDisparities = 208;
    int edge = abs(minDisparity) / 2;
    int edgeL = minDisparity + numDisparities;
    
    int start_x = edgeL;
    int start_y = edge / 2;
    int roi_width = disparity->width - 2 * edgeL;
    int roi_height = disparity->height - edge;
    
    // 动态调整
    double k = (double)roi_width / roi_height;
    if (k > 1.8) {
        int h = (roi_height * 16 / 10) / 2 * 2;
        int offset = (roi_width - h) / 4 * 2;
        start_x += offset;
        roi_width = h;
    } else if (1.0 / k > 1.8) {
        int h = (roi_width * 16 / 10) / 2 * 2;
        int offset = (roi_height - h) / 4 * 2;
        start_y += offset;
        roi_height = h;
    }
    
    // 边界检查
    if (start_x < 0) start_x = 0;
    if (start_y < 0) start_y = 0;
    if (start_x + roi_width > disparity->width) roi_width = disparity->width - start_x;
    if (start_y + roi_height > disparity->height) roi_height = disparity->height - start_y;

    // ========== 双边滤波（与Python一致）==========
    cv::Mat disp_mat(disparity->height, disparity->width, CV_32FC1, disparity->data);
    cv::Mat disp_roi = disp_mat(cv::Rect(start_x, start_y, roi_width, roi_height)).clone();
    cv::Mat disp_filtered;
    cv::bilateralFilter(disp_roi, disp_filtered, 5, 50, 50);  // d=5, sigmaColor=50, sigmaSpace=50

    // 创建滤波后的完整视差图副本
    float* filtered_disparity = (float*)malloc(disparity->width * disparity->height * sizeof(float));
    if (!filtered_disparity) {
        return -1;
    }
    memcpy(filtered_disparity, disparity->data, disparity->width * disparity->height * sizeof(float));

    // 将滤波后的ROI复制回去
    for (int y = 0; y < roi_height; y++) {
        for (int x = 0; x < roi_width; x++) {
            filtered_disparity[(y + start_y) * disparity->width + (x + start_x)] = disp_filtered.at<float>(y, x);
        }
    }
    // ========== 双边滤波结束 ==========

    // 统计最大可能点数
    int max_points = roi_width * roi_height;
    Point3D* point_cloud = (Point3D*)malloc(max_points * sizeof(Point3D));
    if (!point_cloud) {
        free(filtered_disparity);
        return -1;
    }

    int valid_count = 0;
    precise_reprojection(filtered_disparity, (unsigned char*)left_img, disparity->width, disparity->height,
                        Q, point_cloud, &valid_count, start_x, start_y, roi_width, roi_height,
                        min_disparity, max_disparity, max_z, black_threshold);

    if (valid_count == 0) {
        free(point_cloud);
        free(filtered_disparity);
        return -1;
    }

    // 如果提供了output_ctx，填充裁剪后的数据
    if (output_ctx) {
        // 分配内存
        output_ctx->width = roi_width;
        output_ctx->height = roi_height;   
        output_ctx->depth3d = (float*)malloc(roi_width * roi_height * 3 * sizeof(float));
        output_ctx->undistort_image = (unsigned char*)malloc(roi_width * roi_height * 3);
        output_ctx->disp = (float*)malloc(roi_width * roi_height * sizeof(float));

        if (!output_ctx->depth3d || !output_ctx->undistort_image || !output_ctx->disp) {
            // 内存分配失败，清理已分配的内存
            if (output_ctx->depth3d) free(output_ctx->depth3d);
            if (output_ctx->undistort_image) free(output_ctx->undistort_image);
            if (output_ctx->disp) free(output_ctx->disp);
            free(point_cloud);
            return -1;
        }

        // 初始化为0
        memset(output_ctx->depth3d, 0, roi_width * roi_height * 3 * sizeof(float));
        memset(output_ctx->undistort_image, 0, roi_width * roi_height * 3);
        memset(output_ctx->disp, 0, roi_width * roi_height * sizeof(float));

        // 填充数据
        for (int y = 0; y < roi_height; y++) {
            for (int x = 0; x < roi_width; x++) {
                int src_y = y + start_y;
                int src_x = x + start_x;

                if (src_y >= disparity->height || src_x >= disparity->width) continue;

                int dst_idx = y * roi_width + x;
                int src_idx = src_y * disparity->width + src_x;

                // 复制视差值（使用滤波后的视差）
                float d = filtered_disparity[src_idx];
                output_ctx->disp[dst_idx] = d;

                // 索引计算
                int img_src_idx = src_idx * 3;
                int img_dst_idx = dst_idx * 3;

                // 应用完整过滤逻辑（与 Python batch_process_igev.py 一致）
                // 只有通过所有过滤条件的像素才会赋值 XYZ 和 RGB
                if (d > 0 && d >= min_disparity && d <= max_disparity) {
                    // 黑色背景过滤 - 直接从 left_img 读取
                    unsigned char r = left_img[img_src_idx + 0];
                    unsigned char g = left_img[img_src_idx + 1];
                    unsigned char b = left_img[img_src_idx + 2];

                    if (r > black_threshold || g > black_threshold || b > black_threshold) {
                        // 重投影计算3D坐标
                        double w = Q[3][0] * src_x + Q[3][1] * src_y + Q[3][2] * d + Q[3][3];
                        if (fabs(w) > 1e-12) {
                            double X = (Q[0][0] * src_x + Q[0][1] * src_y + Q[0][2] * d + Q[0][3]) / w;
                            double Y = (Q[1][0] * src_x + Q[1][1] * src_y + Q[1][2] * d + Q[1][3]) / w;
                            double Z = (Q[2][0] * src_x + Q[2][1] * src_y + Q[2][2] * d + Q[2][3]) / w;

                            if (isfinite(X) && isfinite(Y) && isfinite(Z) && Z > 0 && Z < max_z) {
                                // 同时赋值 XYZ 和 RGB（与 Python 的 pointmap[mask] 一致）
                                int depth_idx = dst_idx * 3;
                                output_ctx->depth3d[depth_idx + 0] = (float)X;
                                output_ctx->depth3d[depth_idx + 1] = (float)Y;
                                output_ctx->depth3d[depth_idx + 2] = (float)Z;

                                // 只为有效点复制 RGB（转换为 float32，范围 0-255）
                                output_ctx->undistort_image[img_dst_idx + 0] = r;
                                output_ctx->undistort_image[img_dst_idx + 1] = g;
                                output_ctx->undistort_image[img_dst_idx + 2] = b;
                            }
                        }
                    }
                }
            }
        }

        // 如果提供了 bin_output_path，保存 pointmap 为二进制格式
        if (bin_output_path) {
            if (save_pointmap_binary(output_ctx, bin_output_path) != 0) {
                // bin 文件保存失败，但不影响后续 PLY 文件的保存
                printf("Warning: Failed to save binary pointmap to: %s\n", bin_output_path);
            }
        }
    }

    // 保存PLY文件 - 二进制格式
    FILE* ply_file = fopen(output_path, "wb");  // 使用二进制写入模式
    if (!ply_file) {
        free(point_cloud);
        return -1;
    }
    
    // 写入PLY头部
    fprintf(ply_file, "ply\n");
    fprintf(ply_file, "format binary_little_endian 1.0\n");  // 
    fprintf(ply_file, "comment PCL generated\n");  // PCL库标准注释
    fprintf(ply_file, "element vertex %d\n", valid_count);
    fprintf(ply_file, "property float x\n");
    fprintf(ply_file, "property float y\n");
    fprintf(ply_file, "property float z\n");
    fprintf(ply_file, "property uchar red\n");
    fprintf(ply_file, "property uchar green\n");
    fprintf(ply_file, "property uchar blue\n");
    fprintf(ply_file, "end_header\n");
    
    // 写入点云数据 - 二进制格式
    for (int i = 0; i < valid_count; i++) {
        // 写入浮点数坐标 (x, y, z)
        float x = (float)point_cloud[i].x;
        float y = (float)point_cloud[i].y;
        float z = (float)point_cloud[i].z;
        fwrite(&x, sizeof(float), 1, ply_file);
        fwrite(&y, sizeof(float), 1, ply_file);
        fwrite(&z, sizeof(float), 1, ply_file);
        
        // 写入颜色值 (r, g, b)
        fwrite(&point_cloud[i].r, sizeof(unsigned char), 1, ply_file);
        fwrite(&point_cloud[i].g, sizeof(unsigned char), 1, ply_file);
        fwrite(&point_cloud[i].b, sizeof(unsigned char), 1, ply_file);
    }
    
    fclose(ply_file);
    free(point_cloud);
    free(filtered_disparity);

    return 0;
}



// 释放视差图内存
void stereo_matching_free_disparity(DisparityMap* disparity) {
    if (disparity) {
        if (disparity->data) {
            free(disparity->data);
        }
        free(disparity);
    }
}

// 释放输出上下文内存
void stereo_matching_free_output_context(StereoOutputContext* output_ctx) {
    if (output_ctx) {
        if (output_ctx->depth3d) {
            free(output_ctx->depth3d);
            output_ctx->depth3d = NULL;
        }
        if (output_ctx->undistort_image) {
            free(output_ctx->undistort_image);
            output_ctx->undistort_image = NULL;
        }
        if (output_ctx->disp) {
            free(output_ctx->disp);
            output_ctx->disp = NULL;
        }
        output_ctx->width = 0;
        output_ctx->height = 0;
    }
}

// 销毁立体匹配上下文
void stereo_matching_destroy(StereoMatchingContext* ctx) {
    if (ctx) {
        if (ctx->rknn_ctx) {
            rknn_destroy(ctx->rknn_ctx);
        }
        free(ctx);
    }
}

// 获取错误信息
const char* stereo_matching_get_error(StereoMatchingContext* ctx) {
    return ctx ? ctx->error_msg : "Context is NULL";
}