#include <stdio.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <unistd.h> // 用于删除临时文件的remove函数
#include "rknn_api.h"
#include "stereo_matching.h"
#include "pointcloud_processor.h"
#include "morphological_filter.h"
#include <opencv2/opencv.hpp>

// 工具函数：从路径中提取文件名
std::string get_filename_from_path(const std::string& full_path) {
    size_t pos_slash = full_path.find_last_of("/");
    size_t pos_backslash = full_path.find_last_of("\\");
    size_t split_pos = std::max(pos_slash, pos_backslash);
    
    if (split_pos == std::string::npos) {
        return full_path;
    } else {
        return full_path.substr(split_pos + 1);
    }
}

// 工具函数：生成默认输出路径（输入文件名+_processed.ply）
std::string get_default_output_path(const std::string& input_path) {
    std::string filename = get_filename_from_path(input_path);
    size_t dot_pos = filename.find_last_of(".");
    std::string base_name;
    if (dot_pos != std::string::npos) {
        base_name = filename.substr(0, dot_pos);
    } else {
        base_name = filename;
    }
    return base_name + "_processed.ply";
}

// 工具函数：生成形态学滤波后的临时BIN路径
std::string get_morph_filtered_bin_path(const std::string& input_bin_path) {
    std::string filename = get_filename_from_path(input_bin_path);
    size_t dot_pos = filename.find_last_of(".");
    std::string base_name;
    if (dot_pos != std::string::npos) {
        base_name = filename.substr(0, dot_pos);
    } else {
        base_name = filename;
    }
    return base_name + "_morph.bin";
}

int main(int argc, char* argv[]) {
    // ====================== 第一步：解析命令行参数 ======================
    bool enable_post_processing = false;
    std::string processed_output_path;
    std::string default_input_bin = "pointmap.bin";
    std::string input_bin_path = default_input_bin;

    // 兼容原有立体匹配参数 + 新增后处理参数
    if (argc == 7 || argc == 8) {
        enable_post_processing = true;
        if (argc == 7) {
            std::string original_output = argv[5];
            processed_output_path = get_default_output_path(original_output);
        } else {
            processed_output_path = argv[6];
            input_bin_path = (argc == 8) ? argv[7] : default_input_bin;
        }
    } else if (argc == 2 || argc == 3) {
        // 仅后处理模式：直接传入bin文件/输出路径
        enable_post_processing = true;
        input_bin_path = (argc >= 2) ? argv[1] : default_input_bin;
        processed_output_path = (argc == 3) ? argv[2] : get_default_output_path(input_bin_path);
    } else if (argc != 6) {
        printf("=== 双目立体匹配 + 点云后处理工具 ===\n");
        printf("使用方式1（完整流程：立体匹配+后处理）：\n");
        printf("  %s <model.rknn> <left.png> <right.png> <output_disparity.png> <output_pointcloud.ply> [processed_pointcloud.ply] [input_bin_path]\n", argv[0]);
        printf("使用方式2（仅后处理）：\n");
        printf("  %s <input_bin_path> [output_ply_path]\n", argv[0]);
        printf("示例：\n");
        printf("  %s model.rknn left.png right.png disp.png pointcloud.ply processed.ply\n", argv[0]);
        printf("  %s pointmap.bin result.ply\n", argv[0]);
        return -1;
    }

    // ====================== 第二步：原有立体匹配核心流程 ======================
    const char* model_path = nullptr;
    const char* left_img_path = nullptr;
    const char* right_img_path = nullptr;
    const char* output_img_path = nullptr;
    const char* output_ply_path = nullptr;
    std::string bin_path; // 点云临时BIN文件路径

    if (argc >= 6) {
        // 完整立体匹配模式
        model_path = argv[1];
        left_img_path = argv[2];
        right_img_path = argv[3];
        output_img_path = argv[4];
        output_ply_path = argv[5];

        printf("=== 双目立体匹配流程 ===\n");
        printf("模型路径: %s\n", model_path);
        printf("左图路径: %s\n", left_img_path);
        printf("右图路径: %s\n", right_img_path);
        printf("视差图输出: %s\n", output_img_path);
        printf("原始点云输出: %s\n", output_ply_path);
        if (enable_post_processing) {
            printf("后处理点云输出: %s\n", processed_output_path.c_str());
        }
        printf("\n");

        // 1. 创建立体匹配上下文
        printf("[1/4] 初始化立体匹配上下文...\n");
        StereoMatchingContext* ctx = stereo_matching_create(model_path);
        if (!ctx) {
            printf("错误：创建立体匹配上下文失败: %s\n", stereo_matching_get_error(ctx));
            return -1;
        }

        // 2. 加载左右图像
        printf("[2/4] 加载图像...\n");
        cv::Mat left_img_cv = cv::imread(left_img_path, cv::IMREAD_COLOR);
        cv::Mat right_img_cv = cv::imread(right_img_path, cv::IMREAD_COLOR);
        if (left_img_cv.empty() || right_img_cv.empty()) {
            printf("错误：加载图像失败\n");
            stereo_matching_destroy(ctx);
            return -1;
        }
        int width = left_img_cv.cols;
        int height = left_img_cv.rows;
        printf("图像尺寸: %dx%d, 通道数: %d\n", width, height, left_img_cv.channels());

        // 3. 执行立体匹配推理
        printf("[3/4] 执行立体匹配推理...\n");
        double inference_time;
        unsigned char* left_img = left_img_cv.data;
        unsigned char* right_img = right_img_cv.data;
        DisparityMap* disparity = stereo_matching_infer(ctx, left_img, right_img, width, height, &inference_time);
        if (!disparity) {
            printf("错误：推理失败: %s\n", stereo_matching_get_error(ctx));
            stereo_matching_destroy(ctx);
            return -1;
        }
        printf("推理完成，耗时: %.2f ms\n", inference_time);
        printf("视差图尺寸: %dx%d\n", disparity->width, disparity->height);

        // 4. 保存视差图和原始点云
        printf("[4/4] 保存立体匹配结果...\n");
        if (stereo_matching_save_disparity_png(disparity, output_img_path) != 0) {
            printf("警告：保存视差图失败\n");
        } else {
            printf("视差图已保存至: %s\n", output_img_path);
        }

        // 点云生成参数（与Python batch_process_igev.py一致）
        float min_disparity = 5.0f;    // 最小视差阈值
        float max_disparity = 300.0f;  // 最大视差阈值
        float max_z = 200.0f;          // 最大深度阈值(mm)
        int black_threshold = 0;       // 黑色背景阈值
        double Q[4][4] = {
            {1.0, 0.0, 0.0, -312.7411},
            {0.0, 1.0, 0.0, -663.5256},
            {0.0, 0.0, 0.0, 877.7027},
            {0.0, 0.0, 0.3976856, 0.0}
        };

        // 生成临时BIN文件路径（用于后处理）
        if (enable_post_processing) {
            bin_path = std::string(output_ply_path);
            size_t dot_pos = bin_path.find_last_of(".");
            if (dot_pos != std::string::npos) {
                bin_path = bin_path.substr(0, dot_pos) + "_temp.bin";
            } else {
                bin_path += "_temp.bin";
            }
            input_bin_path = bin_path; // 后处理使用该临时BIN
        }

        // 生成原始点云
        StereoOutputContext output_ctx = {0};
        if (stereo_matching_generate_point_cloud(disparity, left_img, output_ply_path,
                                               min_disparity, max_disparity, max_z,
                                               black_threshold, Q, &output_ctx,
                                               enable_post_processing ? bin_path.c_str() : NULL) != 0) {
            printf("警告：生成原始点云失败\n");
        } else {
            printf("原始点云已保存至: %s\n", output_ply_path);
            if (enable_post_processing) {
                printf("临时BIN文件已保存至: %s\n", bin_path.c_str());
            }
            stereo_matching_free_output_context(&output_ctx);
        }

        // 清理立体匹配资源
        stereo_matching_free_disparity(disparity);
        stereo_matching_destroy(ctx);
    }

    // ====================== 第三步：新增形态学滤波 + 点云后处理 ======================
    if (enable_post_processing) {
        printf("\n=== 点云后处理流程 ===\n");
        printf("输入BIN文件: %s\n", input_bin_path.c_str());
        printf("后处理输出PLY: %s\n", processed_output_path.c_str());

        // 生成形态学滤波临时BIN路径
        std::string morph_filtered_bin_path = get_morph_filtered_bin_path(input_bin_path);
        printf("形态学滤波临时BIN: %s\n", morph_filtered_bin_path.c_str());

        // 1. 执行形态学滤波
        printf("\n[Step 1/2] 执行形态学滤波...\n");
        MorphologicalFilterProcessor morph_processor;
        morph_processor.setVerbose(true);

        MorphologicalFilterConfig morph_config;
        morph_config.kernel_size = 31;
        morph_config.op_type = MorphOpType::MORPH_OPEN;
        morph_config.z_valid_eps = 1e-6f;
        morph_config.update_xy = true;
        morph_config.gradient_threshold = 0.1f;
        morph_config.gradient_kernel_size = 3;
        morph_config.gaussian_kernel_size = 7;
        morph_config.gaussian_sigma = 10.0;
        morph_config.output_ply_path = "morph_filtered.ply";

        MorphologicalFilterResult morph_result = morph_processor.process(
            input_bin_path, 
            morph_filtered_bin_path, 
            morph_config
        );

        if (!morph_result.success) {
            printf("错误：形态学滤波失败: %s\n", morph_processor.getLastError().c_str());
            return -1;
        }
        printf("形态学滤波完成！\n");
        printf("  输入有效点数量: %zu\n", morph_result.input_points);
        printf("  更新Z值点数量: %zu\n", morph_result.updated_points);
        printf("  输出点数量: %zu\n", morph_result.output_points);
        printf("  耗时: %.3f 秒\n", morph_result.total_time);

        // 2. 执行点云增强后处理（使用滤波后的BIN）
        printf("\n[Step 2/2] 执行点云增强...\n");
        PointCloudProcessor processor;
        processor.setVerbose(true);

        PointCloudConfig config;
        config.min_noise_area = 200;
        config.gradient_ratio = 0.01;
        config.max_allowed_depth_ratio = 0.01;
        config.verbose = true;
        config.enable_memory_stats = false;

        ProcessResult result = processor.process(morph_filtered_bin_path, processed_output_path, config);
        if (result.success) {
            printf("点云增强完成！\n");
            printf("处理结果已保存至: %s\n", processed_output_path.c_str());
            printf("统计信息:\n");
            printf("  输入点数量（滤波后）: %zu\n", result.input_points);
            printf("  输出点数量: %zu\n", result.output_points);
            printf("  移除噪声点数量: %zu\n", result.removed_points);
            printf("  插值补充点数量: %zu\n", result.interpolated_points);
            printf("  总耗时: %.3f 秒\n", result.total_time);

            // 打印中心像素3D坐标
            if (!result.depth3d.empty() && result.depth3d_width > 0 && result.depth3d_height > 0) {
                printf("\n=== 去噪后深度信息 ===\n");
                printf("深度图尺寸: %dx%d\n", result.depth3d_width, result.depth3d_height);
                int center_x = result.depth3d_width / 2;
                int center_y = result.depth3d_height / 2;
                size_t depth3d_idx = (size_t)center_y * result.depth3d_width * 3 + center_x * 3;
                if (depth3d_idx + 2 < result.depth3d.size()) {
                    printf("中心像素(%d,%d) 3D坐标: (%.3f, %.3f, %.3f)\n",
                           center_x, center_y,
                           result.depth3d[depth3d_idx],
                           result.depth3d[depth3d_idx+1],
                           result.depth3d[depth3d_idx+2]);
                }
                result.clearDepth3d(); // 释放内存
            }
        } else {
            printf("错误：点云增强失败: %s\n", processor.getLastError().c_str());
            return -1;
        }

        // 清理临时文件
        if (!bin_path.empty() && remove(bin_path.c_str()) == 0) {
            printf("已删除临时BIN文件: %s\n", bin_path.c_str());
        }
        if (remove(morph_filtered_bin_path.c_str()) == 0) {
            printf("已删除形态学滤波临时BIN文件: %s\n", morph_filtered_bin_path.c_str());
        }
    }

    printf("\n=== 所有流程执行完成 ===\n");
    return 0;
}
