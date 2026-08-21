cmake_minimum_required(VERSION 3.10)
project(DisparityToPointCloud)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 包含目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 创建库
add_library(disparity_to_pointcloud
    disparity_to_pointcloud.cpp
    disparity_to_pointcloud.h
    npy_reader.cpp
    npy_reader.h
)

# 链接OpenCV
target_link_libraries(disparity_to_pointcloud ${OpenCV_LIBS})

# 创建示例可执行文件
add_executable(example_disparity_to_pointcloud
    example_disparity_to_pointcloud.cpp
)

# 链接库
target_link_libraries(example_disparity_to_pointcloud
    disparity_to_pointcloud
    ${OpenCV_LIBS}
)

# 创建裁剪测试可执行文件
add_executable(test_crop_example
    test_crop_example.cpp
)

# 链接库
target_link_libraries(test_crop_example
    disparity_to_pointcloud
    ${OpenCV_LIBS}
)

# 打印OpenCV信息
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
