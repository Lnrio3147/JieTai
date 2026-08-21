# Pointmap 自定义二进制格式说明

## 概述

为了方便 Python 和 C++ 之间共享点云数据，我们设计了一种简单的二进制格式 `.bin`，包含完整的 XYZ 坐标和 RGB 颜色信息。

---

## 文件格式规范

### 文件结构

```
+--------------------+
|   头部 (64 bytes)   |
+--------------------+
|   数据部分          |
|  (H×W×6×4 bytes)   |
+--------------------+
```

### 头部格式（64 bytes）

| 字段           | 偏移量 | 大小    | 类型      | 说明                          |
|----------------|--------|---------|-----------|-------------------------------|
| 魔法数         | 0      | 4       | char[4]   | "PMAP" (ASCII)                |
| 版本号         | 4      | 4       | uint32    | 当前版本: 1                   |
| 高度           | 8      | 4       | uint32    | 图像高度 H                    |
| 宽度           | 12     | 4       | uint32    | 图像宽度 W                    |
| 通道数         | 16     | 4       | uint32    | 固定为 6 (XYZ + RGB)          |
| 数据类型       | 20     | 4       | uint32    | 0 = float32                   |
| 通道顺序       | 24     | 24      | char[24]  | "XYZRGB" + 0填充              |
| 保留字段       | 48     | 20      | char[20]  | 全0，用于未来扩展             |

### 数据部分

- **存储顺序**: 行优先 (row-major)
- **每个像素**: 6 个 float32 值（24 bytes）
  - `[0]` X 坐标 (mm)
  - `[1]` Y 坐标 (mm)
  - `[2]` Z 坐标 (mm) - **深度值**
  - `[3]` R 通道 (0-255)
  - `[4]` G 通道 (0-255)
  - `[5]` B 通道 (0-255)

- **无效点**: Z = 0 或 Z = NaN，表示该像素无有效深度
- **字节序**: 小端 (little-endian)，适配 x86/x64 架构
- **总大小**: 64 + H × W × 6 × 4 bytes

---

## Python 使用

### 1. 保存 Pointmap

`batch_process_igev.py` 已集成自动保存功能：

```python
# 运行批处理脚本，会同时生成 .npy 和 .bin 文件
python batch_process_igev.py
```

输出文件：
- `pointmap.npy` - NumPy 格式（Python 方便）
- `pointmap.bin` - 自定义二进制格式（C++ 方便）

### 2. 读取 Pointmap (Python)

```python
import numpy as np

# 从 .npy 文件读取（推荐）
pointmap = np.load("pointmap.npy")  # shape: (H, W, 6)

# 提取有效点
mask = pointmap[..., 2] > 0  # Z > 0 表示有效
points_xyz = pointmap[mask, :3]  # (N, 3)
colors_rgb = pointmap[mask, 3:]  # (N, 3)
```

### 3. 可视化 Pointmap (Python)

```bash
python test_pointmap.py
```

---

## C++ 使用

### 1. 编译

**方法 1: 使用批处理脚本（Windows）**

```bash
build_and_run.bat
```

**方法 2: 手动编译**

```bash
# MinGW / GCC
g++ -std=c++17 test_pointmap.cpp -o test_pointmap.exe -O2

# MSVC
cl /std:c++17 /O2 test_pointmap.cpp
```

### 2. 运行

```bash
# 使用默认路径
test_pointmap.exe

# 指定文件路径
test_pointmap.exe "D:\path\to\pointmap.bin"
```

### 3. 输出

程序会：
1. 读取并验证 `.bin` 文件
2. 打印头部信息和统计数据
3. 导出为 `output_from_cpp.ply`（可用 MeshLab / CloudCompare 打开）

### 4. C++ 读取示例

```cpp
#include <fstream>
#include <vector>

struct PointmapHeader {
    char magic[4];
    uint32_t version;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t data_type;
    char channel_order[24];
    char reserved[20];
};

// 读取文件
std::ifstream file("pointmap.bin", std::ios::binary);
PointmapHeader header;
file.read(reinterpret_cast<char*>(&header), 64);

// 读取数据
std::vector<float> data(header.height * header.width * 6);
file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));

// 访问像素 (y, x) 的数据
int idx = (y * header.width + x) * 6;
float x = data[idx + 0];
float y = data[idx + 1];
float z = data[idx + 2];  // 深度
float r = data[idx + 3];
float g = data[idx + 4];
float b = data[idx + 5];
```

---

## 文件示例

### 典型文件大小

对于 1280 × 720 的图像：
- 数据大小 = 1280 × 720 × 6 × 4 = 22,118,400 bytes ≈ 21.1 MB
- 加上头部 = 21.1 MB

### 存储效率对比

| 格式           | 文件大小   | Python 读取 | C++ 读取 | 压缩 |
|----------------|------------|-------------|----------|------|
| `.npy`         | ~21 MB     | ✅ 原生支持 | ❌ 复杂   | 否   |
| `.bin` (自定义) | ~21 MB     | ✅ 可读     | ✅ 简单   | 否   |
| `.ply` (ASCII) | ~50 MB     | ⚠️ 慢       | ⚠️ 慢     | 否   |
| `.ply` (二进制) | ~21 MB     | ✅          | ✅        | 否   |

---

## 注意事项

1. **字节序**: 使用小端 (little-endian)，在大端系统上需要转换
2. **无效点**: Z = 0 或 NaN 表示该像素无有效深度
3. **颜色范围**: RGB 虽然存储为 float32，但值域为 [0, 255]
4. **坐标单位**: XYZ 坐标单位为毫米 (mm)

---

## 扩展性

### 头部保留字段（20 bytes）

未来可用于：
- 相机内参 (fx, fy, cx, cy)
- 时间戳
- 帧 ID
- 压缩标志

### 版本控制

通过 `version` 字段区分不同版本的格式：
- Version 1: 当前格式
- Version 2+: 可添加新字段（需更新头部大小或使用保留字段）

---

## 相关文件

- `batch_process_igev.py` - 批处理脚本（自动生成 .bin 和 .npy）
- `test_pointmap.py` - Python 可视化测试
- `test_pointmap.cpp` - C++ 读取和导出示例
- `build_and_run.bat` - Windows 编译脚本
