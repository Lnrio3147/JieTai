import os
import re
import shutil
from PIL import Image

# —— 配置区 —— #
# source_dir = r'D:\Desktop\stereo_project\数据集\1280x720螺纹测量文件\measure'
source_dir = r'D:\Desktop\20260205\20260205'
output_dir = os.path.join(source_dir, 'output')  # 输出根目录
os.makedirs(output_dir, exist_ok=True)

# 图像文件正则
# 旧格式1 (适配格式: 1-202506281603-0001_L.png)
# img_pattern = re.compile(r'\d+-(\d{12})-(\d{4})_([LR])\.png$', re.IGNORECASE)

# 旧格式2 (适配格式: 656565-0001_L.png)
# img_pattern = re.compile(r'(\d+)-(\d{4})_([LR])\.png$', re.IGNORECASE)

# 新格式 (适配格式: camera-202512081522-0004_L.png)
# img_pattern = re.compile(r'camera-(\d{12})-(\d{4})_([LR])\.png$', re.IGNORECASE)

# ooooo格式 (适配格式: ooooo-20260204-0317_L.png)
img_pattern = re.compile(r'ooooo-(\d{8})-(\d{4})_([LR])\.png$', re.IGNORECASE)

# 点云文件正则
# 旧格式1 (适配格式: 1-202506281603-0001.ply 和 1-202506281603-0001_old.ply)
# ply_pattern = re.compile(r'\d+-(\d{12})-(\d{4})(_old)?\.ply$', re.IGNORECASE)

# 旧格式2 (适配格式: 656565-0001.ply 和 656565-0001_old.ply)
# ply_pattern = re.compile(r'(\d+)-(\d{4})(_old)?\.ply$', re.IGNORECASE)

# 新格式 (适配格式: camera-202512081522-0004.ply 和 camera-202512081522-0004_old.ply)
# ply_pattern = re.compile(r'camera-(\d{12})-(\d{4})(_old)?\.ply$', re.IGNORECASE)

# ooooo格式 (适配格式: ooooo-20260204-0317.ply)
ply_pattern = re.compile(r'ooooo-(\d{8})-(\d{4})(_old)?\.ply$', re.IGNORECASE)

# 存储匹配结果：字典的key是 (时间戳+序号)
image_pairs = {}

print("—— 目录下所有文件 ——")
for fn in os.listdir(source_dir):
    print(fn)
print("\n—— 以下是能被正则匹配到的图像文件 ——")

# 处理图像文件
for filename in os.listdir(source_dir):
    m = img_pattern.search(filename)
    if m:
        timestamp = m.group(1)  # 202512081522 (时间戳)
        seq = m.group(2)  # 0004 (序号)
        side = m.group(3).upper()  # 'L' 或 'R'
        key = f"ooooo-{timestamp}-{seq}"  # 组合唯一标识符

        print(f"Matched: {filename} → timestamp={timestamp}, seq={seq}, side={side}")
        image_pairs.setdefault(key, {})[side] = os.path.join(source_dir, filename)

print("\n—— 匹配到的成对图像 ——")
for key, sides in image_pairs.items():
    print(f"{key}: 包含{list(sides.keys())}图像")

# 处理成对图像
for key, paths in image_pairs.items():
    if 'L' in paths and 'R' in paths:
        pair_dir = os.path.join(output_dir, key)
        os.makedirs(pair_dir, exist_ok=True)

        # 保存左图(L)为im0.png
        img_left = Image.open(paths['L'])
        rotated_left = img_left.transpose(Image.ROTATE_90)
        rotated_left.save(os.path.join(pair_dir, 'im0.png'))

        # 保存右图(R)为im1.png
        img_right = Image.open(paths['R'])
        rotated_right = img_right.transpose(Image.ROTATE_90)
        rotated_right.save(os.path.join(pair_dir, 'im1.png'))

        print(f"已处理图像: {key} → 左图:{os.path.basename(paths['L'])} 右图:{os.path.basename(paths['R'])}")

print("\n—— 处理点云文件 ——")
# 处理点云文件
for filename in os.listdir(source_dir):
    m = ply_pattern.match(filename)
    if m:
        timestamp = m.group(1)  # 202512081522 (时间戳)
        seq = m.group(2)  # 0004 (序号)
        is_old_suffix = m.group(3)  # 可能是"_old"或None
        key = f"ooooo-{timestamp}-{seq}"

        # 仅当该key存在图像pair目录时才处理
        if key in image_pairs:
            pair_dir = os.path.join(output_dir, key)
            if os.path.exists(pair_dir):
                # 构建新文件名 (0004.ply 或 0004_old.ply)
                new_name = f"{seq}{is_old_suffix or ''}.ply"
                dest_path = os.path.join(pair_dir, new_name)

                # 复制点云文件
                # shutil.copy2(os.path.join(source_dir, filename), dest_path)
                print(f"点云已保存: {filename} → {new_name}")
            else:
                print(f"警告: 目录不存在 {pair_dir}, 跳过点云文件 {filename}")
        else:
            print(f"跳过点云: {filename} (无对应图像pair)")

print("\n处理完成! 所有文件保存在:", output_dir)