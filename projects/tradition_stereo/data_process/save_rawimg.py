import os
from PIL import Image
import re

# —— 配置区 —— #
source_dir = r'D:\Desktop\项目\数据集\发动机叶片测量照片采集32325D002002\JMP-LF6020\DE0548'
output_dir = os.path.join(source_dir, 'output_right')  # 输出根目录
os.makedirs(output_dir, exist_ok=True)

# 修改正则表达式适配新格式：1-202506281603-0001_L.png
# 提取：时间戳(202506281603) + 序号(0001)作为key，并捕获L/R
# pattern = re.compile(r'\d+-(\d{12})-(\d{4})_([LR])\.png$', re.IGNORECASE)

pattern = re.compile(r'^(?:.*-)?(\d{12})-(\d{4})_([LR])\.png$', re.IGNORECASE)


# 存储匹配结果：字典的key是 (时间戳+序号)
image_pairs = {}

print("—— 目录下所有文件 ——")
for fn in os.listdir(source_dir):
    print(fn)
print("\n—— 以下是能被正则匹配到的文件 ——")

for filename in os.listdir(source_dir):
    m = pattern.search(filename)
    if m:
        timestamp = m.group(1)  # 202506281603
        seq = m.group(2)  # 0001
        side = m.group(3).upper()  # 'L' 或 'R'
        key = f"{timestamp}-{seq}"  # 组合唯一标识符

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

        print(f"已处理: {key} → 左图:{os.path.basename(paths['L'])} 右图:{os.path.basename(paths['R'])}")

print("\n处理完成! 所有图像保存在:", output_dir)