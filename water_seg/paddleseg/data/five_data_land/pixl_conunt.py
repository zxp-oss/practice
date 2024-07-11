from PIL import Image
import os
from collections import Counter
import tqdm

# 设置mask图片所在的文件夹路径
folder_path = './labels'

# 初始化一个Counter用于统计像素点值的占比
pixel_counter = Counter()

# 遍历文件夹中的所有图片

for file_name in tqdm.tqdm(os.listdir(folder_path)):
    # 判断是否为图片文件
    if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        # 打开图片文件
        image = Image.open(os.path.join(folder_path, file_name))

        # 统计像素点值的占比
        pixel_counter.update(image.getdata())

# 计算各个像素点值的占比
total_pixels = sum(pixel_counter.values())
pixel_percentages = {pixel: count / total_pixels for pixel, count in pixel_counter.items()}

# 输出结果
for pixel, percentage in pixel_percentages.items():
    print(f'Pixel: {pixel}, Percentage: {percentage}')