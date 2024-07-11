import os
import random
from PIL import Image

# 设置文件夹路径
folder_path = 'VOCdevkit/images/test2007'

# 创建一个 10000x10000 像素的空白图像
master_image = Image.new('RGB', (10000, 10000), (255, 255, 255))

# 获取文件夹中所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# 如果图片文件数量少于 100,则重复使用
if len(image_files) < 100:
    image_files = image_files * (100 // len(image_files) + 1)
    image_files = image_files[:100]

# 随机选择 100 张图片,并对其进行处理
for i in range(100):
    # 随机选择一张图片
    image_file = random.choice(image_files)
    image_files.remove(image_file)

    # 打开图片
    image = Image.open(os.path.join(folder_path, image_file))

    # 对图片进行随机旋转和缩放
    angle = random.randint(-20, 25)
    scale = random.uniform(0.5, 2.0)
    image = image.rotate(angle, resample=Image.BICUBIC).resize((int(image.width * scale), int(image.height * scale)),
                                                               resample=Image.BICUBIC)

    # 随机选择一个位置,并将图片贴到master_image上
    x = random.randint(0, 9900)
    y = random.randint(0, 9900)
    master_image.paste(image, (x, y))

# 保存最终的master_image
master_image.save('master_image.jpg')