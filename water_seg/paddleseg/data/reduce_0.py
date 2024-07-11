# import os
# from PIL import Image
# import numpy as np
# from tqdm import tqdm
#
# # 文件夹路径
# folder_path = './five_data_land/labels'
# # 阈值，0到1之间
# threshold = 0.75
#
# # 获取文件夹中的所有文件
# file_list = os.listdir(folder_path)
# socre=[]
# num=0
# # 使用tqdm显示进度条
# for file_name in tqdm(file_list, desc="Processing images"):
#     # 构造文件的完整路径
#     file_path = os.path.join(folder_path, file_name)
#
#     # 检查文件是否为图像文件
#     if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         # 打开图像
#         with Image.open(file_path) as img:
#             # 将图像转换为灰度图
#             img = img.convert('L')
#             # 将图像数据转换为NumPy数组
#             img_data = np.array(img)
#             # 计算总像素点数
#             total_pixels = img_data.size
#             # 计算数值为0的像素点数
#             zero_pixels = np.sum(img_data == 0)
#             # 计算比例
#             zero_ratio = zero_pixels / total_pixels
#
#             if zero_ratio>0.9:
#                 num+=1
#
#             socre.append(zero_ratio)
#
#             # print(zero_ratio)
#
#             # 如果比例大于阈值，删除图片
#             # if zero_ratio > threshold:
#             #     os.remove(file_path)
#             #     tqdm.write(
#             #         f"Deleted {file_name} because zero ratio {zero_ratio:.2f} is greater than threshold {threshold:.2f}")
#
#
#
# print(np.mean(socre))
# print(num)
# print("Process completed.")

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# 文件夹路径
folder_path = './DeepGlobe/eval_labels'
# 阈值，0到1之间
threshold = 0.85

# 获取文件夹中的所有文件
file_list = os.listdir(folder_path)
socre=[]
num=0
reduce_num=0
# 使用tqdm显示进度条
for file_name in tqdm(file_list, desc="Processing images"):
    # 构造文件的完整路径
    file_path = os.path.join(folder_path, file_name)

    # 检查文件是否为图像文件
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 打开图像
        with Image.open(file_path) as img:
            # 将图像转换为灰度图
            img = img.convert('L')
            # 将图像数据转换为NumPy数组
            img_data = np.array(img)
            # 计算总像素点数
            total_pixels = img_data.size
            # 计算数值为0的像素点数
            zero_pixels = np.sum(img_data == 0)
            # 计算比例
            zero_ratio = zero_pixels / total_pixels

            if zero_ratio>0.9:
                num+=1

            socre.append(zero_ratio)

            # print(zero_ratio)

            # 如果比例大于阈值，删除图片
            if zero_ratio > threshold:
                reduce_num+=1
                os.remove(file_path)
                # os.remove(os.path.join("five_data_land/images",file_name.replace(".png",".jpg")))
                tqdm.write(
                    f"Deleted {file_name} because zero ratio {zero_ratio:.2f} is greater than threshold {threshold:.2f}")



print(np.mean(socre))
print(num)
print(reduce_num)
print("Process completed.")

# import os
# print(len(os.listdir("five_data_land/labels")))
# print(len(os.listdir("five_data_land/images")))
