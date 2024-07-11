import os

import tqdm
from PIL import Image
import numpy as np

# 提供的RGB对应标签字典
# rgb_to_label = {
#     (255, 255, 0): 1,
#     (255, 0, 255): 2,
#     (0, 255 , 0): 3
# }
rgb_to_label = {
    (0, 0,255): 255,
}

def rgb_to_label_conversion(image_path, rgb_to_label,output_path):
    # 打开RGB图像
    img = Image.open(image_path)
    img = img.convert("RGB")

    # 获取图像的像素数据
    img_data = np.array(img)

    # 初始化标签数据为0
    label_data = np.zeros_like(img_data[:, :, 0])

    # 转换RGB值为对应的标签值
    for rgb_val, label_val in rgb_to_label.items():
        mask = np.all(img_data == np.array(rgb_val), axis=2)
        label_data[mask] = label_val

    # 保存标签图像
    label_img = Image.fromarray(label_data)
    label_img.save(output_path)



ori_label_path="ori_labels"
labels_path="eval_labels"
ori_label_list=os.listdir(ori_label_path)
for filename in tqdm.tqdm(ori_label_list):
    img_path=os.path.join(ori_label_path,filename)
    output_path=os.path.join(labels_path,filename)

    rgb_to_label_conversion(img_path, rgb_to_label,output_path)