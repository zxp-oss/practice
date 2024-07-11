import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def process_masks(input_folder, output_folder, keep_classes, new_labels):
    """
    处理mask图像，只保留指定类别并重新标注其标签值，其他标签转换为背景。

    参数：
    - input_folder: 输入文件夹，包含原始mask图像
    - output_folder: 输出文件夹，保存处理后的mask图像
    - keep_classes: 字典，键为保留的类别原始标签，值为新的标签值
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for filename in tqdm(files, desc="Processing masks"):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image_array = np.array(image)

        # 创建一个新的数组，初始值全部为背景标签
        new_image_array = np.zeros_like(image_array)

        # 更新新数组中的保留类别标签
        for original_label, new_label in keep_classes.items():
            new_image_array[image_array == original_label] = new_label

        new_image = Image.fromarray(new_image_array)
        new_image.save(os.path.join(output_folder, filename))


# 定义输入文件夹和输出文件夹
input_folder = 'ori_labels'
output_folder = 'labels'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义需要保留的类别及其新的标签值
# 例如：保留水体(4)和森林(6)，将它们分别重新标注为1和2，其他标签都转为背景(0)
keep_classes = {
    11: 1,
    13: 2,
    
}

# 执行批量处理
process_masks(input_folder, output_folder, keep_classes, new_labels={0: 'Background', 1: 'Water', 2: 'Forest',3: 'agriculture'})
