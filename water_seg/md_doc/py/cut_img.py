import os
from PIL import Image
from tqdm import tqdm

def save_rgb_subimage(image, x_offset, y_offset, width, height, output_path):
    # 裁剪子图像
    sub_image = image.crop((x_offset, y_offset, x_offset + width, y_offset + height))
    # 将子图像转换为CMYK模式以便提取通道
    sub_image_cmyk = sub_image.convert('CMYK')
    # 提取R、G、B通道
    _, R, G, B = sub_image_cmyk.split()
    # 合并R、G、B通道为RGB图像
    sub_image_rgb = Image.merge('RGB', (R, G, B))
    # 保存RGB子图像为PNG格式
    sub_image_rgb.save(output_path, 'PNG')

def main(input_path, output_dir, window_size=1024, overlap=256):
    print('Input path:', input_path)

    img_name = input_path.split("\\")[-1].split(".")[0]
    # 读取输入图像
    image = Image.open(input_path)
    img_width, img_height = image.size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_step = window_size - overlap
    y_step = window_size - overlap

    # 计算总的子图像数量，用于进度条
    total_tiles = 90

    count = 0
    with tqdm(total=total_tiles, desc="Processing") as pbar:
        for y in range(0, img_height, y_step):
            if y + window_size > img_height:
                y = img_height - window_size  # 调整y以对齐底部边缘
            for x in range(0, img_width, x_step):
                if x + window_size > img_width:
                    x = img_width - window_size  # 调整x以对齐右侧边缘

                output_path = os.path.join(output_dir, f"{img_name}_tile_{count}.jpg")
                save_rgb_subimage(image, x, y, window_size, window_size, output_path)
                count += 1
                pbar.update(1)

if __name__ == "__main__":
    input_path = "data/img"
    output_dir = 'caijian/img'
    input_list=os.listdir(input_path)

    for index,i in enumerate(input_list):
        print(f"正在处理：{index+1}/{len(input_list)}")
        input_path = os.path.join(input_path, i)
        main(input_path, output_dir, window_size=1024, overlap=256)