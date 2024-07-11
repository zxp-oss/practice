import os
from PIL import Image
from tqdm import tqdm


def save_mask_subimage(mask_image, x_offset, y_offset, width, height, output_path):
    # 裁剪子图像
    sub_image = mask_image.crop((x_offset, y_offset, x_offset + width, y_offset + height))
    # 保存子图像为PNG格式
    sub_image.save(output_path, 'PNG')


def main(mask_input_path, output_dir, window_size=1024, overlap=256):
    print('Processing mask:', mask_input_path)

    # img_name = mask_input_path.split("\\")[-1].split(".")[0]
    # print(img_name)
    img_name = mask_input_path.split(".")[0].split("/")[1]
    print(img_name)
    # 读取输入mask图像
    mask_image = Image.open(mask_input_path)
    img_width, img_height = mask_image.size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_step = window_size - overlap
    y_step = window_size - overlap

    # 计算总的子图像数量，用于进度条
    total_tiles = ((img_width - overlap) // x_step + 1) * ((img_height - overlap) // y_step + 1)

    count = 0
    with tqdm(total=total_tiles, desc="Processing") as pbar:
        for y in range(0, img_height, y_step):
            if y + window_size > img_height:
                y = img_height - window_size  # 调整y以对齐底部边缘
            for x in range(0, img_width, x_step):
                if x + window_size > img_width:
                    x = img_width - window_size  # 调整x以对齐右侧边缘

                mask_output_path = os.path.join(output_dir, f"{img_name}_tile_{count}.png")

                save_mask_subimage(mask_image, x, y, window_size, window_size, mask_output_path)

                count += 1
                pbar.update(1)


if __name__ == "__main__":
    mask_input_path = "big_labels"
    output_dir = "labels"
    mask_input_list = os.listdir(mask_input_path)

    for index, mask_file in enumerate(mask_input_list):
        print(f"正在处理：{index + 1}/{len(mask_input_list)}")
        mask_full_path = os.path.join(mask_input_path, mask_file)
        main(mask_full_path, output_dir, window_size=512, overlap=128)
