import os
import shutil
import tqdm

def move_and_rename_images(source_folder, jpg_folder, png_folder):
    # 创建目标文件夹
    os.makedirs(jpg_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)

    # 遍历源文件夹中的文件
    for filename in tqdm.tqdm(os.listdir(source_folder)):
        if filename.endswith(".jpg"):
            print(filename)
            # 如果是jpg文件，移动并重命名
            new_filename =filename.split("_")[0]+".jpg"
            shutil.move(os.path.join(source_folder, filename), os.path.join(jpg_folder, new_filename))
        elif filename.endswith(".png"):
            print(filename)
            # 如果是png文件，移动并重命名
            new_filename =filename.split("_")[0]+".png"
            shutil.move(os.path.join(source_folder, filename), os.path.join(png_folder, new_filename))

# 设置源文件夹和目标文件夹
source_folder = "land-train/land-train"
jpg_folder = "images"
png_folder = "ori_labels"
move_and_rename_images(source_folder,jpg_folder,png_folder)