import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm

def calculate_average_histogram(image_folder):
    # 初始化三个通道的累积直方图
    hist_r = np.zeros(256)
    hist_g = np.zeros(256)
    hist_b = np.zeros(256)
    image_count = 0

    for filename in tqdm.tqdm(os.listdir(image_folder)):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')
            np_image = np.array(image)

            # 分别计算每个通道的直方图
            r_hist, _ = np.histogram(np_image[:, :, 0], bins=256, range=(0, 256))
            g_hist, _ = np.histogram(np_image[:, :, 1], bins=256, range=(0, 256))
            b_hist, _ = np.histogram(np_image[:, :, 2], bins=256, range=(0, 256))

            # 累加到总直方图中
            hist_r += r_hist
            hist_g += g_hist
            hist_b += b_hist
            image_count += 1

    # 计算平均直方图
    hist_r /= image_count
    hist_g /= image_count
    hist_b /= image_count

    return hist_r, hist_g, hist_b

def plot_histograms(hist_r, hist_g, hist_b):
    plt.figure(figsize=(10, 5))
    plt.plot(hist_r, color='red', alpha=0.6, label='Red channel')
    plt.plot(hist_g, color='green', alpha=0.6, label='Green channel')
    plt.plot(hist_b, color='blue', alpha=0.6, label='Blue channel')
    plt.title('Average RGB Histogram')
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def save_histograms(hist_r, hist_g, hist_b, output_folder):
    np.save(os.path.join(output_folder, 'hist_r.npy'), hist_r)
    np.save(os.path.join(output_folder, 'hist_g.npy'), hist_g)
    np.save(os.path.join(output_folder, 'hist_b.npy'), hist_b)

def load_histograms(input_folder):
    hist_r = np.load(os.path.join(input_folder, 'hist_r.npy'))
    hist_g = np.load(os.path.join(input_folder, 'hist_g.npy'))
    hist_b = np.load(os.path.join(input_folder, 'hist_b.npy'))
    return hist_r, hist_g, hist_b

# 设置图片文件夹路径
image_folder = 'DeepGlobe/images'
# 设置保存直方图数据的文件夹路径
output_folder = './'

# 计算平均直方图
hist_r, hist_g, hist_b = calculate_average_histogram(image_folder)

# 保存直方图数据
save_histograms(hist_r, hist_g, hist_b, output_folder)

