import os

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_histograms(input_folder):
    hist_r = np.load(os.path.join(input_folder, 'hist_r.npy'))
    hist_g = np.load(os.path.join(input_folder, 'hist_g.npy'))
    hist_b = np.load(os.path.join(input_folder, 'hist_b.npy'))
    return hist_r, hist_g, hist_b


def calculate_cdf(hist):
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]
    return cdf_normalized


def match_histograms(image, reference_histograms):
    matched_image = np.zeros_like(image)
    for i in range(3):  # 对于每个通道
        image_channel = image[:, :, i]
        image_hist, _ = np.histogram(image_channel, bins=256, range=(0, 256))
        image_cdf = calculate_cdf(image_hist)

        reference_cdf = calculate_cdf(reference_histograms[i])

        # 创建映射表
        mapping = np.interp(image_cdf, reference_cdf, np.arange(256))

        # 应用映射表
        matched_image[:, :, i] = np.interp(image_channel.flatten(), np.arange(256), mapping).reshape(
            image_channel.shape)

    return matched_image


def plot_histogram(image, title):
    plt.figure()
    plt.hist(image.ravel(), bins=256, histtype='step', color='black')
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.show()

def process_image(image,hist_folder):
    # 设置保存直方图数据的文件夹路径
    input_folder = './land'

    # 从文件加载直方图数据
    loaded_hist_r, loaded_hist_g, loaded_hist_b = load_histograms(hist_folder)

    # 读取测试图像
    test_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    np_test_image = np.array(test_image)

    # 对每个通道进行直方图匹配
    reference_histograms = [loaded_hist_r, loaded_hist_g, loaded_hist_b]
    matched_image = match_histograms(np_test_image, reference_histograms)



    image_bgr = cv2.cvtColor(matched_image, cv2.COLOR_RGB2BGR)


    # # 显示使用OpenCV读取的图像
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image_bgr)
    cv2.imwrite("img/hist_r.jpg", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image_bgr
    #
    # # # 绘制并展示匹配前后的直方图
    # # plot_histogram(np_test_image[:, :, 0], 'Red Channel Histogram (Before)')
    # # plot_histogram(matched_image[:, :, 0], 'Red Channel Histogram (After)')
    # #
    # # plot_histogram(np_test_image[:, :, 1], 'Green Channel Histogram (Before)')
    # # plot_histogram(matched_image[:, :, 1], 'Green Channel Histogram (After)')
    # #
    # # plot_histogram(np_test_image[:, :, 2], 'Blue Channel Histogram (Before)')
    # # plot_histogram(matched_image[:, :, 2], 'Blue Channel Histogram (After)')
    #
    # # 显示匹配前后的图像
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title('Test Image (Before Matching)')
    # plt.imshow(test_image)
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.title('Test Image (After Matching)')
    # plt.imshow(matched_image)
    # plt.axis('off')
    #
    # plt.show()
if __name__ == '__main__':
    IMG_PATH='eval/img/428597.jpg'
    image = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    image = process_image(image, "./water")