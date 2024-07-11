import cv2
import os
import math
import numpy as np
import onnxruntime as ort
import json
import argparse

import tqdm

IMG_MEAN = np.array(
    (0.5, 0.5, 0.5),
    dtype=np.float32)
IMG_VARS = np.array(
    (0.5, 0.5, 0.5),
    dtype=np.float32)


land_cover_colors = {
    0: (0, 0, 0),  # 背景
    1: (255, 255, 0),  # 农田
    2: (255, 0, 255),  # 除了森林、农田以外的绿色场景
    3: (0, 255, 0),  # 森林
}

# 定义函数：将mask图像转换为彩色图像
def convert_mask_to_color(image_array, land_cover_colors):
    # 创建一个新的彩色图像
    color_image = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    # 将每个像素值转换为对应的颜色
    for value, color in land_cover_colors.items():
        mask = (image_array == value)
        color_image[mask] = color

    return color_image

def onnx_infer(img, onnx_path):
    img = cv2.resize(img, (512, 512))
    img_cache = img.astype(np.float32) / 255.0 - IMG_MEAN
    img_cache = img_cache / IMG_VARS
    image_tra = np.transpose(img_cache, (2, 0, 1))
    image_exdim = np.expand_dims(image_tra, axis=0)

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_exdim})

    img_result = outputs[0]
    img_result = np.transpose(img_result, (1, 2, 0))


    return img_result



def slid_windows(image, window_size, stride):
    height, width, _ = image.shape

    num_h = math.ceil((image.shape[0] - window_size) / (window_size - stride)) + 1
    num_w = math.ceil((image.shape[1] - window_size) / (window_size - stride)) + 1

    # reconstructed_image = np.zeros(image.shape, dtype=np.uint8)
    reconstructed_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    for h in tqdm.tqdm(range(num_h)):
        for w in range(num_w):
            top = h * (window_size - stride)
            left = w * (window_size - stride)
            bottom = top + window_size
            right = left + window_size

            if bottom > image.shape[0]:
                top = image.shape[0] - window_size
                bottom = image.shape[0]
            if right > image.shape[1]:
                left = image.shape[1] - window_size
                right = image.shape[1]

            cropped_image = image[top:bottom, left:right]

            img = onnx_infer(cropped_image, ONNX_PATH)
            reconstructed_image[top:bottom, left:right] = img

    return reconstructed_image


def apply_mask_with_transparency(original_image, mask_image, transparency):

    # Ensure the mask image is the same size as the original image
    if original_image.shape[:2] != mask_image.shape[:2]:
        raise ValueError("The mask image must be the same size as the original image.")

    # Blend the mask with the original image
    result_image = cv2.addWeighted(original_image.astype(np.float32), 1.0, mask_image.astype(np.float32), transparency, 0)

    # Convert the result back to uint8
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    return result_image

def main(IMG_PATH):
    filename=IMG_PATH.split('/')[-1].split('.')[0]
    image = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    reconstructed_image=slid_windows(image, 512, 128)
    reconstructed_image=reconstructed_image.squeeze()


    color_image_array = convert_mask_to_color(reconstructed_image, land_cover_colors)

    cv2.namedWindow('image0', cv2.WINDOW_NORMAL)

    cv2.imshow('image0', color_image_array)
    cv2.imwrite(f"output/land/{filename}_rgb_mask.jpg", color_image_array)
    cv2.waitKey(0)

    result_img=apply_mask_with_transparency(image, color_image_array,0.5)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    cv2.imshow('image', result_img)
    cv2.imwrite(f"output/land/{filename}_result_img.jpg", result_img)
    cv2.waitKey(0)





if __name__ == '__main__':
    ONNX_PATH="weights/infer_onnx_vegetation.onnx"
    img_path="img/hist_r.jpg"
    main(img_path)
