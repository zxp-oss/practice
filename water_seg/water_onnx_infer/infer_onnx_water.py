import cv2
import os
import math
import numpy as np
import onnxruntime as ort
import json
import argparse
from hist_process import process_image

import tqdm

IMG_MEAN = np.array(
    (0.5, 0.5, 0.5),
    dtype=np.float32)
IMG_VARS = np.array(
    (0.5, 0.5, 0.5),
    dtype=np.float32)


land_cover_colors = {
    0: (0, 0, 0),  # 背景
    1: (255, 255, 0),  # 河流
    2: (255, 0, 255),  # 湖泊
    3: (0, 255, 0),  # 池塘
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
    img = cv2.resize(img, (1024, 1024))
    img_cache = img.astype(np.float32) / 255.0 - IMG_MEAN
    img_cache = img_cache / IMG_VARS
    image_tra = np.transpose(img_cache, (2, 0, 1))
    image_exdim = np.expand_dims(image_tra, axis=0)

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_exdim})

    img_result = outputs[0]
    img_result = np.transpose(img_result, (1, 2, 0))
    img_result[img_result == 1] = 255
    img_result[img_result == 2] = 255
    img_result[img_result == 3] = 255
    # img_result = cv2.resize(img_result, (2048, 2048))
    # cv2.imshow('image3', img_result.squeeze().astype(np.uint8))
    # cv2.waitKey(0)

    return img_result

def slide_inference(im, crop_size, stride):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """

    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # print(rows, cols)
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = None
    count = np.zeros([1, 1, h_im, w_im])
    for r in tqdm.tqdm(range(rows)):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            # 去掉批次维度，变成 (3, 1024, 1024)
            im_crop = np.squeeze(im_crop, axis=0)

            # 调整维度顺序，变成 (1024, 1024, 3)
            im_crop = np.transpose(im_crop, (1, 2, 0))


            logits = onnx_infer(im_crop, ONNX_PATH)

            logits = np.transpose(logits, (2, 0, 1))
            logit = np.expand_dims(logits, axis=0)



            # logit = logits[0].numpy()
            if final_logit is None:
                final_logit = np.zeros([1, logit.shape[1], h_im, w_im])
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
    cv2.imshow('image3', final_logit.squeeze().astype(np.uint8))
    cv2.waitKey(0)

    return final_logit

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
    filename = IMG_PATH.split('/')[-1].split('.')[0]
    image = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    image=process_image(image,"./water")
    reconstructed_image=slide_inference(image, (1024,1024),(512,512) )
    reconstructed_image=reconstructed_image.squeeze()

    cv2.imwrite(f"output/water/{filename}_1d_mask1.png", reconstructed_image)


    #
    color_image_array = convert_mask_to_color(reconstructed_image, land_cover_colors)
    cv2.namedWindow('image0', cv2.WINDOW_NORMAL)

    cv2.imshow('image0', color_image_array)
    cv2.imwrite(f"output/water/{filename}_rgb_mask.jpg", color_image_array)
    cv2.waitKey(0)

    result_img=apply_mask_with_transparency(image, color_image_array,0.5)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    cv2.imshow('image', result_img)
    cv2.imwrite(f"output/water/{filename}_result_img.jpg", result_img)
    cv2.waitKey(0)





if __name__ == '__main__':
    ONNX_PATH="weights/segformer_water.onnx"
    img_path="img/water/GF2_PMS1__L1A0000564539-MSS1_tile_0.jpg"
    main(img_path)