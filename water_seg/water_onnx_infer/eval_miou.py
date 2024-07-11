import cv2
import numpy as np
from PIL import Image


def calculate_iou_per_class(mask, gt):
    """
    计算每个类别的IoU

    参数:
    - mask: 预测的掩码图像（numpy数组）
    - gt: 真实的掩码图像（numpy数组）

    返回:
    - iou_per_class: 每个类别的IoU字典
    """
    classes = np.unique(np.concatenate((mask, gt)))
    iou_per_class = {}

    for cls in classes:
        mask_cls = (mask == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(mask_cls, gt_cls).sum()
        union = np.logical_or(mask_cls, gt_cls).sum()

        if union != 0:
            iou_per_class[cls] = intersection / union
        else:
            iou_per_class[cls] = 0.0

    return iou_per_class


def calculate_miou(mask, gt):
    """
    计算mIoU和每个类别的IoU

    参数:
    - mask: 预测的掩码图像（numpy数组）
    - gt: 真实的掩码图像（numpy数组）

    返回:
    - miou: 平均IoU
    - iou_per_class: 每个类别的IoU字典
    """
    iou_per_class = calculate_iou_per_class(mask, gt)
    miou = np.mean(list(iou_per_class.values()))

    return miou, iou_per_class


def load_image_as_array(filepath):
    """
    加载图像并转换为numpy数组

    参数:
    - filepath: 图像文件路径

    返回:
    - img_array: 图像的numpy数组表示
    """
    img = Image.open(filepath)
    img_array = np.array(img)
    return img_array


# 示例用法
if __name__ == "__main__":
    # 加载掩码和真实标签图像
    # mask_path = 'eval/iBRAIN_pre/GF2_PMS1__L1A0000564539-MSS1_tile_0_VEGETATION_INFERENCE.tif'
    # mask_path="eval/my_pre/GF2_PMS1__L1A0000564539-MSS1_tile_0_1d_mask1.png"
    # mask_path = "eval/my_pre/GF2_PMS1__L1A0000564539-MSS1_tile_0_1d_mask1_no_hist.png"
    mask_path="eval/my_pre/GF2_PMS1__L1A0000564539-MSS1_ocrnet_land.png"
    gt_path = 'eval/labels/GF2_PMS1__L1A0000564539-MSS1_land.png'

    mask = load_image_as_array(mask_path)
    gt = load_image_as_array(gt_path)

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow("mask", mask)

    cv2.namedWindow('gt', cv2.WINDOW_NORMAL)
    cv2.imshow("gt", gt)
    cv2.waitKey(0)
    miou, iou_per_class = calculate_miou(mask, gt)
    print("Mean IoU:", miou)
    print("IoU per class:", iou_per_class)
