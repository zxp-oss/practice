from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse
import tqdm

class Predictor(BaseEngine):
    def __init__(self, engine_path,confidence_thres, iou_thres):
        super(Predictor, self).__init__(engine_path,confidence_thres, iou_thres)
        self.n_classes = 20  # your model classes


def sliding_window_inference(model,image_path, window_size, step_size,conf_thres,iou_thres):
    # 读取输入大图像
    large_image = cv2.imread(image_path)
    large_image_height, large_image_width = large_image.shape[:2]

    all_detections = []


    for y in tqdm.tqdm(range(0, large_image_height, step_size[1])):
        for x in range(0, large_image_width, step_size[0]):
            # 提取滑窗区域
            window = large_image[y:y + window_size[1], x:x + window_size[0]]

            # 如果滑窗尺寸不符合，进行填充
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                padded_window = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
                padded_window[:window.shape[0], :window.shape[1], :] = window
                window = padded_window

            # 预处理滑窗图像
            output = model.inference(window, end2end=args.end2end,is_slide=True)

            # Transpose and squeeze the output to match the expected shape
            outputs = np.transpose(np.squeeze(output[0]))
            x_offset, y_offset = x,y

            # Get the number of rows in the outputs array
            rows = outputs.shape[0]

            # Lists to store the bounding boxes, scores, and class IDs of the detections
            boxes = []
            scores = []
            class_ids = []
            rotations = []
            ratation_boxes = []

            # Calculate the scaling factors for the bounding box coordinates
            x_factor = window.shape[1] / 640
            y_factor = window.shape[0] / 640

            # Iterate over each row in the outputs array
            for i in range(rows):
                # Extract the class scores from the current row
                classes_scores = outputs[i][4:-1]

                # Find the maximum score among the class scores
                max_score = np.amax(classes_scores)

                # If the maximum score is above the confidence threshold
                if max_score >= conf_thres:
                    # Get the class ID with the highest score
                    class_id = np.argmax(classes_scores)

                    # Extract the bounding box coordinates from the current row
                    x_center, y_center, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                    # Calculate the scaled coordinates of the bounding box
                    left = int((x_center - w / 2) * x_factor)
                    top = int((y_center - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # Add the class ID, score, and box coordinates to the respective lists
                    class_ids.append(class_id)
                    scores.append(max_score)
                    boxes.append([left, top, width, height])
                    ratation_boxes.append(
                        (((2 * left + width) // 2, (2 * top + height) // 2), (width, height),
                         outputs[i][-1] * 180 / np.pi))
                    rotations.append(outputs[i][-1])

            # Apply non-maximum suppression to filter out overlapping bounding boxes
            indices = cv2.dnn.NMSBoxesRotated(ratation_boxes, scores, conf_thres, iou_thres)

            detections = []

            # Iterate over the selected indices after non-maximum suppression
            for i in indices:
                # Get the box, score, and class ID corresponding to the index
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                rotation = rotations[i]

                x0, y0, w, h = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = x0, y0, x0 + w, y0 + h
                x1 += x_offset
                y1 += y_offset
                x2 += x_offset
                y2 += y_offset

                detections.append([[x1, y1, x2 - x1, y2 - y1], score, class_id, rotation])

            all_detections.extend(detections)

    big_box = []
    big_score = []
    big_class_id = []
    big_rotation = []
    big_ratation_boxes = []

    for detection in all_detections:
        left, top, width, height = detection[0][0], detection[0][1], detection[0][2], detection[0][3]
        big_box.append(detection[0])
        big_score.append(detection[1])
        big_class_id.append(detection[2])
        big_rotation.append(detection[3])
        big_ratation_boxes.append(
            (((2 * left + width) // 2, (2 * top + height) // 2), (width, height), detection[3] * 180 / np.pi))

    indexs = cv2.dnn.NMSBoxesRotated(big_ratation_boxes, big_score, conf_thres, 0.2)

    # Iterate over the selected indices after non-maximum suppression
    for i in indexs:
        # Get the box, score, and class ID corresponding to the index
        box = big_box[i]
        score = big_score[i]
        class_id = big_class_id[i]
        rotation = big_rotation[i]
        model.draw_detections(large_image, box, score, class_id, rotation)


    return large_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default="v8m-obb-best.trt", help="TRT engine Path")
    parser.add_argument("-i", "--image", default="img/big_test.jpg", help="image path")
    parser.add_argument("-v", "--video", help="video path or camera index ")
    parser.add_argument("--end2end", default=True, action="store_true", help="use end2end engine")
    parser.add_argument("--is_slide", default=True, action="store_true", help="use sliding window inference")
    parser.add_argument("--window_size", type=tuple, default=(640, 640),
                        help="Size of the sliding window (width, height)")
    parser.add_argument("--step_size", type=tuple, default=(640, 640),
                        help="Step size for sliding window (width, height)")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IoU threshold")

    args = parser.parse_args()
    print(args)

    pred = Predictor(engine_path=args.engine,confidence_thres=args.conf_thres, iou_thres=args.iou_thres)
    pred.get_fps()
    img_path = args.image
    video = args.video
    if img_path:
        if args.is_slide:
            result_img = sliding_window_inference(pred, img_path, args.window_size, args.step_size, args.conf_thres,
                                                  args.iou_thres)
        else:
            result_img = pred.inference(img_path, end2end=args.end2end)

        cv2.imshow('window', result_img)
        cv2.imwrite('detections.jpg', result_img)
        cv2.waitKey(0)
    if video:
        pred.detect_video(video, conf=0.1, end2end=args.end2end)  # set 0 use a webcam
