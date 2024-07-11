#tensorrt中yolov10的水平检测输出的是两个点的坐标：x1,y1,x2,y2,不是坐标加长宽
import tqdm
from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80  # your model classes

    def iou(self, box1, box2):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1 (list): First bounding box in the format (x1, y1, x2, y2).
            box2 (list): Second bounding box in the format (x1, y1, x2, y2).

        Returns:
            float: IoU value.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area


    def merge_nearby_detections(self,final_detections, window_size, step_size, iou_threshold,distance_threshold):
        """
        Merge nearby detections at the boundary of sliding windows.

        Args:
            final_detections (list): List of detections in the format [x1, y1, x2, y2, score, class_id].
            window_size (tuple): Size of the sliding window (width, height).
            step_size (tuple): Step size for sliding the window (width, height).
            iou_threshold (float): Minimum IoU between boxes to consider merging.

        Returns:
            list: Merged detections.
        """

        def merge_boxes(box1, box2):
            """Merge two boxes into one."""
            x1_min, y1_min, x1_max, y1_max = box1[:4]
            x2_min, y2_min, x2_max, y2_max = box2[:4]

            new_x_min = min(x1_min, x2_min)
            new_y_min = min(y1_min, y2_min)
            new_x_max = max(x1_max, x2_max)
            new_y_max = max(y1_max, y2_max)

            return [new_x_min, new_y_min, new_x_max, new_y_max, max(box1[4], box2[4]), box1[5]]

        def is_near_boundary(box, window_size, step_size, distance_threshold):
            """Check if the box is near the boundary of a sliding window."""
            x_min, y_min, x_max, y_max = box[:4]

            # 计算框与窗口边缘的横向和纵向距离
            distance_to_left = x_min % step_size[0]
            distance_to_right = step_size[0] - (x_max % step_size[0])
            distance_to_top = y_min % step_size[1]
            distance_to_bottom = step_size[1] - (y_max % step_size[1])


            # 检查框的任意边缘是否在距离阈值之内
            near_horizontal_boundary = distance_to_left <= distance_threshold or distance_to_right <= distance_threshold
            near_vertical_boundary = distance_to_top <= distance_threshold or distance_to_bottom <= distance_threshold

            return near_horizontal_boundary or near_vertical_boundary


        merged_detections = []
        used_indices = set()

        for i, box1 in enumerate(final_detections):

            if i in used_indices:
                continue
            merged_box = box1
            for j, box2 in enumerate(final_detections):
                if j <= i or j in used_indices:
                    continue
                if box1[5] == box2[5] and is_near_boundary(box1, window_size, step_size,
                                                           distance_threshold) and is_near_boundary(box2, window_size,
                                                                                               step_size,
                                                                                               distance_threshold) and self.iou(
                        box1, box2) > iou_threshold:
                    merged_box = merge_boxes(merged_box, box2)
                    used_indices.add(j)
            merged_detections.append(merged_box)
            used_indices.add(i)

        return merged_detections

    def non_max_suppression(self,boxes, iou_threshold, confidence_threshold):
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        Args:
            boxes (list): List of bounding boxes in the format (x1, y1, x2, y2, score, class_id).
            iou_threshold (float): Intersection over Union (IoU) threshold for NMS.
            confidence_threshold (float): Confidence threshold for filtering boxes.

        Returns:
            list: Filtered list of bounding boxes after applying NMS and confidence filtering.
        """
        boxes = list(boxes)
        if not boxes:
            return []

        # Filter boxes by confidence threshold
        boxes = [box for box in boxes if box[4] >= confidence_threshold]

        # If no boxes pass the confidence threshold, return an empty list
        if not boxes:
            return []

        # Sort boxes by score in descending order
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        filtered_boxes = []

        while boxes:
            best_box = boxes.pop(0)
            filtered_boxes.append(best_box)
            boxes = [box for box in boxes if self.iou(best_box[:4], box[:4]) < iou_threshold]

        return filtered_boxes

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
            detections = model.inference(window, conf=0.5, end2end=args.end2end,is_slide=True)

            detections = model.non_max_suppression(detections, iou_thres,conf_thres)

            new_detections=[]
            if len(detections) ==0:
                continue
            else:
                for detection in detections:
                    x1, y1, x2, y2, score, class_id =detection[0],detection[1],detection[2],detection[3],detection[4],detection[5]
                    x1 += x
                    y1 += y
                    x2 += x
                    y2 += y


                    new_detections.append([x1,y1,x2,y2,score,class_id])
                all_detections.extend(new_detections)

            # boxes = detections[:,:4]
            # scores = detections[:,4]
            # class_ids = detections[:,5]
            #
            # new_detections = []
            #
            #
            #
            #
            # indexs=cv2.dnn.NMSBoxes(boxes, scores, conf_thres,iou_thres)
            # if len(indexs) > 0:
            #     for i in indexs:
            #         x1,y1,x2,y2,score,class_id = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],scores[i],class_ids[i]
            #
            #         x1 += x
            #         y1 += y
            #         x2 += x
            #         y2 += y
            #
            #
            #         new_detections.append([x1,y1,x2,y2,score,class_id])


                # all_detections.extend(detections)



    all_detections = np.array(all_detections)
    print(len(all_detections))
    # indexs = cv2.dnn.NMSBoxes(all_detections[:,:4], all_detections[:,4], 0.5,0.9)
    # print(len(indexs))
    #
    # final_detections = np.array([all_detections[i] for i in indexs])

    final_detections=model.non_max_suppression(all_detections, iou_thres, conf_thres)
    print(len(final_detections))
    final_detections = model.merge_nearby_detections(final_detections, window_size, step_size, iou_threshold=0.2,
                                                    distance_threshold=140)
    print(len(final_detections))

    final_detections=np.array(final_detections)



    large_image = vis(large_image, final_detections[:,:4], final_detections[:, 4], final_detections[:, 5],
                     conf=conf_thres, class_names=model.class_names)



            # print(detections)



    # # 进行非极大值抑制（NMS）来合并重叠检测框
    # print(len(all_detections))
    # final_detections = self.non_max_suppression(all_detections, self.iou_thres)
    # print(len(all_detections))
    # final_detections=self.merge_nearby_detections(final_detections,window_size,step_size,iou_threshold=0.2,distance_threshold=140)
    # print(len(all_detections))
    # # 在大图上绘制最终的检测框
    # for detection in final_detections:
    #     box = detection[:4]
    #     score = detection[4]
    #     class_id = detection[5]
    #     self.draw_detections(large_image, box, score, class_id)

    return large_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default="yolov10.trt",help="TRT engine Path")
    parser.add_argument("-i", "--image", default="img/big_test.jpg",help="image path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("--end2end", default=True, action="store_true",help="use end2end engine")
    parser.add_argument("--is_slide", default=True, action="store_true", help="use sliding window inference")
    parser.add_argument("--window_size", type=tuple, default=(640, 640),help="Size of the sliding window (width, height)")
    parser.add_argument("--step_size", type=tuple, default=(500, 500),help="Step size for sliding window (width, height)")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IoU threshold")

    args = parser.parse_args()
    print(args)

    pred = Predictor(engine_path=args.engine)
    pred.get_fps()
    img_path = args.image
    video = args.video
    if img_path:
        if args.is_slide:
            result_img=sliding_window_inference(pred,img_path,args.window_size,args.step_size,args.conf_thres,args.iou_thres)
        else:
            result_img = pred.inference(img_path, conf=0.1, end2end=args.end2end)


        cv2.imshow('window', result_img)
        cv2.imwrite('detections.jpg', result_img)
        cv2.waitKey(0)
    if video:
      pred.detect_video(video, conf=0.1, end2end=args.end2end) # set 0 use a webcam
