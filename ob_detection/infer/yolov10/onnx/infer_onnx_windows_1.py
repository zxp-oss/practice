import argparse
import cv2
import numpy as np
import onnxruntime as ort
import tqdm

CLASS_NAMES = {0: 'A2', 1: 'A10', 2: 'A3', 3: 'A19', 4: 'A1', 5: 'A13', 6: 'A20', 7: 'truck',
               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant'}





class YOLOv10:
    """YOLOv10 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv10 class.

        Args:
            onnx_model: Path to the ONNX model.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = [CLASS_NAMES[i] for i in range(20)]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

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

    # Example usage:
    # final_detections = [[x1, y1, x2, y2, score, class_id], ...]
    # merged_detections = merge_nearby_detections(final_detections, window_size, step_size, iou_threshold=0.5)

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = box
        x1, y1, w, h=int(x1), int(y1), int(x2-x1), int(y2-y1)

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)



    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        self.img = img
        self.img_height, self.img_width = self.img.shape[:2]

        obj_shape = max(self.img_height, self.img_width)
        self.real_shape = obj_shape
        top_pad = (obj_shape - self.img_height) // 2
        bottom_pad = obj_shape - self.img_height - top_pad
        left_pad = (obj_shape - self.img_width) // 2
        right_pad = obj_shape - self.img_width - left_pad

        img = cv2.copyMakeBorder(self.img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                 value=[127, 127, 127])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        image_data = (np.array(img)) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
            boxes (np.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).
            ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                calculated based on the size difference between the two images.
            padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
                rescaling.
            xywh (bool): The box format is xywh or not, default=False.

        Returns:
            boxes (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding
            if not xywh:
                boxes[..., 2] -= pad[0]  # x padding
                boxes[..., 3] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)

    def clip_boxes(self, boxes, shape):
        """
        Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

        Args:
            boxes (np.ndarray): the bounding boxes to clip
            shape (tuple): the shape of the image

        Returns:
            (np.ndarray): Clipped boxes
        """
        boxes[..., 0] = np.clip(boxes[..., 0], 0, shape[1])  # x1
        boxes[..., 1] = np.clip(boxes[..., 1], 0, shape[0])  # y1
        boxes[..., 2] = np.clip(boxes[..., 2], 0, shape[1])  # x2
        boxes[..., 3] = np.clip(boxes[..., 3], 0, shape[0])  # y2
        return boxes

    def postprocess(self, input_image, output, window_offset):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.
            window_offset (tuple): The (x, y) offset of the current sliding window.

        Returns:
            list: List of detections in the format (x1, y1, x2, y2, score, class_id).
        """
        outputs = output[0][0]
        rows = outputs.shape[0]
        boxes = []
        x_offset, y_offset = window_offset

        for i in range(rows):
            max_score = outputs[i, 4]
            if max_score >= self.confidence_thres:
                class_id = int(outputs[i, 5])
                new_bbox = self.scale_boxes([640, 640], np.array([outputs[i, :4]]), (self.img_height, self.img_width),
                                            xywh=False)
                x1, y1, x2, y2 = new_bbox[0]
                x1 += x_offset
                y1 += y_offset
                x2 += x_offset
                y2 += y_offset
                boxes.append([x1, y1, x2, y2, max_score, class_id])
        return boxes

    def non_max_suppression(self, boxes, iou_threshold):
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        Args:
            boxes (list): List of bounding boxes in the format (x1, y1, x2, y2, score, class_id).
            iou_threshold (float): Intersection over Union (IoU) threshold for NMS.

        Returns:
            list: Filtered list of bounding boxes after applying NMS.
        """
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        filtered_boxes = []

        while boxes:
            best_box = boxes.pop(0)
            filtered_boxes.append(best_box)
            boxes = [box for box in boxes if self.iou(best_box, box) < iou_threshold]

        return filtered_boxes

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

    def sliding_window_inference(self, image_path, window_size, step_size):
        """
        Performs sliding window inference on a large image.

        Args:
            image_path (str): Path to the input image.
            window_size (tuple): Size of the sliding window (width, height).
            step_size (tuple): Step size for sliding the window (width, height).

        Returns:
            numpy.ndarray: The output image with drawn detections.
        """
        # 读取输入大图像
        large_image = cv2.imread(image_path)
        large_image_height, large_image_width = large_image.shape[:2]

        all_detections = []
        # 创建ONNX推理会话
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider"])

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
                img_data = self.preprocess(window)

                # 获取模型输入
                model_inputs = session.get_inputs()

                # 运行推理
                outputs = session.run(None, {model_inputs[0].name: img_data})

                # 后处理输出，映射检测框到大图
                detections = self.postprocess(large_image, outputs, (x, y))
                all_detections.extend(detections)

        # 进行非极大值抑制（NMS）来合并重叠检测框
        print(len(all_detections))
        final_detections = self.non_max_suppression(all_detections, self.iou_thres)
        print(len(all_detections))
        final_detections=self.merge_nearby_detections(final_detections,window_size,step_size,iou_threshold=0.2,distance_threshold=140)
        print(len(all_detections))
        # 在大图上绘制最终的检测框
        for detection in final_detections:
            box = detection[:4]
            score = detection[4]
            class_id = detection[5]
            self.draw_detections(large_image, box, score, class_id)

        return large_image


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./bestx.onnx",
                        help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default="img/big_test.jpg",
                        help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--window-size", type=tuple, default=(640, 640),
                        help="Size of the sliding window (width, height)")
    parser.add_argument("--step-size", type=tuple, default=(500, 500),
                        help="Step size for sliding window (width, height)")
    args = parser.parse_args()

    # 创建YOLOv10类的实例
    detection = YOLOv10(args.model, args.conf_thres, args.iou_thres)

    # 在大图上执行滑窗推理
    output_image = detection.sliding_window_inference(args.img, args.window_size, args.step_size)

    # 显示输出图像
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)

    # 保存结果图像
    cv2.imwrite("result.jpg", output_image)

    # 等待按键以退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()
