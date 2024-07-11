# Ultralytics YOLO 馃殌, AGPL-3.0 license

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import tqdm


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = ["A2", "A10", "A3", "A19", "A1", "A13", "A20", "A15", "A16", "A17", "A12", "A5", "A14", "A7", "A9", "A4", "A18","A8", "A11", "A6"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    import cv2
    import numpy as np

    def draw_detections(self, img, box, score, class_id, rotation):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box (x_center, y_center, width, height).
            score: Corresponding detection score.
            class_id: Class ID for the detected object.
            rotation: Rotation angle of the detected object in degrees.

        Returns:
            None
        """

        # Extract the center coordinates, width, and height of the bounding box
        x, y, w, h = box[0], box[1], box[2], box[3]
        x_center, y_center=(2*x+w)//2, (2*y+h)//2

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]
        rotation_degrees = np.degrees(rotation)

        # Create a rotated rectangle
        rect = ((x_center, y_center), (w, h), rotation_degrees)

        # Get the 4 points of the rotated rectangle
        box_points = cv2.boxPoints(rect)
        box_points = np.intp(box_points)

        # Draw the rotated bounding box on the image
        cv2.polylines(img, [box_points], isClosed=True, color=color, thickness=2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = int(x_center - label_width / 2)
        label_y = int(y_center - h / 2 - 10 if y_center - h / 2 - 10 > label_height else y_center + h / 2 + 10)

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self,input_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = input_image

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output,window_offset):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))
        x_offset, y_offset = window_offset

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        rotations=[]
        ratation_boxes=[]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:-1]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)


            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)


                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                ratation_boxes.append(
                    (((2 * left + width) // 2, (2 * top + height) // 2), (width, height), outputs[i][-1] * 180 / np.pi))
                rotations.append(outputs[i][-1])


        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxesRotated(ratation_boxes, scores, self.confidence_thres, self.iou_thres)

        detections=[]

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:

            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            rotation=rotations[i]

            x, y, w, h = box[0], box[1], box[2], box[3]
            x1,y1,x2,y2 = x,y,x+w,y+h
            x1 += x_offset
            y1 += y_offset
            x2 += x_offset
            y2 += y_offset



            detections.append([[x1,y1,x2-x1,y2-y1],score,class_id,rotation])



        # Return the modified input image
        return detections


    def sliding_window_inference(self, window_size, step_size):
        """
        Performs sliding window inference on a large image.

        Args:
            image_path (str): Path to the input image.
            window_size (tuple): Size of the sliding window (width, height).
            step_size (tuple): Step size for sliding the window (width, height).

        Returns:
            numpy.ndarray: The output image with drawn detections.
        """
        # 创建ONNX推理会话
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider"])

        # 读取输入大图像
        large_image = cv2.imread(self.input_image)
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

                # 获取模型输入
                model_inputs = session.get_inputs()
                input_shape = model_inputs[0].shape
                self.input_width = input_shape[2]
                self.input_height = input_shape[3]

                # 预处理滑窗图像
                img_data = self.preprocess(window)





                # 运行推理
                outputs = session.run(None, {model_inputs[0].name: img_data})

                # 后处理输出，映射检测框到大图
                detections = self.postprocess(window, outputs, (x, y))
                all_detections.extend(detections)

        print(len(all_detections))


        big_box=[]
        big_score=[]
        big_class_id=[]
        big_rotation=[]
        big_ratation_boxes = []

        for detection in all_detections:
            left,top,width,height = detection[0][0], detection[0][1], detection[0][2], detection[0][3]
            big_box.append(detection[0])
            big_score.append(detection[1])
            big_class_id.append(detection[2])
            big_rotation.append(detection[3])
            big_ratation_boxes.append(
                (((2 * left + width) // 2, (2 * top + height) // 2), (width, height), detection[3] * 180 / np.pi))





        indexs = cv2.dnn.NMSBoxesRotated(big_ratation_boxes, big_score, self.confidence_thres, 0.2)

        detections = []

        # Iterate over the selected indices after non-maximum suppression
        for i in indexs:
            # Get the box, score, and class ID corresponding to the index
            box = big_box[i]
            score = big_score[i]
            class_id = big_class_id[i]
            rotation = big_rotation[i]
            detections.append([[box[0],box[1],box[2],box[3]],score,class_id,rotation])

        print(len(detections))



        for detection in all_detections:
            box = detection[0]
            score = detection[1]
            class_id = detection[2]
            rotation = detection[3]
            self.draw_detections(large_image, box, score, class_id,rotation)

        return large_image

    def main(self, window_size, step_size):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers


        reuult_img=self.sliding_window_inference( window_size, step_size)


        # Perform post-processing on the outputs to obtain output image.
        return reuult_img  # output image


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="v8m-obb-best.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default=str("./img/big_test.jpg"), help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--window-size", type=tuple, default=(640, 640),help="Size of the sliding window (width, height)")
    parser.add_argument("--step-size", type=tuple, default=(640, 640),help="Step size for sliding window (width, height)")

    args = parser.parse_args()


    # Create an instance of the YOLOv8 class with the specified arguments
    detection = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    output_image = detection.main( args.window_size, args.step_size)

    # Display the output image in a window
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)
    cv2.imwrite("output.jpg", output_image)

    # Wait for a key press to exit
    cv2.waitKey(0)
