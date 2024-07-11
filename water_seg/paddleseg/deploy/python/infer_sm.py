

import os
import math
import numpy as np
from PIL import Image
import tqdm
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from paddleseg.deploy.infer import DeployConfig
from paddleseg.utils.visualize import get_pseudo_color_map
from paddleseg.utils import logger


class SimplePredictor:
    def __init__(self, cfg_path, device='cpu', use_trt=False, precision='fp32'):
        """
        Initialize the predictor with the configuration and device settings.
        """
        self.cfg = DeployConfig(cfg_path)
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        self.device = device
        self.use_trt = use_trt
        self.precision = precision

        self._init_base_config()
        if device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        self.predictor = create_predictor(self.pred_cfg)

    def _init_base_config(self):
        self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        logger.info("Using CPU")
        self.pred_cfg.disable_gpu()

    def _init_gpu_config(self):
        logger.info("Using GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        if self.use_trt:
            logger.info("Using TensorRT")
            precision_map = {
                "fp16": PrecisionType.Half,
                "fp32": PrecisionType.Float32,
                "int8": PrecisionType.Int8
            }
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=3,
                precision_mode=precision_map[self.precision],
                use_static=False,
                use_calib_mode=False)
            min_input_shape = {"x": [1, 3, 100, 100]}
            max_input_shape = {"x": [1, 3, 2000, 3000]}
            opt_input_shape = {"x": [1, 3, 512, 1024]}
            self.pred_cfg.set_trt_dynamic_shape_info(
                min_input_shape, max_input_shape, opt_input_shape)

    def predict(self, img_path, window_size=512, stride=256):
        img = Image.open(img_path)
        img = np.array(img)
        height, width, _ = img.shape

        num_h = math.ceil((height - window_size) / (window_size - stride)) + 1
        num_w = math.ceil((width - window_size) / (window_size - stride)) + 1

        reconstructed_image = np.zeros((height, width), dtype=np.int32)
        counts = np.zeros((height, width), dtype=np.int32)

        for h in tqdm.tqdm(range(num_h)):
            for w in range(num_w):
                top = h * (window_size - stride)
                left = w * (window_size - stride)
                bottom = top + window_size
                right = left + window_size

                if bottom > height:
                    top = height - window_size
                    bottom = height
                if right > width:
                    left = width - window_size
                    right = width

                cropped_image = img[top:bottom, left:right]
                data = self._preprocess(cropped_image)
                input_names = self.predictor.get_input_names()
                input_handle = self.predictor.get_input_handle(input_names[0])
                output_names = self.predictor.get_output_names()
                output_handle = self.predictor.get_output_handle(output_names[0])

                input_handle.reshape(data.shape)
                input_handle.copy_from_cpu(data)
                self.predictor.run()
                result = output_handle.copy_to_cpu()
                # result = self._postprocess(result)
                result = result.squeeze()

                h_res, w_res = result.shape
                reconstructed_image[top:bottom, left:right] += result
                counts[top:bottom, left:right] += 1

        reconstructed_image = reconstructed_image / counts
        return get_pseudo_color_map(reconstructed_image.astype(np.uint32))

    def _preprocess(self, img):
        data = {'img': img}
        return np.array([self.cfg.transforms(data)['img']])

    def _postprocess(self, results):
        results = np.argmax(results, axis=1)
        return results[0]


def main():
    cfg_path = './infer_model/inference_model/deploy.yaml'
    img_path = '/home/zxp/test_project/PaddleSeg-release-2.9/images/GF2_PMS1__L1A0000564539-MSS1_tile_0.jpg'
    device = 'gpu'  # or 'cpu'
    use_trt = False
    precision = 'fp32'

    predictor = SimplePredictor(cfg_path, device, use_trt, precision)
    result = predictor.predict(img_path, window_size=1024, stride=256)
    result.save('output/result.png')


if __name__ == '__main__':
    main()
