#
import tensorrt as trt
from cuda import cudart
import numpy as np
import cv2
import matplotlib.pyplot as plt

engine_path="yolov8n-obb.trt"


logger = trt.Logger(trt.Logger.WARNING)
logger.min_severity = trt.Logger.Severity.ERROR
runtime = trt.Runtime(logger)
with open(engine_path, "rb") as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()


