zxp_目标检测相关代码
# yolov8infer
里面是用yolov8做的旋转目标检测的推理代码，包括onnx和tensorrt，都使用了cuda
# yolov10-main
## 项目
里面包括了yolov10的基础项目，只能做水平目标检测
## 推理代码
里面包括onnx的推理代码，滑窗onnx推理代码  
tensorrt推理代码  
## 脚本工具
如：  
1.big_size_img.py 小图合成大图  
2.data_convert.py 转换标准VOC数据格式到yolov10可以训练的  

# 注
1.滑窗代码融合的决策不完善，只是使用nms简单融合，后续使用时再完善
