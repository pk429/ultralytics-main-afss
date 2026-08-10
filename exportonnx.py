

from ultralytics import YOLO


# 加载训练好的模型
model = YOLO('/mnt/sda1/xzm/Code/ultralytics-main/runs/detect/runs/train/RIVER_RUBBISH_DETECTED_v11s_1280_0731/weights/best.pt')  
 
# 导出ONNX模型
model.export(
    format='onnx',
    imgsz=(1280, 1280),  # 与训练时相同的输入尺寸
    dynamic=True,     # 固定批次和尺寸以获得更好性能
    simplify=True,     # 启用模型简化
    opset=12,          # ONNX算子集版本
    device='cuda'      # 使用GPU加速导出
)

# # #昇腾

# """ascend版本"""
# from ultralytics.nn.modules import Conv
# import torch.nn as nn

# Conv.default_act = nn.Hardswish()
# from ultralytics import YOLO
 
# model = YOLO(r'/mnt/sda1/xzm/Code/ultralytics-main/runs/obb/cargoship-obbdetect/CARGOSHIP_DETECT_v11s_1280_0724_ascend/weights/0727_caroship_visible_yolov11_1280_best_conv.pt')  # load a custom trained
# # Export the model
# model.export(format='onnx', imgsz = [736, 1280], opset=12, simplify=True)  # export to onnx format, imgsz=[in_h, in_w]    