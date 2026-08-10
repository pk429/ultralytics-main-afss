from ultralytics import YOLO
import torch

model = YOLO('/mnt/sda1/xzm/Code/ultralytics-main/runs/detect/runs/train/FISHER_DETECT_v11n_p2_1920_0527/weights/best.pt')
model.export(format='onnx', imgsz=1280, dynamic=True, simplify=True, opset=12, device='cuda')