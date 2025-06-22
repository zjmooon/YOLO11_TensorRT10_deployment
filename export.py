from ultralytics import YOLO

# Load the YOLO model
model = YOLO("Models/yolo11n.pt")
# Export the model to ONNX format
# export_path = model.export(format="onnx")
export_path = model.export(
    format="onnx",
    nms=True,         # 添加 NMS 后处理到模型中
    simplify=True,    # 简化 ONNX 模型结构
    dynamic=False,    # 是否动态 batch/尺寸，TensorRT 通常推荐 False
    imgsz=640         # 输入图像大小（可选）
)

print(f"Model exported to {export_path}")