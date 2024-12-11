from ultralytics import YOLO
model = YOLO("nopretrain_best.pt")

model.export(format="onnx")