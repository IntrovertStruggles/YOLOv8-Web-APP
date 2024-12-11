from ultralytics import YOLO
from memory_profiler import profile

@profile

def func():
    model = YOLO("yolov8n.yaml")
    results = model.train(data="config.yaml", epochs=300, imgsz=640, pretrained=False, single_cls=True, device='cpu')
    return results

if __name__ == "__main__":
    result = func()








# (.venv2) PS C:\Users\asus\PycharmProjects\pythonProject> pip install memory_profiler
# (.venv2) PS C:\Users\asus\PycharmProjects\pythonProject> pip install ultralytics