from system_usage_dec import monitor_performance
from ultralytics import YOLO


model_used = 'yolov8n'


@monitor_performance(f'../results/{model_used}_memory_usage')
def train_yolo_model():
    model = YOLO(f'{model_used}.pt')

    model.train(data='D:\detection_drones\dataset\data.yaml', epochs=20, imgsz=640)

    metrics = model.val()
    map50 = metrics.box.map50

    return map50


if __name__ == '__main__':
    train_yolo_model()
