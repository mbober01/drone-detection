from system_usage_dec import monitor_performance
from yolov5 import train, val


def train_yolo_model(data_yaml, img_size=640, batch_size=16, epochs=10, name='yolov5s'):
    options = {
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'data': data_yaml,
        'weights': f'{name}.pt',
        'name': name
    }

    x = train.run(**options)
    return f'{x.save_dir}/weights/best.pt'


def evaluate_yolo_model(data_yaml, weights, name='yolov5s'):
    options = {
        'data': data_yaml,
        'weights': weights,
        'name': name,
    }

    y = val.run(**options)

    return y[0][2]


model_used = 'yolov5m'


@monitor_performance(f'../results/{model_used}_memory_usage')
def train_eval():
    yaml_path = 'D:\detection_drones\dataset\data.yaml'
    weights_path = train_yolo_model(yaml_path, name=model_used)
    map50 = evaluate_yolo_model(yaml_path, weights=weights_path, name=model_used)
    return map50


if __name__ == '__main__':
    map50 = train_eval()
