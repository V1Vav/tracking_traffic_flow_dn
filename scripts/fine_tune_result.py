from ultralytics import YOLO
from multiprocessing import freeze_support
import time
import glob
import os

# MODEL_PATH = "../tuning/runs/detect/traffic_20260524_0016_e200/weights/best.pt"
MODEL_PATH = "../tuning/runs/detect/traffic_20260603_2236_e100/weights/best.pt"
# DATA_YAML = "../tuning/Traffic_Count_VN-1/data.yaml"
DATA_YAML = "../tuning/vehicle-u1nn1/data.yaml"
# IMAGE_DIR = "../tuning/Traffic_Count_VN-1/valid/images"   # đổi thành valid/images nếu muốn đo FPS trên tập val
IMAGE_DIR = "../tuning/vehicle-u1nn1/valid/images" 


def evaluate_model():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640,
        batch=8,
        conf=0.001,
        iou=0.6,
        device=0,
        workers=0
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    inference_ms = metrics.speed["inference"]
    fps_from_val = 1000 / inference_ms

    print("\n===== MODEL EVALUATION =====")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"mAP@50        : {map50:.4f}")
    print(f"mAP@50:95     : {map5095:.4f}")
    print(f"F1-score      : {f1:.4f}")

    print("\n===== SPEED FROM YOLO VAL =====")
    print(f"Inference time: {inference_ms:.2f} ms/image")
    print(f"FPS inference : {fps_from_val:.2f}")

    print("\n===== PER CLASS mAP@50:95 =====")
    names = model.names
    for i, class_map in enumerate(metrics.box.maps):
        print(f"{i} - {names[i]}: {class_map:.4f}")


def measure_real_fps():
    model = YOLO(MODEL_PATH)

    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.*"))

    if len(image_paths) == 0:
        print("Không tìm thấy ảnh để đo FPS.")
        return

    for img_path in image_paths[:10]:
        model(img_path, imgsz=640, device=0, verbose=False)

    start = time.time()

    for img_path in image_paths:
        model(img_path, imgsz=640, device=0, verbose=False)

    end = time.time()

    total_time = end - start
    fps = len(image_paths) / total_time

    print("\n===== REAL INFERENCE FPS =====")
    print(f"Number of images : {len(image_paths)}")
    print(f"Total time       : {total_time:.4f} s")
    print(f"FPS inference    : {fps:.2f}")
    print(f"Time per image   : {1000 / fps:.2f} ms")


if __name__ == "__main__":
    freeze_support()

    evaluate_model()
    measure_real_fps()