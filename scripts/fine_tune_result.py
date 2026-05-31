from ultralytics import YOLO
import time
import cv2
import glob
import os

MODEL_PATH = "../tuning/runs/detect/traffic_20260524_0016_e200/weights/best.pt"
DATA_YAML = "../tuning/Traffic_Count_VN-1/data.yaml"
TEST_IMAGE_DIR = "../tuning/Traffic_Count_VN-1/valid/images"   # đổi thành valid/images nếu muốn đo FPS trên tập val

model = YOLO(MODEL_PATH)

metrics = model.val(
    data=DATA_YAML,
    split="val",        # "val" nếu muốn đánh giá trên validation
    imgsz=640,
    batch=16,
    conf=0.001,
    iou=0.6,
    device=0
)

precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map5095 = metrics.box.map

f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("===== MODEL EVALUATION =====")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"mAP@50        : {map50:.4f}")
print(f"mAP@50:95     : {map5095:.4f}")
print(f"F1-score      : {f1:.4f}")

inference_ms = metrics.speed["inference"]
fps_val = 1000 / inference_ms

print("\n===== SPEED FROM YOLO VAL =====")
print(f"Inference time: {inference_ms:.2f} ms/image")
print(f"FPS inference : {fps_val:.2f}")