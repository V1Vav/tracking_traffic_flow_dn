import cv2
import time
import argparse
from ultralytics.models import YOLO

# ===== Argument Parser =====
parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    type=str,
    required=True,
    help="Path to input video"
)
parser.add_argument(
    "--model",
    type=str,
    default="yolov8n.pt",
    help="YOLO model path (yolov8n.pt, yolov8s.pt, ...)"
)

args = parser.parse_args()

# ===== Load model =====
model = YOLO(args.model)

# ===== Load video =====
cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# ===== FPS variables =====
prev_time = 0

# ===== Loop =====
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ===== Inference =====
    results = model(frame, verbose=False)

    # ===== Draw boxes =====
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in [2, 3, 5, 7]:  # car, motorbike, bus, truck
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = {
                2: (0, 255, 0),
                3: (255, 0, 0),
                5: (0, 0, 255),
                7: (0, 255, 255)
            }.get(cls, (255, 255, 255))

            label = f"{cls}:{conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ===== FPS =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("YOLOv8 Traffic", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()