import cv2
import time
import argparse
import csv
import numpy as np
from ultralytics.models import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ===== Argument =====
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--model", type=str, default="yolov8s.pt")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--csv", type=str, default=None)
parser.add_argument("--imgsz", type=int, default=1280,
                    help="YOLO inference image size for better distant detection")
parser.add_argument("--conf", type=float, default=0.15,
                    help="YOLO confidence threshold")
parser.add_argument("--iou", type=float, default=0.45,
                    help="YOLO IOU threshold")
parser.add_argument("--detect-interval", type=int, default=1,
                    help="Detect every N frames")
parser.add_argument("--min-conf", type=float, default=0.18,
                    help="Minimum confidence for car/bus/truck")
args = parser.parse_args()

# ===== Load model =====
model = YOLO(args.model)
model.to("cuda")

# ===== Tracker =====
tracker = DeepSort(
    max_age=60,
    n_init=5,
    max_cosine_distance=0.2,
    nn_budget=100
)

# ===== Video =====

track_smoothing_alpha = 0.4
smoothed_boxes = {}

cap = cv2.VideoCapture(0 if args.video == "0" else args.video)

fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_interval = 1.0 / fps_input

out = None

if args.output is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")

    h, w = frame.shape[:2]
    out = cv2.VideoWriter(args.output, fourcc, fps_input, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ===== CSV Writer =====
csv_file = None
csv_writer = None

if args.csv is not None:
    csv_file = open(args.csv, mode='w', newline='')
    csv_writer = csv.writer(csv_file)

    # header
    csv_writer.writerow([
        "time", "frame", "track_id", "class",
        "x1", "y1", "x2", "y2"
    ])

prev_time = 0
frame_id = 0
detect_interval = args.detect_interval
detections = []

names = model.names  # class names

# ===== Loop =====
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    loop_start = time.time()
    detections = []

    # ===== YOLO detect =====
    if frame_id % detect_interval == 0:
        results = model(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=[2, 3, 5, 7],
            verbose=False
        )

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # ===== Filter vehicle =====
                if cls not in [2, 3, 5, 7]:
                    continue

                # ===== Detail config/thresold =====
                if cls == 3:   # motorbike
                    if conf < 0.1:
                        continue
                else:
                    if conf < args.min_conf:
                        continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                detections.append(([x1, y1, w, h], conf, cls))

    # ===== Tracking =====
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        if track.time_since_update > 5:
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cls = track.det_class

        # ===== Smooth track box =====
        curr_box = np.array([l, t, r, b], dtype=float)
        if track_id in smoothed_boxes:
            prev_box = smoothed_boxes[track_id]
            smoothed_boxes[track_id] = (
                track_smoothing_alpha * curr_box +
                (1 - track_smoothing_alpha) * prev_box
            )
        else:
            smoothed_boxes[track_id] = curr_box

        l, t, r, b = smoothed_boxes[track_id]

        # ===== Add a small padding for more visible boxes =====
        pad = int(max(8, 0.05 * max(r - l, b - t)))
        l = int(l - pad)
        t = int(t - pad)
        r = int(r + pad)
        b = int(b + pad)

        # ===== Clamp =====
        l = max(0, l)
        t = max(0, t)
        r = min(frame.shape[1], r)
        b = min(frame.shape[0], b)
        
        if csv_writer is not None:
            csv_writer.writerow([
                time.time(), frame_id, track_id, cls,
                l, t, r, b
            ])

        # ===== Color =====
        color = {
            2: (0, 255, 0),   # car
            3: (255, 0, 0),   # motorbike
            5: (0, 0, 255),   # bus
            7: (0, 255, 255)  # truck
        }.get(cls, (255, 255, 255))

        label = f"{names[cls]} ID {track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_origin = (l, t - 10)
        text_bg_tl = (l, t - 10 - text_size[1] - 4)
        text_bg_br = (l + text_size[0] + 6, t - 10 + 4)

        cv2.rectangle(frame, (l, t), (r, b), color, 3)
        cv2.rectangle(frame, text_bg_tl, text_bg_br, color, -1)
        cv2.putText(frame, label, (l + 3, t - 12), font, font_scale, (0, 0, 0), thickness)

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

    # ===== Show =====
    if out is not None:
        out.write(frame)
    cv2.imshow("YOLOv8 + DeepSORT (Traffic)", frame)

    if fps_input > 0:
        loop_elapsed = (time.time() - loop_start) * 1000.0
        delay_ms = int(round(max(1.0, 1000.0 / fps_input - loop_elapsed)))
    else:
        delay_ms = 1

    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        break
    
if out is not None:
    out.release()

if csv_file is not None:
    csv_file.close()

cap.release()
cv2.destroyAllWindows()