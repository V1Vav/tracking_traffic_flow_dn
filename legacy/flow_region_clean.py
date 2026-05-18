import csv
import os
import threading
import time
from collections import Counter, deque
from queue import Empty, Full, Queue

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from deep_sort_realtime.deepsort_tracker import DeepSort
from tkinter import filedialog, messagebox, ttk
from ultralytics.models import YOLO


# This class mapping assumes your custom model uses:
# 0=bicycle, 1=motorcycle, 2=car, 3=bus, 4=truck.
# If you use official COCO YOLOv8 weights, the class IDs are different.
CLASS_WEIGHTS = {
    0: 0.2,   # bicycle
    1: 0.3,   # motorcycle
    2: 1.0,   # car
    3: 2.0,   # bus
    4: 2.0,   # truck
}

CLASS_NAMES = {
    0: "bicycle",
    1: "motorcycle",
    2: "car",
    3: "bus",
    4: "truck",
}

BRANCH_ORDER = ("top", "left", "right", "bottom")
VALID_BRANCHES = set(BRANCH_ORDER)
DIRECTIONS = ("in", "out")
DISPLAY_CLASS_IDS = (2, 1)  # car, motorcycle

REGION_NAME_MAP = {
    "up": "top",
    "down": "bottom",
    "left": "left",
    "right": "right",
    "center": "center",
    "none": None,
}

DEFAULT_TEMPLATE_MAPPING = "template.csv"

# Region-counting stability parameters.
# Increase STABLE_REGION_FRAMES if vehicles jitter around polygon edges.
# Decrease it if vehicles pass through small regions too fast.
STABLE_REGION_FRAMES = 4
REGION_HISTORY_LEN = 8
EVENT_COOLDOWN_FRAMES = 10
LOST_OUT_FRAMES = 15


class RegionTemplate:
    def __init__(self, mapping_path):
        self.mapping_path = mapping_path
        self.regions = {}
        self.resolution = None
        self.loaded = False
        self._load_mapping()

    def _load_mapping(self):
        if not self.mapping_path or not os.path.exists(self.mapping_path):
            return

        try:
            with open(self.mapping_path, newline="", encoding="utf-8") as csvfile:
                rows = list(csv.reader(csvfile))

            if not rows:
                return

            if len(rows[0]) >= 2:
                self.resolution = (int(rows[0][0]), int(rows[0][1]))

            for row in rows[1:]:
                if len(row) < 9:
                    continue

                region_name = row[0].strip().lower()
                region = REGION_NAME_MAP.get(region_name, region_name)
                if region is None:
                    continue

                points = []
                for i in range(1, 9, 2):
                    points.append((int(row[i].strip()), int(row[i + 1].strip())))

                if len(points) == 4:
                    self.regions[region] = points
                    print(f"Loaded region {region}: {points}")

            self.loaded = bool(self.regions)
            print(f"RegionTemplate loaded: {self.loaded}, regions: {list(self.regions.keys())}")
        except Exception as exc:
            print(f"Error loading mapping: {exc}")
            self.regions = {}
            self.loaded = False

    def _scale_points(self, points, frame_width, frame_height):
        if not self.resolution:
            return points
        if frame_width == self.resolution[0] and frame_height == self.resolution[1]:
            return points

        scale_x = frame_width / self.resolution[0]
        scale_y = frame_height / self.resolution[1]
        return [(int(x * scale_x), int(y * scale_y)) for x, y in points]

    def get_region(self, centroid, width, height):
        if not self.loaded:
            return None

        x, y = centroid
        if x < 0 or y < 0 or x >= width or y >= height:
            return None

        for region_name, points in self.regions.items():
            scaled = self._scale_points(points, width, height)
            contour = np.array(scaled, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                return region_name

        return None

    def overlay(self, frame):
        if not self.loaded:
            return

        height, width = frame.shape[:2]
        for region_name, points in self.regions.items():
            scaled = self._scale_points(points, width, height)
            contour = np.array(scaled, dtype=np.int32)
            cv2.polylines(frame, [contour], True, (0, 255, 0), 3)

            moment = cv2.moments(contour)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                cv2.putText(
                    frame,
                    region_name.upper(),
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                )


def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_direction_region(centroid, width, height, margin_fraction, template=None):
    """Return raw region from polygon template or fallback margin-based regions."""
    if template and template.loaded:
        region = template.get_region(centroid, width, height)
        return region if region is not None else "center"

    x, y = centroid
    left_margin = int(width * margin_fraction)
    right_margin = int(width * (1.0 - margin_fraction))
    top_margin = int(height * margin_fraction)
    bottom_margin = int(height * (1.0 - margin_fraction))

    if left_margin <= x <= right_margin and top_margin <= y <= bottom_margin:
        return "center"
    if y < top_margin:
        return "top"
    if y > bottom_margin:
        return "bottom"
    if x < left_margin:
        return "left"
    if x > right_margin:
        return "right"
    return "outside"


def draw_region_overlay(frame, margin_fraction, template=None):
    """
    Draw region overlay only when explicitly called.
    If template is unavailable, draw fallback margin regions.
    """
    if template and template.loaded:
        template.overlay(frame)
        return

    height, width = frame.shape[:2]
    left_margin = int(width * margin_fraction)
    right_margin = int(width * (1.0 - margin_fraction))
    top_margin = int(height * margin_fraction)
    bottom_margin = int(height * (1.0 - margin_fraction))

    cv2.line(frame, (left_margin, 0), (left_margin, height), (0, 255, 0), 2)
    cv2.line(frame, (right_margin, 0), (right_margin, height), (0, 255, 0), 2)
    cv2.line(frame, (0, top_margin), (width, top_margin), (0, 255, 0), 2)
    cv2.line(frame, (0, bottom_margin), (width, bottom_margin), (0, 255, 0), 2)


def create_track_meta(frame_id, cls):
    return {
        "raw_history": deque(maxlen=REGION_HISTORY_LEN),
        "stable_region": None,
        "active_branch": None,
        "last_seen_frame": frame_id,
        "last_event_frame": {},
        "current_region": None,
        "weight": CLASS_WEIGHTS.get(cls, 1.0),
        "cls": cls,
    }


def update_stable_region(meta, raw_region):
    """Debounce region changes to avoid edge jitter count noise."""
    meta["raw_history"].append(raw_region)

    if len(meta["raw_history"]) < STABLE_REGION_FRAMES:
        return meta["stable_region"]

    recent = list(meta["raw_history"])[-STABLE_REGION_FRAMES:]
    if all(region == recent[0] for region in recent):
        meta["stable_region"] = recent[0]

    return meta["stable_region"]


def can_emit_event(meta, branch, direction, frame_id):
    key = (branch, direction)
    last_frame = meta["last_event_frame"].get(key, -10**9)
    if frame_id - last_frame < EVENT_COOLDOWN_FRAMES:
        return False

    meta["last_event_frame"][key] = frame_id
    return True


def add_flow_event(branch_event_windows, branch, direction, current_time, frame_id, track_id, cls):
    branch_event_windows[(branch, direction)].append({
        "time": current_time,
        "frame": frame_id,
        "track_id": track_id,
        "cls": cls,
    })


def cleanup_flow_windows(branch_event_windows, current_time, window_seconds):
    for events in branch_event_windows.values():
        while events and current_time - events[0]["time"] > window_seconds:
            events.popleft()


def calc_veh_per_min(events, window_seconds):
    if window_seconds <= 0:
        return 0.0
    return len(events) * 60.0 / window_seconds


def emit_branch_event(
    branch,
    direction,
    meta,
    frame_id,
    current_time,
    track_id,
    branch_count_total,
    branch_class_count_total,
    branch_event_windows,
):
    """Update total counters and sliding-window flow events for one branch event."""
    if not can_emit_event(meta, branch, direction, frame_id):
        return False

    cls = meta.get("cls", 2)
    branch_count_total[(branch, direction)] += 1
    branch_class_count_total[(branch, direction, cls)] += 1
    add_flow_event(branch_event_windows, branch, direction, current_time, frame_id, track_id, cls)
    return True


class FlowApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vehicle Flow Explorer")
        self.root.geometry("1280x780")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_path_var = tk.StringVar(value="")
        self.model_path_var = tk.StringVar(value="models/tuning.pt")
        self.template_mapping_path_var = tk.StringVar(value=DEFAULT_TEMPLATE_MAPPING)
        self.status_var = tk.StringVar(value="Ready")
        self.available_models = [
            "models/yolov8n.pt",
            "models/yolov8s.pt",
            "models/yolov8m.pt",
            "models/yolov8l.pt",
            "models/yolov8x.pt",
            "models/tuning.pt",
        ]

        self.region_template = None
        self.display_template_var = tk.BooleanVar(value=False)

        self.worker_state = self._default_worker_state()
        self.metrics = {
            key: tk.StringVar(value=value)
            for key, value in self.worker_state.items()
            if key != "status"
        }

        self.latest_pil_image = None
        self.latest_photo = None
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()

        self.detect_interval = 1
        self.flow_window = 60
        self.region_margin = 0.25

        self._build_ui()
        self.load_region_template()
        self.root.after(50, self._update_ui)

    def _default_worker_state(self):
        state = {
            "status": "Ready",
            "frame": "0",
            "fps": "0.0",
            "active_tracks": "0",
            "current_pce": "0.0",
            "flow_veh_pm": "0.0",
        }

        for branch in BRANCH_ORDER:
            state.update({
                f"{branch}_pce": "0.0",
                f"{branch}_in_flow": "0.0",
                f"{branch}_out_flow": "0.0",
                f"{branch}_in_count": "0",
                f"{branch}_out_count": "0",
            })
            for cls_id in DISPLAY_CLASS_IDS:
                class_name = CLASS_NAMES[cls_id]
                state[f"{branch}_{class_name}_in"] = "0"
                state[f"{branch}_{class_name}_out"] = "0"

        return state

    def _build_ui(self):
        left_frame = ttk.Frame(self.root)
        right_frame = ttk.Frame(self.root, width=340)

        left_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        right_frame.grid(row=0, column=1, sticky="ns", padx=6, pady=6)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_rowconfigure(0, weight=1)

        self.video_label = ttk.Label(left_frame, anchor="center")
        self.video_label.pack(fill="both", expand=True)

        control_frame = ttk.LabelFrame(right_frame, text="Controls")
        control_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(control_frame, text="Video:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(control_frame, textvariable=self.video_path_var, width=36).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_video).grid(row=1, column=1, sticky="ew")

        ttk.Label(control_frame, text="Model:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(
            control_frame,
            textvariable=self.model_path_var,
            values=self.available_models,
            width=34,
            state="normal",
        ).grid(row=3, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_model).grid(row=3, column=1, sticky="ew")

        ttk.Label(control_frame, text="Template CSV:").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(control_frame, textvariable=self.template_mapping_path_var, width=36).grid(row=5, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_template_mapping).grid(row=5, column=1, sticky="ew")

        ttk.Checkbutton(
            control_frame,
            text="Display Region Template",
            variable=self.display_template_var,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=4)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_processing)
        self.start_button.grid(row=7, column=0, columnspan=2, pady=8, sticky="ew")
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=8, column=0, columnspan=2, sticky="ew")

        for child in control_frame.winfo_children():
            child.grid_configure(padx=2, pady=2)

        metrics_frame = ttk.LabelFrame(right_frame, text="Metrics")
        metrics_frame.pack(fill="x", pady=(0, 8))

        for row, (label_text, var_name) in enumerate([
            ("Frame:", "frame"),
            ("FPS:", "fps"),
            ("Active tracks:", "active_tracks"),
            ("Current PCE:", "current_pce"),
            ("Flow (veh/min):", "flow_veh_pm"),
        ]):
            ttk.Label(metrics_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Label(metrics_frame, textvariable=self.metrics[var_name], width=28).grid(row=row, column=1, sticky="w")

        branch_frame = ttk.LabelFrame(metrics_frame, text="Branch Current PCE + Flow + Count")
        branch_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0), padx=2)
        headers = ["Direction", "PCE now", "In veh/min", "Out veh/min", "In total", "Out total"]
        for col, header in enumerate(headers):
            branch_frame.grid_columnconfigure(col, weight=1)
            ttk.Label(branch_frame, text=header, anchor="center").grid(row=0, column=col, sticky="ew", padx=2)

        for row, branch in enumerate(BRANCH_ORDER, start=1):
            ttk.Label(branch_frame, text=branch.title()).grid(row=row, column=0, sticky="w", padx=4)
            ttk.Label(branch_frame, textvariable=self.metrics[f"{branch}_pce"], anchor="center").grid(row=row, column=1, sticky="ew", padx=2)
            ttk.Label(branch_frame, textvariable=self.metrics[f"{branch}_in_flow"], anchor="center").grid(row=row, column=2, sticky="ew", padx=2)
            ttk.Label(branch_frame, textvariable=self.metrics[f"{branch}_out_flow"], anchor="center").grid(row=row, column=3, sticky="ew", padx=2)
            ttk.Label(branch_frame, textvariable=self.metrics[f"{branch}_in_count"], anchor="center").grid(row=row, column=4, sticky="ew", padx=2)
            ttk.Label(branch_frame, textvariable=self.metrics[f"{branch}_out_count"], anchor="center").grid(row=row, column=5, sticky="ew", padx=2)

        vehicle_frame = ttk.LabelFrame(metrics_frame, text="Vehicle Type Count")
        vehicle_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0), padx=2)
        vehicle_headers = ["Direction", "Car In", "Car Out", "Moto In", "Moto Out"]
        for col, header in enumerate(vehicle_headers):
            vehicle_frame.grid_columnconfigure(col, weight=1)
            ttk.Label(vehicle_frame, text=header, anchor="center").grid(row=0, column=col, sticky="ew", padx=2)

        for row, branch in enumerate(BRANCH_ORDER, start=1):
            ttk.Label(vehicle_frame, text=branch.title()).grid(row=row, column=0, sticky="w", padx=4)
            ttk.Label(vehicle_frame, textvariable=self.metrics[f"{branch}_car_in"], anchor="center").grid(row=row, column=1, sticky="ew", padx=2)
            ttk.Label(vehicle_frame, textvariable=self.metrics[f"{branch}_car_out"], anchor="center").grid(row=row, column=2, sticky="ew", padx=2)
            ttk.Label(vehicle_frame, textvariable=self.metrics[f"{branch}_motorcycle_in"], anchor="center").grid(row=row, column=3, sticky="ew", padx=2)
            ttk.Label(vehicle_frame, textvariable=self.metrics[f"{branch}_motorcycle_out"], anchor="center").grid(row=row, column=4, sticky="ew", padx=2)

        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.status_var, width=36).grid(row=0, column=1, sticky="w")

    def browse_video(self):
        video_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*")],
        )
        if video_path:
            self.video_path_var.set(video_path)

    def browse_model(self):
        model_path = filedialog.askopenfilename(
            title="Select YOLO model file",
            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*")],
        )
        if model_path:
            self.model_path_var.set(model_path)

    def browse_template_mapping(self):
        mapping_path = filedialog.askopenfilename(
            title="Select template mapping CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if mapping_path:
            self.template_mapping_path_var.set(mapping_path)
            self.load_region_template()

    def load_region_template(self):
        mapping_path = self.template_mapping_path_var.get().strip()
        if mapping_path and os.path.exists(mapping_path):
            self.region_template = RegionTemplate(mapping_path)
            if self.region_template.loaded:
                self.status_var.set("Template loaded")
                return

        self.region_template = None
        self.status_var.set("Template unavailable; using margin regions")

    def start_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            return

        video_path = self.video_path_var.get().strip()
        model_path = self.model_path_var.get().strip()

        if not video_path:
            messagebox.showwarning("Missing video", "Please choose a video file first.")
            return
        if not model_path:
            messagebox.showwarning("Missing model", "Please choose a YOLO model file first.")
            return

        self.load_region_template()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        with self.state_lock:
            self.worker_state = self._default_worker_state()
            self.worker_state["status"] = "Loading model..."

        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._process_video,
            args=(video_path, model_path),
            daemon=True,
        )
        self.processing_thread.start()

    def stop_processing(self):
        self.stop_event.set()
        self.stop_button.config(state="disabled")
        self.status_var.set("Stopping...")

    def _process_video(self, video_path, model_path):
        reader_thread = None
        try:
            model = YOLO(model_path)
            try:
                model.to("cuda")
            except Exception:
                model.to("cpu")

            tracker = DeepSort(
                max_age=60,
                n_init=4,
                max_cosine_distance=0.2,
                nn_budget=100,
            )

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Không thể mở video")

            fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_duration = 1.0 / fps_input
            buffer_size = max(8, int(fps_input * 10))
            frame_queue = Queue(maxsize=buffer_size)
            reader_done = threading.Event()

            def video_reader():
                try:
                    while cap.isOpened() and not self.stop_event.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        while not self.stop_event.is_set():
                            try:
                                frame_queue.put(frame, timeout=0.1)
                                break
                            except Full:
                                continue
                finally:
                    reader_done.set()
                    cap.release()

            with self.state_lock:
                self.worker_state["status"] = "Buffering video..."

            reader_thread = threading.Thread(target=video_reader, daemon=True)
            reader_thread.start()

            while (
                frame_queue.qsize() < min(buffer_size, 4)
                and not reader_done.is_set()
                and not self.stop_event.is_set()
            ):
                time.sleep(0.05)

            with self.state_lock:
                self.worker_state["status"] = f"Buffer ready @ {fps_input:.1f} FPS"

            if self.region_template and self.region_template.loaded:
                valid_branches = set(self.region_template.regions.keys()) & VALID_BRANCHES
            else:
                valid_branches = set(VALID_BRANCHES)

            branch_count_total = Counter()
            branch_class_count_total = Counter()
            branch_event_windows = {
                (branch, direction): deque()
                for branch in valid_branches
                for direction in DIRECTIONS
            }

            track_meta = {}
            prev_time = time.time()
            playback_start = time.time()
            frame_id = 0

            while not self.stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=0.1)
                except Empty:
                    if reader_done.is_set() and frame_queue.empty():
                        break
                    continue

                frame_id += 1
                current_time = frame_id / fps_input  # video-time, not wall-clock-time
                display_frame = frame.copy()

                detections = []
                if frame_id % self.detect_interval == 0:
                    results = model(
                        frame,
                        imgsz=1280,
                        conf=0.25,
                        iou=0.45,
                        classes=list(CLASS_WEIGHTS.keys()),
                        verbose=False,
                    )

                    for result in results:
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls not in CLASS_WEIGHTS:
                                continue

                            # Motorcycles and bicycles can be small/dense in Vietnam traffic footage.
                            if cls == 2 and conf < 0.30:
                                continue
                            if cls != 2 and conf < 0.15:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1
                            if w <= 0 or h <= 0:
                                continue

                            detections.append(([x1, y1, w, h], conf, cls))

                tracks = tracker.update_tracks(detections, frame=frame)
                active_tracks = 0
                active_track_ids = set()

                print(f"[DeepSort] Frame={frame_id} detections={len(detections)} total_tracks={len(tracks)}")

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 5:
                        continue
                    active_track_ids.add(track.track_id)

                # Tracks missing for only a few frames are kept to avoid false OUT events.
                tracks_to_remove = []
                for track_id, meta in list(track_meta.items()):
                    if track_id in active_track_ids:
                        continue

                    missing_frames = frame_id - meta.get("last_seen_frame", frame_id)
                    if missing_frames < LOST_OUT_FRAMES:
                        continue

                    active_branch = meta.get("active_branch")
                    if active_branch in valid_branches:
                        emitted = emit_branch_event(
                            active_branch,
                            "out",
                            meta,
                            frame_id,
                            current_time,
                            track_id,
                            branch_count_total,
                            branch_class_count_total,
                            branch_event_windows,
                        )
                        if emitted:
                            print(f"[Flow] OUT confirmed lost: branch={active_branch} track_id={track_id}")

                    tracks_to_remove.append(track_id)

                for track_id in tracks_to_remove:
                    del track_meta[track_id]

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 5:
                        print(
                            f"[DeepSort]   Skipping track {track.track_id} "
                            f"confirmed={track.is_confirmed()} time_since_update={track.time_since_update}"
                        )
                        continue

                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    det_cls = track.det_class
                    if det_cls is None:
                        det_cls = track_meta.get(track_id, {}).get("cls", 2)

                    cls = int(det_cls)
                    if cls not in CLASS_WEIGHTS:
                        continue

                    active_tracks += 1
                    centroid = centroid_from_box((l, t, r, b))
                    raw_region = get_direction_region(
                        centroid,
                        frame.shape[1],
                        frame.shape[0],
                        self.region_margin,
                        self.region_template,
                    )

                    meta = track_meta.setdefault(track_id, create_track_meta(frame_id, cls))
                    meta["last_seen_frame"] = frame_id
                    meta["cls"] = cls
                    meta["weight"] = CLASS_WEIGHTS.get(cls, 1.0)

                    stable_region = update_stable_region(meta, raw_region)
                    meta["current_region"] = stable_region

                    if stable_region is not None:
                        current_is_branch = stable_region in valid_branches
                        active_branch = meta.get("active_branch")

                        if active_branch in valid_branches and stable_region != active_branch:
                            emitted = emit_branch_event(
                                active_branch,
                                "out",
                                meta,
                                frame_id,
                                current_time,
                                track_id,
                                branch_count_total,
                                branch_class_count_total,
                                branch_event_windows,
                            )
                            if emitted:
                                print(
                                    f"[Flow] OUT stable: branch={active_branch} -> {stable_region} "
                                    f"cls={cls} track_id={track_id}"
                                )
                            meta["active_branch"] = None

                        if current_is_branch and meta.get("active_branch") != stable_region:
                            emitted = emit_branch_event(
                                stable_region,
                                "in",
                                meta,
                                frame_id,
                                current_time,
                                track_id,
                                branch_count_total,
                                branch_class_count_total,
                                branch_event_windows,
                            )
                            if emitted:
                                print(f"[Flow] IN stable: branch={stable_region} cls={cls} track_id={track_id}")
                            meta["active_branch"] = stable_region

                    color = {
                        0: (0, 255, 0),      # bicycle
                        1: (255, 0, 0),      # motorcycle
                        2: (0, 0, 255),      # car
                        3: (0, 255, 255),    # bus
                        4: (0, 165, 255),    # truck
                    }.get(cls, (255, 255, 255))

                    label = f"{CLASS_NAMES.get(cls, cls)} #{track_id}"
                    region_label = stable_region if stable_region is not None else raw_region
                    if region_label:
                        label += f" {region_label}"

                    cv2.rectangle(display_frame, (l, t), (r, b), color, 2)
                    cv2.putText(display_frame, label, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(display_frame, centroid, 3, color, -1)

                    print(
                        f"[DeepSort]   Track={track_id} cls={cls} label={CLASS_NAMES.get(cls, cls)} "
                        f"ltrb=({l},{t},{r},{b}) centroid={centroid} raw={raw_region} "
                        f"stable={stable_region} active_branch={meta.get('active_branch')} "
                        f"confirmed={track.is_confirmed()} time_since_update={track.time_since_update}"
                    )

                if self.display_template_var.get():
                    draw_region_overlay(display_frame, self.region_margin, self.region_template)

                cleanup_flow_windows(branch_event_windows, current_time, self.flow_window)

                branch_current_pce = {branch: 0.0 for branch in valid_branches}
                for meta in track_meta.values():
                    if meta.get("last_seen_frame") != frame_id:
                        continue
                    region = meta.get("current_region")
                    if region in valid_branches:
                        branch_current_pce[region] += meta.get("weight", 0.0)

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
                prev_time = curr_time

                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((880, 620), Image.LANCZOS)

                total_current_pce = 0.0
                total_in_flow = 0.0
                metric_updates = {}

                for branch in BRANCH_ORDER:
                    if branch not in valid_branches:
                        metric_updates[f"{branch}_pce"] = "0.0"
                        metric_updates[f"{branch}_in_flow"] = "0.0"
                        metric_updates[f"{branch}_out_flow"] = "0.0"
                        metric_updates[f"{branch}_in_count"] = "0"
                        metric_updates[f"{branch}_out_count"] = "0"
                        for cls_id in DISPLAY_CLASS_IDS:
                            class_name = CLASS_NAMES[cls_id]
                            metric_updates[f"{branch}_{class_name}_in"] = "0"
                            metric_updates[f"{branch}_{class_name}_out"] = "0"
                        continue

                    pce_now = branch_current_pce.get(branch, 0.0)
                    in_flow = calc_veh_per_min(branch_event_windows[(branch, "in")], self.flow_window)
                    out_flow = calc_veh_per_min(branch_event_windows[(branch, "out")], self.flow_window)

                    metric_updates[f"{branch}_pce"] = f"{pce_now:.1f}"
                    metric_updates[f"{branch}_in_flow"] = f"{in_flow:.1f}"
                    metric_updates[f"{branch}_out_flow"] = f"{out_flow:.1f}"
                    metric_updates[f"{branch}_in_count"] = str(branch_count_total[(branch, "in")])
                    metric_updates[f"{branch}_out_count"] = str(branch_count_total[(branch, "out")])

                    for cls_id in DISPLAY_CLASS_IDS:
                        class_name = CLASS_NAMES[cls_id]
                        metric_updates[f"{branch}_{class_name}_in"] = str(branch_class_count_total[(branch, "in", cls_id)])
                        metric_updates[f"{branch}_{class_name}_out"] = str(branch_class_count_total[(branch, "out", cls_id)])

                    total_current_pce += pce_now
                    total_in_flow += in_flow

                with self.state_lock:
                    self.latest_pil_image = pil_image
                    self.worker_state["status"] = "Running"
                    self.worker_state["frame"] = str(frame_id)
                    self.worker_state["fps"] = f"{fps:.1f}"
                    self.worker_state["active_tracks"] = str(active_tracks)
                    self.worker_state["current_pce"] = f"{total_current_pce:.1f}"
                    self.worker_state["flow_veh_pm"] = f"{total_in_flow:.1f}"
                    self.worker_state.update(metric_updates)

                expected_display = playback_start + (frame_id - 1) * frame_duration
                delay = expected_display - time.time()
                if delay > 0:
                    time.sleep(delay)

            with self.state_lock:
                self.worker_state["status"] = "Stopped" if self.stop_event.is_set() else "Finished"

        except Exception as exc:
            with self.state_lock:
                self.worker_state["status"] = f"Error: {exc}"
        finally:
            if reader_thread is not None:
                try:
                    reader_thread.join(timeout=1.0)
                except Exception:
                    pass

    def _update_ui(self):
        with self.state_lock:
            if self.latest_pil_image is not None:
                self.latest_photo = ImageTk.PhotoImage(self.latest_pil_image)
                self.video_label.configure(image=self.latest_photo)
                self.video_label.image = self.latest_photo
                self.latest_pil_image = None

            for key, var in self.metrics.items():
                var.set(self.worker_state.get(key, var.get()))
            self.status_var.set(self.worker_state.get("status", self.status_var.get()))

        if not self.processing_thread or not self.processing_thread.is_alive():
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")

        self.root.after(50, self._update_ui)

    def on_close(self):
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = FlowApp()
    app.run()
