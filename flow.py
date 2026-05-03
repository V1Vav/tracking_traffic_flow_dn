import threading
import time
import os
import csv
import numpy as np
from collections import Counter, deque
from queue import Empty, Queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.models import YOLO


CLASS_WEIGHTS = {
    0: 0.2,   # bicycle
    1: 0.3,   # motorcycle
    2: 1.0,   # car
    3: 2.0,   # bus
    4: 2.0    # truck
}

CLASS_NAMES = {
    0: "bicycle",
    1: "motorcycle",
    2: "car",
    3: "bus",
    4: "truck"
}

VALID_BRANCHES = {"top", "bottom", "left", "right"} 

COLOR_NAME_TO_BGR = {
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "orange": (0, 165, 255),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}
REGION_NAME_MAP = {
    "up": "top",
    "down": "bottom",
    "left": "left",
    "right": "right",
    "center": "center",
    "none": None,
}
DEFAULT_TEMPLATE_IMAGE = "template.png"
DEFAULT_TEMPLATE_MAPPING = "template.csv"


class RegionTemplate:
    def __init__(self, image_path, mapping_path):
        self.image_path = image_path
        self.mapping_path = mapping_path
        self.template = None
        self.color_to_region = {}
        self.loaded = False
        self.resize_cache = None
        self.resized_template = None
        self.label_positions = {}
        self._load_mapping()
        self._load_image()

    def _load_mapping(self):
        if not self.mapping_path or not os.path.exists(self.mapping_path):
            return
        try:
            with open(self.mapping_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) < 2:
                        continue
                    color_name = row[0].strip().lower()
                    region_name = row[1].strip().lower()
                    region = REGION_NAME_MAP.get(region_name)
                    color_bgr = COLOR_NAME_TO_BGR.get(color_name)
                    if region and color_bgr is not None:
                        self.color_to_region[color_bgr] = region
        except Exception:
            self.color_to_region = {}

    def _load_image(self):
        if not self.image_path or not os.path.exists(self.image_path):
            return
        template = cv2.imread(self.image_path)
        if template is None:
            return
        self.template = template
        self.loaded = bool(self.template is not None and self.color_to_region)

    def _ensure_resized(self, width, height):
        if self.resize_cache == (width, height):
            return
        self.resize_cache = (width, height)
        self.resized_template = cv2.resize(
            self.template, (width, height), interpolation=cv2.INTER_NEAREST
        )
        self.label_positions = {}
        for color, region in self.color_to_region.items():
            lower = np.array(color, dtype=np.uint8)
            upper = np.array(color, dtype=np.uint8)
            mask = cv2.inRange(self.resized_template, lower, upper)
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                self.label_positions[region] = (cx, cy)

    def get_region(self, centroid, width, height):
        if not self.loaded:
            return None
        x, y = centroid
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        self._ensure_resized(width, height)
        pixel = tuple(int(v) for v in self.resized_template[y, x])
        region = self.color_to_region.get(pixel)
        if region:
            return region
        best_region = None
        best_dist = float('inf')
        px = np.array(pixel, dtype=np.int32)
        for color, region_name in self.color_to_region.items():
            dist = np.linalg.norm(px - np.array(color, dtype=np.int32))
            if dist < best_dist:
                best_dist = dist
                best_region = region_name
        return best_region if best_dist < 30 else None

    def overlay(self, frame, alpha=0.15):
        if not self.loaded:
            return
        height, width = frame.shape[:2]
        self._ensure_resized(width, height)
        overlay = self.resized_template.copy()
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
        for region, pos in self.label_positions.items():
            cv2.putText(frame, region.upper(), (pos[0] - 40, pos[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)


def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_direction_region(centroid, width, height, margin_fraction, template=None):
    if template and template.loaded:
        region = template.get_region(centroid, width, height)
        if region:
            return region
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
    if template and template.loaded:
        template.overlay(frame, alpha=0.20)
        return
    height, width = frame.shape[:2]
    left_margin = int(width * margin_fraction)
    right_margin = int(width * (1.0 - margin_fraction))
    top_margin = int(height * margin_fraction)
    bottom_margin = int(height * (1.0 - margin_fraction))

    color = (220, 220, 220)
    thickness = 1
    cv2.line(frame, (left_margin, 0), (left_margin, height), color, thickness)
    cv2.line(frame, (right_margin, 0), (right_margin, height), color, thickness)
    cv2.line(frame, (0, top_margin), (width, top_margin), color, thickness)
    cv2.line(frame, (0, bottom_margin), (width, bottom_margin), color, thickness)

    text_color = (240, 240, 240)
    text_scale = 0.6
    text_thickness = 2

    cv2.putText(frame, "TOP", (left_margin + 10, top_margin // 2), cv2.FONT_HERSHEY_SIMPLEX,
                text_scale, text_color, text_thickness)
    cv2.putText(frame, "LEFT", (10, bottom_margin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                text_scale, text_color, text_thickness)
    cv2.putText(frame, "CENTER", (left_margin + 10, top_margin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                text_scale, text_color, text_thickness)
    cv2.putText(frame, "RIGHT", (right_margin + 10, bottom_margin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                text_scale, text_color, text_thickness)
    cv2.putText(frame, "BOTTOM", (left_margin + 10, bottom_margin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                text_scale, text_color, text_thickness)


def format_flow_metrics(event_window, window_seconds, active_track_ids):
    filtered_events = [event for event in event_window if event["track_id"] in active_track_ids]
    count = len(filtered_events)
    weighted_sum = sum(event["weight"] for event in filtered_events)
    if window_seconds <= 0:
        return 0.0, 0.0
    flow_pce_pm = weighted_sum * 60.0 / window_seconds
    flow_veh_pm = count * 60.0 / window_seconds
    return flow_pce_pm, flow_veh_pm


def cleanup_old_events(event_window, current_time, window_seconds):
    while event_window and current_time - event_window[0]["time"] > window_seconds:
        event_window.popleft()


class FlowApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vehicle Flow Explorer")
        self.root.geometry("1280x780")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_path_var = tk.StringVar(value="")
        self.model_path_var = tk.StringVar(value="models/tuning.pt")
        self.template_image_path_var = tk.StringVar(value=DEFAULT_TEMPLATE_IMAGE)
        self.template_mapping_path_var = tk.StringVar(value=DEFAULT_TEMPLATE_MAPPING)
        self.status_var = tk.StringVar(value="Ready")
        self.available_models = ["models/yolov8n.pt", "models/yolov8s.pt", "models/yolov8m.pt", "models/yolov8l.pt", "models/yolov8x.pt", "models/tuning.pt"]

        self.region_template = None
        self.display_template_var = tk.BooleanVar(value=False)

        self.metrics = {
            "frame": tk.StringVar(value="0"),
            "fps": tk.StringVar(value="0.0"),
            "active_tracks": tk.StringVar(value="0"),
            "flow_pce_pm": tk.StringVar(value="0.0"),
            "flow_veh_pm": tk.StringVar(value="0.0"),
            "top_in": tk.StringVar(value="0.0"),
            "top_out": tk.StringVar(value="0.0"),
            "top_in_count": tk.StringVar(value="0"),
            "top_out_count": tk.StringVar(value="0"),
            "left_in": tk.StringVar(value="0.0"),
            "left_out": tk.StringVar(value="0.0"),
            "left_in_count": tk.StringVar(value="0"),
            "left_out_count": tk.StringVar(value="0"),
            "right_in": tk.StringVar(value="0.0"),
            "right_out": tk.StringVar(value="0.0"),
            "right_in_count": tk.StringVar(value="0"),
            "right_out_count": tk.StringVar(value="0"),
            "bottom_in": tk.StringVar(value="0.0"),
            "bottom_out": tk.StringVar(value="0.0"),
            "bottom_in_count": tk.StringVar(value="0"),
            "bottom_out_count": tk.StringVar(value="0"),
        }

        self.worker_state = {
            "status": "Ready",
            "frame": "0",
            "fps": "0.0",
            "active_tracks": "0",
            "flow_pce_pm": "0.0",
            "flow_veh_pm": "0.0",
            "top_in": "0.0",
            "top_out": "0.0",
            "top_in_count": "0",
            "top_out_count": "0",
            "left_in": "0.0",
            "left_out": "0.0",
            "left_in_count": "0",
            "left_out_count": "0",
            "right_in": "0.0",
            "right_out": "0.0",
            "right_in_count": "0",
            "right_out_count": "0",
            "bottom_in": "0.0",
            "bottom_out": "0.0",
            "bottom_in_count": "0",
            "bottom_out_count": "0",
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

    def _build_ui(self):
        left_frame = ttk.Frame(self.root)
        right_frame = ttk.Frame(self.root, width=320)

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
        video_entry = ttk.Entry(control_frame, textvariable=self.video_path_var, width=36)
        video_entry.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_video).grid(row=1, column=1, sticky="ew")

        ttk.Label(control_frame, text="Model:").grid(row=2, column=0, sticky="w", pady=4)
        model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_path_var,
            values=self.available_models,
            width=34,
            state="normal"
        )
        model_combo.grid(row=3, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_model).grid(row=3, column=1, sticky="ew")

        ttk.Label(control_frame, text="Template image:").grid(row=4, column=0, sticky="w", pady=4)
        template_entry = ttk.Entry(control_frame, textvariable=self.template_image_path_var, width=36)
        template_entry.grid(row=5, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_template).grid(row=5, column=1, sticky="ew")

        ttk.Label(control_frame, text="Template CSV:").grid(row=6, column=0, sticky="w", pady=4)
        mapping_entry = ttk.Entry(control_frame, textvariable=self.template_mapping_path_var, width=36)
        mapping_entry.grid(row=7, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_template_mapping).grid(row=7, column=1, sticky="ew")

        ttk.Checkbutton(control_frame, text="Display Region Template",
                            variable=self.display_template_var).grid(row=8, column=0, columnspan=2, sticky="ew", pady=4)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_processing)
        self.start_button.grid(row=9, column=0, columnspan=2, pady=8, sticky="ew")
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=10, column=0, columnspan=2, sticky="ew")

        for child in control_frame.winfo_children():
            child.grid_configure(padx=2, pady=2)

        metrics_frame = ttk.LabelFrame(right_frame, text="Metrics")
        metrics_frame.pack(fill="x", pady=(0, 8))

        row = 0
        for label_text, var_name in [
            ("Frame:", "frame"),
            ("FPS:", "fps"),
            ("Active tracks:", "active_tracks"),
            ("Flow (PCE/min):", "flow_pce_pm"),
            ("Flow (veh/min):", "flow_veh_pm"),
        ]:
            ttk.Label(metrics_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Label(metrics_frame, textvariable=self.metrics[var_name], width=28).grid(row=row, column=1, sticky="w")
            row += 1

        branch_frame = ttk.LabelFrame(metrics_frame, text="Branch PCE + Count")
        branch_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0), padx=2)
        branch_frame.grid_columnconfigure(1, weight=1)
        branch_frame.grid_columnconfigure(2, weight=1)
        branch_frame.grid_columnconfigure(3, weight=1)
        branch_frame.grid_columnconfigure(4, weight=1)

        ttk.Label(branch_frame, text="Direction").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(branch_frame, text="In PCE", anchor="center").grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(branch_frame, text="Out PCE", anchor="center").grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Label(branch_frame, text="In cnt", anchor="center").grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Label(branch_frame, text="Out cnt", anchor="center").grid(row=0, column=4, sticky="ew", padx=4)

        ttk.Label(branch_frame, text="Top").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["top_in"], anchor="center").grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["top_out"], anchor="center").grid(row=1, column=2, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["top_in_count"], anchor="center").grid(row=1, column=3, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["top_out_count"], anchor="center").grid(row=1, column=4, sticky="ew", padx=4)

        ttk.Label(branch_frame, text="Left").grid(row=2, column=0, sticky="w", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["left_in"], anchor="center").grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["left_out"], anchor="center").grid(row=2, column=2, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["left_in_count"], anchor="center").grid(row=2, column=3, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["left_out_count"], anchor="center").grid(row=2, column=4, sticky="ew", padx=4)

        ttk.Label(branch_frame, text="Right").grid(row=3, column=0, sticky="w", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["right_in"], anchor="center").grid(row=3, column=1, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["right_out"], anchor="center").grid(row=3, column=2, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["right_in_count"], anchor="center").grid(row=3, column=3, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["right_out_count"], anchor="center").grid(row=3, column=4, sticky="ew", padx=4)

        ttk.Label(branch_frame, text="Bottom").grid(row=4, column=0, sticky="w", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["bottom_in"], anchor="center").grid(row=4, column=1, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["bottom_out"], anchor="center").grid(row=4, column=2, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["bottom_in_count"], anchor="center").grid(row=4, column=3, sticky="ew", padx=4)
        ttk.Label(branch_frame, textvariable=self.metrics["bottom_out_count"], anchor="center").grid(row=4, column=4, sticky="ew", padx=4)

        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.status_var, width=36).grid(row=0, column=1, sticky="w")

    def browse_video(self):
        video_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*")]
        )
        if video_path:
            self.video_path_var.set(video_path)

    def browse_model(self):
        model_path = filedialog.askopenfilename(
            title="Select YOLO model file",
            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*")]
        )
        if model_path:
            self.model_path_var.set(model_path)

    def browse_template(self):
        template_path = filedialog.askopenfilename(
            title="Select template image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*")]
        )
        if template_path:
            self.template_image_path_var.set(template_path)
            self.load_region_template()

    def browse_template_mapping(self):
        mapping_path = filedialog.askopenfilename(
            title="Select template mapping CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")]
        )
        if mapping_path:
            self.template_mapping_path_var.set(mapping_path)
            self.load_region_template()

    def load_region_template(self):
        image_path = self.template_image_path_var.get().strip()
        mapping_path = self.template_mapping_path_var.get().strip()
        if image_path and mapping_path and os.path.exists(image_path) and os.path.exists(mapping_path):
            self.region_template = RegionTemplate(image_path, mapping_path)
            if self.region_template.loaded:
                self.status_var.set("Template loaded")
                return
        self.region_template = None
        self.status_var.set("Template unavailable")

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
                        frame_queue.put(frame)
                finally:
                    reader_done.set()
                    cap.release()

            with self.state_lock:
                self.worker_state["status"] = "Buffering video..."

            reader_thread = threading.Thread(target=video_reader, daemon=True)
            reader_thread.start()

            while frame_queue.qsize() < min(buffer_size, 4) and not reader_done.is_set() and not self.stop_event.is_set():
                time.sleep(0.05)

            with self.state_lock:
                self.worker_state["status"] = f"Buffer ready @ {fps_input:.1f} FPS"

            flow_events = deque()
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
                current_time = frame_id / fps_input
                if self.display_template_var.get():
                    draw_region_overlay(frame, self.region_margin, self.region_template)
                new_events = []
                detections = []

                if frame_id % self.detect_interval == 0:
                    results = model(
                        frame,
                        imgsz=1280,
                        conf=0.25,
                        iou=0.45,
                        classes=[0, 1, 2, 3],
                        verbose=False,
                    )
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls not in CLASS_WEIGHTS:
                                continue
                            if cls == 2 and conf < 0.3:
                                continue
                            if cls != 2 and conf < 0.15:
                                continue
                            # 0: 0.2,   # bicycle
                            # 1: 0.3,   # motorcycle
                            # 2: 1.0,   # car
                            # 3: 2.0,   # bus
                            # 4: 2.0    # truck
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1
                            detections.append(([x1, y1, w, h], conf, cls))

                tracks = tracker.update_tracks(detections, frame=frame)
                active_tracks = 0
                active_track_ids = set()

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 5:
                        continue

                    active_tracks += 1
                    track_id = track.track_id
                    active_track_ids.add(track_id)
                    l, t, r, b = map(int, track.to_ltrb())
                    cls = track.det_class
                    centroid = centroid_from_box((l, t, r, b))
                    current_region = get_direction_region(
                        centroid, frame.shape[1], frame.shape[0], self.region_margin, self.region_template
                    )

                    meta = track_meta.setdefault(track_id, {
                        "last_region": current_region,
                        "seen_center": current_region == "center",
                    })

                    direction = None
                    branch = None

                    if current_region == "center" and meta["last_region"] in VALID_BRANCHES:
                        direction = "in"
                        branch = meta["last_region"]
                    elif current_region in VALID_BRANCHES and meta["last_region"] == "center":
                        direction = "out"
                        branch = current_region

                    if direction is not None and branch is not None:
                        weight = CLASS_WEIGHTS.get(cls, 1.0)
                        event = {
                            "time": current_time,
                            "track_id": track_id,
                            "class": cls,
                            "branch": branch,
                            "direction": direction,
                            "weight": weight,
                            "bbox": (l, t, r, b),
                        }
                        flow_events.append(event)
                        new_events.append(event)

                    if current_region == "outside":
                        meta["seen_center"] = False

                    if current_region == "center":
                        meta["seen_center"] = True

                    meta["last_region"] = current_region
                    color = {
                        0: (0, 255, 0),
                        1: (255, 0, 0),
                        2: (0, 0, 255),
                        3: (0, 255, 255),
                    }.get(cls, (255, 255, 255))
                    label = f"{CLASS_NAMES.get(cls, cls)} #{track_id}"
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    cv2.putText(frame, label, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(frame, centroid, 3, color, -1)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0.0
                prev_time = curr_time

                active_track_ids = {track.track_id for track in tracks if track.is_confirmed() and track.time_since_update <= 5}
                cleanup_old_events(flow_events, current_time, self.flow_window)
                flow_pce_pm, flow_veh_pm = format_flow_metrics(flow_events, self.flow_window, active_track_ids)
                branch_pce = {
                    (branch, direction): 0.0
                    for branch in VALID_BRANCHES
                    for direction in ("in", "out")
                }
                branch_count = {
                    (branch, direction): 0
                    for branch in VALID_BRANCHES
                    for direction in ("in", "out")
                }
                for event in flow_events:
                    if event["track_id"] not in active_track_ids:
                        continue
                    key = (event["branch"], event["direction"])
                    if key in branch_pce:
                        branch_pce[key] += event["weight"]
                        branch_count[key] += 1

                curr_time = time.time()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((880, 620), Image.LANCZOS)

                with self.state_lock:
                    self.latest_pil_image = pil_image
                    self.worker_state["status"] = "Running"
                    self.worker_state["frame"] = str(frame_id)
                    self.worker_state["fps"] = f"{fps:.1f}"
                    self.worker_state["active_tracks"] = str(active_tracks)
                    self.worker_state["flow_pce_pm"] = f"{flow_pce_pm:.1f}"
                    self.worker_state["flow_veh_pm"] = f"{flow_veh_pm:.0f}"
                    self.worker_state["top_in"] = f"{branch_pce[('top', 'in')] * 60.0 / self.flow_window:.1f}"
                    self.worker_state["top_out"] = f"{branch_pce[('top', 'out')] * 60.0 / self.flow_window:.1f}"
                    self.worker_state["top_in_count"] = str(branch_count[('top', 'in')])
                    self.worker_state["top_out_count"] = str(branch_count[('top', 'out')])
                    self.worker_state["left_in"] = f"{branch_pce[('left', 'in')] * 60.0 / self.flow_window:.1f}"
                    self.worker_state["left_out"] = f"{branch_pce[('left', 'out')] * 60.0 / self.flow_window:.1f}"
                    self.worker_state["left_in_count"] = str(branch_count[('left', 'in')])
                    self.worker_state["left_out_count"] = str(branch_count[('left', 'out')])
                    self.worker_state["right_in"] = f"{branch_pce[('right', 'in')] * 60.0 / self.flow_window:.1f}"
                    self.worker_state["right_out"] = f"{branch_pce[('right', 'out')] * 60.0 / self.flow_window:.1f}"
                    self.worker_state["right_in_count"] = str(branch_count[('right', 'in')])
                    self.worker_state["right_out_count"] = str(branch_count[('right', 'out')])
                    self.worker_state["bottom_in"] = f"{branch_pce[('bottom', 'in')] * 60.0 / self.flow_window:.1f}"
                    self.worker_state["bottom_out"] = f"{branch_pce[('bottom', 'out')] * 60.0 / self.flow_window:.1f}"
                    self.worker_state["bottom_in_count"] = str(branch_count[('bottom', 'in')])
                    self.worker_state["bottom_out_count"] = str(branch_count[('bottom', 'out')])

                expected_display = playback_start + (frame_id - 1) * frame_duration
                delay = expected_display - time.time()
                if delay > 0:
                    time.sleep(delay)

            with self.state_lock:
                if not self.stop_event.is_set():
                    self.worker_state["status"] = "Finished"
                else:
                    self.worker_state["status"] = "Stopped"
        except Exception as exc:
            with self.state_lock:
                self.worker_state["status"] = f"Error: {exc}"
        finally:
            if "reader_thread" in locals() and reader_thread is not None:
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

