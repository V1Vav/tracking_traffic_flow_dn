import threading
import time
import os
import csv
import numpy as np
from collections import deque
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
    def __init__(self, mapping_path):
        self.mapping_path = mapping_path
        self.regions = {}  # region_name -> list of (x,y) points
        self.resolution = None
        self.loaded = False
        self._load_mapping()

    def _load_mapping(self):
        if not self.mapping_path or not os.path.exists(self.mapping_path):
            return
        try:
            with open(self.mapping_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                if not lines:
                    return
                # First line: resolution
                res_line = lines[0]
                if len(res_line) >= 2:
                    self.resolution = (int(res_line[0]), int(res_line[1]))
                # Subsequent lines: region_name, x1,y1,x2,y2,x3,y3,x4,y4
                for row in lines[1:]:
                    if len(row) < 9:
                        continue
                    region_name = row[0].strip().lower()
                    points = []
                    for i in range(1, 9, 2):
                        x = int(row[i].strip())
                        y = int(row[i+1].strip())
                        points.append((x, y))
                    if len(points) == 4:
                        region = REGION_NAME_MAP.get(region_name, region_name)
                        self.regions[region] = points
                        print(f"Loaded region {region}: {points}")
            self.loaded = bool(self.regions)
            print(f"RegionTemplate loaded: {self.loaded}, regions: {list(self.regions.keys())}")
        except Exception as e:
            print(f"Error loading mapping: {e}")
            self.regions = {}

    def _scale_points(self, points, frame_width, frame_height):
        if not self.resolution or (frame_width == self.resolution[0] and frame_height == self.resolution[1]):
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
        scaled_regions = {name: self._scale_points(pts, width, height) for name, pts in self.regions.items()}
        for region_name, points in scaled_regions.items():
            contour = np.array(points, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                return region_name
        return None

    def overlay(self, frame):
        if not self.loaded:
            return
        height, width = frame.shape[:2]
        scaled_regions = {name: self._scale_points(pts, width, height) for name, pts in self.regions.items()}
        for region_name, points in scaled_regions.items():
            contour = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [contour], True, (0, 255, 0), 3)
            # Label position: centroid of the polygon
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, region_name.upper(), (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 0), 2)


def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_direction_region(centroid, width, height, margin_fraction, template=None):
    if template and template.loaded:
        region = template.get_region(centroid, width, height)
        if region is not None:
            return region
        # If a custom template is loaded but the centroid is not inside any mapped region,
        # treat it as center instead of falling back to margin-based branch classification.
        return "center"
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
        template.overlay(frame)
        return


def format_flow_metrics(event_window, window_seconds, active_track_ids=None):
    if active_track_ids is None:
        filtered_events = list(event_window)
    else:
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
            "top_car_in": tk.StringVar(value="0"),
            "top_car_out": tk.StringVar(value="0"),
            "top_moto_in": tk.StringVar(value="0"),
            "top_moto_out": tk.StringVar(value="0"),
            "left_car_in": tk.StringVar(value="0"),
            "left_car_out": tk.StringVar(value="0"),
            "left_moto_in": tk.StringVar(value="0"),
            "left_moto_out": tk.StringVar(value="0"),
            "right_car_in": tk.StringVar(value="0"),
            "right_car_out": tk.StringVar(value="0"),
            "right_moto_in": tk.StringVar(value="0"),
            "right_moto_out": tk.StringVar(value="0"),
            "bottom_car_in": tk.StringVar(value="0"),
            "bottom_car_out": tk.StringVar(value="0"),
            "bottom_moto_in": tk.StringVar(value="0"),
            "bottom_moto_out": tk.StringVar(value="0"),
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
            "top_car_in": "0",
            "top_car_out": "0",
            "top_moto_in": "0",
            "top_moto_out": "0",
            "left_car_in": "0",
            "left_car_out": "0",
            "left_moto_in": "0",
            "left_moto_out": "0",
            "right_car_in": "0",
            "right_car_out": "0",
            "right_moto_in": "0",
            "right_moto_out": "0",
            "bottom_car_in": "0",
            "bottom_car_out": "0",
            "bottom_moto_in": "0",
            "bottom_moto_out": "0",
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

        ttk.Label(control_frame, text="Template CSV:").grid(row=4, column=0, sticky="w", pady=4)
        template_entry = ttk.Entry(control_frame, textvariable=self.template_mapping_path_var, width=36)
        template_entry.grid(row=5, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_template_mapping).grid(row=5, column=1, sticky="ew")

        ttk.Checkbutton(control_frame, text="Display Region Template",
                            variable=self.display_template_var).grid(row=6, column=0, columnspan=2, sticky="ew", pady=4)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_processing)
        self.start_button.grid(row=7, column=0, columnspan=2, pady=8, sticky="ew")
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=8, column=0, columnspan=2, sticky="ew")

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

        vehicle_frame = ttk.LabelFrame(metrics_frame, text="Vehicle Type Count")
        vehicle_frame.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(8, 0), padx=2)
        vehicle_frame.grid_columnconfigure(1, weight=1)
        vehicle_frame.grid_columnconfigure(2, weight=1)
        vehicle_frame.grid_columnconfigure(3, weight=1)
        vehicle_frame.grid_columnconfigure(4, weight=1)

        ttk.Label(vehicle_frame, text="Direction").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(vehicle_frame, text="Car In", anchor="center").grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, text="Car Out", anchor="center").grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, text="Moto In", anchor="center").grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, text="Moto Out", anchor="center").grid(row=0, column=4, sticky="ew", padx=4)

        ttk.Label(vehicle_frame, text="Top").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["top_car_in"], anchor="center").grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["top_car_out"], anchor="center").grid(row=1, column=2, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["top_moto_in"], anchor="center").grid(row=1, column=3, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["top_moto_out"], anchor="center").grid(row=1, column=4, sticky="ew", padx=4)

        ttk.Label(vehicle_frame, text="Left").grid(row=2, column=0, sticky="w", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["left_car_in"], anchor="center").grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["left_car_out"], anchor="center").grid(row=2, column=2, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["left_moto_in"], anchor="center").grid(row=2, column=3, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["left_moto_out"], anchor="center").grid(row=2, column=4, sticky="ew", padx=4)

        ttk.Label(vehicle_frame, text="Right").grid(row=3, column=0, sticky="w", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["right_car_in"], anchor="center").grid(row=3, column=1, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["right_car_out"], anchor="center").grid(row=3, column=2, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["right_moto_in"], anchor="center").grid(row=3, column=3, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["right_moto_out"], anchor="center").grid(row=3, column=4, sticky="ew", padx=4)

        ttk.Label(vehicle_frame, text="Bottom").grid(row=4, column=0, sticky="w", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["bottom_car_in"], anchor="center").grid(row=4, column=1, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["bottom_car_out"], anchor="center").grid(row=4, column=2, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["bottom_moto_in"], anchor="center").grid(row=4, column=3, sticky="ew", padx=4)
        ttk.Label(vehicle_frame, textvariable=self.metrics["bottom_moto_out"], anchor="center").grid(row=4, column=4, sticky="ew", padx=4)

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

    def browse_template_mapping(self):
        mapping_path = filedialog.askopenfilename(
            title="Select template mapping CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")]
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

            valid_branches = set(self.region_template.regions.keys()) if self.region_template else set()
            track_meta = {}
            # Persistent counters for each branch
            # branch_pce_current: current PCE (recalc each frame based on vehicles in region) - display only
            branch_pce_current = {
                (branch, direction): 0.0
                for branch in valid_branches
                for direction in ("in", "out")
            }
            # branch_count_total: accumulate vehicle count entered/exited
            branch_count_total = {
                (branch, direction): 0
                for branch in valid_branches
                for direction in ("in", "out")
            }
            branch_class_count_total = {
                (branch, direction, cls_id): 0
                for branch in valid_branches
                for direction in ("in", "out")
                for cls_id in CLASS_WEIGHTS.keys()
            }
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
                overlay_template = self.region_template if self.display_template_var.get() else None
                draw_region_overlay(
                    frame,
                    self.region_margin,
                    overlay_template,
                )
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
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1
                            detections.append(([x1, y1, w, h], conf, cls))

                tracks = tracker.update_tracks(detections, frame=frame)
                active_tracks = 0
                active_track_ids = set()

                print(f"[DeepSort] Frame={frame_id} detections={len(detections)} total_tracks={len(tracks)}")

                # First pass: collect active track ids
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 5:
                        continue
                    active_track_ids.add(track.track_id)

                # Second pass: cleanup old tracks and detect exits
                tracks_to_remove = []
                for track_id, meta in track_meta.items():
                    if track_id not in active_track_ids:
                        # Track lost - count out if it was in a region
                        if meta["previous_region"] in valid_branches:
                            branch_count_total[(meta["previous_region"], "out")] += 1
                            cls = meta.get("cls", 2)  # default to car
                            branch_class_count_total[(meta["previous_region"], "out", cls)] += 1
                            print(f"[Flow] Count OUT (lost): {meta['previous_region']} track_id={track_id}")
                        tracks_to_remove.append(track_id)

                # Remove lost tracks from metadata
                for track_id in tracks_to_remove:
                    del track_meta[track_id]

                # Process active tracks
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 5:
                        print(f"[DeepSort]   Skipping track {track.track_id} confirmed={track.is_confirmed()} time_since_update={track.time_since_update}")
                        continue

                    active_tracks += 1
                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    cls = track.det_class
                    centroid = centroid_from_box((l, t, r, b))
                    current_region = get_direction_region(
                        centroid,
                        frame.shape[1],
                        frame.shape[0],
                        self.region_margin,
                        self.region_template,
                    )

                    meta = track_meta.setdefault(track_id, {
                        "first_region": None,
                        "previous_region": None,
                        "current_region": None,
                        "weight": 0.0,
                        "counted_in": False,
                        "cls": cls,
                    })

                    # Initialize on first detection
                    if meta["first_region"] is None:
                        meta["first_region"] = current_region
                        meta["previous_region"] = current_region
                        meta["cls"] = cls

                    # Update current region and weight
                    weight = CLASS_WEIGHTS.get(cls, 1.0)
                    meta["weight"] = weight
                    
                    # Check if vehicle entered a region (first time in valid branch)
                    if current_region in valid_branches and not meta["counted_in"]:
                        branch_count_total[(current_region, "in")] += 1
                        branch_class_count_total[(current_region, "in", cls)] += 1
                        meta["counted_in"] = True
                        print(f"[Flow] Count IN: {current_region} cls={cls} track_id={track_id}")
                    
                    # Check if vehicle left a region while still active (region changed, not lost)
                    if meta["previous_region"] in valid_branches and current_region != meta["previous_region"] and meta["counted_in"]:
                        branch_count_total[(meta["previous_region"], "out")] += 1
                        branch_class_count_total[(meta["previous_region"], "out", cls)] += 1
                        meta["counted_in"] = False  # Reset for potential re-entry
                        print(f"[Flow] Count OUT (region change): {meta['previous_region']} -> {current_region} cls={cls} track_id={track_id}")
                    
                    # Update previous region for next frame
                    meta["previous_region"] = current_region
                    meta["current_region"] = current_region
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
                    print(
                        f"[DeepSort]   Track={track_id} cls={cls} label={CLASS_NAMES.get(cls, cls)} "
                        f"ltrb=({l},{t},{r},{b}) centroid={centroid} region={current_region} confirmed={track.is_confirmed()} "
                        f"time_since_update={track.time_since_update}"
                    )

                # Recalculate current PCE based on vehicles currently in each region
                for branch in valid_branches:
                    branch_pce_current[(branch, "in")] = sum(meta["weight"] for meta in track_meta.values() 
                                                             if meta["current_region"] == branch)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0.0
                prev_time = curr_time

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((880, 620), Image.LANCZOS)

                with self.state_lock:
                    self.latest_pil_image = pil_image
                    self.worker_state["status"] = "Running"
                    self.worker_state["frame"] = str(frame_id)
                    self.worker_state["fps"] = f"{fps:.1f}"
                    self.worker_state["active_tracks"] = str(active_tracks)
                    # Total flow (current PCE + accumulated count)
                    total_pce = sum(v for k, v in branch_pce_current.items() if k[1] == "in")
                    total_veh = sum(v for k, v in branch_count_total.items() if k[1] == "in")
                    self.worker_state["flow_pce_pm"] = f"{total_pce:.1f}"
                    self.worker_state["flow_veh_pm"] = f"{total_veh:.0f}"
                    # Update metrics for each branch (current PCE, accumulated count)
                    for branch in valid_branches:
                        self.worker_state[f"{branch}_in"] = f"{branch_pce_current[(branch, 'in')]:.1f}"
                        self.worker_state[f"{branch}_out"] = f"{branch_pce_current[(branch, 'out')]:.1f}"
                        self.worker_state[f"{branch}_in_count"] = str(branch_count_total[(branch, 'in')])
                        self.worker_state[f"{branch}_out_count"] = str(branch_count_total[(branch, 'out')])
                        self.worker_state[f"{branch}_car_in"] = str(branch_class_count_total.get((branch, 'in', 2), 0))
                        self.worker_state[f"{branch}_car_out"] = str(branch_class_count_total.get((branch, 'out', 2), 0))
                        self.worker_state[f"{branch}_moto_in"] = str(branch_class_count_total.get((branch, 'in', 1), 0))
                        self.worker_state[f"{branch}_moto_out"] = str(branch_class_count_total.get((branch, 'out', 1), 0))

                expected_display = playback_start + (frame_id - 1) * frame_duration
                delay = expected_display - time.time()
                if delay > 0:
                    time.sleep(delay)

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

