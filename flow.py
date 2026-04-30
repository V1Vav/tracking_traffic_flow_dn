import threading
import time
from collections import Counter, deque
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.models import YOLO


CLASS_WEIGHTS = {
    2: 1.0,   # car
    3: 0.5,   # motorbike / motorcycle
    5: 2.0,   # bus
    7: 2.5    # truck
}

CLASS_NAMES = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck"
}


def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_direction_region(centroid, width, height, margin_fraction):
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


def format_flow_metrics(event_window, window_seconds):
    count = len(event_window)
    weighted_sum = sum(event["weight"] for event in event_window)
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
        self.model_path_var = tk.StringVar(value="yolov8n.pt")
        self.status_var = tk.StringVar(value="Ready")

        self.metrics = {
            "frame": tk.StringVar(value="0"),
            "fps": tk.StringVar(value="0.0"),
            "active_tracks": tk.StringVar(value="0"),
            "flow_pce_pm": tk.StringVar(value="0.0"),
            "flow_veh_pm": tk.StringVar(value="0.0"),
            "directions": tk.StringVar(value="none"),
        }

        self.worker_state = {
            "status": "Ready",
            "frame": "0",
            "fps": "0.0",
            "active_tracks": "0",
            "flow_pce_pm": "0.0",
            "flow_veh_pm": "0.0",
            "directions": "none",
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
        model_entry = ttk.Entry(control_frame, textvariable=self.model_path_var, width=36)
        model_entry.grid(row=3, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Browse...", command=self.browse_model).grid(row=3, column=1, sticky="ew")

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_processing)
        self.start_button.grid(row=4, column=0, columnspan=2, pady=8, sticky="ew")
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=5, column=0, columnspan=2, sticky="ew")

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
            ("Directions:", "directions"),
        ]:
            ttk.Label(metrics_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Label(metrics_frame, textvariable=self.metrics[var_name], width=28).grid(row=row, column=1, sticky="w")
            row += 1

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
            frames = []

            with self.state_lock:
                self.worker_state["status"] = "Loading video into memory..."

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                raise RuntimeError("Video không có frame nào")

            with self.state_lock:
                self.worker_state["status"] = f"Loaded {len(frames)} frames @ {fps_input:.1f} FPS"

            flow_events = deque()
            track_meta = {}
            prev_time = time.time()
            playback_start = time.time()

            for frame_id, frame in enumerate(frames, start=1):
                if self.stop_event.is_set():
                    break

                current_time = frame_id / fps_input
                new_events = []
                detections = []

                if frame_id % self.detect_interval == 0:
                    results = model(
                        frame,
                        imgsz=1280,
                        conf=0.15,
                        iou=0.45,
                        classes=[2, 3, 5, 7],
                        verbose=False,
                    )
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls not in CLASS_WEIGHTS:
                                continue
                            if cls == 3 and conf < 0.1:
                                continue
                            if cls != 3 and conf < 0.18:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1
                            detections.append(([x1, y1, w, h], conf, cls))

                tracks = tracker.update_tracks(detections, frame=frame)
                active_tracks = 0

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 5:
                        continue

                    active_tracks += 1
                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    cls = track.det_class
                    centroid = centroid_from_box((l, t, r, b))
                    current_region = get_direction_region(
                        centroid, frame.shape[1], frame.shape[0], self.region_margin
                    )

                    meta = track_meta.setdefault(track_id, {
                        "last_region": current_region,
                        "origin_region": current_region if current_region != "center" else None,
                        "seen_center": current_region == "center",
                    })

                    if current_region == "center":
                        meta["seen_center"] = True

                    if current_region in {"top", "bottom", "left", "right"} and meta["seen_center"]:
                        origin = meta["origin_region"] or meta["last_region"]
                        if origin in {"top", "bottom", "left", "right"} and current_region != origin:
                            direction_label = f"{origin}->{current_region}"
                            weight = CLASS_WEIGHTS.get(cls, 1.0)
                            event = {
                                "time": current_time,
                                "track_id": track_id,
                                "class": cls,
                                "direction": direction_label,
                                "weight": weight,
                                "bbox": (l, t, r, b),
                            }
                            flow_events.append(event)
                            new_events.append(event)
                            meta["seen_center"] = False
                            meta["origin_region"] = current_region

                    if current_region == "outside":
                        meta["seen_center"] = False

                    if current_region in {"top", "bottom", "left", "right"} and meta["origin_region"] is None:
                        meta["origin_region"] = current_region

                    meta["last_region"] = current_region
                    color = {
                        2: (0, 255, 0),
                        3: (255, 0, 0),
                        5: (0, 0, 255),
                        7: (0, 255, 255),
                    }.get(cls, (255, 255, 255))
                    label = f"{CLASS_NAMES.get(cls, cls)} #{track_id}"
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    cv2.putText(frame, label, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(frame, centroid, 3, color, -1)

                cleanup_old_events(flow_events, current_time, self.flow_window)
                flow_pce_pm, flow_veh_pm = format_flow_metrics(flow_events, self.flow_window)
                direction_counts = Counter(event["direction"] for event in flow_events)
                direction_summary = (
                    " ".join(f"{d}:{c}" for d, c in direction_counts.items())
                    if direction_counts else "none"
                )

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
                    self.worker_state["flow_pce_pm"] = f"{flow_pce_pm:.1f}"
                    self.worker_state["flow_veh_pm"] = f"{flow_veh_pm:.0f}"
                    self.worker_state["directions"] = direction_summary

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
