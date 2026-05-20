"""Tkinter UI application for vehicle flow analysis."""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from .config import (
    BRANCH_ORDER,
    CLASS_NAMES,
    DEFAULT_AVAILABLE_MODELS,
    DEFAULT_EXPORT_ROOT,
    DEFAULT_FLUID_BIN_SECONDS,
    DEFAULT_FLUID_SMOOTH_SECONDS,
    DEFAULT_INFER_HIDDEN_LEFT,
    DEFAULT_LEFT_GATE_X_RATIO,
    DEFAULT_MODEL_PATH,
    DEFAULT_PERFORMANCE_PROFILE,
    PERFORMANCE_PROFILES,
    DEFAULT_REGION_STATE_SAMPLE_SECONDS,
    DEFAULT_TEMPLATE_MAPPING,
    DEFAULT_TRACK_SAMPLE_SECONDS,
    DISPLAY_CLASS_IDS,
)
from .regions import RegionTemplate
from .video_worker import process_video


class FlowApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vehicle Flow Explorer")
        self.root.geometry("1280x780")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_path_var = tk.StringVar(value="")
        self.model_path_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self.template_mapping_path_var = tk.StringVar(value=DEFAULT_TEMPLATE_MAPPING)
        self.status_var = tk.StringVar(value="Ready")
        self.available_models = DEFAULT_AVAILABLE_MODELS
        self.performance_profile_var = tk.StringVar(value=DEFAULT_PERFORMANCE_PROFILE)

        self.region_template = None
        self.display_template_var = tk.BooleanVar(value=False)

        # Fluid-flow export options. These are intentionally kept simple so
        # this branch can export replay data without changing the existing
        # tracking/counting UI semantics.
        self.export_fluid_var = tk.BooleanVar(value=False)
        self.infer_hidden_left_var = tk.BooleanVar(value=DEFAULT_INFER_HIDDEN_LEFT)
        self.export_root_var = tk.StringVar(value=DEFAULT_EXPORT_ROOT)
        self.fluid_bin_seconds_var = tk.StringVar(value=str(DEFAULT_FLUID_BIN_SECONDS))
        self.fluid_smooth_seconds_var = tk.StringVar(value=str(DEFAULT_FLUID_SMOOTH_SECONDS))
        self.track_sample_seconds_var = tk.StringVar(value=str(DEFAULT_TRACK_SAMPLE_SECONDS))
        self.region_state_sample_seconds_var = tk.StringVar(value=str(DEFAULT_REGION_STATE_SAMPLE_SECONDS))
        self.left_gate_x_ratio_var = tk.StringVar(value=str(DEFAULT_LEFT_GATE_X_RATIO))

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
            "flow_veh_pm": "0",
        }

        for branch in BRANCH_ORDER:
            state.update({
                f"{branch}_pce": "0.0",
                f"{branch}_count": "0",
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

        ttk.Label(control_frame, text="Video / Camera / Stream:").grid(row=0, column=0, sticky="w", pady=4)
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
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Checkbutton(
            control_frame,
            text="Export Flow Data",
            variable=self.export_fluid_var,
        ).grid(row=7, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Label(control_frame, text="Performance:").grid(row=8, column=0, sticky="w", pady=2)
        ttk.Combobox(
            control_frame,
            textvariable=self.performance_profile_var,
            values=list(PERFORMANCE_PROFILES.keys()),
            width=12,
            state="readonly",
        ).grid(row=8, column=1, sticky="ew", pady=2)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_processing)
        self.start_button.grid(row=9, column=0, columnspan=2, pady=(6, 2), sticky="ew")
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=10, column=0, columnspan=2, sticky="ew")

        control_frame.grid_columnconfigure(0, weight=1)
        for child in control_frame.winfo_children():
            child.grid_configure(padx=2, pady=2)

        metrics_frame = ttk.LabelFrame(right_frame, text="Metrics")
        metrics_frame.pack(fill="x", pady=(0, 8))

        for row, (label_text, var_name) in enumerate([
            ("Frame:", "frame"),
            ("FPS:", "fps"),
            ("Active tracks:", "active_tracks"),
            ("Current PCE:", "current_pce"),
            ("Total veh:", "flow_veh_pm"),
        ]):
            ttk.Label(metrics_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Label(metrics_frame, textvariable=self.metrics[var_name], width=28).grid(row=row, column=1, sticky="w")

        branch_frame = ttk.LabelFrame(metrics_frame, text="Branch Current PCE + Count")
        branch_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0), padx=2)
        headers = ["Direction", "PCE now", "Count now"]
        for col, header in enumerate(headers):
            branch_frame.grid_columnconfigure(col, weight=1)
            ttk.Label(branch_frame, text=header, anchor="center").grid(row=0, column=col, sticky="ew", padx=2)

        for row, branch in enumerate(BRANCH_ORDER, start=1):
            ttk.Label(branch_frame, text=branch.title()).grid(row=row, column=0, sticky="w", padx=4)
            ttk.Label(branch_frame, textvariable=self.metrics[f"{branch}_pce"], anchor="center").grid(row=row, column=1, sticky="ew", padx=2)
            ttk.Label(branch_frame, textvariable=self.metrics[f"{branch}_count"], anchor="center").grid(row=row, column=2, sticky="ew", padx=2)

        vehicle_frame = ttk.LabelFrame(metrics_frame, text="Vehicle Type Count Total")
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

    def _float_from_var(self, var, default, min_value=None, max_value=None):
        try:
            value = float(var.get())
        except Exception:
            value = float(default)
        if min_value is not None:
            value = max(value, float(min_value))
        if max_value is not None:
            value = min(value, float(max_value))
        var.set(str(value))
        return value

    def _normalize_export_settings(self):
        self.fluid_bin_seconds = self._float_from_var(self.fluid_bin_seconds_var, DEFAULT_FLUID_BIN_SECONDS, 0.1, 60.0)
        self.fluid_smooth_seconds = self._float_from_var(self.fluid_smooth_seconds_var, DEFAULT_FLUID_SMOOTH_SECONDS, 0.0, 300.0)
        self.track_sample_seconds = self._float_from_var(self.track_sample_seconds_var, DEFAULT_TRACK_SAMPLE_SECONDS, 0.0, 60.0)
        self.region_state_sample_seconds = self._float_from_var(self.region_state_sample_seconds_var, DEFAULT_REGION_STATE_SAMPLE_SECONDS, 0.1, 60.0)
        self.left_gate_x_ratio = self._float_from_var(self.left_gate_x_ratio_var, DEFAULT_LEFT_GATE_X_RATIO, 0.01, 0.95)

    def _normalize_performance_settings(self):
        profile = self.performance_profile_var.get().strip().lower()
        if profile not in PERFORMANCE_PROFILES:
            profile = DEFAULT_PERFORMANCE_PROFILE
        self.performance_profile = profile
        self.performance_cfg = PERFORMANCE_PROFILES[profile].copy()
        self.detect_interval = int(self.performance_cfg.get("detect_interval", self.detect_interval))
        self.performance_profile_var.set(profile)

    def start_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            return

        video_path = self.video_path_var.get().strip()
        model_path = self.model_path_var.get().strip()

        if not video_path:
            messagebox.showwarning("Missing video", "Please choose a video file, type camera index like 0, or paste RTSP/HTTP stream URL.")
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

        self._normalize_export_settings()
        self._normalize_performance_settings()

        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=process_video,
            args=(self, video_path, model_path),
            daemon=True,
        )
        self.processing_thread.start()

    def stop_processing(self):
        self.stop_event.set()
        self.stop_button.config(state="disabled")
        self.status_var.set("Stopping...")

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
