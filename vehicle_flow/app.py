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
    REGION_DISPLAY_NAMES,
)
from .regions import RegionTemplate
from .video_worker import process_video


UI = {
    "bg": "#f1f5f9",
    "card": "#ffffff",
    "card_2": "#f8fafc",
    "border": "#cbd5e1",
    "border_soft": "#e2e8f0",
    "text": "#0f172a",
    "muted": "#64748b",
    "accent": "#2563eb",
    "accent_dark": "#1d4ed8",
    "success": "#16a34a",
    "danger": "#dc2626",
    "video_bg": "#020617",
    "video_text": "#94a3b8",
}


class FlowApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vehicle Flow Explorer · AI Traffic Analytics")
        self.root.geometry("1360x820")
        self.root.minsize(1120, 720)
        self.root.configure(bg=UI["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._setup_style()

        self.video_path_var = tk.StringVar(value="")
        self.model_path_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self.template_mapping_path_var = tk.StringVar(value=DEFAULT_TEMPLATE_MAPPING)
        self.status_var = tk.StringVar(value="Ready")
        self.available_models = DEFAULT_AVAILABLE_MODELS
        self.performance_profile_var = tk.StringVar(value=DEFAULT_PERFORMANCE_PROFILE)

        self.region_template = None
        self.display_template_var = tk.BooleanVar(value=True)

        # Fluid-flow export options. Keep the UI compact: only the checkbox is
        # exposed in the main screen. Advanced values still have safe defaults.
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

    def _setup_style(self):
        self.root.option_add("*Font", ("Segoe UI", 10))
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=("Segoe UI", 10), background=UI["bg"], foreground=UI["text"])
        style.configure("App.TFrame", background=UI["bg"])
        style.configure("Card.TFrame", background=UI["card"], relief="flat")
        style.configure("Subtle.TFrame", background=UI["card_2"], relief="flat")
        style.configure("Header.TFrame", background=UI["bg"])

        style.configure("TLabel", background=UI["bg"], foreground=UI["text"])
        style.configure("Card.TLabel", background=UI["card"], foreground=UI["text"])
        style.configure("Muted.TLabel", background=UI["card"], foreground=UI["muted"])
        style.configure("Title.TLabel", background=UI["bg"], foreground=UI["text"], font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=UI["bg"], foreground=UI["muted"], font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=UI["card"], foreground=UI["text"], font=("Segoe UI", 11, "bold"))
        style.configure("MetricValue.TLabel", background=UI["card_2"], foreground=UI["accent_dark"], font=("Segoe UI", 13, "bold"))
        style.configure("MetricName.TLabel", background=UI["card_2"], foreground=UI["muted"], font=("Segoe UI", 8, "bold"))
        style.configure("MetricCompactName.TLabel", background=UI["card"], foreground=UI["muted"], font=("Segoe UI", 8, "bold"))
        style.configure("MetricCompactValue.TLabel", background=UI["card"], foreground=UI["accent_dark"], font=("Segoe UI", 10, "bold"))
        style.configure("TableHeader.TLabel", background="#e0ecff", foreground=UI["accent_dark"], font=("Segoe UI", 8, "bold"))
        style.configure("TableCell.TLabel", background=UI["card"], foreground=UI["text"], font=("Segoe UI", 9))
        style.configure("TableValue.TLabel", background=UI["card"], foreground=UI["text"], font=("Segoe UI", 9, "bold"))
        style.configure("Status.TLabel", background=UI["card_2"], foreground=UI["muted"], font=("Segoe UI", 9))
        style.configure("StatusValue.TLabel", background=UI["card_2"], foreground=UI["text"], font=("Segoe UI", 9, "bold"))

        style.configure("TLabelframe", background=UI["card"], foreground=UI["text"], bordercolor=UI["border_soft"], relief="solid")
        style.configure("TLabelframe.Label", background=UI["card"], foreground=UI["text"], font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background=UI["card"], foreground=UI["text"])
        style.map("TCheckbutton", background=[("active", UI["card"])] )

        style.configure("TEntry", fieldbackground="#ffffff", foreground=UI["text"], bordercolor=UI["border"])
        style.configure("TCombobox", fieldbackground="#ffffff", foreground=UI["text"], bordercolor=UI["border"])
        style.configure("TNotebook", background=UI["card"], borderwidth=0, tabmargins=(0, 2, 0, 0))
        style.configure("TNotebook.Tab", background=UI["card_2"], foreground=UI["muted"], padding=(10, 5), font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#e0ecff"), ("active", "#f1f5f9")], foreground=[("selected", UI["accent_dark"])])

        style.configure("Accent.TButton", background=UI["success"], foreground="#ffffff", font=("Segoe UI", 10, "bold"), borderwidth=0, padding=(8, 6))
        style.map("Accent.TButton", background=[("active", "#15803d"), ("disabled", "#94a3b8")])
        style.configure("Danger.TButton", background=UI["danger"], foreground="#ffffff", font=("Segoe UI", 10, "bold"), borderwidth=0, padding=(8, 6))
        style.map("Danger.TButton", background=[("active", "#b91c1c"), ("disabled", "#94a3b8")])
        style.configure("Tool.TButton", background="#e2e8f0", foreground=UI["text"], borderwidth=0, padding=(8, 5))
        style.map("Tool.TButton", background=[("active", "#cbd5e1")])

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

    def _metric_card(self, parent, row, col, title, var_name):
        card = ttk.Frame(parent, style="Subtle.TFrame", padding=(8, 5))
        card.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)
        ttk.Label(card, text=title.upper(), style="MetricName.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=self.metrics[var_name], style="MetricValue.TLabel").pack(anchor="w")
        return card

    def _compact_metric(self, parent, row, col, title, var_name):
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.grid(row=row, column=col, sticky="ew", padx=(0, 8), pady=1)
        cell.grid_columnconfigure(1, weight=1)
        ttk.Label(cell, text=f"{title}:", style="MetricCompactName.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Label(cell, textvariable=self.metrics[var_name], style="MetricCompactValue.TLabel").grid(row=0, column=1, sticky="w")
        return cell

    def _table_label(self, parent, text=None, variable=None, style="TableCell.TLabel", anchor="center"):
        label = ttk.Label(parent, text=text, textvariable=variable, style=style, anchor=anchor)
        return label

    def _vehicle_header_name(self, class_name, direction):
        short_names = {
            "bicycle": "Bicycle",
            "bus": "Bus",
            "car": "Car",
            "motorbike": "Motorbike",
            "motorcycle": "Moto",
        }
        return f"{short_names.get(class_name, class_name.title())} {direction}"

    def _build_ui(self):
        shell = ttk.Frame(self.root, style="App.TFrame", padding=10)
        shell.grid(row=0, column=0, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        header = ttk.Frame(shell, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(header, text="Vehicle Flow Explorer", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="YOLO + DeepSORT traffic counting, 8-lane region analysis, and fluid-flow export",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        content = ttk.Frame(shell, style="App.TFrame")
        content.grid(row=1, column=0, sticky="nsew")
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(1, weight=1)

        left_frame = ttk.Frame(content, style="App.TFrame")
        right_frame = ttk.Frame(content, style="Card.TFrame", padding=8, width=390)

        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_frame.grid(row=0, column=1, sticky="ns")
        right_frame.grid_propagate(False)

        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)
        content.grid_rowconfigure(0, weight=1)

        video_card = ttk.Frame(left_frame, style="Card.TFrame", padding=10)
        video_card.pack(fill="both", expand=True)
        video_card.grid_columnconfigure(0, weight=1)
        video_card.grid_rowconfigure(1, weight=1)

        video_header = ttk.Frame(video_card, style="Card.TFrame")
        video_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        video_header.grid_columnconfigure(0, weight=1)
        ttk.Label(video_header, text="Live Preview", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(video_header, text="Regions are highlighted when Show Regions is enabled", style="Muted.TLabel").grid(row=1, column=0, sticky="w")

        self.video_label = tk.Label(
            video_card,
            text="Select a video, camera index, or stream URL\nthen press Start",
            anchor="center",
            justify="center",
            bg=UI["video_bg"],
            fg=UI["video_text"],
            font=("Segoe UI", 14, "bold"),
            bd=0,
            highlightthickness=1,
            highlightbackground=UI["border"],
        )
        self.video_label.grid(row=1, column=0, sticky="nsew")

        # ---------- Controls ----------
        control_frame = ttk.LabelFrame(right_frame, text="  Controls  ", padding=8)
        control_frame.pack(fill="x", pady=(0, 6))
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=0)

        ttk.Label(control_frame, text="Video / Camera / Stream", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.video_path_var, width=34).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(1, 4))
        ttk.Button(control_frame, text="Browse", command=self.browse_video, style="Tool.TButton").grid(row=1, column=1, sticky="ew", pady=(1, 4))

        ttk.Label(control_frame, text="YOLO model", style="Card.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            control_frame,
            textvariable=self.model_path_var,
            values=self.available_models,
            width=32,
            state="normal",
        ).grid(row=3, column=0, sticky="ew", padx=(0, 6), pady=(1, 4))
        ttk.Button(control_frame, text="Browse", command=self.browse_model, style="Tool.TButton").grid(row=3, column=1, sticky="ew", pady=(1, 4))

        ttk.Label(control_frame, text="Region template CSV", style="Card.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.template_mapping_path_var, width=34).grid(row=5, column=0, sticky="ew", padx=(0, 6), pady=(1, 4))
        ttk.Button(control_frame, text="Browse", command=self.browse_template_mapping, style="Tool.TButton").grid(row=5, column=1, sticky="ew", pady=(1, 4))

        options = ttk.Frame(control_frame, style="Card.TFrame")
        options.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        options.grid_columnconfigure(0, weight=1)
        options.grid_columnconfigure(1, weight=1)
        ttk.Checkbutton(options, text="Show Regions", variable=self.display_template_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options, text="Export Flow Data", variable=self.export_fluid_var).grid(row=0, column=1, sticky="w")

        perf_frame = ttk.Frame(control_frame, style="Card.TFrame")
        perf_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        perf_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(perf_frame, text="Performance", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Combobox(
            perf_frame,
            textvariable=self.performance_profile_var,
            values=list(PERFORMANCE_PROFILES.keys()),
            width=12,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew")

        button_frame = ttk.Frame(control_frame, style="Card.TFrame")
        button_frame.grid(row=8, column=0, columnspan=2, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        self.start_button = ttk.Button(button_frame, text="▶ Start", command=self.start_processing, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.stop_button = ttk.Button(button_frame, text="■ Stop", command=self.stop_processing, state="disabled", style="Danger.TButton")
        self.stop_button.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # ---------- Metrics ----------
        # Compact layout: keep this short so count tables stay visible.
        metrics_frame = ttk.LabelFrame(right_frame, text="  Realtime Metrics  ", padding=(8, 5))
        metrics_frame.pack(fill="x", pady=(0, 6))
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)

        metric_items = [
            ("Frame", "frame"),
            ("FPS", "fps"),
            ("Tracks", "active_tracks"),
            ("PCE", "current_pce"),
            ("Total", "flow_veh_pm"),
        ]
        for idx, (title, var_name) in enumerate(metric_items):
            self._compact_metric(metrics_frame, idx // 2, idx % 2, title, var_name)

        # ---------- Count tables ----------
        # Only one long 8-lane table is visible at a time, so the right panel
        # stays compact and Vehicle Type Count Total is no longer hidden.
        tables_notebook = ttk.Notebook(right_frame)
        tables_notebook.pack(fill="both", expand=True, pady=(0, 6))

        branch_frame = ttk.Frame(tables_notebook, style="Card.TFrame", padding=(6, 5))
        vehicle_frame = ttk.Frame(tables_notebook, style="Card.TFrame", padding=(6, 5))
        tables_notebook.add(branch_frame, text="Current PCE + Count")
        tables_notebook.add(vehicle_frame, text="Vehicle Type Total")

        headers = ["Region", "PCE now", "Count now"]
        for col, header_text in enumerate(headers):
            branch_frame.grid_columnconfigure(col, weight=1)
            self._table_label(branch_frame, text=header_text, style="TableHeader.TLabel").grid(row=0, column=col, sticky="ew", padx=1, pady=(0, 2))

        for row, branch in enumerate(BRANCH_ORDER, start=1):
            self._table_label(branch_frame, text=REGION_DISPLAY_NAMES.get(branch, branch.upper()), style="TableCell.TLabel", anchor="w").grid(row=row, column=0, sticky="ew", padx=1, pady=1)
            self._table_label(branch_frame, variable=self.metrics[f"{branch}_pce"], style="TableValue.TLabel").grid(row=row, column=1, sticky="ew", padx=1, pady=1)
            self._table_label(branch_frame, variable=self.metrics[f"{branch}_count"], style="TableValue.TLabel").grid(row=row, column=2, sticky="ew", padx=1, pady=1)

        vehicle_headers = ["Region"]
        for cls_id in DISPLAY_CLASS_IDS:
            class_name = CLASS_NAMES[cls_id]
            vehicle_headers.append(self._vehicle_header_name(class_name, "In"))
            vehicle_headers.append(self._vehicle_header_name(class_name, "Out"))

        for col, header_text in enumerate(vehicle_headers):
            vehicle_frame.grid_columnconfigure(col, weight=1)
            self._table_label(vehicle_frame, text=header_text, style="TableHeader.TLabel").grid(row=0, column=col, sticky="ew", padx=1, pady=(0, 2))

        for row, branch in enumerate(BRANCH_ORDER, start=1):
            self._table_label(vehicle_frame, text=REGION_DISPLAY_NAMES.get(branch, branch.upper()), style="TableCell.TLabel", anchor="w").grid(row=row, column=0, sticky="ew", padx=1, pady=1)
            col = 1
            for cls_id in DISPLAY_CLASS_IDS:
                class_name = CLASS_NAMES[cls_id]
                self._table_label(vehicle_frame, variable=self.metrics[f"{branch}_{class_name}_in"], style="TableValue.TLabel").grid(row=row, column=col, sticky="ew", padx=1, pady=1)
                self._table_label(vehicle_frame, variable=self.metrics[f"{branch}_{class_name}_out"], style="TableValue.TLabel").grid(row=row, column=col + 1, sticky="ew", padx=1, pady=1)
                col += 2

        status_frame = ttk.Frame(right_frame, style="Subtle.TFrame", padding=(8, 5))
        status_frame.pack(fill="x")
        status_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Status", style="Status.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(status_frame, textvariable=self.status_var, style="StatusValue.TLabel", wraplength=300).grid(row=0, column=1, sticky="w")

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

        if not mapping_path:
            self.region_template = None
            self.status_var.set("No template selected; using margin regions")
            return

        if os.path.exists(mapping_path):
            self.region_template = RegionTemplate(mapping_path)
            if self.region_template.loaded:
                loaded_regions = ", ".join(self.region_template.regions.keys())
                self.status_var.set(f"Template loaded: {loaded_regions}")
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
                self.video_label.configure(image=self.latest_photo, text="")
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
