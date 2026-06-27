"""Ứng dụng giao diện Tkinter cho phân tích lưu lượng phương tiện."""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from .config import (
    BRANCH_ORDER,
    CLASS_COLORS,
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    COLOR_LEGEND_CLASS_IDS,
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
    DISPLAY_BRANCH_ORDER,
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

PERFORMANCE_PROFILE_LABELS = {
    "realtime": "Thời gian thực",
    "balanced": "Cân bằng",
    "quality": "Chất lượng",
}
PERFORMANCE_PROFILE_VALUE_BY_LABEL = {
    label.lower(): key for key, label in PERFORMANCE_PROFILE_LABELS.items()
}



class FlowApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hệ thống phân tích lưu lượng giao thông")
        self.root.geometry("1360x820")
        self.root.minsize(1120, 720)
        self.root.configure(bg=UI["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._setup_style()

        self.video_path_var = tk.StringVar(value="")
        self.model_path_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self.template_mapping_path_var = tk.StringVar(value=DEFAULT_TEMPLATE_MAPPING)
        self.status_var = tk.StringVar(value="Chưa bắt đầu")
        self.performance_profile_var = tk.StringVar(value=PERFORMANCE_PROFILE_LABELS.get(DEFAULT_PERFORMANCE_PROFILE, DEFAULT_PERFORMANCE_PROFILE))

        self.region_template = None
        self.display_template_var = tk.BooleanVar(value=True)

        # Tùy chọn xuất dữ liệu flow dạng chất lỏng. Giữ UI gọn: chỉ hiển thị checkbox
        # trên màn hình chính. Các giá trị nâng cao vẫn dùng mặc định an toàn.
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
        self.pcu_bar_widgets = {}
        self.pcu_bar_scale = 6.0
        self.flow_bar_widgets = {}
        self.flow_bar_scale = 10.0
        # Kích thước thật của vùng video trên Tkinter. Worker chỉ đọc giá trị này,
        # không gọi trực tiếp Tkinter từ thread nền để tránh lỗi thread-safety.
        self.video_display_size = (880, 620)
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()

        self.detect_interval = 1
        self.flow_window = 60
        self.region_margin = 0.25

        self._build_ui()
        self.load_region_template()
        self.root.after(0, self._maximize_window)
        self.root.after(50, self._update_ui)

    def _maximize_window(self):
        """Mở ứng dụng ở trạng thái maximize, không phải fullscreen."""
        try:
            # Windows: dùng trạng thái zoomed chuẩn của Tk.
            self.root.state("zoomed")
            return
        except tk.TclError:
            pass

        try:
            # Linux/X11: một số window manager hỗ trợ thuộc tính -zoomed.
            self.root.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass

        # Fallback cho môi trường không hỗ trợ maximize: dùng kích thước màn hình.
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

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
        style.configure("Hint.TLabel", background=UI["card_2"], foreground=UI["muted"], font=("Segoe UI", 8))
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
            "status": "Chưa bắt đầu",
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

    def _pcu_bar(self, parent, branch):
        """Thanh PCU dạng volume để nhìn nhanh mức chiếm dụng từng hướng."""
        canvas = tk.Canvas(
            parent,
            width=126,
            height=18,
            bg=UI["card"],
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        bg_rect = canvas.create_rectangle(2, 5, 124, 14, fill=UI["border_soft"], outline="")
        fill_rect = canvas.create_rectangle(2, 5, 2, 14, fill=UI["success"], outline="")
        text_item = canvas.create_text(63, 9, text="0.0", fill=UI["text"], font=("Segoe UI", 8, "bold"))
        self.pcu_bar_widgets[branch] = (canvas, bg_rect, fill_rect, text_item)
        return canvas

    def _safe_float(self, value, default=0.0):
        try:
            return float(str(value).replace(",", "."))
        except Exception:
            return float(default)

    def _update_pcu_bars(self):
        if not self.pcu_bar_widgets:
            return

        values = {
            branch: max(0.0, self._safe_float(self.worker_state.get(f"{branch}_pce", "0.0")))
            for branch in self.pcu_bar_widgets
        }
        max_value = max(values.values(), default=0.0)
        self.pcu_bar_scale = max(6.0, max_value * 1.15)

        for branch, (canvas, bg_rect, fill_rect, text_item) in self.pcu_bar_widgets.items():
            value = values.get(branch, 0.0)
            width = max(36, int(canvas.winfo_width() or 126))
            height = max(16, int(canvas.winfo_height() or 18))
            x1, x2 = 2, width - 2
            y1, y2 = max(4, height // 2 - 4), min(height - 3, height // 2 + 5)
            ratio = 0.0 if self.pcu_bar_scale <= 0 else min(1.0, value / self.pcu_bar_scale)
            if value > 0.0:
                ratio = max(0.07, ratio)
            fill_x2 = x1 + int((x2 - x1) * ratio)
            canvas.coords(bg_rect, x1, y1, x2, y2)
            canvas.coords(fill_rect, x1, y1, fill_x2, y2)
            canvas.coords(text_item, width // 2, height // 2)
            canvas.itemconfigure(text_item, text=f"{value:.1f}")

    def _flow_bar(self, parent, branch):
        """Thanh lưu lượng dạng volume để nhìn nhanh mật độ xe theo hướng."""
        canvas = tk.Canvas(
            parent,
            width=126,
            height=18,
            bg=UI["card"],
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        bg_rect = canvas.create_rectangle(2, 5, 124, 14, fill=UI["border_soft"], outline="")
        fill_rect = canvas.create_rectangle(2, 5, 2, 14, fill=UI["accent"], outline="")
        text_item = canvas.create_text(63, 9, text="0.0", fill=UI["text"], font=("Segoe UI", 8, "bold"))
        self.flow_bar_widgets[branch] = (canvas, bg_rect, fill_rect, text_item)
        return canvas

    def _update_flow_bars(self):
        if not self.flow_bar_widgets:
            return

        values = {
            branch: max(0.0, self._safe_float(self.worker_state.get(f"{branch}_in_flow", "0.0")))
            for branch in self.flow_bar_widgets
        }
        max_value = max(values.values(), default=0.0)
        self.flow_bar_scale = max(10.0, max_value * 1.15)

        for branch, (canvas, bg_rect, fill_rect, text_item) in self.flow_bar_widgets.items():
            value = values.get(branch, 0.0)
            width = max(36, int(canvas.winfo_width() or 126))
            height = max(16, int(canvas.winfo_height() or 18))
            x1, x2 = 2, width - 2
            y1, y2 = max(4, height // 2 - 4), min(height - 3, height // 2 + 5)
            ratio = 0.0 if self.flow_bar_scale <= 0 else min(1.0, value / self.flow_bar_scale)
            if value > 0.0:
                ratio = max(0.07, ratio)
            fill_x2 = x1 + int((x2 - x1) * ratio)
            canvas.coords(bg_rect, x1, y1, x2, y2)
            canvas.coords(fill_rect, x1, y1, fill_x2, y2)
            canvas.coords(text_item, width // 2, height // 2)
            canvas.itemconfigure(text_item, text=f"{value:.1f}")

    def _vehicle_header_name(self, class_name, direction=None):
        short_names = {
            "bicycle": "Xe đạp",
            "bus": "Xe buýt",
            "bus_truck": "Xe buýt",
            "truck": "Xe buýt",
            "car": "Ô tô",
            "motorbike": "Xe máy",
            "motorcycle": "Xe máy",
        }

        base_name = short_names.get(class_name, class_name.title())
        if direction is None or str(direction).strip() == "":
            return base_name

        direction_names = {"In": "Vào", "Out": "Ra", "in": "Vào", "out": "Ra"}
        return f"{base_name} {direction_names.get(direction, direction)}"

    def _bgr_to_hex(self, color):
        """Đổi màu BGR OpenCV sang RGB hex cho Tkinter legend."""
        try:
            b, g, r = [max(0, min(255, int(v))) for v in color]
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#94a3b8"

    def _vehicle_display_name(self, cls_id):
        return CLASS_DISPLAY_NAMES.get(cls_id, self._vehicle_header_name(CLASS_NAMES.get(cls_id, str(cls_id)), "").strip())

    def _build_color_legend(self, parent):
        """Chú thích màu gọn một dòng để không che bảng thống kê bên dưới."""
        legend_frame = ttk.Frame(parent, style="Card.TFrame")
        legend_frame.pack(fill="x", pady=(0, 4))
        legend_frame.grid_columnconfigure(len(COLOR_LEGEND_CLASS_IDS) * 2 + 1, weight=1)

        ttk.Label(legend_frame, text="Chú thích:", style="MetricCompactName.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )

        col = 1
        for cls_id in COLOR_LEGEND_CLASS_IDS:
            color = self._bgr_to_hex(CLASS_COLORS.get(cls_id, (160, 160, 160)))
            swatch = tk.Frame(
                legend_frame,
                width=12,
                height=10,
                bg=color,
                highlightthickness=1,
                highlightbackground=UI["border"],
            )
            swatch.grid(row=0, column=col, sticky="w", padx=(0, 3))
            swatch.grid_propagate(False)
            ttk.Label(
                legend_frame,
                text=self._vehicle_display_name(cls_id),
                style="MetricCompactName.TLabel",
            ).grid(row=0, column=col + 1, sticky="w", padx=(0, 8))
            col += 2

    def _on_video_label_configure(self, event):
        """Lưu kích thước vùng hiển thị để worker resize video đúng tỉ lệ."""
        width = max(1, int(getattr(event, "width", 1)))
        height = max(1, int(getattr(event, "height", 1)))
        with self.state_lock:
            self.video_display_size = (width, height)

    def _build_ui(self):
        shell = ttk.Frame(self.root, style="App.TFrame", padding=10)
        shell.grid(row=0, column=0, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        header = ttk.Frame(shell, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(header, text="Hệ thống phân tích lưu lượng giao thông", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Theo dõi phương tiện và trực quan hóa lưu lượng tại nút giao.",
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
        ttk.Label(video_header, text="Khu vực quan sát", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        self.video_label = tk.Label(
            video_card,
            text="Chọn nguồn video để bắt đầu phân tích",
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
        self.video_label.bind("<Configure>", self._on_video_label_configure)

        # ---------- Điều khiển ----------
        control_frame = ttk.LabelFrame(right_frame, text="  Nguồn dữ liệu  ", padding=8)
        control_frame.pack(fill="x", pady=(0, 6))
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=0)

        ttk.Label(control_frame, text="Nguồn video", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.video_path_var, width=34).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(1, 4))
        ttk.Button(control_frame, text="Chọn video", command=self.browse_video, style="Tool.TButton").grid(row=1, column=1, sticky="ew", pady=(1, 4))

        mode_frame = ttk.Frame(control_frame, style="Card.TFrame")
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(1, 4))
        mode_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(mode_frame, text="Chế độ xử lý", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.performance_profile_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.performance_profile_var,
            values=(
                PERFORMANCE_PROFILE_LABELS["realtime"],
                PERFORMANCE_PROFILE_LABELS["balanced"],
                PERFORMANCE_PROFILE_LABELS["quality"],
            ),
            state="readonly",
            width=18,
        )
        self.performance_profile_combo.grid(row=0, column=1, sticky="ew")

        options = ttk.Frame(control_frame, style="Card.TFrame")
        options.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        options.grid_columnconfigure(0, weight=1)
        options.grid_columnconfigure(1, weight=1)
        ttk.Checkbutton(options, text="Hiển thị vùng phân tích", variable=self.display_template_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options, text="Lưu kết quả", variable=self.export_fluid_var).grid(row=0, column=1, sticky="w")

        button_frame = ttk.Frame(control_frame, style="Card.TFrame")
        button_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        self.start_button = ttk.Button(button_frame, text="▶ Bắt đầu phân tích", command=self.start_processing, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.stop_button = ttk.Button(button_frame, text="■ Dừng", command=self.stop_processing, state="disabled", style="Danger.TButton")
        self.stop_button.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # ---------- Chỉ số ----------
        # Bố cục gọn để các bảng đếm vẫn nhìn thấy rõ.
        metrics_frame = ttk.LabelFrame(right_frame, text="  Tổng quan  ", padding=(8, 5))
        metrics_frame.pack(fill="x", pady=(0, 6))
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)

        metric_items = [
            ("PCU hiện tại", "current_pce"),
            ("Tổng lượt xe", "flow_veh_pm"),
        ]
        for idx, (title, var_name) in enumerate(metric_items):
            self._compact_metric(metrics_frame, idx // 2, idx % 2, title, var_name)

        self._build_color_legend(right_frame)

        # ---------- Bảng đếm ----------
        # Chỉ hiển thị một bảng dài 8 làn tại một thời điểm để panel bên phải
        # luôn gọn và bảng tổng theo loại xe không bị khuất.
        tables_notebook = ttk.Notebook(right_frame)
        tables_notebook.pack(fill="both", expand=True, pady=(0, 6))

        branch_frame = ttk.Frame(tables_notebook, style="Card.TFrame", padding=(6, 5))
        flow_frame = ttk.Frame(tables_notebook, style="Card.TFrame", padding=(6, 5))
        vehicle_frame = ttk.Frame(tables_notebook, style="Card.TFrame", padding=(6, 5))
        tables_notebook.add(branch_frame, text="PCU theo hướng")
        tables_notebook.add(flow_frame, text="Lưu lượng theo hướng")
        tables_notebook.add(vehicle_frame, text="Số lượng phương tiện")

        headers = ["Hướng", "PCU", "Mức", "Số xe"]
        weights = [2, 1, 3, 1]
        for col, header_text in enumerate(headers):
            branch_frame.grid_columnconfigure(col, weight=weights[col])
            self._table_label(branch_frame, text=header_text, style="TableHeader.TLabel").grid(row=0, column=col, sticky="ew", padx=1, pady=(0, 2))

        for row, branch in enumerate(DISPLAY_BRANCH_ORDER, start=1):
            self._table_label(branch_frame, text=REGION_DISPLAY_NAMES.get(branch, branch.upper()), style="TableCell.TLabel", anchor="w").grid(row=row, column=0, sticky="ew", padx=1, pady=1)
            self._table_label(branch_frame, variable=self.metrics[f"{branch}_pce"], style="TableValue.TLabel").grid(row=row, column=1, sticky="ew", padx=1, pady=1)
            self._pcu_bar(branch_frame, branch).grid(row=row, column=2, sticky="ew", padx=2, pady=1)
            self._table_label(branch_frame, variable=self.metrics[f"{branch}_count"], style="TableValue.TLabel").grid(row=row, column=3, sticky="ew", padx=1, pady=1)

        flow_headers = ["Hướng", "Lượt/phút", "Mức", "Tổng"]
        flow_weights = [2, 1, 3, 1]
        for col, header_text in enumerate(flow_headers):
            flow_frame.grid_columnconfigure(col, weight=flow_weights[col])
            self._table_label(flow_frame, text=header_text, style="TableHeader.TLabel").grid(row=0, column=col, sticky="ew", padx=1, pady=(0, 2))

        for row, branch in enumerate(DISPLAY_BRANCH_ORDER, start=1):
            self._table_label(flow_frame, text=REGION_DISPLAY_NAMES.get(branch, branch.upper()), style="TableCell.TLabel", anchor="w").grid(row=row, column=0, sticky="ew", padx=1, pady=1)
            self._table_label(flow_frame, variable=self.metrics[f"{branch}_in_flow"], style="TableValue.TLabel").grid(row=row, column=1, sticky="ew", padx=1, pady=1)
            self._flow_bar(flow_frame, branch).grid(row=row, column=2, sticky="ew", padx=2, pady=1)
            self._table_label(flow_frame, variable=self.metrics[f"{branch}_in_count"], style="TableValue.TLabel").grid(row=row, column=3, sticky="ew", padx=1, pady=1)

        # Bảng tổng theo loại xe chỉ hiển thị tổng lượt phương tiện đi vào từng vùng.
        # Không tách thêm cột Vào/Ra vì hướng di chuyển đã được mã hóa ngay trong
        # tên vùng: t1/l1/r1/b1 là làn vào, t2/l2/r2/b2 là làn ra.
        vehicle_headers = ["Hướng"]
        for cls_id in DISPLAY_CLASS_IDS:
            class_name = CLASS_NAMES[cls_id]
            vehicle_headers.append(self._vehicle_header_name(class_name))

        for col, header_text in enumerate(vehicle_headers):
            vehicle_frame.grid_columnconfigure(col, weight=1)
            self._table_label(vehicle_frame, text=header_text, style="TableHeader.TLabel").grid(row=0, column=col, sticky="ew", padx=1, pady=(0, 2))

        for row, branch in enumerate(DISPLAY_BRANCH_ORDER, start=1):
            self._table_label(vehicle_frame, text=REGION_DISPLAY_NAMES.get(branch, branch.upper()), style="TableCell.TLabel", anchor="w").grid(row=row, column=0, sticky="ew", padx=1, pady=1)
            col = 1
            for cls_id in DISPLAY_CLASS_IDS:
                class_name = CLASS_NAMES[cls_id]
                self._table_label(vehicle_frame, variable=self.metrics[f"{branch}_{class_name}_in"], style="TableValue.TLabel").grid(row=row, column=col, sticky="ew", padx=1, pady=1)
                col += 1

        status_frame = ttk.Frame(right_frame, style="Subtle.TFrame", padding=(8, 5))
        status_frame.pack(fill="x")
        status_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Trạng thái:", style="Status.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(status_frame, textvariable=self.status_var, style="StatusValue.TLabel", wraplength=300).grid(row=0, column=1, sticky="w")

    def browse_video(self):
        video_path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("File video", "*.mp4 *.avi *.mov *.mkv"), ("Tất cả file", "*")],
        )
        if video_path:
            self.video_path_var.set(video_path)

    def load_region_template(self):
        mapping_path = self.template_mapping_path_var.get().strip()

        if not mapping_path:
            self.region_template = None
            self.status_var.set("Sẵn sàng phân tích")
            return

        if os.path.exists(mapping_path):
            self.region_template = RegionTemplate(mapping_path)
            if self.region_template.loaded:
                self.status_var.set("Sẵn sàng phân tích")
                return

        self.region_template = None
        self.status_var.set("Sẵn sàng phân tích")

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
        raw_profile = self.performance_profile_var.get().strip()
        profile = PERFORMANCE_PROFILE_VALUE_BY_LABEL.get(raw_profile.lower(), raw_profile.lower())
        if profile not in PERFORMANCE_PROFILES:
            profile = DEFAULT_PERFORMANCE_PROFILE
        self.performance_profile = profile
        self.performance_cfg = PERFORMANCE_PROFILES[profile].copy()
        self.detect_interval = int(self.performance_cfg.get("detect_interval", self.detect_interval))
        self.performance_profile_var.set(PERFORMANCE_PROFILE_LABELS.get(profile, profile))

    def start_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            return

        video_path = self.video_path_var.get().strip()
        model_path = self.model_path_var.get().strip() or DEFAULT_MODEL_PATH
        self.model_path_var.set(model_path)

        if not video_path:
            messagebox.showwarning("Thiếu nguồn video", "Vui lòng chọn file video, nhập chỉ số camera hoặc dán URL luồng video.")
            return

        self.load_region_template()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        with self.state_lock:
            self.worker_state = self._default_worker_state()
            self.worker_state["status"] = "Đang chuẩn bị phân tích..."

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
        self.status_var.set("Đang dừng phân tích...")

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
            self._update_pcu_bars()
            self._update_flow_bars()

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
