"""Chạy xử lý video không cần giao diện để export flow cho video dài."""

import os
import threading
from dataclasses import dataclass

from .config import (
    DEFAULT_EXPORT_ROOT,
    DEFAULT_FLUID_BIN_SECONDS,
    DEFAULT_FLUID_SMOOTH_SECONDS,
    DEFAULT_LEFT_GATE_X_RATIO,
    DEFAULT_REGION_STATE_SAMPLE_SECONDS,
    DEFAULT_TEMPLATE_MAPPING,
    DEFAULT_TRACK_SAMPLE_SECONDS,
    PERFORMANCE_PROFILES,
)
from .regions import RegionTemplate
from .video_worker import process_video


class SimpleVar:
    """Biến đơn giản mô phỏng StringVar/BooleanVar của Tkinter cho chế độ headless."""

    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


@dataclass
class HeadlessOptions:
    video: str
    model: str
    template: str = DEFAULT_TEMPLATE_MAPPING
    performance: str = "quality"
    export_root: str = DEFAULT_EXPORT_ROOT
    target_fps: float | None = None
    imgsz: int | None = None
    process_width: int | None = None
    max_det: int | None = None
    track_person: bool = False
    realtime_skip: bool = False
    allow_fallback_regions: bool = False
    infer_hidden_left: bool = False
    left_gate_x_ratio: float = DEFAULT_LEFT_GATE_X_RATIO
    fluid_bin_seconds: float = DEFAULT_FLUID_BIN_SECONDS
    fluid_smooth_seconds: float = DEFAULT_FLUID_SMOOTH_SECONDS
    track_sample_seconds: float = DEFAULT_TRACK_SAMPLE_SECONDS
    region_state_sample_seconds: float = DEFAULT_REGION_STATE_SAMPLE_SECONDS
    flow_window: int = 60
    region_margin: float = 0.25
    progress_interval: float = 10.0


class HeadlessFlowApp:
    """Đối tượng trạng thái tối thiểu để dùng lại worker xử lý video của app."""

    def __init__(self, options: HeadlessOptions):
        if not options.video:
            raise ValueError("Thiếu đường dẫn video cho chế độ headless")
        if not options.model:
            raise ValueError("Thiếu đường dẫn model cho chế độ headless")

        self.headless = True
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.worker_state = {"status": "Sẵn sàng"}
        self.latest_pil_image = None
        self.export_output_dir = None

        self.video_path_var = SimpleVar(options.video)
        self.model_path_var = SimpleVar(options.model)
        self.template_mapping_path_var = SimpleVar(options.template or "")
        self.export_root_var = SimpleVar(options.export_root or DEFAULT_EXPORT_ROOT)
        self.export_fluid_var = SimpleVar(True)  # Headless mặc định luôn export flow.
        self.display_template_var = SimpleVar(False)
        self.infer_hidden_left_var = SimpleVar(bool(options.infer_hidden_left))
        self.left_gate_x_ratio_var = SimpleVar(str(options.left_gate_x_ratio))

        self.fluid_bin_seconds = float(options.fluid_bin_seconds)
        self.fluid_smooth_seconds = float(options.fluid_smooth_seconds)
        self.track_sample_seconds = float(options.track_sample_seconds)
        self.region_state_sample_seconds = float(options.region_state_sample_seconds)
        self.left_gate_x_ratio = float(options.left_gate_x_ratio)
        self.flow_window = int(options.flow_window)
        self.region_margin = float(options.region_margin)
        self.headless_progress_interval = float(options.progress_interval)

        profile = str(options.performance or "quality").strip().lower()
        if profile not in PERFORMANCE_PROFILES:
            raise ValueError(f"Performance không hợp lệ: {profile}. Hợp lệ: {', '.join(PERFORMANCE_PROFILES)}")

        self.performance_profile = profile
        self.performance_cfg = PERFORMANCE_PROFILES[profile].copy()

        # Headless dùng để xuất dữ liệu thật cho video dài, nên mặc định không bỏ frame.
        # Nếu muốn replay realtime và chấp nhận mất frame, truyền --realtime-skip.
        self.performance_cfg["drop_frames_when_slow"] = bool(options.realtime_skip)

        # Không cần tạo ảnh preview ở chế độ headless.
        self.performance_cfg["display_every_n"] = 10**9
        self.performance_cfg["async_display"] = False
        self.performance_cfg["display_width"] = 0
        self.performance_cfg["display_height"] = 0

        if options.target_fps is not None:
            self.performance_cfg["target_process_fps"] = float(options.target_fps)
        if options.imgsz is not None:
            self.performance_cfg["model_imgsz"] = int(options.imgsz)
        if options.process_width is not None:
            self.performance_cfg["process_width"] = int(options.process_width)
        if options.max_det is not None:
            self.performance_cfg["max_det"] = int(options.max_det)
        if options.track_person:
            # Person vẫn không tính flow/count, nhưng có thể đưa vào DeepSORT nếu cần debug.
            self.performance_cfg["track_ignored_classes"] = True

        self.detect_interval = int(self.performance_cfg.get("detect_interval", 1))
        self.allow_fallback_regions = bool(options.allow_fallback_regions)
        self.region_template = None
        self.load_region_template()

    def load_region_template(self):
        mapping_path = str(self.template_mapping_path_var.get() or "").strip()
        if not mapping_path:
            if self.allow_fallback_regions:
                self.region_template = None
                print("[HEADLESS] Không có template, dùng vùng biên mặc định.")
                return
            raise FileNotFoundError("Headless cần --template hoặc dùng --allow-fallback-regions")

        if not os.path.exists(mapping_path):
            if self.allow_fallback_regions:
                self.region_template = None
                print(f"[HEADLESS] Không tìm thấy template '{mapping_path}', dùng vùng biên mặc định.")
                return
            raise FileNotFoundError(f"Không tìm thấy template: {mapping_path}")

        template = RegionTemplate(mapping_path)
        if not template.loaded:
            if self.allow_fallback_regions:
                self.region_template = None
                print(f"[HEADLESS] Template không hợp lệ '{mapping_path}', dùng vùng biên mặc định.")
                return
            raise RuntimeError(f"Template không hợp lệ hoặc không có vùng 8 làn: {mapping_path}")

        self.region_template = template
        print(f"[HEADLESS] Đã tải template: {', '.join(template.regions.keys())}")


def run_headless(options: HeadlessOptions):
    """Chạy worker video trực tiếp trong terminal và trả về thư mục export."""
    app = HeadlessFlowApp(options)
    print("[HEADLESS] Bắt đầu xử lý")
    print(f"[HEADLESS] Video    : {options.video}")
    print(f"[HEADLESS] Model    : {options.model}")
    print(f"[HEADLESS] Template : {options.template}")
    print(f"[HEADLESS] Export   : {options.export_root}")
    print(f"[HEADLESS] Profile  : {app.performance_profile}")
    print(f"[HEADLESS] Cấu hình : {app.performance_cfg}")

    try:
        process_video(app, options.video, options.model)
    except KeyboardInterrupt:
        app.stop_event.set()
        print("\n[HEADLESS] Nhận Ctrl+C, đang dừng...")
        raise

    status = app.worker_state.get("status", "Hoàn tất")
    print(f"[HEADLESS] {status}")
    if app.export_output_dir:
        print(f"[HEADLESS] Thư mục export: {app.export_output_dir}")
    return app.export_output_dir
