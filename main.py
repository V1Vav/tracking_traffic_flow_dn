"""Entry point cho app GUI và chế độ headless export flow."""

import argparse
import sys

from vehicle_flow.config import (
    DEFAULT_EXPORT_ROOT,
    DEFAULT_LEFT_GATE_X_RATIO,
    DEFAULT_MODEL_PATH,
    DEFAULT_PERFORMANCE_PROFILE,
    DEFAULT_REGION_STATE_SAMPLE_SECONDS,
    DEFAULT_TEMPLATE_MAPPING,
    DEFAULT_TRACK_SAMPLE_SECONDS,
    PERFORMANCE_PROFILES,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phân tích lưu lượng phương tiện bằng YOLO + ByteTrack.",
    )
    parser.add_argument("--headless", action="store_true", help="Chạy không mở UI, dùng cho video dài và export flow.")
    parser.add_argument("--video", help="Đường dẫn video/camera/RTSP dùng khi --headless.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Đường dẫn model YOLO .pt khi --headless.")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE_MAPPING, help="Đường dẫn template.csv 8 vùng/làn khi --headless.")
    parser.add_argument(
        "--performance",
        choices=tuple(PERFORMANCE_PROFILES.keys()),
        default="quality",
        help="Preset hiệu năng headless. Mặc định quality để không bỏ frame khi export dữ liệu.",
    )
    parser.add_argument("--export-root", default=DEFAULT_EXPORT_ROOT, help="Thư mục gốc lưu flow_exports khi --headless.")
    parser.add_argument("--target-fps", type=float, default=None, help="FPS lấy mẫu trước khi xử lý, ví dụ 30. Mặc định theo preset.")
    parser.add_argument("--imgsz", type=int, default=None, help="Kích thước ảnh YOLO, ví dụ 640/736/960/1280.")
    parser.add_argument("--process-width", type=int, default=None, help="Resize frame về chiều rộng này trước khi xử lý. 0 = giữ nguyên.")
    parser.add_argument("--max-det", type=int, default=None, help="Số detection tối đa mỗi frame.")
    parser.add_argument("--track-person", action="store_true", help="Vẫn đưa person vào DeepSORT để debug. Person vẫn không tính flow/count.")
    parser.add_argument("--realtime-skip", action="store_true", help="Cho phép bỏ frame khi xử lý không kịp. Không khuyến nghị cho export dữ liệu kiểm chứng.")
    parser.add_argument("--allow-fallback-regions", action="store_true", help="Cho phép chạy không template hợp lệ và dùng vùng biên mặc định.")
    parser.add_argument("--infer-hidden-left", action="store_true", help="Bật suy luận left ẩn. Mặc định tắt để chỉ xuất dữ liệu thật.")
    parser.add_argument("--left-gate-x-ratio", type=float, default=DEFAULT_LEFT_GATE_X_RATIO, help="Ngưỡng cổng trái nếu bật --infer-hidden-left.")
    parser.add_argument("--track-sample-seconds", type=float, default=DEFAULT_TRACK_SAMPLE_SECONDS, help="Chu kỳ ghi track_replay.csv.")
    parser.add_argument("--region-state-sample-seconds", type=float, default=DEFAULT_REGION_STATE_SAMPLE_SECONDS, help="Chu kỳ ghi region_state_timeseries.csv.")
    parser.add_argument("--progress-interval", type=float, default=10.0, help="Số giây giữa hai dòng log tiến độ headless.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.headless:
        if not args.video:
            parser.error("--headless cần --video")
        from vehicle_flow.headless import HeadlessOptions, run_headless

        options = HeadlessOptions(
            video=args.video,
            model=args.model,
            template=args.template,
            performance=args.performance,
            export_root=args.export_root,
            target_fps=args.target_fps,
            imgsz=args.imgsz,
            process_width=args.process_width,
            max_det=args.max_det,
            track_person=args.track_person,
            realtime_skip=args.realtime_skip,
            allow_fallback_regions=args.allow_fallback_regions,
            infer_hidden_left=args.infer_hidden_left,
            left_gate_x_ratio=args.left_gate_x_ratio,
            track_sample_seconds=args.track_sample_seconds,
            region_state_sample_seconds=args.region_state_sample_seconds,
            progress_interval=args.progress_interval,
        )
        run_headless(options)
        return 0

    from vehicle_flow.app import FlowApp

    app = FlowApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
