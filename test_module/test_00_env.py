from __future__ import annotations

import argparse
import os
import platform
import sys

from common import ensure_output_dir, get_profile_config, save_json, print_summary_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra môi trường CUDA/OpenCV/Ultralytics/DeepSORT và cấu hình profile.")
    parser.add_argument("--model", default=None, help="Model .pt để đọc model.names, không bắt buộc.")
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "profile": args.profile,
        "profile_cfg": get_profile_config(args.profile),
    }

    try:
        import cv2
        data["opencv_version"] = cv2.__version__
        try:
            data["opencv_threads"] = cv2.getNumThreads()
        except Exception:
            data["opencv_threads"] = None
    except Exception as exc:
        data["opencv_error"] = repr(exc)

    try:
        import torch
        data["torch_version"] = torch.__version__
        data["torch_cuda_available"] = bool(torch.cuda.is_available())
        data["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        data["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            data["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            data["gpu_total_memory_mb"] = round(props.total_memory / 1024 / 1024, 1)
    except Exception as exc:
        data["torch_error"] = repr(exc)

    try:
        import ultralytics
        data["ultralytics_version"] = getattr(ultralytics, "__version__", None)
    except Exception as exc:
        data["ultralytics_error"] = repr(exc)

    try:
        import deep_sort_realtime
        data["deep_sort_realtime_version"] = getattr(deep_sort_realtime, "__version__", "unknown")
    except Exception as exc:
        data["deep_sort_realtime_error"] = repr(exc)

    if args.model:
        try:
            from ultralytics import YOLO
            model = YOLO(args.model)
            data["model_path"] = args.model
            data["model_names"] = getattr(model, "names", {})
        except Exception as exc:
            data["model_error"] = repr(exc)

    out_dir = ensure_output_dir(args.output_dir)
    save_json(out_dir / "00_env.json", data)
    print_summary_table(
        "ENV / PROFILE",
        data,
        [
            "python", "platform", "cpu_count", "opencv_version", "opencv_threads",
            "torch_version", "torch_cuda_available", "torch_cuda_version", "cudnn_version",
            "gpu_name", "gpu_total_memory_mb", "ultralytics_version",
            "deep_sort_realtime_version", "model_path", "model_names", "profile_cfg",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '00_env.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
