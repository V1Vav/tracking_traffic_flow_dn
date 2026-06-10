"""Hàm dùng chung cho các script benchmark module realtime.

Các script trong thư mục này cố ý chạy độc lập với GUI để đo từng khối:
Video I/O -> resize -> YOLO -> DeepSORT -> region -> display/overlay.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test_results"


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def ensure_output_dir(path: Optional[str] = None) -> Path:
    out = Path(path).expanduser().resolve() if path else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summarize(values: Sequence[float], name: str) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            f"{name}_count": 0,
            f"{name}_avg_ms": 0.0,
            f"{name}_median_ms": 0.0,
            f"{name}_p95_ms": 0.0,
            f"{name}_max_ms": 0.0,
        }
    return {
        f"{name}_count": len(vals),
        f"{name}_avg_ms": mean(vals),
        f"{name}_median_ms": median(vals),
        f"{name}_p95_ms": percentile(vals, 95),
        f"{name}_max_ms": max(vals),
    }


def get_profile_config(profile: str) -> Dict[str, Any]:
    from vehicle_flow.config import DEFAULT_PERFORMANCE_PROFILE, PERFORMANCE_PROFILES

    profile = (profile or DEFAULT_PERFORMANCE_PROFILE).strip().lower()
    if profile not in PERFORMANCE_PROFILES:
        raise ValueError(f"Profile không hợp lệ: {profile}. Hợp lệ: {', '.join(PERFORMANCE_PROFILES)}")
    return PERFORMANCE_PROFILES[profile].copy()


def resize_keep_aspect_bgr(frame, process_width: int):
    import cv2

    try:
        process_width = int(process_width or 0)
    except Exception:
        process_width = 0
    if process_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= process_width:
        return frame
    scale = process_width / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (process_width, new_h), interpolation=cv2.INTER_AREA)


def resolve_video_source(source: str):
    text = str(source).strip()
    return int(text) if text.isdigit() else text


def open_capture(video: str):
    import cv2

    cap = cv2.VideoCapture(resolve_video_source(video))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video/camera: {video}")
    return cap


def get_capture_info(cap) -> Dict[str, Any]:
    import cv2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return {"fps": fps, "width": width, "height": height, "frame_count": count}


def cuda_sync_if_needed(device: str = "") -> None:
    try:
        import torch

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
    except Exception:
        pass


def choose_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def format_ms(value: float) -> str:
    return f"{float(value):7.2f} ms"


def print_summary_table(title: str, summary: Dict[str, Any], keys: Iterable[str]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key in keys:
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, float):
            print(f"{key:32s}: {value:10.3f}")
        else:
            print(f"{key:32s}: {value}")


def diagnose_stage(stage_ms: Dict[str, float], target_fps: float) -> List[str]:
    """Trả về nhận xét ngắn về stage nào vượt ngân sách thời gian."""
    if target_fps <= 0:
        target_fps = 30.0
    budget = 1000.0 / target_fps
    notes = [f"Ngân sách realtime mục tiêu: {budget:.2f} ms/frame cho {target_fps:.1f} FPS."]
    total = sum(max(0.0, float(v)) for v in stage_ms.values())
    notes.append(f"Tổng trung bình các stage đo được: {total:.2f} ms/frame (~{1000.0 / total:.2f} FPS)" if total > 0 else "Chưa có số liệu stage.")
    if not stage_ms:
        return notes
    sorted_items = sorted(stage_ms.items(), key=lambda kv: float(kv[1]), reverse=True)
    top_name, top_value = sorted_items[0]
    notes.append(f"Stage nặng nhất: {top_name} = {top_value:.2f} ms/frame.")
    if total > budget:
        notes.append("Pipeline đang vượt ngân sách realtime; cần giảm stage nặng nhất hoặc bỏ frame cũ.")
    else:
        notes.append("Pipeline trung bình nằm trong ngân sách, nếu vẫn giật hãy xem p95/max hoặc queue latency.")
    return notes


def add_common_args(parser) -> None:
    parser.add_argument("--video", required=True, help="Đường dẫn video, camera index hoặc RTSP/HTTP URL.")
    parser.add_argument("--profile", default="realtime", help="Preset trong vehicle_flow.config.PERFORMANCE_PROFILES.")
    parser.add_argument("--frames", type=int, default=300, help="Số frame dùng để test.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Thư mục lưu CSV/JSON kết quả.")
