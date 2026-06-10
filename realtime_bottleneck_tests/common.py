"""
Common helpers for realtime bottleneck phase tests.
These scripts are intentionally independent from the GUI so they can isolate each phase.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def perf_ms() -> float:
    return time.perf_counter() * 1000.0


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_size(text: str | None) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    raw = text.lower().replace(" ", "")
    if "x" not in raw:
        raise ValueError("size must look like 1280x720")
    w, h = raw.split("x", 1)
    return int(w), int(h)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize_ms(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "p50": float(percentile(values, 50)),
        "p90": float(percentile(values, 90)),
        "p95": float(percentile(values, 95)),
        "p99": float(percentile(values, 99)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def print_table(title: str, rows: List[Tuple[str, Any]]) -> None:
    print("\n" + title)
    print("=" * len(title))
    if not rows:
        print("(empty)")
        return
    width = max(len(str(k)) for k, _ in rows)
    for k, v in rows:
        if isinstance(v, float):
            print(f"{k:<{width}} : {v:.3f}")
        else:
            print(f"{k:<{width}} : {v}")


def sync_cuda_if_needed(device: str | None = None) -> None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and (device is None or str(device) != "cpu"):
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class PhaseTimer:
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)

    def summary(self, ms_columns: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, float]]:
        if not self.rows:
            return {}
        if ms_columns is None:
            ms_columns = [k for k in self.rows[0].keys() if k.endswith("_ms")]
        out: Dict[str, Dict[str, float]] = {}
        for col in ms_columns:
            vals: List[float] = []
            for row in self.rows:
                try:
                    vals.append(float(row[col]))
                except Exception:
                    pass
            out[col] = summarize_ms(vals)
        return out

    def save(self, report_dir: str | Path, name: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[Path, Path]:
        report_dir = ensure_dir(report_dir)
        csv_path = write_csv(report_dir / f"{name}.csv", self.rows)
        json_path = write_json(report_dir / f"{name}_summary.json", {
            "metadata": metadata or {},
            "summary_ms": self.summary(),
            "rows": len(self.rows),
        })
        return csv_path, json_path


class ResourceMonitor:
    """Low-frequency CPU/RAM/GPU sampler. It avoids per-frame nvidia-smi overhead."""

    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self.samples: List[Dict[str, Any]] = []
        self.thread: Optional[threading.Thread] = None

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _run(self) -> None:
        psutil = None
        try:
            import psutil as _psutil  # type: ignore
            psutil = _psutil
        except Exception:
            pass
        while not self._stop.is_set():
            sample: Dict[str, Any] = {"time": time.time()}
            if psutil:
                try:
                    sample["cpu_percent"] = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory()
                    sample["ram_percent"] = mem.percent
                    sample["ram_used_mb"] = round(mem.used / 1024 / 1024, 1)
                except Exception as e:
                    sample["psutil_error"] = str(e)
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=0.4,
                )
                if out.returncode == 0 and out.stdout.strip():
                    gpu, used, total, temp = [x.strip() for x in out.stdout.splitlines()[0].split(",")[:4]]
                    sample["gpu_percent"] = float(gpu)
                    sample["gpu_mem_used_mb"] = float(used)
                    sample["gpu_mem_total_mb"] = float(total)
                    sample["gpu_temp_c"] = float(temp)
            except Exception:
                pass
            self.samples.append(sample)
            self._stop.wait(self.interval_sec)

    def save(self, report_dir: str | Path, name: str = "resource_samples") -> Optional[Path]:
        if not self.samples:
            return None
        return write_csv(Path(report_dir) / f"{name}.csv", self.samples)


def get_video_info(video_path: str | Path) -> Dict[str, Any]:
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "video": str(video_path),
        "opened": bool(cap.isOpened()),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC) or 0),
    }
    cap.release()
    return info


def iter_video_frames(video_path: str | Path, max_frames: int = 500, start_frame: int = 0, stride: int = 1):
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    produced = 0
    frame_idx = start_frame
    while produced < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if stride <= 1 or ((frame_idx - start_frame) % stride == 0):
            yield frame_idx, frame
            produced += 1
        frame_idx += 1
    cap.release()


def infer_device_string(device: str) -> str:
    if device.lower() in {"cpu", "none"}:
        return "cpu"
    return device


def try_import_ultralytics_yolo(model_path: str | Path):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing ultralytics. Install/use the same environment as your app: pip install ultralytics") from e
    return YOLO(str(model_path))


def cv2_font_scale(frame_w: int) -> float:
    return max(0.45, min(0.8, frame_w / 1600.0))
