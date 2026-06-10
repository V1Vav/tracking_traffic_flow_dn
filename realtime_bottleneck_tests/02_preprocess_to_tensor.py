from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2  # type: ignore
import numpy as np

from common import PhaseTimer, ResourceMonitor, infer_device_string, iter_video_frames, perf_ms, print_table, summarize_ms, sync_cuda_if_needed


def letterbox_bgr(frame: np.ndarray, imgsz: int, pad_value: int = 114) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2: measure letterbox/transpose/CPU->GPU tensor transfer.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="0/cuda:0 or cpu")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--reports", default="bottleneck_reports")
    args = ap.parse_args()

    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("This test needs torch in the same environment as the app.") from e

    device = torch.device("cpu" if infer_device_string(args.device) == "cpu" else f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    dtype = torch.float16 if args.half and str(device).startswith("cuda") else torch.float32

    timer = PhaseTimer()
    with ResourceMonitor() as mon:
        for frame_idx, frame in iter_video_frames(args.video, args.max_frames, args.start_frame, args.stride):
            t0 = perf_ms()
            lb0 = perf_ms()
            img = letterbox_bgr(frame, args.imgsz)
            lb1 = perf_ms()

            tr0 = perf_ms()
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR HWC -> RGB CHW
            img = np.ascontiguousarray(img)
            tr1 = perf_ms()

            ten0 = perf_ms()
            tensor = torch.from_numpy(img).unsqueeze(0)
            ten1 = perf_ms()

            sync_cuda_if_needed(str(device))
            to0 = perf_ms()
            tensor = tensor.to(device=device, dtype=dtype, non_blocking=True) / 255.0
            sync_cuda_if_needed(str(device))
            to1 = perf_ms()

            total_ms = perf_ms() - t0
            timer.add(
                frame=frame_idx,
                letterbox_resize_pad_ms=lb1 - lb0,
                transpose_contiguous_ms=tr1 - tr0,
                torch_from_numpy_ms=ten1 - ten0,
                to_device_normalize_ms=to1 - to0,
                total_phase_ms=total_ms,
                shape=str(tuple(tensor.shape)),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
            )
        mon.save(args.reports, "02_resource_samples")

    csv_path, json_path = timer.save(args.reports, "02_preprocess_to_tensor", vars(args))
    s = summarize_ms([float(r["total_phase_ms"]) for r in timer.rows])
    print_table("Phase 2 - Preprocess/tensor", [
        ("frames", len(timer.rows)),
        ("mean total ms", s["mean"]),
        ("p95 total ms", s["p95"]),
        ("phase fps", 1000.0 / s["mean"] if s["mean"] > 0 else 0.0),
        ("csv", str(csv_path)),
        ("summary", str(json_path)),
    ])


if __name__ == "__main__":
    main()
