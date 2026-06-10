from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore

from common import PhaseTimer, ResourceMonitor, parse_size, perf_ms, print_table, summarize_ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1: measure video open/read/decode/resize/color conversion.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="Read all frames but only record/process every Nth frame.")
    ap.add_argument("--resize", default=None, help="Optional resize output, e.g. 1280x720 or 640x360")
    ap.add_argument("--bgr2rgb", action="store_true", help="Also measure BGR->RGB conversion like GUI/PIL pipeline.")
    ap.add_argument("--reports", default="bottleneck_reports")
    args = ap.parse_args()

    resize_to = parse_size(args.resize)
    timer = PhaseTimer()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    frame_idx = args.start_frame
    processed = 0
    t_all0 = perf_ms()
    with ResourceMonitor() as mon:
        while processed < args.max_frames:
            t0 = perf_ms()
            ok, frame = cap.read()
            t1 = perf_ms()
            if not ok:
                break

            if args.stride > 1 and ((frame_idx - args.start_frame) % args.stride != 0):
                frame_idx += 1
                continue

            resize_ms = 0.0
            if resize_to:
                r0 = perf_ms()
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_LINEAR)
                resize_ms = perf_ms() - r0

            bgr2rgb_ms = 0.0
            if args.bgr2rgb:
                c0 = perf_ms()
                _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bgr2rgb_ms = perf_ms() - c0

            total_ms = perf_ms() - t0
            timer.add(frame=frame_idx, read_decode_ms=t1 - t0, resize_ms=resize_ms, bgr2rgb_ms=bgr2rgb_ms, total_phase_ms=total_ms)
            processed += 1
            frame_idx += 1
        mon.save(args.reports, "01_resource_samples")
    cap.release()
    total_elapsed = perf_ms() - t_all0

    csv_path, json_path = timer.save(args.reports, "01_video_io_decode", vars(args))
    vals = [float(r["total_phase_ms"]) for r in timer.rows]
    s = summarize_ms(vals)
    fps = 1000.0 / s["mean"] if s["mean"] > 0 else 0.0
    print_table("Phase 1 - Video IO/decode", [
        ("frames", len(timer.rows)),
        ("mean total ms", s["mean"]),
        ("p95 total ms", s["p95"]),
        ("phase fps", fps),
        ("wall fps", len(timer.rows) * 1000.0 / total_elapsed if total_elapsed > 0 else 0.0),
        ("csv", str(csv_path)),
        ("summary", str(json_path)),
    ])


if __name__ == "__main__":
    main()
