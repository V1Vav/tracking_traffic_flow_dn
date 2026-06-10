from __future__ import annotations

import argparse
import time

from common import (
    add_common_args,
    ensure_output_dir,
    get_capture_info,
    get_profile_config,
    open_capture,
    resize_keep_aspect_bgr,
    save_csv,
    save_json,
    summarize,
    print_summary_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark đọc video/camera và resize trước khi YOLO.")
    add_common_args(parser)
    parser.add_argument("--process-width", type=int, default=None, help="Override process_width. Mặc định lấy từ profile.")
    args = parser.parse_args()

    profile_cfg = get_profile_config(args.profile)
    process_width = args.process_width if args.process_width is not None else int(profile_cfg.get("process_width", 0) or 0)

    cap = open_capture(args.video)
    cap_info = get_capture_info(cap)

    rows = []
    read_ms = []
    resize_ms = []
    total_start = time.perf_counter()

    for idx in range(1, args.frames + 1):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        before_shape = tuple(frame.shape[:2])
        t2 = time.perf_counter()
        frame = resize_keep_aspect_bgr(frame, process_width)
        t3 = time.perf_counter()
        after_shape = tuple(frame.shape[:2])
        r_ms = (t1 - t0) * 1000.0
        z_ms = (t3 - t2) * 1000.0
        read_ms.append(r_ms)
        resize_ms.append(z_ms)
        rows.append({
            "frame": idx,
            "read_ms": r_ms,
            "resize_ms": z_ms,
            "src_h": before_shape[0],
            "src_w": before_shape[1],
            "proc_h": after_shape[0],
            "proc_w": after_shape[1],
        })

    cap.release()
    elapsed = time.perf_counter() - total_start
    processed = len(rows)
    summary = {
        "video": args.video,
        "profile": args.profile,
        "process_width": process_width,
        "source_info": cap_info,
        "processed_frames": processed,
        "wall_time_s": elapsed,
        "effective_fps": processed / elapsed if elapsed > 0 else 0.0,
    }
    summary.update(summarize(read_ms, "read"))
    summary.update(summarize(resize_ms, "resize"))

    out_dir = ensure_output_dir(args.output_dir)
    save_csv(out_dir / "01_video_io.csv", rows)
    save_json(out_dir / "01_video_io.json", summary)
    print_summary_table(
        "VIDEO I/O",
        summary,
        [
            "video", "profile", "process_width", "source_info", "processed_frames",
            "effective_fps", "read_avg_ms", "read_p95_ms", "resize_avg_ms", "resize_p95_ms",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '01_video_io.csv'} và {out_dir / '01_video_io.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
