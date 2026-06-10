from __future__ import annotations

import argparse
import queue
import threading
import time

from common import ensure_output_dir, get_capture_info, get_profile_config, open_capture, save_csv, save_json, summarize, print_summary_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Mô phỏng reader queue realtime để xem có bị tích delay/frame cũ không.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--queue-size", type=int, default=None, help="Mặc định lấy realtime_queue_size từ profile.")
    parser.add_argument("--process-ms", type=float, default=40.0, help="Giả lập thời gian xử lý mỗi frame, ví dụ 40ms.")
    parser.add_argument("--drop-old", action="store_true", help="Khi queue đầy, bỏ frame cũ và giữ frame mới.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    profile_cfg = get_profile_config(args.profile)
    qsize = args.queue_size if args.queue_size is not None else int(profile_cfg.get("realtime_queue_size", 1) or 1)
    cap = open_capture(args.video)
    info = get_capture_info(cap)
    source_fps = info.get("fps") or 30.0
    read_interval = 1.0 / max(1.0, float(source_fps))

    frame_queue = queue.Queue(maxsize=max(1, qsize))
    stop = threading.Event()
    stats = {"read": 0, "put": 0, "dropped": 0, "queue_full": 0}

    def reader():
        while not stop.is_set() and stats["read"] < args.frames:
            t0 = time.perf_counter()
            ok, _frame = cap.read()
            if not ok:
                break
            item = {"source_id": stats["read"] + 1, "capture_time": time.perf_counter()}
            stats["read"] += 1
            try:
                frame_queue.put_nowait(item)
                stats["put"] += 1
            except queue.Full:
                stats["queue_full"] += 1
                if args.drop_old:
                    try:
                        frame_queue.get_nowait()
                        stats["dropped"] += 1
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait(item)
                        stats["put"] += 1
                    except queue.Full:
                        stats["dropped"] += 1
                else:
                    # block như pipeline không bỏ frame: gây tăng latency.
                    try:
                        frame_queue.put(item, timeout=0.5)
                        stats["put"] += 1
                    except queue.Full:
                        stats["dropped"] += 1
            # Giả lập tốc độ camera/file realtime.
            elapsed = time.perf_counter() - t0
            if elapsed < read_interval:
                time.sleep(read_interval - elapsed)
        stop.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    rows = []
    latency_ms = []
    queue_depths = []
    processed = 0
    total_start = time.perf_counter()
    while not stop.is_set() or not frame_queue.empty():
        try:
            item = frame_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        start_proc = time.perf_counter()
        delay = (start_proc - item["capture_time"]) * 1000.0
        time.sleep(max(0.0, args.process_ms) / 1000.0)
        end_proc = time.perf_counter()
        processed += 1
        qdepth = frame_queue.qsize()
        latency_ms.append(delay)
        queue_depths.append(qdepth)
        rows.append({
            "processed_index": processed,
            "source_id": item["source_id"],
            "queue_delay_ms": delay,
            "process_ms": (end_proc - start_proc) * 1000.0,
            "queue_depth_after_get": qdepth,
        })
        if processed >= args.frames:
            stop.set()
            break

    stop.set()
    th.join(timeout=1.0)
    cap.release()
    elapsed = time.perf_counter() - total_start

    summary = {
        "video": args.video,
        "profile": args.profile,
        "source_info": info,
        "queue_size": qsize,
        "process_ms_simulated": args.process_ms,
        "drop_old": args.drop_old,
        "reader_stats": stats,
        "processed_frames": processed,
        "wall_time_s": elapsed,
        "effective_fps": processed / elapsed if elapsed > 0 else 0.0,
        "avg_queue_depth_after_get": sum(queue_depths) / len(queue_depths) if queue_depths else 0.0,
    }
    summary.update(summarize(latency_ms, "queue_delay"))

    out_dir = ensure_output_dir(args.output_dir)
    save_csv(out_dir / "07_queue_latency.csv", rows)
    save_json(out_dir / "07_queue_latency.json", summary)
    print_summary_table(
        "QUEUE LATENCY SIMULATION",
        summary,
        [
            "source_info", "queue_size", "process_ms_simulated", "drop_old", "reader_stats",
            "processed_frames", "effective_fps", "queue_delay_avg_ms", "queue_delay_p95_ms",
            "queue_delay_max_ms", "avg_queue_depth_after_get",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '07_queue_latency.csv'} và {out_dir / '07_queue_latency.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
