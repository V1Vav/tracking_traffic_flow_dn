from __future__ import annotations

import argparse
import time

from common import (
    add_common_args,
    choose_device,
    cuda_sync_if_needed,
    ensure_output_dir,
    get_profile_config,
    open_capture,
    resize_keep_aspect_bgr,
    save_csv,
    save_json,
    summarize,
    print_summary_table,
)


def _build_tracker_configs(profile_cfg):
    deep_cfg = dict(profile_cfg)
    deep_cfg["tracker_type"] = "deepsort"

    byte_cfg = dict(profile_cfg)
    byte_cfg["tracker_type"] = "bytetrack_lite"
    byte_cfg.setdefault("bytetrack_high_thresh", 0.45)
    byte_cfg.setdefault("bytetrack_low_thresh", 0.10)
    byte_cfg.setdefault("bytetrack_new_track_thresh", 0.45)
    byte_cfg.setdefault("bytetrack_match_thresh", 0.30)
    byte_cfg.setdefault("bytetrack_low_match_thresh", 0.20)
    byte_cfg.setdefault("bytetrack_max_age", 45)
    byte_cfg.setdefault("bytetrack_n_init", 1)
    return deep_cfg, byte_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="So sánh thời gian update DeepSORT và ByteTrackLite trên cùng detection YOLO.")
    add_common_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--track-person", action="store_true")
    parser.add_argument("--no-half", action="store_true")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO
    from vehicle_flow.config import COUNTED_CLASS_IDS, MODEL_CONF, MODEL_IOU
    from vehicle_flow.video_worker import (
        _build_model_class_map,
        _create_tracker,
        _passes_detection_filters,
        _suppress_duplicate_detections,
    )

    profile_cfg = get_profile_config(args.profile)
    deep_cfg, byte_cfg = _build_tracker_configs(profile_cfg)

    device = choose_device()
    half = bool(profile_cfg.get("half_cuda", True)) and device.startswith("cuda") and not args.no_half
    imgsz = args.imgsz if args.imgsz is not None else int(profile_cfg.get("model_imgsz", 640))
    max_det = args.max_det if args.max_det is not None else int(profile_cfg.get("max_det", 300))
    process_width = args.process_width if args.process_width is not None else int(profile_cfg.get("process_width", 0) or 0)
    detection_conf_scale = float(profile_cfg.get("detection_conf_scale", 1.0) or 1.0)

    model = YOLO(args.model)
    try:
        model.to(device)
    except Exception:
        pass
    model_to_internal, ignored, unknown = _build_model_class_map(model)

    deep_tracker = None
    deep_error = None
    try:
        deep_tracker = _create_tracker(deep_cfg, device, half)
    except Exception as exc:
        deep_error = str(exc)

    byte_tracker = _create_tracker(byte_cfg, device, half)

    cap = open_capture(args.video)
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Không đọc được frame đầu.")
    first_frame = resize_keep_aspect_bgr(first_frame, process_width)
    with torch.inference_mode():
        _ = model(first_frame, imgsz=imgsz, conf=MODEL_CONF, iou=MODEL_IOU, max_det=max_det, device=device, half=half, verbose=False)
    cuda_sync_if_needed(device)
    try:
        cap.set(1, 0)
    except Exception:
        pass

    rows = []
    yolo_ms = []
    map_filter_ms = []
    suppress_ms = []
    deep_ms = []
    byte_ms = []
    det_counts = []
    deep_track_counts = []
    byte_track_counts = []

    total_start = time.perf_counter()
    for idx in range(1, args.frames + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect_bgr(frame, process_width)

        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            results = model(frame, imgsz=imgsz, conf=MODEL_CONF, iou=MODEL_IOU, max_det=max_det, device=device, half=half, verbose=False)
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()

        frame_h, frame_w = frame.shape[:2]
        detections = []
        raw = 0
        mapped = 0
        t2 = time.perf_counter()
        for result in results:
            for box in result.boxes:
                raw += 1
                model_cls = int(box.cls[0])
                cls = model_to_internal.get(model_cls)
                if cls is None:
                    continue
                mapped += 1
                if (cls not in COUNTED_CLASS_IDS) and not args.track_person:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(0, min(x2, frame_w - 1))
                y2 = max(0, min(y2, frame_h - 1))
                if not _passes_detection_filters(cls, conf, x1, y1, x2, y2, frame_w, frame_h, conf_scale=detection_conf_scale):
                    continue
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        t3 = time.perf_counter()
        detections = _suppress_duplicate_detections(detections)
        t4 = time.perf_counter()

        deep_tracks = []
        deep_update_ms = None
        if deep_tracker is not None:
            td0 = time.perf_counter()
            deep_tracks = deep_tracker.update_tracks(detections, frame=frame)
            td1 = time.perf_counter()
            deep_update_ms = (td1 - td0) * 1000.0
            deep_ms.append(deep_update_ms)
            deep_track_counts.append(len(deep_tracks))

        tb0 = time.perf_counter()
        byte_tracks = byte_tracker.update_tracks(detections, frame=frame)
        tb1 = time.perf_counter()
        byte_update_ms = (tb1 - tb0) * 1000.0
        byte_ms.append(byte_update_ms)
        byte_track_counts.append(len(byte_tracks))

        yolo_ms.append((t1 - t0) * 1000.0)
        map_filter_ms.append((t3 - t2) * 1000.0)
        suppress_ms.append((t4 - t3) * 1000.0)
        det_counts.append(len(detections))
        rows.append({
            "frame": idx,
            "yolo_ms": yolo_ms[-1],
            "map_filter_ms": map_filter_ms[-1],
            "suppress_ms": suppress_ms[-1],
            "detections": len(detections),
            "raw_boxes": raw,
            "mapped_boxes": mapped,
            "deepsort_ms": deep_update_ms,
            "deepsort_tracks": len(deep_tracks) if deep_tracker is not None else None,
            "bytetrack_lite_ms": byte_update_ms,
            "bytetrack_lite_tracks": len(byte_tracks),
        })

    cap.release()
    elapsed = time.perf_counter() - total_start
    processed = len(rows)
    summary = {
        "video": args.video,
        "model": args.model,
        "profile": args.profile,
        "device": device,
        "half": half,
        "imgsz": imgsz,
        "process_width": process_width,
        "max_det": max_det,
        "detection_conf_scale": detection_conf_scale,
        "deep_error": deep_error,
        "processed_frames": processed,
        "wall_time_s": elapsed,
        "effective_fps_including_yolo": processed / elapsed if elapsed > 0 else 0.0,
        "avg_detections": sum(det_counts) / len(det_counts) if det_counts else 0.0,
        "avg_deepsort_tracks": sum(deep_track_counts) / len(deep_track_counts) if deep_track_counts else 0.0,
        "avg_bytetrack_lite_tracks": sum(byte_track_counts) / len(byte_track_counts) if byte_track_counts else 0.0,
    }
    for name, values in [
        ("yolo", yolo_ms),
        ("map_filter", map_filter_ms),
        ("suppress", suppress_ms),
        ("deepsort", deep_ms),
        ("bytetrack_lite", byte_ms),
    ]:
        summary.update(summarize(values, name))

    if summary.get("deepsort_avg_ms") and summary.get("bytetrack_lite_avg_ms"):
        b = summary["bytetrack_lite_avg_ms"]
        d = summary["deepsort_avg_ms"]
        summary["tracker_speedup_ratio"] = d / b if b > 0 else None

    out_dir = ensure_output_dir(args.output_dir)
    save_csv(out_dir / "08_tracker_compare.csv", rows)
    save_json(out_dir / "08_tracker_compare.json", summary)
    print_summary_table(
        "DEEPSORT vs BYTETRACK_LITE",
        summary,
        [
            "device", "half", "imgsz", "process_width", "processed_frames", "avg_detections",
            "yolo_avg_ms", "deepsort_avg_ms", "deepsort_p95_ms",
            "bytetrack_lite_avg_ms", "bytetrack_lite_p95_ms", "tracker_speedup_ratio",
            "effective_fps_including_yolo", "deep_error",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '08_tracker_compare.csv'} và {out_dir / '08_tracker_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
