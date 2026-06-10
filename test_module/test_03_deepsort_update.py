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


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark YOLO -> format bbox -> DeepSORT update_tracks.")
    add_common_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--track-person", action="store_true", help="Đưa person/pedestrian vào DeepSORT để so sánh bottleneck.")
    parser.add_argument("--no-half", action="store_true")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO
    from vehicle_flow.config import COUNTED_CLASS_IDS, MODEL_CONF, MODEL_IOU
    from vehicle_flow.video_worker import (
        _build_model_class_map,
        _create_deepsort_tracker,
        _passes_detection_filters,
        _suppress_duplicate_detections,
    )

    profile_cfg = get_profile_config(args.profile)
    device = choose_device()
    half = bool(profile_cfg.get("half_cuda", True)) and device.startswith("cuda") and not args.no_half
    imgsz = args.imgsz if args.imgsz is not None else int(profile_cfg.get("model_imgsz", 640))
    max_det = args.max_det if args.max_det is not None else int(profile_cfg.get("max_det", 300))
    process_width = args.process_width if args.process_width is not None else int(profile_cfg.get("process_width", 0) or 0)

    model = YOLO(args.model)
    try:
        model.to(device)
    except Exception:
        pass
    model_to_internal, ignored, unknown = _build_model_class_map(model)
    tracker = _create_deepsort_tracker(profile_cfg, device, half)

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
    read_ms = []
    resize_ms = []
    yolo_ms = []
    map_filter_ms = []
    suppress_ms = []
    deepsort_ms = []
    raw_counts = []
    mapped_counts = []
    det_counts = []
    track_counts = []

    total_start = time.perf_counter()
    for idx in range(1, args.frames + 1):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        frame = resize_keep_aspect_bgr(frame, process_width)
        t2 = time.perf_counter()

        cuda_sync_if_needed(device)
        t3 = time.perf_counter()
        with torch.inference_mode():
            results = model(frame, imgsz=imgsz, conf=MODEL_CONF, iou=MODEL_IOU, max_det=max_det, device=device, half=half, verbose=False)
        cuda_sync_if_needed(device)
        t4 = time.perf_counter()

        frame_h, frame_w = frame.shape[:2]
        detections = []
        raw = 0
        mapped = 0
        t5 = time.perf_counter()
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
                if not _passes_detection_filters(cls, conf, x1, y1, x2, y2, frame_w, frame_h):
                    continue
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        t6 = time.perf_counter()
        detections = _suppress_duplicate_detections(detections)
        t7 = time.perf_counter()
        tracks = tracker.update_tracks(detections, frame=frame)
        t8 = time.perf_counter()

        r_ms = (t1 - t0) * 1000.0
        z_ms = (t2 - t1) * 1000.0
        y_ms = (t4 - t3) * 1000.0
        m_ms = (t6 - t5) * 1000.0
        s_ms = (t7 - t6) * 1000.0
        d_ms = (t8 - t7) * 1000.0
        read_ms.append(r_ms)
        resize_ms.append(z_ms)
        yolo_ms.append(y_ms)
        map_filter_ms.append(m_ms)
        suppress_ms.append(s_ms)
        deepsort_ms.append(d_ms)
        raw_counts.append(raw)
        mapped_counts.append(mapped)
        det_counts.append(len(detections))
        track_counts.append(len(tracks))
        rows.append({
            "frame": idx,
            "read_ms": r_ms,
            "resize_ms": z_ms,
            "yolo_ms": y_ms,
            "map_filter_ms": m_ms,
            "suppress_ms": s_ms,
            "deepsort_ms": d_ms,
            "raw_boxes": raw,
            "mapped_boxes": mapped,
            "tracker_detections": len(detections),
            "tracks": len(tracks),
        })

    cap.release()
    elapsed = time.perf_counter() - total_start
    processed = len(rows)
    summary = {
        "video": args.video,
        "model": args.model,
        "model_names": getattr(model, "names", {}),
        "model_to_internal": model_to_internal,
        "ignored_model_classes": ignored,
        "unknown_model_classes": unknown,
        "profile": args.profile,
        "device": device,
        "half": half,
        "imgsz": imgsz,
        "process_width": process_width,
        "max_det": max_det,
        "track_person": args.track_person,
        "processed_frames": processed,
        "wall_time_s": elapsed,
        "effective_fps": processed / elapsed if elapsed > 0 else 0.0,
        "avg_raw_boxes": sum(raw_counts) / len(raw_counts) if raw_counts else 0.0,
        "avg_mapped_boxes": sum(mapped_counts) / len(mapped_counts) if mapped_counts else 0.0,
        "avg_tracker_detections": sum(det_counts) / len(det_counts) if det_counts else 0.0,
        "avg_tracks": sum(track_counts) / len(track_counts) if track_counts else 0.0,
    }
    for name, values in [
        ("read", read_ms),
        ("resize", resize_ms),
        ("yolo", yolo_ms),
        ("map_filter", map_filter_ms),
        ("suppress", suppress_ms),
        ("deepsort", deepsort_ms),
    ]:
        summary.update(summarize(values, name))

    stage_avg = {
        "read": summary.get("read_avg_ms", 0.0),
        "resize": summary.get("resize_avg_ms", 0.0),
        "yolo": summary.get("yolo_avg_ms", 0.0),
        "map_filter": summary.get("map_filter_avg_ms", 0.0),
        "suppress": summary.get("suppress_avg_ms", 0.0),
        "deepsort": summary.get("deepsort_avg_ms", 0.0),
    }
    summary["stage_avg_ms"] = stage_avg

    out_dir = ensure_output_dir(args.output_dir)
    save_csv(out_dir / "03_deepsort_update.csv", rows)
    save_json(out_dir / "03_deepsort_update.json", summary)
    print_summary_table(
        "YOLO + DEEPSORT",
        summary,
        [
            "model_names", "model_to_internal", "ignored_model_classes", "unknown_model_classes",
            "device", "half", "imgsz", "process_width", "track_person", "processed_frames",
            "avg_raw_boxes", "avg_mapped_boxes", "avg_tracker_detections", "avg_tracks",
            "yolo_avg_ms", "yolo_p95_ms", "deepsort_avg_ms", "deepsort_p95_ms", "effective_fps",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '03_deepsort_update.csv'} và {out_dir / '03_deepsort_update.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
