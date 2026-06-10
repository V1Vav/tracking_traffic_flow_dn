from __future__ import annotations

import argparse
import time

from common import (
    add_common_args,
    choose_device,
    cuda_sync_if_needed,
    diagnose_stage,
    ensure_output_dir,
    get_profile_config,
    open_capture,
    resize_keep_aspect_bgr,
    save_csv,
    save_json,
    summarize,
    print_summary_table,
)


def safe_track_ltrb(track):
    try:
        return tuple(map(int, track.to_ltrb(orig=True, orig_strict=False)))
    except TypeError:
        return tuple(map(int, track.to_ltrb()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pipeline gần giống realtime: read -> resize -> YOLO -> DeepSORT -> region -> display.")
    add_common_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", default="template.csv")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--display-width", type=int, default=None)
    parser.add_argument("--display-height", type=int, default=None)
    parser.add_argument("--detect-interval", type=int, default=None)
    parser.add_argument("--display-every-n", type=int, default=None)
    parser.add_argument("--track-person", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-region", action="store_true")
    parser.add_argument("--no-half", action="store_true")
    args = parser.parse_args()

    import cv2
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from vehicle_flow.config import COUNTED_CLASS_IDS, MODEL_CONF, MODEL_IOU, USE_BOTTOM_CENTER_FOR_REGION
    from vehicle_flow.regions import RegionTemplate, draw_region_overlay, get_direction_region, region_point_from_box
    from vehicle_flow.video_worker import (
        _build_model_class_map,
        _create_tracker,
        _passes_detection_filters,
        _resolve_target_process_fps,
        _sanitize_capture_fps,
        _suppress_duplicate_detections,
    )

    profile_cfg = get_profile_config(args.profile)
    device = choose_device()
    half = bool(profile_cfg.get("half_cuda", True)) and device.startswith("cuda") and not args.no_half
    imgsz = args.imgsz if args.imgsz is not None else int(profile_cfg.get("model_imgsz", 640))
    max_det = args.max_det if args.max_det is not None else int(profile_cfg.get("max_det", 300))
    process_width = args.process_width if args.process_width is not None else int(profile_cfg.get("process_width", 0) or 0)
    display_width = args.display_width if args.display_width is not None else int(profile_cfg.get("display_width", 900) or 900)
    display_height = args.display_height if args.display_height is not None else int(profile_cfg.get("display_height", 620) or 620)
    detect_interval = max(1, args.detect_interval if args.detect_interval is not None else int(profile_cfg.get("detect_interval", 1) or 1))
    display_every_n = max(1, args.display_every_n if args.display_every_n is not None else int(profile_cfg.get("display_every_n", 1) or 1))
    target_fps = float(profile_cfg.get("target_process_fps", 30.0) or 30.0)
    source_fps_used = 0.0
    fps_metadata_valid = False

    model = YOLO(args.model)
    try:
        model.to(device)
    except Exception:
        pass
    model_to_internal, ignored, unknown = _build_model_class_map(model)
    tracker = _create_tracker(profile_cfg, device, half)
    template = None if args.no_region else RegionTemplate(args.template)

    cap = open_capture(args.video)
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_fps_used, fps_metadata_valid = _sanitize_capture_fps(raw_fps, fallback=30.0)
    target_fps, _ = _resolve_target_process_fps(source_fps_used, profile_cfg)
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

    last_detections = []
    rows = []
    timers = {name: [] for name in [
        "read", "resize", "yolo", "map_filter", "suppress", "tracker", "region", "draw", "display_convert", "total"
    ]}
    counts = []
    track_region_state = {}

    total_start = time.perf_counter()
    for frame_id in range(1, args.frames + 1):
        frame_t0 = time.perf_counter()
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        frame = resize_keep_aspect_bgr(frame, process_width)
        t2 = time.perf_counter()

        detections = []
        raw = 0
        mapped = 0
        if frame_id % detect_interval == 0:
            cuda_sync_if_needed(device)
            t3 = time.perf_counter()
            with torch.inference_mode():
                results = model(frame, imgsz=imgsz, conf=MODEL_CONF, iou=MODEL_IOU, max_det=max_det, device=device, half=half, verbose=False)
            cuda_sync_if_needed(device)
            t4 = time.perf_counter()

            frame_h, frame_w = frame.shape[:2]
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
                    if not _passes_detection_filters(cls, conf, x1, y1, x2, y2, frame_w, frame_h, conf_scale=float(profile_cfg.get("detection_conf_scale", 1.0) or 1.0)):
                        continue
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
            t6 = time.perf_counter()
            detections = _suppress_duplicate_detections(detections)
            t7 = time.perf_counter()
            last_detections = detections
            timers["yolo"].append((t4 - t3) * 1000.0)
            timers["map_filter"].append((t6 - t5) * 1000.0)
            timers["suppress"].append((t7 - t6) * 1000.0)
        else:
            detections = []
            # Tracker vẫn nhận [] để predict, đúng hành vi khi bỏ detect frame.
            timers["yolo"].append(0.0)
            timers["map_filter"].append(0.0)
            timers["suppress"].append(0.0)

        t8 = time.perf_counter()
        tracks = tracker.update_tracks(detections, frame=frame)
        t9 = time.perf_counter()

        t10 = time.perf_counter()
        active_tracks = 0
        regions_checked = 0
        h, w = frame.shape[:2]
        for track in tracks:
            try:
                if not track.is_confirmed():
                    continue
            except Exception:
                pass
            box_xyxy = safe_track_ltrb(track)
            point = region_point_from_box(box_xyxy, use_bottom_center=USE_BOTTOM_CENTER_FOR_REGION)
            track_id = getattr(track, "track_id", None)
            prev_point, prev_region = track_region_state.get(track_id, (None, None))
            region = get_direction_region(point, w, h, 0.25, template=template, previous_centroid=prev_point, current_region=prev_region)
            track_region_state[track_id] = (point, region)
            active_tracks += 1
            regions_checked += 1
        t11 = time.perf_counter()

        should_display = (not args.no_display) and (frame_id % display_every_n == 0)
        t12 = time.perf_counter()
        if should_display:
            if not args.no_region:
                draw_region_overlay(frame, 0.25, template=template)
            for track in tracks:
                box_xyxy = safe_track_ltrb(track)
                x1, y1, x2, y2 = box_xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        t13 = time.perf_counter()
        if should_display:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src_h, src_w = rgb.shape[:2]
            scale = min(display_width / max(1, src_w), display_height / max(1, src_h))
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            _ = Image.fromarray(rgb)
        t14 = time.perf_counter()

        frame_t1 = time.perf_counter()
        timers["read"].append((t1 - t0) * 1000.0)
        timers["resize"].append((t2 - t1) * 1000.0)
        timers["tracker"].append((t9 - t8) * 1000.0)
        timers["region"].append((t11 - t10) * 1000.0)
        timers["draw"].append((t13 - t12) * 1000.0)
        timers["display_convert"].append((t14 - t13) * 1000.0)
        timers["total"].append((frame_t1 - frame_t0) * 1000.0)
        counts.append((raw, mapped, len(last_detections), len(tracks), active_tracks, regions_checked))
        rows.append({
            "frame": frame_id,
            "read_ms": timers["read"][-1],
            "resize_ms": timers["resize"][-1],
            "yolo_ms": timers["yolo"][-1],
            "map_filter_ms": timers["map_filter"][-1],
            "suppress_ms": timers["suppress"][-1],
            "tracker_ms": timers["tracker"][-1],
            "region_ms": timers["region"][-1],
            "draw_ms": timers["draw"][-1],
            "display_convert_ms": timers["display_convert"][-1],
            "total_ms": timers["total"][-1],
            "raw_boxes": raw,
            "mapped_boxes": mapped,
            "detections": len(last_detections),
            "tracks": len(tracks),
            "active_tracks": active_tracks,
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
        "detect_interval": detect_interval,
        "display_every_n": display_every_n,
        "source_fps_used": source_fps_used,
        "fps_metadata_valid": fps_metadata_valid,
        "target_process_fps": target_fps,
        "track_person": args.track_person,
        "display_enabled": not args.no_display,
        "region_enabled": not args.no_region,
        "processed_frames": processed,
        "wall_time_s": elapsed,
        "effective_fps": processed / elapsed if elapsed > 0 else 0.0,
        "tracker_type": str(profile_cfg.get("tracker_type", "deepsort")),
    }
    for name, values in timers.items():
        summary.update(summarize(values, name))
    if counts:
        summary["avg_raw_boxes"] = sum(v[0] for v in counts) / len(counts)
        summary["avg_mapped_boxes"] = sum(v[1] for v in counts) / len(counts)
        summary["avg_detections"] = sum(v[2] for v in counts) / len(counts)
        summary["avg_tracks"] = sum(v[3] for v in counts) / len(counts)
        summary["avg_active_tracks"] = sum(v[4] for v in counts) / len(counts)

    stage_avg = {
        "read": summary.get("read_avg_ms", 0.0),
        "resize": summary.get("resize_avg_ms", 0.0),
        "yolo": summary.get("yolo_avg_ms", 0.0),
        "map_filter": summary.get("map_filter_avg_ms", 0.0),
        "suppress": summary.get("suppress_avg_ms", 0.0),
        "tracker": summary.get("tracker_avg_ms", 0.0),
        "region": summary.get("region_avg_ms", 0.0),
        "draw": summary.get("draw_avg_ms", 0.0),
        "display_convert": summary.get("display_convert_avg_ms", 0.0),
    }
    summary["stage_avg_ms"] = stage_avg
    summary["diagnosis"] = diagnose_stage(stage_avg, target_fps)

    out_dir = ensure_output_dir(args.output_dir)
    save_csv(out_dir / "06_full_pipeline_profile.csv", rows)
    save_json(out_dir / "06_full_pipeline_profile.json", summary)
    print_summary_table(
        "FULL PIPELINE PROFILE",
        summary,
        [
            "model_names", "model_to_internal", "device", "half", "imgsz", "process_width",
            "detect_interval", "display_every_n", "source_fps_used", "target_process_fps", "tracker_type", "processed_frames",
            "avg_raw_boxes", "avg_detections", "avg_tracks", "avg_active_tracks",
            "read_avg_ms", "resize_avg_ms", "yolo_avg_ms", "tracker_avg_ms",
            "region_avg_ms", "draw_avg_ms", "display_convert_avg_ms", "total_avg_ms",
            "total_p95_ms", "effective_fps", "diagnosis",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '06_full_pipeline_profile.csv'} và {out_dir / '06_full_pipeline_profile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
