from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore

from common import PhaseTimer, ResourceMonitor, infer_device_string, iter_video_frames, perf_ms, print_table, summarize_ms, sync_cuda_if_needed, try_import_ultralytics_yolo
from region_utils import classify_region, load_regions
from tracking_utils import MiniCentroidTracker
from importlib import import_module


def build_tracker(kind: str, device: str, half: bool):
    if kind in {"auto", "deepsort"}:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
            return "deepsort_realtime", DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, embedder="mobilenet", half=half, bgr=True, embedder_gpu=(device != "cpu"))
        except Exception as e:
            if kind == "deepsort":
                raise
            print(f"DeepSORT unavailable ({e}); using centroid fallback.")
    if kind == "none":
        return "none", None
    return "centroid", MiniCentroidTracker()


def draw_basic(frame, tracks_out):
    for tid, box, cls, conf in tracks_out:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, f"{tid}:{cls}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end probe: read -> YOLO -> tracker -> region -> optional render.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", default=None)
    ap.add_argument("--tracker", default="auto", choices=["auto", "deepsort", "centroid", "none"])
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--render", action="store_true", help="Include cv2 drawing overhead.")
    ap.add_argument("--reports", default="bottleneck_reports")
    args = ap.parse_args()

    model = try_import_ultralytics_yolo(args.model)
    device = infer_device_string(args.device)
    tracker_name, tracker = build_tracker(args.tracker, device, args.half)

    # Region template will be loaded after first frame size is known.
    regions = None
    prev_region: Dict[int, str] = {}
    timer = PhaseTimer()
    seen = 0

    with ResourceMonitor() as mon:
        for frame_idx, frame in iter_video_frames(args.video, args.max_frames + args.warmup, args.start_frame, args.stride):
            if regions is None:
                h, w = frame.shape[:2]
                regions = load_regions(args.template, w, h)
            row = {"frame": frame_idx}
            total0 = perf_ms()

            sync_cuda_if_needed(device)
            d0 = perf_ms()
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device, half=args.half, verbose=False)
            sync_cuda_if_needed(device)
            d1 = perf_ms()
            res = results[0]
            speed = getattr(res, "speed", {}) or {}
            detections = []
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                confs = res.boxes.conf.detach().cpu().numpy()
                clss = res.boxes.cls.detach().cpu().numpy()
                for i, b in enumerate(xyxy):
                    detections.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(confs[i]), int(clss[i])))

            tr0 = perf_ms()
            tracks_out = []
            if tracker_name == "deepsort_realtime":
                ds_dets = []
                for box, conf, cls in detections:
                    x1, y1, x2, y2 = box
                    ds_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, str(cls)))
                tracks = tracker.update_tracks(ds_dets, frame=frame)  # type: ignore[union-attr]
                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    box = tuple(float(v) for v in t.to_ltrb())
                    try:
                        cls = int(t.det_class)
                    except Exception:
                        cls = -1
                    tracks_out.append((int(t.track_id), box, cls, float(getattr(t, "det_conf", 0.0) or 0.0)))
            elif tracker_name == "centroid":
                tracks = tracker.update(detections)  # type: ignore[union-attr]
                for t in tracks:
                    tracks_out.append((t.tid, t.bbox, t.cls, t.conf))
            else:
                for i, (box, conf, cls) in enumerate(detections):
                    tracks_out.append((i, box, cls, conf))
            tr1 = perf_ms()

            rg0 = perf_ms()
            transitions = 0
            for tid, box, cls, conf in tracks_out:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                region = classify_region(cx, cy, regions or [])
                if region:
                    old = prev_region.get(tid)
                    if old and old != region:
                        transitions += 1
                    prev_region[tid] = region
            rg1 = perf_ms()

            rd0 = perf_ms()
            if args.render:
                _ = draw_basic(frame.copy(), tracks_out)
            rd1 = perf_ms()

            if seen >= args.warmup:
                total_ms = perf_ms() - total0
                row.update({
                    "yolo_total_ms": d1 - d0,
                    "yolo_preprocess_ms": float(speed.get("preprocess", 0.0)),
                    "yolo_inference_ms": float(speed.get("inference", 0.0)),
                    "yolo_postprocess_ms": float(speed.get("postprocess", 0.0)),
                    "tracker_ms": tr1 - tr0,
                    "region_logic_ms": rg1 - rg0,
                    "render_ms": rd1 - rd0,
                    "total_pipeline_ms": total_ms,
                    "detections": len(detections),
                    "tracks": len(tracks_out),
                    "transitions": transitions,
                    "tracker_name": tracker_name,
                })
                timer.add(**row)
            seen += 1
        mon.save(args.reports, "07_resource_samples")

    csv_path, json_path = timer.save(args.reports, "07_full_pipeline_probe", {**vars(args), "tracker_name": tracker_name})
    s = summarize_ms([float(r["total_pipeline_ms"]) for r in timer.rows])
    print_table("Phase 7 - Full pipeline probe", [
        ("tracker", tracker_name),
        ("render included", args.render),
        ("frames", len(timer.rows)),
        ("mean total ms", s["mean"]),
        ("p95 total ms", s["p95"]),
        ("estimated fps", 1000.0 / s["mean"] if s["mean"] > 0 else 0.0),
        ("csv", str(csv_path)),
        ("summary", str(json_path)),
    ])


if __name__ == "__main__":
    main()
