from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Tuple

from common import PhaseTimer, ResourceMonitor, infer_device_string, iter_video_frames, perf_ms, print_table, summarize_ms, sync_cuda_if_needed, try_import_ultralytics_yolo, write_csv
from tracking_utils import MiniCentroidTracker


def build_tracker(kind: str, device: str, half: bool):
    if kind in {"auto", "deepsort"}:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
            tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.2,
                embedder="mobilenet",
                half=half,
                bgr=True,
                embedder_gpu=(device != "cpu"),
            )
            return "deepsort_realtime", tracker
        except Exception as e:
            if kind == "deepsort":
                raise RuntimeError("deep_sort_realtime is not installed or failed to initialize.") from e
            print(f"DeepSORT not available ({e}); using MiniCentroidTracker fallback.")
    if kind == "none":
        return "none", None
    return "centroid", MiniCentroidTracker()


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4: measure YOLO + tracker cost.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
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
    ap.add_argument("--save-tracks", action="store_true", help="Save 04_tracks.csv for region/render tests.")
    ap.add_argument("--reports", default="bottleneck_reports")
    args = ap.parse_args()

    model = try_import_ultralytics_yolo(args.model)
    device = infer_device_string(args.device)
    tracker_name, tracker = build_tracker(args.tracker, device, args.half)

    timer = PhaseTimer()
    track_rows = []
    seen = 0
    with ResourceMonitor() as mon:
        for frame_idx, frame in iter_video_frames(args.video, args.max_frames + args.warmup, args.start_frame, args.stride):
            sync_cuda_if_needed(device)
            d0 = perf_ms()
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device, half=args.half, verbose=False)
            sync_cuda_if_needed(device)
            d1 = perf_ms()
            res = results[0]
            speed = getattr(res, "speed", {}) or {}
            detections_xyxy: List[Tuple[Tuple[float, float, float, float], float, int]] = []
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                confs = res.boxes.conf.detach().cpu().numpy()
                clss = res.boxes.cls.detach().cpu().numpy()
                for i, box in enumerate(xyxy):
                    detections_xyxy.append(((float(box[0]), float(box[1]), float(box[2]), float(box[3])), float(confs[i]), int(clss[i])))

            tr0 = perf_ms()
            tracks_out = []
            if tracker_name == "deepsort_realtime":
                ds_dets = []
                for box, conf, cls in detections_xyxy:
                    x1, y1, x2, y2 = box
                    ds_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, str(cls)))
                tracks = tracker.update_tracks(ds_dets, frame=frame)  # type: ignore[union-attr]
                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    l, ttop, r, b = t.to_ltrb()
                    cls_raw = t.det_class if hasattr(t, "det_class") else -1
                    try:
                        cls_i = int(cls_raw)
                    except Exception:
                        cls_i = -1
                    tracks_out.append((int(t.track_id), (float(l), float(ttop), float(r), float(b)), cls_i, float(getattr(t, "det_conf", 0.0) or 0.0)))
            elif tracker_name == "centroid":
                tracks = tracker.update(detections_xyxy)  # type: ignore[union-attr]
                for t in tracks:
                    tracks_out.append((t.tid, t.bbox, t.cls, t.conf))
            else:
                for i, (box, conf, cls) in enumerate(detections_xyxy):
                    tracks_out.append((i, box, cls, conf))
            tr1 = perf_ms()

            if seen >= args.warmup:
                timer.add(
                    frame=frame_idx,
                    yolo_total_ms=d1 - d0,
                    yolo_preprocess_ms=float(speed.get("preprocess", 0.0)),
                    yolo_inference_ms=float(speed.get("inference", 0.0)),
                    yolo_postprocess_ms=float(speed.get("postprocess", 0.0)),
                    tracker_ms=tr1 - tr0,
                    total_detect_track_ms=(d1 - d0) + (tr1 - tr0),
                    detections=len(detections_xyxy),
                    tracks=len(tracks_out),
                    tracker_name=tracker_name,
                )
                if args.save_tracks:
                    for tid, box, cls, conf in tracks_out:
                        track_rows.append({
                            "frame": frame_idx,
                            "track_id": tid,
                            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                            "conf": conf, "cls": cls,
                        })
            seen += 1
        mon.save(args.reports, "04_resource_samples")

    csv_path, json_path = timer.save(args.reports, "04_yolo_plus_tracker_speed", {**vars(args), "tracker_name": tracker_name})
    if args.save_tracks:
        write_csv(Path(args.reports) / "04_tracks.csv", track_rows)
    s = summarize_ms([float(r["total_detect_track_ms"]) for r in timer.rows])
    ts = summarize_ms([float(r["tracker_ms"]) for r in timer.rows])
    print_table("Phase 4 - YOLO + tracker", [
        ("tracker", tracker_name),
        ("frames", len(timer.rows)),
        ("mean total ms", s["mean"]),
        ("p95 total ms", s["p95"]),
        ("mean tracker ms", ts["mean"]),
        ("p95 tracker ms", ts["p95"]),
        ("phase fps", 1000.0 / s["mean"] if s["mean"] > 0 else 0.0),
        ("csv", str(csv_path)),
        ("summary", str(json_path)),
    ])


if __name__ == "__main__":
    main()
