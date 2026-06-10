from __future__ import annotations

import argparse
from pathlib import Path

from common import PhaseTimer, ResourceMonitor, infer_device_string, iter_video_frames, perf_ms, print_table, summarize_ms, sync_cuda_if_needed, try_import_ultralytics_yolo, write_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: measure YOLO detect speed including Ultralytics preprocess/inference/postprocess.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--save-detections", action="store_true", help="Save boxes to 03_detections.csv for later region/render tests.")
    ap.add_argument("--reports", default="bottleneck_reports")
    args = ap.parse_args()

    model = try_import_ultralytics_yolo(args.model)
    device = infer_device_string(args.device)

    timer = PhaseTimer()
    det_rows = []
    seen = 0
    with ResourceMonitor() as mon:
        for frame_idx, frame in iter_video_frames(args.video, args.max_frames + args.warmup, args.start_frame, args.stride):
            sync_cuda_if_needed(device)
            t0 = perf_ms()
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device, half=args.half, verbose=False)
            sync_cuda_if_needed(device)
            t1 = perf_ms()
            res = results[0]
            speed = getattr(res, "speed", {}) or {}
            n_boxes = int(len(res.boxes)) if getattr(res, "boxes", None) is not None else 0
            if seen >= args.warmup:
                timer.add(
                    frame=frame_idx,
                    yolo_total_ms=t1 - t0,
                    yolo_preprocess_ms=float(speed.get("preprocess", 0.0)),
                    yolo_inference_ms=float(speed.get("inference", 0.0)),
                    yolo_postprocess_ms=float(speed.get("postprocess", 0.0)),
                    boxes=n_boxes,
                )
                if args.save_detections and n_boxes > 0:
                    xyxy = res.boxes.xyxy.detach().cpu().numpy()
                    conf = res.boxes.conf.detach().cpu().numpy()
                    cls = res.boxes.cls.detach().cpu().numpy()
                    for i, box in enumerate(xyxy):
                        det_rows.append({
                            "frame": frame_idx,
                            "det_id": i,
                            "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                            "conf": float(conf[i]), "cls": int(cls[i]),
                        })
            seen += 1
        mon.save(args.reports, "03_resource_samples")

    csv_path, json_path = timer.save(args.reports, "03_yolo_detect_speed", vars(args))
    if args.save_detections:
        write_csv(Path(args.reports) / "03_detections.csv", det_rows)
    s = summarize_ms([float(r["yolo_total_ms"]) for r in timer.rows])
    print_table("Phase 3 - YOLO detect", [
        ("frames", len(timer.rows)),
        ("mean total ms", s["mean"]),
        ("p95 total ms", s["p95"]),
        ("phase fps", 1000.0 / s["mean"] if s["mean"] > 0 else 0.0),
        ("csv", str(csv_path)),
        ("summary", str(json_path)),
    ])


if __name__ == "__main__":
    main()
