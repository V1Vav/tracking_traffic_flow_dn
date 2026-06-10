from __future__ import annotations

import argparse
import time

from common import (
    add_common_args,
    choose_device,
    cuda_sync_if_needed,
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


def parse_classes(text):
    if text is None or str(text).strip() == "":
        return None
    if str(text).strip().lower() in {"none", "all"}:
        return None
    return [int(x.strip()) for x in str(text).split(",") if x.strip() != ""]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark riêng YOLO inference, chưa chạy DeepSORT.")
    add_common_args(parser)
    parser.add_argument("--model", required=True, help="Đường dẫn model YOLO .pt")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--classes", default="all", help="VD: 0,1,3,5 hoặc all. Nên để all để tránh lọc sai class.")
    parser.add_argument("--no-half", action="store_true")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO
    from vehicle_flow.config import MODEL_CONF, MODEL_IOU

    profile_cfg = get_profile_config(args.profile)
    device = choose_device()
    half = bool(profile_cfg.get("half_cuda", True)) and device.startswith("cuda") and not args.no_half
    imgsz = args.imgsz if args.imgsz is not None else int(profile_cfg.get("model_imgsz", 640))
    max_det = args.max_det if args.max_det is not None else int(profile_cfg.get("max_det", 300))
    conf = args.conf if args.conf is not None else float(MODEL_CONF)
    iou = args.iou if args.iou is not None else float(MODEL_IOU)
    process_width = args.process_width if args.process_width is not None else int(profile_cfg.get("process_width", 0) or 0)
    classes = parse_classes(args.classes)

    model = YOLO(args.model)
    try:
        model.to(device)
    except Exception:
        pass

    cap = open_capture(args.video)
    cap_info = get_capture_info(cap)

    rows = []
    yolo_ms = []
    read_ms = []
    resize_ms = []
    raw_boxes = []

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Không đọc được frame đầu.")
    first_frame = resize_keep_aspect_bgr(first_frame, process_width)
    # Warmup để không tính CUDA kernel init vào benchmark.
    with torch.inference_mode():
        _ = model(first_frame, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, half=half, classes=classes, verbose=False)
    cuda_sync_if_needed(device)

    # Quay lại đầu file nếu là file; camera có thể không seek được, không sao.
    try:
        cap.set(1, 0)
    except Exception:
        pass

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
            results = model(frame, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, half=half, classes=classes, verbose=False)
        cuda_sync_if_needed(device)
        t4 = time.perf_counter()
        box_count = sum(len(r.boxes) for r in results)

        r_ms = (t1 - t0) * 1000.0
        z_ms = (t2 - t1) * 1000.0
        y_ms = (t4 - t3) * 1000.0
        read_ms.append(r_ms)
        resize_ms.append(z_ms)
        yolo_ms.append(y_ms)
        raw_boxes.append(box_count)
        rows.append({
            "frame": idx,
            "read_ms": r_ms,
            "resize_ms": z_ms,
            "yolo_ms": y_ms,
            "raw_boxes": box_count,
        })

    cap.release()
    elapsed = time.perf_counter() - total_start
    processed = len(rows)
    summary = {
        "video": args.video,
        "model": args.model,
        "model_names": getattr(model, "names", {}),
        "profile": args.profile,
        "device": device,
        "half": half,
        "imgsz": imgsz,
        "process_width": process_width,
        "max_det": max_det,
        "conf": conf,
        "iou": iou,
        "classes": classes if classes is not None else "all",
        "source_info": cap_info,
        "processed_frames": processed,
        "wall_time_s": elapsed,
        "effective_fps_including_read_resize": processed / elapsed if elapsed > 0 else 0.0,
        "avg_raw_boxes": sum(raw_boxes) / len(raw_boxes) if raw_boxes else 0.0,
        "max_raw_boxes": max(raw_boxes) if raw_boxes else 0,
    }
    summary.update(summarize(read_ms, "read"))
    summary.update(summarize(resize_ms, "resize"))
    summary.update(summarize(yolo_ms, "yolo"))
    summary["yolo_only_fps_avg"] = 1000.0 / summary["yolo_avg_ms"] if summary.get("yolo_avg_ms", 0) > 0 else 0.0

    out_dir = ensure_output_dir(args.output_dir)
    save_csv(out_dir / "02_yolo_inference.csv", rows)
    save_json(out_dir / "02_yolo_inference.json", summary)
    print_summary_table(
        "YOLO INFERENCE",
        summary,
        [
            "model_names", "device", "half", "imgsz", "process_width", "max_det", "classes",
            "processed_frames", "avg_raw_boxes", "effective_fps_including_read_resize",
            "yolo_avg_ms", "yolo_p95_ms", "yolo_only_fps_avg",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '02_yolo_inference.csv'} và {out_dir / '02_yolo_inference.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
