from __future__ import annotations

import argparse
import random
import time

from common import (
    add_common_args,
    ensure_output_dir,
    get_profile_config,
    open_capture,
    resize_keep_aspect_bgr,
    save_csv,
    save_json,
    summarize,
    print_summary_table,
)


def make_fake_boxes(width: int, height: int, count: int, seed: int):
    rng = random.Random(seed)
    boxes = []
    for _ in range(count):
        w = rng.randint(max(12, width // 45), max(28, width // 12))
        h = rng.randint(max(12, height // 45), max(28, height // 10))
        x1 = rng.randint(0, max(1, width - w - 1))
        y1 = rng.randint(0, max(1, height - h - 1))
        boxes.append((x1, y1, x1 + w, y1 + h, rng.choice([0, 1, 2, 4])))
    return boxes


def resize_keep_aspect_rgb(rgb_frame, target_width, target_height):
    import cv2
    import numpy as np

    h, w = rgb_frame.shape[:2]
    if target_width <= 0 or target_height <= 0 or w <= 0 or h <= 0:
        return rgb_frame
    scale = min(target_width / w, target_height / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb_frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((target_height, target_width, 3), (2, 6, 23), dtype=rgb_frame.dtype)
    x0 = (target_width - nw) // 2
    y0 = (target_height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark riêng vẽ overlay/box, BGR->RGB, resize preview, PIL conversion.")
    add_common_args(parser)
    parser.add_argument("--template", default="template.csv")
    parser.add_argument("--fake-boxes", type=int, default=100, help="Số bbox giả để vẽ mỗi frame.")
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--display-width", type=int, default=None)
    parser.add_argument("--display-height", type=int, default=None)
    parser.add_argument("--no-region", action="store_true")
    parser.add_argument("--no-pil", action="store_true", help="Không tạo PIL Image để đo riêng OpenCV draw/convert.")
    args = parser.parse_args()

    import cv2
    from PIL import Image
    from vehicle_flow.config import CLASS_COLORS
    from vehicle_flow.regions import RegionTemplate, draw_region_overlay

    profile_cfg = get_profile_config(args.profile)
    process_width = args.process_width if args.process_width is not None else int(profile_cfg.get("process_width", 0) or 0)
    display_width = args.display_width if args.display_width is not None else int(profile_cfg.get("display_width", 900) or 900)
    display_height = args.display_height if args.display_height is not None else int(profile_cfg.get("display_height", 620) or 620)

    template = None if args.no_region else RegionTemplate(args.template)
    cap = open_capture(args.video)

    rows = []
    read_ms = []
    resize_ms = []
    draw_ms = []
    convert_ms = []
    pil_ms = []

    total_start = time.perf_counter()
    for idx in range(1, args.frames + 1):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        frame = resize_keep_aspect_bgr(frame, process_width)
        t2 = time.perf_counter()
        h, w = frame.shape[:2]
        boxes = make_fake_boxes(w, h, args.fake_boxes, seed=idx)

        t3 = time.perf_counter()
        if not args.no_region:
            draw_region_overlay(frame, 0.25, template=template)
        for x1, y1, x2, y2, cls in boxes:
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(frame, str(cls), (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        t4 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = resize_keep_aspect_rgb(rgb, display_width, display_height)
        t5 = time.perf_counter()
        if not args.no_pil:
            _ = Image.fromarray(rgb)
        t6 = time.perf_counter()

        r_ms = (t1 - t0) * 1000.0
        z_ms = (t2 - t1) * 1000.0
        d_ms = (t4 - t3) * 1000.0
        c_ms = (t5 - t4) * 1000.0
        p_ms = (t6 - t5) * 1000.0
        read_ms.append(r_ms)
        resize_ms.append(z_ms)
        draw_ms.append(d_ms)
        convert_ms.append(c_ms)
        pil_ms.append(p_ms)
        rows.append({
            "frame": idx,
            "read_ms": r_ms,
            "resize_ms": z_ms,
            "draw_ms": d_ms,
            "convert_resize_rgb_ms": c_ms,
            "pil_ms": p_ms,
            "fake_boxes": args.fake_boxes,
        })

    cap.release()
    elapsed = time.perf_counter() - total_start
    processed = len(rows)
    summary = {
        "profile": args.profile,
        "template": args.template,
        "region_overlay": not args.no_region,
        "process_width": process_width,
        "display_width": display_width,
        "display_height": display_height,
        "fake_boxes": args.fake_boxes,
        "processed_frames": processed,
        "wall_time_s": elapsed,
        "effective_fps": processed / elapsed if elapsed > 0 else 0.0,
    }
    for name, values in [
        ("read", read_ms), ("resize", resize_ms), ("draw", draw_ms),
        ("convert_resize_rgb", convert_ms), ("pil", pil_ms),
    ]:
        summary.update(summarize(values, name))

    out_dir = ensure_output_dir(args.output_dir)
    save_csv(out_dir / "05_display_overlay.csv", rows)
    save_json(out_dir / "05_display_overlay.json", summary)
    print_summary_table(
        "DISPLAY / OVERLAY",
        summary,
        [
            "region_overlay", "process_width", "display_width", "display_height", "fake_boxes",
            "processed_frames", "draw_avg_ms", "draw_p95_ms", "convert_resize_rgb_avg_ms",
            "convert_resize_rgb_p95_ms", "pil_avg_ms", "effective_fps",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '05_display_overlay.csv'} và {out_dir / '05_display_overlay.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
