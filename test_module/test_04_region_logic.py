from __future__ import annotations

import argparse
import random
import time

from common import ensure_output_dir, open_capture, get_capture_info, save_csv, save_json, summarize, print_summary_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark riêng logic xác định region/polygon, không chạy YOLO/DeepSORT.")
    parser.add_argument("--video", required=True, help="Dùng để lấy kích thước frame.")
    parser.add_argument("--template", default="template.csv", help="template.csv 8 vùng/làn.")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--points-per-frame", type=int, default=300, help="Số điểm giả lập mỗi frame.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from vehicle_flow.regions import RegionTemplate, get_direction_region

    cap = open_capture(args.video)
    info = get_capture_info(cap)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Không đọc được frame đầu.")
    height, width = frame.shape[:2]

    t_load0 = time.perf_counter()
    template = RegionTemplate(args.template)
    t_load1 = time.perf_counter()

    rng = random.Random(1234)
    rows = []
    region_ms = []
    counts = {}
    prev = None
    current_region = None

    for frame_id in range(1, args.frames + 1):
        points = [(rng.randint(0, width - 1), rng.randint(0, height - 1)) for _ in range(args.points_per_frame)]
        t0 = time.perf_counter()
        for point in points:
            region = get_direction_region(
                point,
                width,
                height,
                0.25,
                template=template,
                previous_centroid=prev,
                current_region=current_region,
            )
            counts[region or "none"] = counts.get(region or "none", 0) + 1
            prev = point
            current_region = region
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        region_ms.append(ms)
        rows.append({
            "frame": frame_id,
            "points": args.points_per_frame,
            "region_total_ms": ms,
            "region_ms_per_point": ms / max(1, args.points_per_frame),
        })

    summary = {
        "video": args.video,
        "template": args.template,
        "template_loaded": bool(template.loaded),
        "template_regions": list(template.regions.keys()),
        "frame_width": width,
        "frame_height": height,
        "source_info": info,
        "frames": args.frames,
        "points_per_frame": args.points_per_frame,
        "template_load_ms": (t_load1 - t_load0) * 1000.0,
        "region_counts": counts,
        "avg_region_ms_per_point": (sum(region_ms) / len(region_ms) / max(1, args.points_per_frame)) if region_ms else 0.0,
    }
    summary.update(summarize(region_ms, "region_total"))

    out_dir = ensure_output_dir(args.output_dir)
    save_csv(out_dir / "04_region_logic.csv", rows)
    save_json(out_dir / "04_region_logic.json", summary)
    print_summary_table(
        "REGION LOGIC",
        summary,
        [
            "template_loaded", "template_regions", "frame_width", "frame_height",
            "frames", "points_per_frame", "template_load_ms", "region_total_avg_ms",
            "region_total_p95_ms", "avg_region_ms_per_point", "region_counts",
        ],
    )
    print(f"\nĐã lưu: {out_dir / '04_region_logic.csv'} và {out_dir / '04_region_logic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
