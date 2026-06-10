from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from common import percentile, print_table, summarize_ms, write_json


def load_ms_columns(csv_path: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return out
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if not k.endswith("_ms"):
                    continue
                try:
                    out.setdefault(k, []).append(float(v))
                except Exception:
                    pass
    return out


def recommendation(phase: str, p95: float, fps_target: float) -> str:
    budget = 1000.0 / fps_target
    name = phase.lower()
    if "read" in name or "decode" in name:
        return "Video IO/decode cao: thử video H.264 nhẹ hơn/MJPEG, giảm resolution, dùng CAP_FFMPEG, đặt video trên SSD, hoặc tách thread đọc frame."
    if "preprocess" in name or "to_device" in name or "letterbox" in name:
        return "Preprocess/transfer cao: giảm imgsz, tránh resize lặp, dùng frame contiguous, dùng half/CUDA, gom bước resize đúng một lần."
    if "yolo_inference" in name or "yolo_total" in name:
        return "YOLO là bottleneck: dùng yolov8n/best nhỏ hơn, imgsz 512/416, half=True, TensorRT/ONNX, tăng conf để giảm boxes, hoặc skip frame có kiểm soát."
    if "postprocess" in name:
        return "Postprocess/NMS cao: tăng conf, giới hạn class cần detect, giảm max_det, tránh nhiều object quá nhỏ."
    if "tracker" in name:
        return "Tracker cao: DeepSORT embedder thường nặng; giảm số detection, tăng conf, giảm max_age/n_init, chỉ chạy appearance mỗi N frame hoặc dùng tracker nhẹ hơn."
    if "region" in name or "classify" in name:
        return "Region logic cao bất thường: cache polygon, ưu tiên bounding-box quick check trước point-in-polygon, không duyệt vùng không cần thiết."
    if "render" in name or "draw" in name or "pil" in name or "imshow" in name:
        return "Render/UI cao: giảm overlay text, update UI mỗi 2-3 frame, tránh PIL/ImageTk conversion dư, giữ đúng aspect ratio canvas, không vẽ center."
    if p95 > budget:
        return "P95 vượt frame budget; cần tối ưu phase này hoặc giảm FPS mục tiêu."
    return "Ổn, chưa phải bottleneck chính."


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze bottleneck report CSVs and rank expensive phases.")
    ap.add_argument("--reports", default="bottleneck_reports")
    ap.add_argument("--target-fps", type=float, default=30.0)
    args = ap.parse_args()

    report_dir = Path(args.reports)
    all_rows = []
    for csv_path in sorted(report_dir.glob("*.csv")):
        if csv_path.name.endswith("resource_samples.csv") or "resource_samples" in csv_path.name:
            continue
        ms_cols = load_ms_columns(csv_path)
        for col, vals in ms_cols.items():
            s = summarize_ms(vals)
            all_rows.append({
                "file": csv_path.name,
                "phase": col,
                "count": int(s["count"]),
                "mean_ms": s["mean"],
                "p50_ms": s["p50"],
                "p95_ms": s["p95"],
                "max_ms": s["max"],
                "recommendation": recommendation(col, s["p95"], args.target_fps),
            })

    all_rows.sort(key=lambda r: r["p95_ms"], reverse=True)
    budget = 1000.0 / args.target_fps
    print(f"\nTarget FPS: {args.target_fps:.1f} => frame budget: {budget:.2f} ms/frame")
    print("\nTop bottleneck candidates by P95:")
    print("=" * 35)
    for r in all_rows[:15]:
        marker = "!" if r["p95_ms"] > budget else " "
        print(f'{marker} {r["file"]:<32} {r["phase"]:<28} mean={r["mean_ms"]:7.3f} p95={r["p95_ms"]:7.3f} max={r["max_ms"]:7.3f}')
        print(f'    -> {r["recommendation"]}')

    out = write_json(report_dir / "99_bottleneck_analysis.json", {"target_fps": args.target_fps, "frame_budget_ms": budget, "ranked": all_rows})
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
