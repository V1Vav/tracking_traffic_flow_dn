from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2  # type: ignore
import numpy as np

from common import PhaseTimer, ResourceMonitor, cv2_font_scale, iter_video_frames, parse_size, perf_ms, print_table, read_csv_rows, summarize_ms
from region_utils import load_regions


def synthetic_boxes(frame_idx: int, n: int, w: int, h: int):
    rng = random.Random(frame_idx)
    out = []
    for i in range(n):
        x = rng.randint(0, max(1, w - 80))
        y = rng.randint(0, max(1, h - 80))
        out.append({"track_id": i, "x1": x, "y1": y, "x2": x + 60, "y2": y + 40, "cls": rng.randint(0, 4), "conf": 0.8})
    return out


def draw_overlay(frame, boxes, regions, show_regions=True, show_text=True):
    h, w = frame.shape[:2]
    font_scale = cv2_font_scale(w)
    if show_regions:
        for r in regions:
            if str(r["name"]).lower() == "center":
                continue
            pts = np.array(r["points"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (160, 160, 160), 2)
            x, y = pts.reshape(-1, 2).mean(axis=0).astype(int)
            cv2.putText(frame, str(r["name"]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (230, 230, 230), 1, cv2.LINE_AA)
    for b in boxes:
        x1, y1, x2, y2 = int(float(b["x1"])), int(float(b["y1"])), int(float(b["x2"])), int(float(b["y2"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        if show_text:
            label = f'id:{b.get("track_id", "-")} c:{b.get("cls", "-")}'
            cv2.putText(frame, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    if show_text:
        cv2.rectangle(frame, (8, 8), (380, 78), (0, 0, 0), -1)
        cv2.putText(frame, f"objects: {len(boxes)}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "render bottleneck test", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6: measure overlay drawing and GUI-like conversion cost.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", default=None)
    ap.add_argument("--tracks-csv", default=None)
    ap.add_argument("--synthetic-boxes", type=int, default=80)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--resize", default=None, help="Optional display size, e.g. 1280x720")
    ap.add_argument("--no-regions", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--pil-convert", action="store_true", help="Measure BGR->RGB + PIL Image.fromarray like Tkinter display path.")
    ap.add_argument("--imshow", action="store_true", help="Actually show frames with cv2.imshow; skip on headless machines.")
    ap.add_argument("--reports", default="bottleneck_reports")
    args = ap.parse_args()

    resize_to = parse_size(args.resize)
    track_by_frame: Dict[int, List[Dict]] = defaultdict(list)
    if args.tracks_csv:
        for r in read_csv_rows(args.tracks_csv):
            try:
                track_by_frame[int(float(r["frame"]))].append(r)
            except Exception:
                pass

    timer = PhaseTimer()
    with ResourceMonitor() as mon:
        for frame_idx, frame in iter_video_frames(args.video, args.max_frames, args.start_frame, args.stride):
            h0, w0 = frame.shape[:2]
            regions = load_regions(args.template, w0, h0)
            boxes = track_by_frame.get(frame_idx) if track_by_frame else synthetic_boxes(frame_idx, args.synthetic_boxes, w0, h0)

            t0 = perf_ms()
            resize_ms = 0.0
            if resize_to:
                r0 = perf_ms()
                sx, sy = resize_to[0] / w0, resize_to[1] / h0
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_LINEAR)
                # Keep synthetic/current track coordinates roughly aligned with resized frame.
                for b in boxes:
                    b["x1"] = float(b["x1"]) * sx; b["x2"] = float(b["x2"]) * sx
                    b["y1"] = float(b["y1"]) * sy; b["y2"] = float(b["y2"]) * sy
                resize_ms = perf_ms() - r0

            d0 = perf_ms()
            out = draw_overlay(frame.copy(), boxes, regions, show_regions=not args.no_regions, show_text=not args.no_text)
            d1 = perf_ms()

            pil_ms = 0.0
            if args.pil_convert:
                p0 = perf_ms()
                from PIL import Image  # type: ignore
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                _ = Image.fromarray(rgb)
                pil_ms = perf_ms() - p0

            imshow_ms = 0.0
            if args.imshow:
                s0 = perf_ms()
                cv2.imshow("render_bottleneck_test", out)
                cv2.waitKey(1)
                imshow_ms = perf_ms() - s0

            timer.add(frame=frame_idx, boxes=len(boxes), resize_ms=resize_ms, draw_overlay_ms=d1 - d0, pil_convert_ms=pil_ms, imshow_ms=imshow_ms, total_phase_ms=perf_ms() - t0)
        mon.save(args.reports, "06_resource_samples")

    if args.imshow:
        cv2.destroyAllWindows()
    csv_path, json_path = timer.save(args.reports, "06_render_overlay_speed", vars(args))
    s = summarize_ms([float(r["total_phase_ms"]) for r in timer.rows])
    print_table("Phase 6 - Render/overlay", [
        ("frames", len(timer.rows)),
        ("mean total ms", s["mean"]),
        ("p95 total ms", s["p95"]),
        ("phase fps", 1000.0 / s["mean"] if s["mean"] > 0 else 0.0),
        ("csv", str(csv_path)),
        ("summary", str(json_path)),
    ])


if __name__ == "__main__":
    main()
