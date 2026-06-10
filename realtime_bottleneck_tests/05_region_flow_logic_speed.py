from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common import PhaseTimer, get_video_info, perf_ms, print_table, read_csv_rows, summarize_ms, write_csv
from region_utils import bbox_center, classify_region, load_regions


def synthetic_rows(frames: int, tracks: int, w: int, h: int) -> List[Dict[str, float]]:
    rows = []
    rng = random.Random(42)
    for tid in range(tracks):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        vx = rng.uniform(-5, 5)
        vy = rng.uniform(-3, 3)
        cls = rng.choice([0, 1, 2, 3, 4])
        for f in range(frames):
            x = (x + vx) % w
            y = (y + vy) % h
            rows.append({"frame": f, "track_id": tid, "x1": x - 20, "y1": y - 20, "x2": x + 20, "y2": y + 20, "cls": cls, "conf": 0.9})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5: measure region classification and flow transition logic.")
    ap.add_argument("--template", default=None, help="CSV/JSON region template. If omitted, a default 8-region layout is used.")
    ap.add_argument("--video", default=None, help="Optional only for frame size.")
    ap.add_argument("--tracks-csv", default=None, help="Use 04_tracks.csv output instead of synthetic tracks.")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--synthetic-tracks", type=int, default=120, help="Stress-test track count when no tracks CSV is provided.")
    ap.add_argument("--frame-size", default="1280x720")
    ap.add_argument("--reports", default="bottleneck_reports")
    args = ap.parse_args()

    if args.video:
        info = get_video_info(args.video)
        w, h = int(info["width"] or 1280), int(info["height"] or 720)
    else:
        w, h = [int(x) for x in args.frame_size.lower().split("x")]

    regions = load_regions(args.template, w, h)
    if args.tracks_csv:
        all_rows = read_csv_rows(args.tracks_csv)
    else:
        all_rows = synthetic_rows(args.frames, args.synthetic_tracks, w, h)

    by_frame: Dict[int, List[Dict]] = defaultdict(list)
    for r in all_rows:
        try:
            by_frame[int(float(r["frame"]))].append(r)
        except Exception:
            pass

    prev_region: Dict[int, str] = {}
    transitions = defaultdict(int)
    timer = PhaseTimer()
    for frame in sorted(by_frame.keys())[:args.frames]:
        rows = by_frame[frame]
        c0 = perf_ms()
        current = []
        for r in rows:
            tid = int(float(r.get("track_id", r.get("det_id", 0))))
            cx, cy = bbox_center(r)
            region = classify_region(cx, cy, regions)
            if region:
                old = prev_region.get(tid)
                if old and old != region:
                    transitions[(old, region)] += 1
                prev_region[tid] = region
            current.append(region or "none")
        c1 = perf_ms()
        timer.add(frame=frame, objects=len(rows), classify_and_update_ms=c1 - c0, transitions_total=sum(transitions.values()))

    trans_rows = [{"src": k[0], "dst": k[1], "count": v} for k, v in sorted(transitions.items(), key=lambda x: -x[1])]
    write_csv(Path(args.reports) / "05_region_transitions.csv", trans_rows)
    csv_path, json_path = timer.save(args.reports, "05_region_flow_logic_speed", {**vars(args), "regions": [r["name"] for r in regions]})
    s = summarize_ms([float(r["classify_and_update_ms"]) for r in timer.rows])
    print_table("Phase 5 - Region/flow logic", [
        ("regions", ", ".join(str(r["name"]) for r in regions)),
        ("frames", len(timer.rows)),
        ("mean ms", s["mean"]),
        ("p95 ms", s["p95"]),
        ("objects/frame", sum(int(r["objects"]) for r in timer.rows) / max(1, len(timer.rows))),
        ("csv", str(csv_path)),
        ("summary", str(json_path)),
    ])


if __name__ == "__main__":
    main()
