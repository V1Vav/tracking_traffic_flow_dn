from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, allow_fail=False):
    print("\n$", " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0 and not allow_fail:
        raise SystemExit(p.returncode)
    if p.returncode != 0:
        print(f"Command failed but continuing: exit={p.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all realtime bottleneck phase tests.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", default=None)
    ap.add_argument("--reports", default="bottleneck_reports")
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--tracker", default="auto", choices=["auto", "deepsort", "centroid", "none"])
    ap.add_argument("--render", action="store_true", help="Include render in full pipeline test.")
    ap.add_argument("--resize", default=None, help="Display resize for video/render test, e.g. 1280x720")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    py = sys.executable
    common = ["--reports", args.reports]
    vf = ["--video", args.video, "--max-frames", str(args.max_frames), "--start-frame", str(args.start_frame), "--stride", str(args.stride)]
    yolo = ["--model", args.model, "--imgsz", str(args.imgsz), "--device", args.device, "--conf", str(args.conf), "--iou", str(args.iou)]
    if args.half:
        yolo.append("--half")
    template = ["--template", args.template] if args.template else []
    resize = ["--resize", args.resize] if args.resize else []

    run([py, str(here / "00_collect_env.py"), "--video", args.video, "--model", args.model, *common])
    run([py, str(here / "01_video_io_decode.py"), *vf, *resize, "--bgr2rgb", *common])
    run([py, str(here / "02_preprocess_to_tensor.py"), *vf, "--imgsz", str(args.imgsz), "--device", args.device, *( ["--half"] if args.half else [] ), *common], allow_fail=True)
    run([py, str(here / "03_yolo_detect_speed.py"), *vf, *yolo, "--save-detections", *common])
    run([py, str(here / "04_yolo_plus_tracker_speed.py"), *vf, *yolo, "--tracker", args.tracker, "--save-tracks", *common], allow_fail=True)
    tracks_csv = Path(args.reports) / "04_tracks.csv"
    region_cmd = [py, str(here / "05_region_flow_logic_speed.py"), *template, "--video", args.video, "--frames", str(args.max_frames), *common]
    if tracks_csv.exists():
        region_cmd += ["--tracks-csv", str(tracks_csv)]
    run(region_cmd, allow_fail=True)
    render_cmd = [py, str(here / "06_render_overlay_speed.py"), "--video", args.video, *template, *resize, "--pil-convert", "--max-frames", str(args.max_frames), "--start-frame", str(args.start_frame), "--stride", str(args.stride), *common]
    if tracks_csv.exists():
        render_cmd += ["--tracks-csv", str(tracks_csv)]
    run(render_cmd, allow_fail=True)
    full_cmd = [py, str(here / "07_full_pipeline_probe.py"), *vf, *yolo, *template, "--tracker", args.tracker, *common]
    if args.render:
        full_cmd.append("--render")
    run(full_cmd, allow_fail=True)
    run([py, str(here / "99_analyze_reports.py"), "--reports", args.reports, "--target-fps", str(args.target_fps)])


if __name__ == "__main__":
    main()
