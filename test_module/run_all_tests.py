from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd):
    print("\n" + "#" * 88)
    print("RUN:", " ".join(str(x) for x in cmd))
    print("#" * 88)
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy lần lượt các test module realtime.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", default="template.csv")
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output-dir", default="test_results")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    common = ["--profile", args.profile, "--output-dir", args.output_dir]
    jobs = [
        [py, str(SCRIPT_DIR / "test_00_env.py"), "--model", args.model, "--profile", args.profile, "--output-dir", args.output_dir],
        [py, str(SCRIPT_DIR / "test_01_video_io.py"), "--video", args.video, "--frames", str(args.frames), *common],
        [py, str(SCRIPT_DIR / "test_09_fps_lock.py"), "--video", args.video, "--frames", str(args.frames), *common],
        [py, str(SCRIPT_DIR / "test_02_yolo_inference.py"), "--video", args.video, "--model", args.model, "--frames", str(args.frames), *common],
        [py, str(SCRIPT_DIR / "test_03_deepsort_update.py"), "--video", args.video, "--model", args.model, "--frames", str(args.frames), *common],
        [py, str(SCRIPT_DIR / "test_08_tracker_compare.py"), "--video", args.video, "--model", args.model, "--frames", str(args.frames), *common],
        [py, str(SCRIPT_DIR / "test_04_region_logic.py"), "--video", args.video, "--template", args.template, "--frames", str(max(60, args.frames // 2)), "--output-dir", args.output_dir],
        [py, str(SCRIPT_DIR / "test_05_display_overlay.py"), "--video", args.video, "--template", args.template, "--frames", str(max(60, args.frames // 2)), *common],
        [py, str(SCRIPT_DIR / "test_06_full_pipeline_profile.py"), "--video", args.video, "--model", args.model, "--template", args.template, "--frames", str(args.frames), *common],
        [py, str(SCRIPT_DIR / "test_07_queue_latency.py"), "--video", args.video, "--frames", str(args.frames), "--profile", args.profile, "--output-dir", args.output_dir, "--drop-old"],
    ]

    failed = []
    for job in jobs:
        code = run(job)
        if code != 0:
            failed.append((job[1], code))
            if not args.continue_on_error:
                print(f"\nDừng vì lỗi ở {job[1]} code={code}")
                return code

    if failed:
        print("\nCó test lỗi:")
        for path, code in failed:
            print(f"- {path}: code={code}")
        return 1

    print(f"\nHoàn tất. Xem CSV/JSON trong: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
