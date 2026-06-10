from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from common import ensure_dir, get_video_info, print_table, write_json


def run_cmd(cmd):
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=3)
        return out.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect runtime environment info for realtime bottleneck tests.")
    ap.add_argument("--video", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reports", default="bottleneck_reports")
    args = ap.parse_args()

    data = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
    }

    for name in ["cv2", "numpy", "torch", "ultralytics", "PIL", "psutil"]:
        try:
            mod = __import__(name)
            data[name] = getattr(mod, "__version__", "installed")
        except Exception as e:
            data[name] = f"missing/error: {e}"

    try:
        import cv2  # type: ignore
        data["opencv_build_head"] = cv2.getBuildInformation().splitlines()[:60]
    except Exception:
        pass

    try:
        import torch  # type: ignore
        data["torch_cuda_available"] = bool(torch.cuda.is_available())
        data["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            data["torch_cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        data["torch_cuda_error"] = str(e)

    data["nvidia_smi"] = run_cmd(["nvidia-smi"])

    if args.video:
        try:
            data["video_info"] = get_video_info(args.video)
        except Exception as e:
            data["video_info_error"] = str(e)
    if args.model:
        p = Path(args.model)
        data["model"] = {"path": str(p), "exists": p.exists(), "size_mb": round(p.stat().st_size / 1024 / 1024, 2) if p.exists() else None}

    out = write_json(Path(args.reports) / "00_env.json", data)
    print_table("Environment", [("python", sys.version.split()[0]), ("platform", platform.platform()), ("report", str(out))])
    if "video_info" in data:
        vi = data["video_info"]
        print_table("Video", [("opened", vi["opened"]), ("size", f'{vi["width"]}x{vi["height"]}'), ("fps", vi["fps"]), ("frames", vi["frame_count"])])
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
