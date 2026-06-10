from __future__ import annotations

import argparse

from common import add_common_args, ensure_output_dir, get_capture_info, get_profile_config, open_capture, save_json, print_summary_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra logic khóa FPS realtime theo min(video_fps, 30).")
    add_common_args(parser)
    args = parser.parse_args()

    import cv2
    from vehicle_flow.video_worker import _resolve_target_process_fps, _sanitize_capture_fps

    profile_cfg = get_profile_config(args.profile)
    cap = open_capture(args.video)
    info = get_capture_info(cap)
    raw_fps = float(info.get("fps", 0.0) or 0.0)
    source_fps, metadata_valid = _sanitize_capture_fps(raw_fps, fallback=30.0)
    effective_fps, downsample_enabled = _resolve_target_process_fps(source_fps, profile_cfg)

    # Mô phỏng đúng thuật toán lấy mẫu trong video_worker.video_reader cho file video.
    sampled = []
    target_period = 1.0 / effective_fps if effective_fps > 0 else 0.0
    next_sample_time = 0.0
    total_frames = int(info.get("frame_count", 0) or 0)
    max_source_frames = args.frames if args.frames > 0 else min(total_frames or 300, 300)
    for source_frame_id in range(1, max_source_frames + 1):
        source_time = (source_frame_id - 1) / source_fps if source_fps > 0 else 0.0
        take = True
        if downsample_enabled and target_period > 0:
            if source_time + 1e-9 < next_sample_time:
                take = False
            else:
                while next_sample_time <= source_time + 1e-9:
                    next_sample_time += target_period
        if take:
            sampled.append({"source_frame_id": source_frame_id, "source_time_s": source_time})

    duration_s = 0.0
    if sampled:
        duration_s = max(sampled[-1]["source_time_s"] - sampled[0]["source_time_s"], 1e-9)
    sampled_fps_est = (len(sampled) - 1) / duration_s if len(sampled) > 1 and duration_s > 0 else 0.0

    summary = {
        "profile": args.profile,
        "raw_fps_from_opencv": raw_fps,
        "source_fps_used": source_fps,
        "fps_metadata_valid": metadata_valid,
        "lock_to_source_fps": bool(profile_cfg.get("lock_to_source_fps", False)),
        "fps_lock_max": float(profile_cfg.get("fps_lock_max", profile_cfg.get("target_process_fps", 0.0)) or 0.0),
        "target_process_fps": effective_fps,
        "target_frame_budget_ms": 1000.0 / effective_fps if effective_fps > 0 else 0.0,
        "downsample_enabled": downsample_enabled,
        "source_frames_checked": max_source_frames,
        "sampled_frames": len(sampled),
        "sampled_fps_estimate": sampled_fps_est,
        "expected_rule": "target_process_fps = min(source_fps, fps_lock_max) khi lock_to_source_fps=True",
    }

    out = ensure_output_dir(args.output_dir)
    save_json(out / "09_fps_lock.json", summary)
    save_json(out / "09_fps_lock_sampled_frames.json", {"sampled": sampled[:200]})

    print_summary_table(
        "09_fps_lock",
        summary,
        [
            "profile",
            "raw_fps_from_opencv",
            "source_fps_used",
            "fps_metadata_valid",
            "lock_to_source_fps",
            "fps_lock_max",
            "target_process_fps",
            "target_frame_budget_ms",
            "downsample_enabled",
            "sampled_fps_estimate",
        ],
    )

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
