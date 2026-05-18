"""Video processing worker for YOLO detection, DeepSORT tracking, and metric updates."""

import threading
import time
from collections import Counter, deque
from queue import Empty, Full, Queue

import cv2
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.models import YOLO

from .config import (
    BRANCH_ORDER,
    CLASS_ASPECT_RATIO_LIMITS,
    CLASS_COLORS,
    CLASS_CONF_THRESHOLDS,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    DIRECTIONS,
    DISPLAY_CLASS_IDS,
    EXPECTED_MODEL_NAMES,
    LOST_OUT_FRAMES,
    MIN_BOX_AREA_RATIO,
    MIN_BOX_WH,
    MODEL_CONF,
    MODEL_IMGSZ,
    MODEL_IOU,
    VALID_BRANCHES,
)
from .flow_logic import (
    calc_veh_per_min,
    cleanup_flow_windows,
    create_track_meta,
    emit_branch_event,
    update_stable_class,
    update_stable_region,
)
from .regions import centroid_from_box, draw_region_overlay, get_direction_region


def _normalize_model_names(names):
    if isinstance(names, dict):
        return {int(k): str(v).lower() for k, v in names.items()}
    return {idx: str(name).lower() for idx, name in enumerate(names)}


def _validate_model_class_mapping(model):
    """Warn early if the fine-tuned model class order differs from config.py."""
    model_names = _normalize_model_names(getattr(model, "names", {}))
    expected = {idx: name.lower() for idx, name in EXPECTED_MODEL_NAMES.items()}
    mismatches = []

    for idx, expected_name in expected.items():
        actual_name = model_names.get(idx)
        if actual_name != expected_name:
            mismatches.append(f"id {idx}: expected={expected_name}, actual={actual_name}")

    return mismatches


def _passes_detection_filters(cls, conf, x1, y1, x2, y2, frame_width, frame_height):
    """Class-specific confidence + loose geometry filters for stable detection."""
    if cls not in CLASS_WEIGHTS:
        return False

    min_conf = CLASS_CONF_THRESHOLDS.get(cls, 0.25)
    if conf < min_conf:
        return False

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False

    min_w, min_h = MIN_BOX_WH.get(cls, (8, 8))
    if w < min_w or h < min_h:
        return False

    frame_area = max(frame_width * frame_height, 1)
    if (w * h) / frame_area < MIN_BOX_AREA_RATIO:
        return False

    aspect_ratio = w / max(h, 1)
    min_ratio, max_ratio = CLASS_ASPECT_RATIO_LIMITS.get(cls, (0.15, 8.0))
    if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
        return False

    return True


def process_video(app, video_path, model_path):
    """
    Process a video in a background thread.

    This function intentionally receives the FlowApp instance so the original logic can be
    preserved with minimal changes. UI updates are still done safely through app.worker_state
    and app.latest_pil_image under app.state_lock.
    """
    reader_thread = None
    try:
        model = YOLO(model_path)
        try:
            model.to("cuda")
        except Exception:
            model.to("cpu")

        mapping_warnings = _validate_model_class_mapping(model)
        if mapping_warnings:
            warning_text = "Class mapping mismatch: " + "; ".join(mapping_warnings)
            print("[Model Warning]", warning_text)
            with app.state_lock:
                app.worker_state["status"] = warning_text

        tracker = DeepSort(
            max_age=60,
            n_init=4,
            max_cosine_distance=0.2,
            nn_budget=100,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Không thể mở video")

        fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration = 1.0 / fps_input
        buffer_size = max(8, int(fps_input * 10))
        frame_queue = Queue(maxsize=buffer_size)
        reader_done = threading.Event()

        def video_reader():
            try:
                while cap.isOpened() and not app.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    while not app.stop_event.is_set():
                        try:
                            frame_queue.put(frame, timeout=0.1)
                            break
                        except Full:
                            continue
            finally:
                reader_done.set()
                cap.release()

        with app.state_lock:
            app.worker_state["status"] = "Buffering video..."

        reader_thread = threading.Thread(target=video_reader, daemon=True)
        reader_thread.start()

        while (
            frame_queue.qsize() < min(buffer_size, 4)
            and not reader_done.is_set()
            and not app.stop_event.is_set()
        ):
            time.sleep(0.05)

        with app.state_lock:
            app.worker_state["status"] = f"Buffer ready @ {fps_input:.1f} FPS"

        if app.region_template and app.region_template.loaded:
            valid_branches = set(app.region_template.regions.keys()) & VALID_BRANCHES
        else:
            valid_branches = set(VALID_BRANCHES)

        branch_count_total = Counter()
        branch_class_count_total = Counter()
        branch_event_windows = {
            (branch, direction): deque()
            for branch in valid_branches
            for direction in DIRECTIONS
        }

        track_meta = {}
        prev_time = time.time()
        playback_start = time.time()
        frame_id = 0

        while not app.stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except Empty:
                if reader_done.is_set() and frame_queue.empty():
                    break
                continue

            frame_id += 1
            current_time = frame_id / fps_input  # video-time, not wall-clock-time
            display_frame = frame.copy()

            detections = []
            if frame_id % app.detect_interval == 0:
                results = model(
                    frame,
                    imgsz=MODEL_IMGSZ,
                    conf=MODEL_CONF,
                    iou=MODEL_IOU,
                    classes=list(CLASS_WEIGHTS.keys()),
                    verbose=False,
                )

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        frame_h, frame_w = frame.shape[:2]
                        x1 = max(0, min(x1, frame_w - 1))
                        y1 = max(0, min(y1, frame_h - 1))
                        x2 = max(0, min(x2, frame_w - 1))
                        y2 = max(0, min(y2, frame_h - 1))

                        if not _passes_detection_filters(cls, conf, x1, y1, x2, y2, frame_w, frame_h):
                            continue

                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], conf, cls))

            tracks = tracker.update_tracks(detections, frame=frame)
            active_tracks = 0
            active_track_ids = set()

            print(f"[DeepSort] Frame={frame_id} detections={len(detections)} total_tracks={len(tracks)}")

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 5:
                    continue
                active_track_ids.add(track.track_id)

            # Tracks missing for only a few frames are kept to avoid false OUT events.
            tracks_to_remove = []
            for track_id, meta in list(track_meta.items()):
                if track_id in active_track_ids:
                    continue

                missing_frames = frame_id - meta.get("last_seen_frame", frame_id)
                if missing_frames < LOST_OUT_FRAMES:
                    continue

                active_branch = meta.get("active_branch")
                if active_branch in valid_branches:
                    emitted = emit_branch_event(
                        active_branch,
                        "out",
                        meta,
                        frame_id,
                        current_time,
                        track_id,
                        branch_count_total,
                        branch_class_count_total,
                        branch_event_windows,
                    )
                    if emitted:
                        print(f"[Flow] OUT confirmed lost: branch={active_branch} track_id={track_id}")

                tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del track_meta[track_id]

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 5:
                    print(
                        f"[DeepSort]   Skipping track {track.track_id} "
                        f"confirmed={track.is_confirmed()} time_since_update={track.time_since_update}"
                    )
                    continue

                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                det_cls = getattr(track, "det_class", None)
                det_conf = getattr(track, "det_conf", None)
                if det_cls is None:
                    det_cls = track_meta.get(track_id, {}).get("stable_cls", 2)

                if int(det_cls) not in CLASS_WEIGHTS:
                    continue

                active_tracks += 1
                centroid = centroid_from_box((l, t, r, b))
                raw_region = get_direction_region(
                    centroid,
                    frame.shape[1],
                    frame.shape[0],
                    app.region_margin,
                    app.region_template,
                )

                meta = track_meta.setdefault(track_id, create_track_meta(frame_id, int(det_cls)))
                meta["last_seen_frame"] = frame_id
                cls = update_stable_class(meta, det_cls, det_conf)

                stable_region = update_stable_region(meta, raw_region)
                meta["current_region"] = stable_region

                if stable_region is not None:
                    current_is_branch = stable_region in valid_branches
                    active_branch = meta.get("active_branch")

                    if active_branch in valid_branches and stable_region != active_branch:
                        emitted = emit_branch_event(
                            active_branch,
                            "out",
                            meta,
                            frame_id,
                            current_time,
                            track_id,
                            branch_count_total,
                            branch_class_count_total,
                            branch_event_windows,
                        )
                        if emitted:
                            print(
                                f"[Flow] OUT stable: branch={active_branch} -> {stable_region} "
                                f"cls={cls} track_id={track_id}"
                            )
                        meta["active_branch"] = None

                    if current_is_branch and meta.get("active_branch") != stable_region:
                        emitted = emit_branch_event(
                            stable_region,
                            "in",
                            meta,
                            frame_id,
                            current_time,
                            track_id,
                            branch_count_total,
                            branch_class_count_total,
                            branch_event_windows,
                        )
                        if emitted:
                            print(f"[Flow] IN stable: branch={stable_region} cls={cls} track_id={track_id}")
                        meta["active_branch"] = stable_region

                color = CLASS_COLORS.get(cls, (255, 255, 255))

                label = f"{CLASS_NAMES.get(cls, cls)} #{track_id}"
                region_label = stable_region if stable_region is not None else raw_region
                if region_label:
                    label += f" {region_label}"

                cv2.rectangle(display_frame, (l, t), (r, b), color, 2)
                cv2.putText(display_frame, label, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(display_frame, centroid, 3, color, -1)

                print(
                    f"[DeepSort]   Track={track_id} stable_cls={cls} raw_cls={det_cls} "
                    f"conf={det_conf} label={CLASS_NAMES.get(cls, cls)} "
                    f"ltrb=({l},{t},{r},{b}) centroid={centroid} raw={raw_region} "
                    f"stable={stable_region} active_branch={meta.get('active_branch')} "
                    f"confirmed={track.is_confirmed()} time_since_update={track.time_since_update}"
                )

            if app.display_template_var.get():
                draw_region_overlay(display_frame, app.region_margin, app.region_template)

            cleanup_flow_windows(branch_event_windows, current_time, app.flow_window)

            branch_current_pce = {branch: 0.0 for branch in valid_branches}
            for meta in track_meta.values():
                if meta.get("last_seen_frame") != frame_id:
                    continue
                region = meta.get("current_region")
                if region in valid_branches:
                    branch_current_pce[region] += meta.get("weight", 0.0)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time

            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((880, 620), Image.LANCZOS)

            total_current_pce = 0.0
            total_in_flow = 0.0
            metric_updates = {}

            for branch in BRANCH_ORDER:
                if branch not in valid_branches:
                    metric_updates[f"{branch}_pce"] = "0.0"
                    metric_updates[f"{branch}_in_flow"] = "0.0"
                    metric_updates[f"{branch}_out_flow"] = "0.0"
                    metric_updates[f"{branch}_in_count"] = "0"
                    metric_updates[f"{branch}_out_count"] = "0"
                    for cls_id in DISPLAY_CLASS_IDS:
                        class_name = CLASS_NAMES[cls_id]
                        metric_updates[f"{branch}_{class_name}_in"] = "0"
                        metric_updates[f"{branch}_{class_name}_out"] = "0"
                    continue

                pce_now = branch_current_pce.get(branch, 0.0)
                in_flow = calc_veh_per_min(branch_event_windows[(branch, "in")], app.flow_window)
                out_flow = calc_veh_per_min(branch_event_windows[(branch, "out")], app.flow_window)

                metric_updates[f"{branch}_pce"] = f"{pce_now:.1f}"
                metric_updates[f"{branch}_in_flow"] = f"{in_flow:.1f}"
                metric_updates[f"{branch}_out_flow"] = f"{out_flow:.1f}"
                metric_updates[f"{branch}_in_count"] = str(branch_count_total[(branch, "in")])
                metric_updates[f"{branch}_out_count"] = str(branch_count_total[(branch, "out")])

                for cls_id in DISPLAY_CLASS_IDS:
                    class_name = CLASS_NAMES[cls_id]
                    metric_updates[f"{branch}_{class_name}_in"] = str(branch_class_count_total[(branch, "in", cls_id)])
                    metric_updates[f"{branch}_{class_name}_out"] = str(branch_class_count_total[(branch, "out", cls_id)])

                total_current_pce += pce_now
                total_in_flow += in_flow

            with app.state_lock:
                app.latest_pil_image = pil_image
                app.worker_state["status"] = "Running"
                app.worker_state["frame"] = str(frame_id)
                app.worker_state["fps"] = f"{fps:.1f}"
                app.worker_state["active_tracks"] = str(active_tracks)
                app.worker_state["current_pce"] = f"{total_current_pce:.1f}"
                app.worker_state["flow_veh_pm"] = f"{total_in_flow:.1f}"
                app.worker_state.update(metric_updates)

            expected_display = playback_start + (frame_id - 1) * frame_duration
            delay = expected_display - time.time()
            if delay > 0:
                time.sleep(delay)

        with app.state_lock:
            app.worker_state["status"] = "Stopped" if app.stop_event.is_set() else "Finished"

    except Exception as exc:
        with app.state_lock:
            app.worker_state["status"] = f"Error: {exc}"
    finally:
        if reader_thread is not None:
            try:
                reader_thread.join(timeout=1.0)
            except Exception:
                pass
