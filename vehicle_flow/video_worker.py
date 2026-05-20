"""Video processing worker for YOLO detection, DeepSORT tracking, and metric updates."""

import threading
import time
from collections import Counter, deque
from queue import Empty, Full, Queue

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - torch is normally installed with ultralytics
    torch = None
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.models import YOLO

from .config import (
    BRANCH_ORDER,
    CLASS_ASPECT_RATIO_LIMITS,
    CLASS_COLORS,
    CLASS_CONF_THRESHOLDS,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    CV2_NUM_THREADS,
    DEBUG_TRACK_LOGS,
    EDGE_LOST_OUT_FRAMES,
    FRAME_EXIT_MARGIN_RATIO,
    DETECTION_DUPLICATE_CENTER_RATIO,
    DETECTION_DUPLICATE_CONTAINMENT,
    DETECTION_DUPLICATE_IOU,
    DIRECTIONS,
    DISPLAY_CLASS_IDS,
    DEFAULT_PERFORMANCE_PROFILE,
    EXPECTED_MODEL_NAMES,
    FILE_QUEUE_SECONDS,
    LOST_OUT_FRAMES,
    MIN_BOX_AREA_RATIO,
    MIN_BOX_WH,
    MODEL_CONF,
    MODEL_IMGSZ,
    MODEL_IOU,
    PERFORMANCE_PROFILES,
    REALTIME_QUEUE_SIZE,
    REALTIME_SOURCE_PREFIXES,
    TRACK_COUNT_HOLD_FRAMES,
    TRACK_DISPLAY_MAX_AGE,
    TRACK_DUPLICATE_CENTER_RATIO,
    TRACK_DUPLICATE_CONTAINMENT,
    TRACK_DUPLICATE_IOU,
    TRACK_MAX_AGE,
    TRACK_MAX_COSINE_DISTANCE,
    TRACK_N_INIT,
    TRACK_NN_BUDGET,
    TRACK_STALE_MERGE_FRAMES,
    VALID_BRANCHES,
    YOLO_WARMUP,
)
from .flow_logic import (
    calc_veh_per_min,
    cleanup_flow_windows,
    create_track_meta,
    emit_branch_event,
    mark_branch_enter,
    mark_branch_exit,
    update_stable_class,
    update_stable_region,
)
from .regions import centroid_from_box, draw_region_overlay, get_direction_region
from .fluid_export import FluidFlowExporter


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


def _is_realtime_source(source):
    """Return True for webcam indexes and common live stream URLs."""
    source_text = str(source).strip()
    if source_text.isdigit():
        return True
    return source_text.lower().startswith(REALTIME_SOURCE_PREFIXES)


def _open_capture(source):
    """Open file, webcam index, or stream URL with OpenCV."""
    source_text = str(source).strip()
    if source_text.isdigit():
        return cv2.VideoCapture(int(source_text))
    return cv2.VideoCapture(source_text)


def _put_frame(frame_queue, frame, realtime_source, stop_event):
    """Put frame into queue. In real-time mode, drop old frames to reduce lag."""
    if not realtime_source:
        while not stop_event.is_set():
            try:
                frame_queue.put(frame, timeout=0.1)
                return
            except Full:
                continue
        return

    # Live camera/RTSP should prefer low latency over processing every frame.
    while not stop_event.is_set():
        try:
            frame_queue.put_nowait(frame)
            return
        except Full:
            try:
                frame_queue.get_nowait()
            except Empty:
                pass


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


def _xywh_to_ltrb(box):
    x, y, w, h = box
    return int(x), int(y), int(x + w), int(y + h)


def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _iou(a, b):
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(a, b):
    """Intersection over the smaller box area."""
    inter = _intersection_area(a, b)
    smaller = max(min(_area(a), _area(b)), 1)
    return inter / smaller


def _center_distance_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    diag_a = max(((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5, 1.0)
    diag_b = max(((bx2 - bx1) ** 2 + (by2 - by1) ** 2) ** 0.5, 1.0)
    return dist / max(min(diag_a, diag_b), 1.0)


def _boxes_look_duplicate(
    box_a,
    box_b,
    iou_threshold,
    containment_threshold,
    center_ratio_threshold,
):
    iou_value = _iou(box_a, box_b)
    if iou_value >= iou_threshold:
        return True

    if _containment(box_a, box_b) >= containment_threshold:
        return True

    # Avoid removing two adjacent vehicles: require some overlap before using
    # center distance as an additional duplicate cue.
    if iou_value >= 0.12 and _center_distance_ratio(box_a, box_b) <= center_ratio_threshold:
        return True

    return False


def _suppress_duplicate_detections(detections):
    """Class-agnostic duplicate suppression before DeepSORT.

    YOLO NMS is already applied, but fine-tuned small models can still output
    several very similar boxes for one motorcycle/car. If these boxes reach
    DeepSORT, they become separate IDs and remain visible for many frames.
    """
    if len(detections) <= 1:
        return detections

    # Keep high-confidence detections first. For equal confidence, prefer the
    # smaller/tighter box because it is less likely to cover multiple vehicles.
    sorted_dets = sorted(
        detections,
        key=lambda det: (float(det[1]), -_area(_xywh_to_ltrb(det[0]))),
        reverse=True,
    )

    kept = []
    for det in sorted_dets:
        box = _xywh_to_ltrb(det[0])
        duplicate = False
        for kept_det in kept:
            kept_box = _xywh_to_ltrb(kept_det[0])
            if _boxes_look_duplicate(
                box,
                kept_box,
                DETECTION_DUPLICATE_IOU,
                DETECTION_DUPLICATE_CONTAINMENT,
                DETECTION_DUPLICATE_CENTER_RATIO,
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    return kept


def _track_is_usable(track):
    # DeepSORT can internally keep tracks for a long time, but drawing/counting
    # stale boxes causes the exact symptom in the screenshot: many IDs remain
    # around the same vehicle. Only fresh tracks are used for display/counting.
    return track.is_confirmed() and track.time_since_update <= TRACK_DISPLAY_MAX_AGE


def _track_ltrb(track):
    l, t, r, b = map(int, track.to_ltrb())
    return l, t, r, b


def _box_near_frame_edge(box, frame_width, frame_height):
    """Return True when the last known box is close to leaving the camera view."""
    if box is None:
        return False
    x1, y1, x2, y2 = box
    margin_x = max(int(frame_width * FRAME_EXIT_MARGIN_RATIO), 2)
    margin_y = max(int(frame_height * FRAME_EXIT_MARGIN_RATIO), 2)
    return (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= frame_width - margin_x
        or y2 >= frame_height - margin_y
    )


def _emit_active_branch_out(
    meta,
    track_id,
    frame_id,
    current_time,
    valid_branches,
    branch_count_total,
    branch_class_count_total,
    branch_event_windows,
):
    """Close one active branch visit if its IN event was counted.

    This keeps IN/OUT paired per track visit without forcing OUT at the same
    time as IN. It also uses the class saved at IN time, so class smoothing
    changes after entry cannot turn a Moto In into a Car Out.
    """
    active_branch = meta.get("active_branch")
    if active_branch not in valid_branches:
        return False

    counted = bool(meta.get("active_branch_counted", False))
    event_cls = meta.get("active_branch_cls", meta.get("stable_cls", meta.get("cls", 2)))
    emitted = False

    if counted:
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
            event_cls=event_cls,
            force=True,
        )

    mark_branch_exit(meta, active_branch)
    return emitted


def _track_conf(track):
    try:
        conf = getattr(track, "det_conf", None)
        return 0.0 if conf is None else float(conf)
    except Exception:
        return 0.0


def _track_sort_key(track, track_meta):
    meta = track_meta.get(track.track_id, {})
    history_len = len(meta.get("class_history", []))
    stable_age = int(meta.get("stable_cls_age", 0))
    return (
        int(track.time_since_update),
        -_track_conf(track),
        -stable_age,
        -history_len,
        int(track.track_id) if str(track.track_id).isdigit() else 0,
    )


def _suppress_duplicate_tracks(tracks, track_meta):
    """Return primary fresh tracks and duplicate track ids to ignore/delete."""
    fresh_tracks = [track for track in tracks if _track_is_usable(track)]
    if len(fresh_tracks) <= 1:
        return fresh_tracks, set()

    sorted_tracks = sorted(fresh_tracks, key=lambda track: _track_sort_key(track, track_meta))
    kept = []
    duplicate_ids = set()

    for track in sorted_tracks:
        box = _track_ltrb(track)
        is_duplicate = False
        for kept_track in kept:
            kept_box = _track_ltrb(kept_track)
            if _boxes_look_duplicate(
                box,
                kept_box,
                TRACK_DUPLICATE_IOU,
                TRACK_DUPLICATE_CONTAINMENT,
                TRACK_DUPLICATE_CENTER_RATIO,
            ):
                duplicate_ids.add(track.track_id)
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(track)

    return kept, duplicate_ids


def _merge_stale_duplicate_meta(track_meta, primary_tracks, frame_id):
    """Move old metadata to a new ID when DeepSORT switches ID on one vehicle.

    This does not change DeepSORT's internal id, but it preserves stable class,
    active branch and cooldown state, so one physical vehicle is less likely to
    be counted again after an ID switch.
    """
    primary_ids = {track.track_id for track in primary_tracks}

    for track in primary_tracks:
        track_id = track.track_id
        if track_id in track_meta:
            continue

        new_box = _track_ltrb(track)
        best_old_id = None
        best_score = 0.0

        for old_id, meta in list(track_meta.items()):
            if old_id in primary_ids:
                continue

            old_box = meta.get("last_box")
            if old_box is None:
                continue

            missing_frames = frame_id - meta.get("last_seen_frame", frame_id)
            if missing_frames < 1 or missing_frames > TRACK_STALE_MERGE_FRAMES:
                continue

            iou_value = _iou(new_box, old_box)
            containment_value = _containment(new_box, old_box)
            center_value = _center_distance_ratio(new_box, old_box)
            looks_same = (
                iou_value >= TRACK_DUPLICATE_IOU
                or containment_value >= TRACK_DUPLICATE_CONTAINMENT
                or (iou_value >= 0.10 and center_value <= TRACK_DUPLICATE_CENTER_RATIO)
            )
            if not looks_same:
                continue

            score = max(iou_value, containment_value * 0.9, (1.0 - center_value) * 0.7)
            if score > best_score:
                best_score = score
                best_old_id = old_id

        if best_old_id is not None:
            track_meta[track_id] = track_meta.pop(best_old_id)
            track_meta[track_id]["last_seen_frame"] = frame_id
            track_meta[track_id]["last_box"] = new_box


def _near_hidden_left_gate(centroid, frame_width, app):
    """Heuristic gate for the hidden/occluded left branch.

    If a track appears or disappears in the center close to this x threshold,
    the exporter can create an inferred left<->center flow edge. This does not
    modify the real tracking/counting logic.
    """
    if centroid is None or frame_width <= 0:
        return False
    try:
        ratio = float(getattr(app, "left_gate_x_ratio", 0.30))
    except Exception:
        ratio = 0.30
    return float(centroid[0]) <= frame_width * ratio


def _fluid_export_enabled(app):
    try:
        return bool(app.export_fluid_var.get())
    except Exception:
        return False


def _get_performance_cfg(app):
    """Resolve the selected performance preset without changing UI state."""
    profile = str(getattr(app, "performance_profile", DEFAULT_PERFORMANCE_PROFILE)).strip().lower()
    if profile not in PERFORMANCE_PROFILES:
        profile = DEFAULT_PERFORMANCE_PROFILE
    cfg = PERFORMANCE_PROFILES[profile].copy()
    app_cfg = getattr(app, "performance_cfg", None)
    if isinstance(app_cfg, dict):
        cfg.update(app_cfg)
    return profile, cfg


def _configure_runtime():
    """Small runtime setup that can improve GPU/CV throughput without changing logic."""
    if CV2_NUM_THREADS and CV2_NUM_THREADS > 0:
        try:
            cv2.setNumThreads(int(CV2_NUM_THREADS))
        except Exception:
            pass

    if torch is not None:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass


def _select_torch_device():
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
    return "cpu"


def _resize_for_processing(frame, process_width):
    """Resize frame before detection/tracking; region templates scale automatically."""
    try:
        process_width = int(process_width)
    except Exception:
        process_width = 0

    if process_width <= 0:
        return frame

    h, w = frame.shape[:2]
    if w <= process_width:
        return frame

    scale = process_width / float(w)
    process_height = max(1, int(round(h * scale)))
    return cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)


def _predict_yolo(model, frame, *, model_imgsz, device, use_half, max_det):
    """Run YOLO with fast settings, with fallback for older ultralytics versions."""
    kwargs = {
        "imgsz": int(model_imgsz),
        "conf": MODEL_CONF,
        "iou": MODEL_IOU,
        "classes": list(CLASS_WEIGHTS.keys()),
        "verbose": False,
        "max_det": int(max_det),
    }
    if device:
        kwargs["device"] = device
    if use_half:
        kwargs["half"] = True

    try:
        return model(frame, **kwargs)
    except TypeError:
        kwargs.pop("half", None)
        kwargs.pop("max_det", None)
        return model(frame, **kwargs)
    except Exception:
        if use_half:
            kwargs.pop("half", None)
            return model(frame, **kwargs)
        raise


def _warmup_model(model, *, model_imgsz, device, use_half, max_det):
    if not YOLO_WARMUP:
        return
    try:
        warm_size = max(320, min(int(model_imgsz), 960))
        dummy = np.zeros((warm_size, warm_size, 3), dtype=np.uint8)
        _predict_yolo(
            model,
            dummy,
            model_imgsz=model_imgsz,
            device=device,
            use_half=use_half,
            max_det=max_det,
        )
    except Exception as exc:
        print(f"[Perf] YOLO warmup skipped: {exc}")


def _export_track_transition(exporter, *, meta, track_id, frame_id, current_time, from_region, to_region, cls, box, centroid, source="observed", confidence=1.0, reason="region_change"):
    if exporter is None:
        return False
    return exporter.log_transition(
        time_s=current_time,
        frame_id=frame_id,
        track_id=track_id,
        from_region=from_region,
        to_region=to_region,
        cls=cls,
        box=box,
        centroid=centroid,
        source=source,
        confidence=confidence,
        reason=reason,
    )


def process_video(app, video_path, model_path):
    """
    Process a video/live source in a background thread.

    For file input, playback is limited to the source FPS.
    For webcam/RTSP/HTTP input, old frames are dropped when processing is slow so
    the display stays close to real time.
    """
    reader_thread = None
    cap = None
    exporter = None
    try:
        _configure_runtime()
        performance_profile, performance_cfg = _get_performance_cfg(app)
        model_imgsz = int(performance_cfg.get("model_imgsz", MODEL_IMGSZ))
        process_width = int(performance_cfg.get("process_width", 0))
        display_every_n = max(1, int(performance_cfg.get("display_every_n", 1)))
        detect_interval = max(1, int(performance_cfg.get("detect_interval", getattr(app, "detect_interval", 1))))
        max_det = max(1, int(performance_cfg.get("max_det", 300)))

        device = _select_torch_device()
        use_half = bool(performance_cfg.get("half_cuda", True)) and str(device).startswith("cuda")

        model = YOLO(model_path)
        try:
            model.to(device)
        except Exception:
            device = "cpu"
            use_half = False
            model.to("cpu")

        _warmup_model(model, model_imgsz=model_imgsz, device=device, use_half=use_half, max_det=max_det)

        mapping_warnings = _validate_model_class_mapping(model)
        if mapping_warnings:
            warning_text = "Class mapping mismatch: " + "; ".join(mapping_warnings)
            print("[Model Warning]", warning_text)
            with app.state_lock:
                app.worker_state["status"] = warning_text

        tracker = DeepSort(
            max_age=TRACK_MAX_AGE,
            n_init=TRACK_N_INIT,
            max_cosine_distance=TRACK_MAX_COSINE_DISTANCE,
            nn_budget=TRACK_NN_BUDGET,
        )

        realtime_source = _is_realtime_source(video_path)
        cap = _open_capture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Không thể mở video/camera/stream")

        fps_input = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps_input <= 1.0 or fps_input > 240.0:
            fps_input = 30.0

        frame_duration = 1.0 / fps_input
        buffer_size = REALTIME_QUEUE_SIZE if realtime_source else max(8, int(fps_input * FILE_QUEUE_SECONDS))
        frame_queue = Queue(maxsize=buffer_size)
        reader_done = threading.Event()

        def video_reader():
            try:
                while cap.isOpened() and not app.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _put_frame(frame_queue, frame, realtime_source, app.stop_event)
            finally:
                reader_done.set()
                cap.release()

        with app.state_lock:
            app.worker_state["status"] = "Opening live source..." if realtime_source else "Buffering video..."

        reader_thread = threading.Thread(target=video_reader, daemon=True)
        reader_thread.start()

        if not realtime_source:
            while (
                frame_queue.qsize() < min(buffer_size, 4)
                and not reader_done.is_set()
                and not app.stop_event.is_set()
            ):
                time.sleep(0.05)

        with app.state_lock:
            source_mode = "live" if realtime_source else "file"
            app.worker_state["status"] = (
                f"{source_mode.title()} source ready @ {fps_input:.1f} FPS | "
                f"perf={performance_profile}, device={device}, imgsz={model_imgsz}, width={process_width or 'native'}"
            )

        if app.region_template and app.region_template.loaded:
            # Center is always a valid row in the UI. If template.csv has an
            # explicit center polygon, get_direction_region() will use it;
            # otherwise any point outside the outer polygons becomes center.
            valid_branches = (set(app.region_template.regions.keys()) & VALID_BRANCHES) | {"center"}
        else:
            valid_branches = set(VALID_BRANCHES)

        # Cumulative totals use real tracking events:
        # - IN when a tracked vehicle enters a region/branch.
        # - OUT when that same tracked vehicle leaves the region/branch.
        # Therefore IN and OUT are not forced to be equal at the same frame;
        # temporary mismatch means vehicles are currently inside a region.
        branch_count_total = Counter()
        branch_class_count_total = Counter()
        branch_event_windows = {
            (branch, direction): deque()
            for branch in valid_branches
            for direction in DIRECTIONS
        }

        if _fluid_export_enabled(app):
            exporter = FluidFlowExporter(
                root_dir=app.export_root_var.get().strip() or "flow_exports",
                source_path=video_path,
                model_path=model_path,
                fps=fps_input,
                valid_regions=valid_branches,
                template_path=app.template_mapping_path_var.get().strip(),
                bin_seconds=getattr(app, "fluid_bin_seconds", 1.0),
                smooth_seconds=getattr(app, "fluid_smooth_seconds", 5.0),
                track_sample_seconds=getattr(app, "track_sample_seconds", 0.25),
                region_state_sample_seconds=getattr(app, "region_state_sample_seconds", 1.0),
                hidden_left_enabled=bool(app.infer_hidden_left_var.get()),
            )
            with app.state_lock:
                app.worker_state["status"] = f"Exporting flow to {exporter.output_dir}"

        track_meta = {}
        prev_time = time.time()
        playback_start = time.time()
        processing_start = time.time()
        frame_id = 0

        while not app.stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except Empty:
                if reader_done.is_set() and frame_queue.empty():
                    break
                continue

            frame_id += 1
            current_time = time.time() - processing_start if realtime_source else frame_id / fps_input
            frame = _resize_for_processing(frame, process_width)
            should_update_display = (frame_id % display_every_n == 0)
            display_frame = frame.copy() if should_update_display else None

            detections = []
            if frame_id % detect_interval == 0:
                results = _predict_yolo(
                    model,
                    frame,
                    model_imgsz=model_imgsz,
                    device=device,
                    use_half=use_half,
                    max_det=max_det,
                )

                frame_h, frame_w = frame.shape[:2]
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        x1 = max(0, min(x1, frame_w - 1))
                        y1 = max(0, min(y1, frame_h - 1))
                        x2 = max(0, min(x2, frame_w - 1))
                        y2 = max(0, min(y2, frame_h - 1))

                        if not _passes_detection_filters(cls, conf, x1, y1, x2, y2, frame_w, frame_h):
                            continue

                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], conf, cls))

                detections = _suppress_duplicate_detections(detections)

            tracks = tracker.update_tracks(detections, frame=frame)
            primary_tracks, duplicate_track_ids = _suppress_duplicate_tracks(tracks, track_meta)
            for duplicate_id in duplicate_track_ids:
                # If this duplicate/ghost ID had already emitted IN before being
                # recognized as duplicate, close that visit with OUT before
                # removing it. Otherwise Moto In can stay higher than Moto Out.
                duplicate_meta = track_meta.get(duplicate_id)
                if duplicate_meta is not None:
                    _emit_active_branch_out(
                        duplicate_meta,
                        duplicate_id,
                        frame_id,
                        current_time,
                        valid_branches,
                        branch_count_total,
                        branch_class_count_total,
                        branch_event_windows,
                    )
                track_meta.pop(duplicate_id, None)

            _merge_stale_duplicate_meta(track_meta, primary_tracks, frame_id)

            active_tracks = 0
            active_track_ids = {track.track_id for track in primary_tracks}

            if DEBUG_TRACK_LOGS:
                print(
                    f"[DeepSort] Frame={frame_id} detections={len(detections)} "
                    f"total_tracks={len(tracks)} primary_tracks={len(primary_tracks)} "
                    f"suppressed={len(duplicate_track_ids)}"
                )

            # Tracks missing for only a few frames are kept to avoid false OUT events.
            # But if the last box was close to the frame edge, close the visit
            # sooner because the vehicle likely left the camera view.
            tracks_to_remove = []
            frame_h, frame_w = frame.shape[:2]
            for track_id, meta in list(track_meta.items()):
                if track_id in active_track_ids:
                    continue

                missing_frames = frame_id - meta.get("last_seen_frame", frame_id)
                near_edge = _box_near_frame_edge(meta.get("last_box"), frame_w, frame_h)
                lost_limit = EDGE_LOST_OUT_FRAMES if near_edge else LOST_OUT_FRAMES
                if missing_frames < lost_limit:
                    continue

                last_centroid = meta.get("last_centroid")
                if (
                    exporter is not None
                    and bool(app.infer_hidden_left_var.get())
                    and meta.get("last_fluid_region") == "center"
                    and _near_hidden_left_gate(last_centroid, frame_w, app)
                    and not meta.get("fluid_closed_to_left", False)
                ):
                    _export_track_transition(
                        exporter,
                        meta=meta,
                        track_id=track_id,
                        frame_id=frame_id,
                        current_time=current_time,
                        from_region="center",
                        to_region="left",
                        cls=meta.get("active_branch_cls", meta.get("stable_cls", meta.get("cls", 2))),
                        box=meta.get("last_box"),
                        centroid=last_centroid,
                        source="inferred",
                        confidence=0.70,
                        reason="lost_near_left_gate",
                    )
                    meta["fluid_closed_to_left"] = True

                emitted = _emit_active_branch_out(
                    meta,
                    track_id,
                    frame_id,
                    current_time,
                    valid_branches,
                    branch_count_total,
                    branch_class_count_total,
                    branch_event_windows,
                )
                if emitted and DEBUG_TRACK_LOGS:
                    print(f"[Flow] OUT by lost track: track_id={track_id} near_edge={near_edge}")

                tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del track_meta[track_id]

            for track in primary_tracks:
                track_id = track.track_id
                l, t, r, b = _track_ltrb(track)
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
                meta["last_box"] = (l, t, r, b)
                meta["duplicate_of"] = None
                cls = update_stable_class(meta, det_cls, det_conf)

                stable_region = update_stable_region(meta, raw_region)
                meta["current_region"] = stable_region
                meta["last_centroid"] = centroid

                if exporter is not None:
                    exporter.log_track_sample(
                        time_s=current_time,
                        frame_id=frame_id,
                        track_id=track_id,
                        cls=cls,
                        raw_region=raw_region,
                        stable_region=stable_region,
                        active_branch=meta.get("active_branch"),
                        box=(l, t, r, b),
                        source="observed",
                    )

                    if stable_region in valid_branches:
                        previous_fluid_region = meta.get("last_fluid_region")
                        if previous_fluid_region is None:
                            # First stable region for this track. If it appears
                            # in center close to the hidden-left gate, create an
                            # inferred left->center input edge for fluid replay.
                            if (
                                bool(app.infer_hidden_left_var.get())
                                and stable_region == "center"
                                and _near_hidden_left_gate(centroid, frame.shape[1], app)
                                and not meta.get("fluid_started_from_left", False)
                            ):
                                _export_track_transition(
                                    exporter,
                                    meta=meta,
                                    track_id=track_id,
                                    frame_id=frame_id,
                                    current_time=current_time,
                                    from_region="left",
                                    to_region="center",
                                    cls=cls,
                                    box=(l, t, r, b),
                                    centroid=centroid,
                                    source="inferred",
                                    confidence=0.70,
                                    reason="first_seen_center_near_left_gate",
                                )
                                meta["fluid_started_from_left"] = True
                            meta["last_fluid_region"] = stable_region
                        elif previous_fluid_region != stable_region:
                            _export_track_transition(
                                exporter,
                                meta=meta,
                                track_id=track_id,
                                frame_id=frame_id,
                                current_time=current_time,
                                from_region=previous_fluid_region,
                                to_region=stable_region,
                                cls=cls,
                                box=(l, t, r, b),
                                centroid=centroid,
                                source="observed",
                                confidence=1.0,
                                reason="stable_region_change",
                            )
                            meta["last_fluid_region"] = stable_region

                if stable_region is not None:
                    current_is_branch = stable_region in valid_branches
                    active_branch = meta.get("active_branch")

                    if active_branch in valid_branches and stable_region != active_branch:
                        emitted = _emit_active_branch_out(
                            meta,
                            track_id,
                            frame_id,
                            current_time,
                            valid_branches,
                            branch_count_total,
                            branch_class_count_total,
                            branch_event_windows,
                        )
                        if emitted and DEBUG_TRACK_LOGS:
                            print(
                                f"[Flow] OUT: branch={active_branch} -> {stable_region} "
                                f"track_id={track_id}"
                            )

                    if current_is_branch and meta.get("active_branch") != stable_region:
                        event_cls = int(cls)
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
                            event_cls=event_cls,
                        )
                        if emitted and DEBUG_TRACK_LOGS:
                            print(f"[Flow] IN: branch={stable_region} cls={event_cls} track_id={track_id}")
                        mark_branch_enter(meta, stable_region, counted=emitted, event_cls=event_cls)

                if should_update_display and display_frame is not None:
                    color = CLASS_COLORS.get(cls, (255, 255, 255))

                    label = f"{CLASS_NAMES.get(cls, cls)} #{track_id}"
                    region_label = stable_region if stable_region is not None else raw_region
                    if region_label:
                        label += f" {region_label}"

                    cv2.rectangle(display_frame, (l, t), (r, b), color, 2)
                    cv2.putText(display_frame, label, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(display_frame, centroid, 3, color, -1)

                if DEBUG_TRACK_LOGS:
                    print(
                        f"[DeepSort]   Track={track_id} stable_cls={cls} raw_cls={det_cls} "
                        f"conf={det_conf} label={CLASS_NAMES.get(cls, cls)} "
                        f"ltrb=({l},{t},{r},{b}) centroid={centroid} raw={raw_region} "
                        f"stable={stable_region} active_branch={meta.get('active_branch')} "
                        f"confirmed={track.is_confirmed()} time_since_update={track.time_since_update}"
                    )

            if should_update_display and display_frame is not None and app.display_template_var.get():
                draw_region_overlay(display_frame, app.region_margin, app.region_template)

            cleanup_flow_windows(branch_event_windows, current_time, app.flow_window)

            branch_current_pce = {branch: 0.0 for branch in valid_branches}
            branch_current_count = {branch: 0 for branch in valid_branches}
            for meta in track_meta.values():
                missing_frames = frame_id - meta.get("last_seen_frame", frame_id)
                if missing_frames > TRACK_COUNT_HOLD_FRAMES:
                    continue

                region = meta.get("current_region")
                if region in valid_branches:
                    branch_current_pce[region] += meta.get("weight", 0.0)
                    branch_current_count[region] += 1

            if exporter is not None:
                exporter.write_region_state_snapshot(
                    time_s=current_time,
                    frame_id=frame_id,
                    region_current_count=branch_current_count,
                    region_current_pce=branch_current_pce,
                )

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time

            pil_image = None
            if should_update_display and display_frame is not None:
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((880, 620), Image.LANCZOS)

            total_current_pce = 0.0
            total_in_count = 0
            metric_updates = {}

            for branch in BRANCH_ORDER:
                if branch not in valid_branches:
                    metric_updates[f"{branch}_pce"] = "0.0"
                    metric_updates[f"{branch}_count"] = "0"
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
                count_now = branch_current_count.get(branch, 0)
                in_flow = calc_veh_per_min(branch_event_windows[(branch, "in")], app.flow_window)

                metric_updates[f"{branch}_pce"] = f"{pce_now:.1f}"
                metric_updates[f"{branch}_count"] = str(count_now)
                # Kept in worker_state for compatibility, even though app.py no
                # longer shows branch-level veh/min columns.
                metric_updates[f"{branch}_in_flow"] = f"{in_flow:.1f}"
                metric_updates[f"{branch}_out_flow"] = f"{calc_veh_per_min(branch_event_windows[(branch, 'out')], app.flow_window):.1f}"
                metric_updates[f"{branch}_in_count"] = str(branch_count_total[(branch, "in")])
                metric_updates[f"{branch}_out_count"] = str(branch_count_total[(branch, "out")])

                for cls_id in DISPLAY_CLASS_IDS:
                    class_name = CLASS_NAMES[cls_id]
                    metric_updates[f"{branch}_{class_name}_in"] = str(branch_class_count_total[(branch, "in", cls_id)])
                    metric_updates[f"{branch}_{class_name}_out"] = str(branch_class_count_total[(branch, "out", cls_id)])

                total_current_pce += pce_now
                if branch != "center":
                    # Total veh uses outer-direction IN events only. Center is
                    # shown as its own row but excluded to avoid double counting
                    # one vehicle again when it passes through the intersection.
                    total_in_count += branch_count_total[(branch, "in")]

            realtime_ratio = fps / fps_input if fps_input > 0 else 0.0
            with app.state_lock:
                if pil_image is not None:
                    app.latest_pil_image = pil_image
                source_label = "live" if realtime_source else "file"
                app.worker_state["status"] = (
                    f"Running ({source_label}, {performance_profile}, {device}, "
                    f"{realtime_ratio:.2f}x realtime)"
                )
                app.worker_state["frame"] = str(frame_id)
                app.worker_state["fps"] = f"{fps:.1f}"
                app.worker_state["active_tracks"] = str(active_tracks)
                app.worker_state["current_pce"] = f"{total_current_pce:.1f}"
                app.worker_state["flow_veh_pm"] = str(total_in_count)
                app.worker_state.update(metric_updates)

            if not realtime_source:
                expected_display = playback_start + (frame_id - 1) * frame_duration
                delay = expected_display - time.time()
                if delay > 0:
                    time.sleep(delay)

        # For a finished video file, close any still-open counted visits so the
        # final report is balanced. This is not done while manually stopping a
        # live stream because those vehicles may simply still be inside the view.
        if (not app.stop_event.is_set()) and (not realtime_source):
            final_time = frame_id / fps_input if fps_input > 0 else current_time
            for track_id, meta in list(track_meta.items()):
                _emit_active_branch_out(
                    meta,
                    track_id,
                    frame_id + 1,
                    final_time,
                    valid_branches,
                    branch_count_total,
                    branch_class_count_total,
                    branch_event_windows,
                )

            final_updates = {}
            total_in_count = 0
            for branch in BRANCH_ORDER:
                if branch not in valid_branches:
                    final_updates[f"{branch}_pce"] = "0.0"
                    final_updates[f"{branch}_count"] = "0"
                    final_updates[f"{branch}_in_flow"] = "0.0"
                    final_updates[f"{branch}_out_flow"] = "0.0"
                    final_updates[f"{branch}_in_count"] = "0"
                    final_updates[f"{branch}_out_count"] = "0"
                    for cls_id in DISPLAY_CLASS_IDS:
                        class_name = CLASS_NAMES[cls_id]
                        final_updates[f"{branch}_{class_name}_in"] = "0"
                        final_updates[f"{branch}_{class_name}_out"] = "0"
                    continue

                final_updates[f"{branch}_pce"] = "0.0"
                final_updates[f"{branch}_count"] = "0"
                final_updates[f"{branch}_in_flow"] = f"{calc_veh_per_min(branch_event_windows[(branch, 'in')], app.flow_window):.1f}"
                final_updates[f"{branch}_out_flow"] = f"{calc_veh_per_min(branch_event_windows[(branch, 'out')], app.flow_window):.1f}"
                final_updates[f"{branch}_in_count"] = str(branch_count_total[(branch, "in")])
                final_updates[f"{branch}_out_count"] = str(branch_count_total[(branch, "out")])

                for cls_id in DISPLAY_CLASS_IDS:
                    class_name = CLASS_NAMES[cls_id]
                    final_updates[f"{branch}_{class_name}_in"] = str(branch_class_count_total[(branch, "in", cls_id)])
                    final_updates[f"{branch}_{class_name}_out"] = str(branch_class_count_total[(branch, "out", cls_id)])

                if branch != "center":
                    total_in_count += branch_count_total[(branch, "in")]

            with app.state_lock:
                app.worker_state["active_tracks"] = "0"
                app.worker_state["current_pce"] = "0.0"
                app.worker_state["flow_veh_pm"] = str(total_in_count)
                app.worker_state.update(final_updates)

        if exporter is not None:
            exporter.write_region_state_snapshot(
                time_s=(frame_id / fps_input if (not realtime_source and fps_input > 0) else current_time),
                frame_id=frame_id,
                region_current_count={branch: 0 for branch in valid_branches},
                region_current_pce={branch: 0.0 for branch in valid_branches},
                force=True,
            )

        with app.state_lock:
            if exporter is not None:
                app.worker_state["status"] = ("Stopped" if app.stop_event.is_set() else "Finished") + f" | export: {exporter.output_dir}"
            else:
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
        if exporter is not None:
            try:
                exporter.close()
            except Exception as export_exc:
                print(f"[FluidExport] Error while closing exporter: {export_exc}")
        if cap is not None and cap.isOpened():
            cap.release()
