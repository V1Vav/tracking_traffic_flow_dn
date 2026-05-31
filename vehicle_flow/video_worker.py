"""Worker xử lý video cho YOLO detection, DeepSORT tracking và cập nhật chỉ số."""

import os
import threading
import time
from collections import Counter, deque
from queue import Empty, Full, Queue

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - torch thường được cài cùng ultralytics
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
    ASYNC_DISPLAY_CONVERSION,
    CPU_THREAD_COUNT,
    COUNTED_CLASS_IDS,
    DETECT_CLASS_IDS,
    CV2_NUM_THREADS,
    DEBUG_TRACK_LOGS,
    DISPLAY_CONVERSION_QUEUE_SIZE,
    ENABLE_SOURCE_FPS_DOWNSAMPLE,
    EDGE_LOST_OUT_FRAMES,
    FRAME_EXIT_MARGIN_RATIO,
    HIDDEN_LEFT_INBOUND_REGION,
    HIDDEN_LEFT_OUTBOUND_REGION,
    INBOUND_LANE_REGIONS,
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
    TORCH_INTEROP_THREADS,
    TARGET_PROCESS_FPS,
    USE_CPU_THREADS,
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
    """Cảnh báo sớm nếu thứ tự class của model fine-tune khác config.py."""
    model_names = _normalize_model_names(getattr(model, "names", {}))
    expected = {idx: name.lower() for idx, name in EXPECTED_MODEL_NAMES.items()}
    mismatches = []

    for idx, expected_name in expected.items():
        actual_name = model_names.get(idx)
        if actual_name != expected_name:
            mismatches.append(f"id {idx}: kỳ_vọng={expected_name}, thực_tế={actual_name}")

    return mismatches


def _is_realtime_source(source):
    """Trả về True với chỉ số webcam và URL stream phổ biến."""
    source_text = str(source).strip()
    if source_text.isdigit():
        return True
    return source_text.lower().startswith(REALTIME_SOURCE_PREFIXES)


def _open_capture(source):
    """Mở file, chỉ số webcam hoặc URL stream bằng OpenCV."""
    source_text = str(source).strip()
    if source_text.isdigit():
        return cv2.VideoCapture(int(source_text))
    return cv2.VideoCapture(source_text)


def _sanitize_capture_fps(raw_fps, fallback=30.0):
    """Trả về giá trị FPS dùng được từ metadata OpenCV.

    Một số camera/codec báo 0, 1000 hoặc giá trị không thực tế. Với file video
    bình thường, vẫn giữ FPS cao nhưng hợp lệ như 60/120 để bộ giảm FPS có thể
    đưa về đúng FPS xử lý mục tiêu.
    """
    try:
        fps = float(raw_fps or 0.0)
    except Exception:
        fps = 0.0

    if fps < 1.0 or fps > 1000.0:
        return float(fallback), False
    return fps, True


def _resolve_target_process_fps(source_fps, performance_cfg=None):
    """Xác định FPS đưa vào YOLO/DeepSORT sau khi giảm FPS nguồn."""
    cfg = performance_cfg or {}
    try:
        requested = float(cfg.get("target_process_fps", TARGET_PROCESS_FPS))
    except Exception:
        requested = float(TARGET_PROCESS_FPS)

    if (not ENABLE_SOURCE_FPS_DOWNSAMPLE) or requested <= 0:
        return float(source_fps), False

    effective = min(float(source_fps), requested)
    enabled = effective < float(source_fps) - 0.01
    return max(1.0, effective), enabled


def _put_frame(frame_queue, frame, drop_old_frames, stop_event):
    """Đưa một frame vào hàng đợi xử lý.

    Khi drop_old_frames=True, queue hoạt động như bộ đệm frame mới nhất:
    frame cũ bị bỏ thay vì tích lũy độ trễ. Dùng cho nguồn live và replay file
    theo thời gian thực.
    """
    if not drop_old_frames:
        while not stop_event.is_set():
            try:
                frame_queue.put(frame, timeout=0.1)
                return
            except Full:
                continue
        return

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
    """Lọc theo confidence riêng từng class và hình học tương đối để detection ổn định."""
    if cls not in DETECT_CLASS_IDS:
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
    """Phần giao nhau chia cho diện tích box nhỏ hơn."""
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

    # Tránh xóa nhầm hai xe sát nhau: yêu cầu có một phần chồng lấn trước khi dùng
    # khoảng cách tâm làm tín hiệu trùng bổ sung.
    if iou_value >= 0.12 and _center_distance_ratio(box_a, box_b) <= center_ratio_threshold:
        return True

    return False


def _detections_can_suppress(det_a, det_b):
    """Trả về True khi bước khử trùng có thể xóa một detection.

    Giữ chồng lấn person-vs-vehicle vì box người/người lái có thể hợp lệ khi
    chồng lên box xe máy/ô tô. Vehicle-vs-vehicle vẫn không phụ thuộc class để
    gộp các box trùng do class nhảy như car/motorbike trên cùng object.
    """
    cls_a = int(det_a[2])
    cls_b = int(det_b[2])
    counted_a = cls_a in COUNTED_CLASS_IDS
    counted_b = cls_b in COUNTED_CLASS_IDS
    if counted_a != counted_b:
        return False
    return True


def _suppress_duplicate_detections(detections):
    """Khử detection trùng trước DeepSORT.

    Các phương tiện được đếm sẽ được khử trùng không phụ thuộc class để tránh
    một xe thành nhiều ID. Detection person được giữ riêng với vehicle để vẫn
    hiển thị được mà không ảnh hưởng tracking/counting phương tiện.
    """
    if len(detections) <= 1:
        return detections

    # Giữ detection confidence cao trước. Nếu confidence bằng nhau, ưu tiên
    # box nhỏ/gọn hơn vì ít khả năng bao nhiều xe cùng lúc.
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
            if not _detections_can_suppress(det, kept_det):
                continue
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
    # DeepSORT có thể giữ track nội bộ rất lâu, nhưng vẽ/đếm
    # box cũ sẽ gây đúng hiện tượng trong ảnh: nhiều ID còn nằm
    # quanh cùng một xe. Chỉ track mới được dùng để hiển thị/đếm.
    return track.is_confirmed() and track.time_since_update <= TRACK_DISPLAY_MAX_AGE


def _track_ltrb(track):
    l, t, r, b = map(int, track.to_ltrb())
    return l, t, r, b


def _box_near_frame_edge(box, frame_width, frame_height):
    """Trả về True nếu box gần nhất sắp rời khỏi vùng camera."""
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
    """Đóng một lượt branch đang hoạt động nếu event IN đã được đếm.

    Cách này giữ IN/OUT theo từng lượt track mà không ép OUT xảy ra cùng lúc
    với IN. Nó cũng dùng class đã lưu tại thời điểm IN, nên việc làm mượt class
    sau khi vào vùng không thể biến Moto In thành Car Out.
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


def _track_effective_cls(track, track_meta):
    det_cls = getattr(track, "det_class", None)
    if det_cls is not None:
        try:
            return int(det_cls)
        except Exception:
            pass
    meta = track_meta.get(track.track_id, {})
    cls = meta.get("stable_cls", meta.get("cls"))
    try:
        return int(cls) if cls is not None else None
    except Exception:
        return None


def _tracks_can_suppress(track_a, track_b, track_meta):
    cls_a = _track_effective_cls(track_a, track_meta)
    cls_b = _track_effective_cls(track_b, track_meta)
    counted_a = cls_a in COUNTED_CLASS_IDS
    counted_b = cls_b in COUNTED_CLASS_IDS
    if counted_a != counted_b:
        return False
    return True


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
    """Trả về các track chính còn mới và ID track trùng cần bỏ qua/xóa."""
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
            if not _tracks_can_suppress(track, kept_track, track_meta):
                continue
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
    """Chuyển metadata cũ sang ID mới khi DeepSORT đổi ID trên cùng một xe.

    Việc này không đổi ID nội bộ của DeepSORT, nhưng giữ lại class ổn định,
    branch đang hoạt động và trạng thái cooldown, nhờ đó một xe thật ít bị
    đếm lại sau khi bị đổi ID.
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
    """Cổng heuristic cho nhánh trái bị khuất/che.

    Nếu track xuất hiện hoặc biến mất trong center gần ngưỡng x này,
    exporter có thể tạo cạnh flow suy luận left<->center. Việc này không
    thay đổi logic tracking/counting thật.
    """
    if centroid is None or frame_width <= 0:
        return False
    try:
        ratio = float(getattr(app, "left_gate_x_ratio", 0.30))
    except Exception:
        ratio = 0.30
    return float(centroid[0]) <= frame_width * ratio


def _is_counted_cls(cls):
    try:
        return int(cls) in COUNTED_CLASS_IDS
    except Exception:
        return False


def _is_detect_cls(cls):
    try:
        return int(cls) in DETECT_CLASS_IDS
    except Exception:
        return False


def _fluid_export_enabled(app):
    try:
        return bool(app.export_fluid_var.get())
    except Exception:
        return False


def _get_performance_cfg(app):
    """Xác định preset hiệu năng đã chọn mà không đổi trạng thái UI."""
    profile = str(getattr(app, "performance_profile", DEFAULT_PERFORMANCE_PROFILE)).strip().lower()
    if profile not in PERFORMANCE_PROFILES:
        profile = DEFAULT_PERFORMANCE_PROFILE
    cfg = PERFORMANCE_PROFILES[profile].copy()
    app_cfg = getattr(app, "performance_cfg", None)
    if isinstance(app_cfg, dict):
        cfg.update(app_cfg)
    return profile, cfg


def _resolve_cpu_thread_count(performance_cfg=None):
    """Trả về số luồng CPU dùng cho OpenCV/PyTorch CPU."""
    if not USE_CPU_THREADS:
        return 0

    cfg = performance_cfg or {}
    raw_value = cfg.get("cpu_thread_count", CPU_THREAD_COUNT)
    try:
        requested = int(raw_value)
    except Exception:
        requested = 0

    total_cores = os.cpu_count() or 4
    if requested <= 0:
        # Chừa tài nguyên cho UI Tkinter, hệ điều hành, đọc video và lịch GPU driver.
        requested = max(1, total_cores - 2)

    return max(1, min(requested, total_cores))


def _configure_runtime(performance_cfg=None):
    """Thiết lập runtime cho xử lý CPU đa lõi và thông lượng GPU/CV."""
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass

    cpu_threads = _resolve_cpu_thread_count(performance_cfg)

    # OpenCV dùng giá trị này cho decode/resize/chuyển màu/vẽ nếu được hỗ trợ.
    opencv_threads = int(CV2_NUM_THREADS) if CV2_NUM_THREADS and CV2_NUM_THREADS > 0 else cpu_threads
    if opencv_threads > 0:
        try:
            cv2.setNumThreads(int(opencv_threads))
        except Exception:
            pass

    if torch is not None:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Nếu YOLO fallback sang CPU, thiết lập này cho phép PyTorch dùng nhiều core.
        # Khi chạy CUDA, nó vẫn hỗ trợ một số bước tiền/hậu xử lý trên CPU.
        if cpu_threads > 0:
            try:
                torch.set_num_threads(int(cpu_threads))
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(max(1, min(int(TORCH_INTEROP_THREADS), int(cpu_threads))))
            except Exception:
                # PyTorch có thể từ chối đổi interop sau khi đã bắt đầu xử lý song song.
                pass

    return cpu_threads


def _put_display_frame(display_queue, display_frame, display_width, display_height, stop_event):
    """Gửi frame hiển thị sang luồng chuyển đổi và bỏ frame preview cũ."""
    if display_queue is None or display_frame is None:
        return False

    item = (display_frame, int(display_width), int(display_height))
    while not stop_event.is_set():
        try:
            display_queue.put_nowait(item)
            return True
        except Full:
            try:
                display_queue.get_nowait()
            except Empty:
                pass
    return False


def _display_converter_worker(app, display_queue, stop_event):
    """Chuyển frame hiển thị sang ảnh PIL bên ngoài vòng detection/tracking."""
    while not stop_event.is_set():
        try:
            item = display_queue.get(timeout=0.1)
        except Empty:
            continue

        if item is None:
            break

        display_frame, display_width, display_height = item
        try:
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            if display_width > 0 and display_height > 0:
                rgb_frame = cv2.resize(rgb_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            pil_image = Image.fromarray(rgb_frame)
            with app.state_lock:
                app.latest_pil_image = pil_image
        except Exception as exc:
            print(f"[DisplayWorker] bỏ qua frame: {exc}")


def _select_torch_device():
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
    return "cpu"


def _resize_for_processing(frame, process_width):
    """Resize frame trước detection/tracking; template vùng sẽ tự scale."""
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
    """Chạy YOLO với cấu hình nhanh, có fallback cho phiên bản ultralytics cũ."""
    kwargs = {
        "imgsz": int(model_imgsz),
        "conf": MODEL_CONF,
        "iou": MODEL_IOU,
        "classes": list(DETECT_CLASS_IDS),
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
        print(f"[Hiệu năng] Bỏ qua warmup YOLO: {exc}")


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
    """Xử lý video/stream trong worker nền hoặc trong chế độ headless.

    Với GUI, worker cập nhật ảnh preview và chỉ số realtime.
    Với headless, worker bỏ toàn bộ preview, mặc định không bỏ frame và xuất flow CSV.
    """
    reader_thread = None
    display_thread = None
    display_queue = None
    cap = None
    exporter = None
    try:
        headless_mode = bool(getattr(app, "headless", False))
        last_headless_progress_print = 0.0
        performance_profile, performance_cfg = _get_performance_cfg(app)
        cpu_threads = _configure_runtime(performance_cfg)
        model_imgsz = int(performance_cfg.get("model_imgsz", MODEL_IMGSZ))
        process_width = int(performance_cfg.get("process_width", 0))
        display_every_n = max(1, int(performance_cfg.get("display_every_n", 1)))
        detect_interval = max(1, int(performance_cfg.get("detect_interval", getattr(app, "detect_interval", 1))))
        max_det = max(1, int(performance_cfg.get("max_det", 300)))
        track_ignored_classes = bool(performance_cfg.get("track_ignored_classes", True))
        drop_frames_when_slow = bool(performance_cfg.get("drop_frames_when_slow", False))
        display_width = max(1, int(performance_cfg.get("display_width", 880)))
        display_height = max(1, int(performance_cfg.get("display_height", 620)))
        async_display = bool(performance_cfg.get("async_display", ASYNC_DISPLAY_CONVERSION))

        if headless_mode:
            # Chế độ headless chỉ xuất CSV, không tạo ảnh preview để tránh tốn CPU/RAM.
            async_display = False
            display_every_n = 10**9
            display_width = 0
            display_height = 0

        if async_display:
            display_queue = Queue(maxsize=max(1, int(performance_cfg.get("display_queue_size", DISPLAY_CONVERSION_QUEUE_SIZE))))
            display_thread = threading.Thread(
                target=_display_converter_worker,
                args=(app, display_queue, app.stop_event),
                daemon=True,
            )
            display_thread.start()

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
            print("[Cảnh báo model]", warning_text)
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

        raw_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps_input, fps_metadata_valid = _sanitize_capture_fps(raw_fps, fallback=30.0)
        target_process_fps, fps_downsample_enabled = _resolve_target_process_fps(fps_input, performance_cfg)
        target_frame_period = 1.0 / target_process_fps if target_process_fps > 0 else 0.0

        low_latency_queue = realtime_source or drop_frames_when_slow
        realtime_queue_size = max(1, int(performance_cfg.get("realtime_queue_size", REALTIME_QUEUE_SIZE)))
        buffer_size = realtime_queue_size if low_latency_queue else max(8, int(target_process_fps * FILE_QUEUE_SECONDS))
        frame_queue = Queue(maxsize=buffer_size)
        reader_done = threading.Event()

        def video_reader():
            source_frame_id = 0
            next_sample_time = 0.0
            live_last_emit_time = 0.0
            reader_start = time.time()
            file_playback_start = time.time()

            try:
                while cap.isOpened() and not app.stop_event.is_set():
                    if realtime_source:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        source_frame_id += 1
                        now = time.time()
                        # Giới hạn mềm nguồn live để tránh đưa 60/120 FPS
                        # frame camera vào pipeline xử lý 30 FPS. Queue
                        # vẫn bỏ frame cũ nếu xử lý chậm hơn.
                        if target_frame_period > 0 and live_last_emit_time > 0:
                            if now - live_last_emit_time < target_frame_period:
                                continue
                        live_last_emit_time = now
                        source_time = now - reader_start
                        _put_frame(frame_queue, (frame, source_frame_id, source_time), True, app.stop_event)
                        continue

                    # Với file: dùng grab() cho frame bị bỏ để OpenCV tránh
                    # decode đầy đủ các frame không được xử lý. Cách này
                    # nhẹ hơn nhiều với video 60/120 FPS.
                    ret = cap.grab()
                    if not ret:
                        break

                    source_frame_id += 1
                    source_time = (source_frame_id - 1) / fps_input if fps_input > 0 else 0.0

                    if fps_downsample_enabled and target_frame_period > 0:
                        if source_time + 1e-9 < next_sample_time:
                            continue
                        while next_sample_time <= source_time + 1e-9:
                            next_sample_time += target_frame_period

                    if drop_frames_when_slow:
                        # Replay file theo realtime: thời gian input == thời gian output.
                        # Reader chạy theo source_time; nếu worker đang bận,
                        # frame cũ trong queue bị bỏ thay vì tích lũy độ trễ.
                        delay = file_playback_start + source_time - time.time()
                        if delay > 0:
                            time.sleep(delay)

                    ret, frame = cap.retrieve()
                    if not ret:
                        break

                    _put_frame(frame_queue, (frame, source_frame_id, source_time), low_latency_queue, app.stop_event)
            finally:
                reader_done.set()
                cap.release()

        with app.state_lock:
            app.worker_state["status"] = "Đang mở nguồn live..." if realtime_source else "Đang đệm video..."

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
            fps_note = "valid" if fps_metadata_valid else "fallback"
            downsample_note = f", process_fps={target_process_fps:.1f}"
            if fps_downsample_enabled:
                downsample_note += " (downsampled)"
            app.worker_state["status"] = (
                f"{source_mode.title()} source ready @ source_fps={fps_input:.1f} ({fps_note})"
                f"{downsample_note} | "
                f"perf={performance_profile}, device={device}, cpu_threads={cpu_threads}, "
                f"async_display={async_display}, skip_stale={drop_frames_when_slow}, q={buffer_size}, "
                f"display/{display_every_n}, imgsz={model_imgsz}, width={process_width or 'native'}"
            )

        if app.region_template and app.region_template.loaded:
            # Center luôn là một dòng hợp lệ trong UI. Vùng làn chỉ hợp lệ
            # khi có trong template.csv; điều này hỗ trợ template chưa đầy đủ
            # nhưng vẫn giữ bố cục 8 làn trong config.py.
            valid_branches = (set(app.region_template.regions.keys()) & VALID_BRANCHES) | {"center"}
        else:
            valid_branches = set(VALID_BRANCHES)

        # Tổng tích lũy dùng event tracking thật:
        # - IN khi phương tiện được track đi vào vùng/nhánh.
        # - OUT khi chính track đó rời vùng/nhánh.
        # Vì vậy IN và OUT không bị ép bằng nhau tại cùng một frame;
        # lệch tạm thời nghĩa là đang có xe nằm trong vùng.
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
            try:
                app.export_output_dir = exporter.output_dir
            except Exception:
                pass
            with app.state_lock:
                app.worker_state["status"] = f"Đang xuất flow vào {exporter.output_dir}"

        track_meta = {}
        prev_time = time.time()
        playback_start = time.time()
        processing_start = time.time()
        frame_id = 0                 # processed frame id, after FPS downsampling
        source_frame_id = 0          # original video/camera frame id
        last_source_time = 0.0       # original video time for export/replay

        while not app.stop_event.is_set():
            try:
                frame_item = frame_queue.get(timeout=0.1)
            except Empty:
                if reader_done.is_set() and frame_queue.empty():
                    break
                continue

            if isinstance(frame_item, tuple) and len(frame_item) == 3:
                frame, source_frame_id, source_time = frame_item
            else:
                # Fallback tương thích ngược cho item queue cũ.
                frame = frame_item
                source_frame_id = source_frame_id + 1
                source_time = time.time() - processing_start if realtime_source else frame_id / max(fps_input, 1.0)

            frame_id += 1
            current_time = float(source_time)
            last_source_time = current_time
            frame = _resize_for_processing(frame, process_width)
            should_update_display = (frame_id % display_every_n == 0)
            display_frame = frame.copy() if should_update_display else None

            detections = []
            display_only_detections = []
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

                        # Trong cảnh đông, box person có thể rất nhiều và khiến
                        # DeepSORT thành bottleneck dù person không dùng
                        # cho PCE/count/flow. Các profile nhanh vẫn giữ person hiển thị như
                        # box YOLO, nhưng không gán ID DeepSORT cho person.
                        if (cls not in COUNTED_CLASS_IDS) and (not track_ignored_classes):
                            display_only_detections.append(([x1, y1, w, h], conf, cls))
                        else:
                            detections.append(([x1, y1, w, h], conf, cls))

                detections = _suppress_duplicate_detections(detections)

            tracks = tracker.update_tracks(detections, frame=frame)
            primary_tracks, duplicate_track_ids = _suppress_duplicate_tracks(tracks, track_meta)
            for duplicate_id in duplicate_track_ids:
                # Nếu ID trùng/ghost này đã phát IN trước khi được
                # nhận ra là trùng, đóng lượt đó bằng OUT trước khi
                # xóa nó. Nếu không, Moto In có thể cao hơn Moto Out.
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

            # Track mất chỉ vài frame được giữ lại để tránh OUT giả.
            # Nhưng nếu box cuối gần mép frame, đóng lượt
            # sớm hơn vì xe có khả năng đã rời vùng camera.
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
                    and _is_counted_cls(meta.get("active_branch_cls", meta.get("stable_cls", meta.get("cls", -1))))
                ):
                    _export_track_transition(
                        exporter,
                        meta=meta,
                        track_id=track_id,
                        frame_id=frame_id,
                        current_time=current_time,
                        from_region="center",
                        to_region=HIDDEN_LEFT_OUTBOUND_REGION,
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
                    print(f"[Flow] OUT do mất track: track_id={track_id} gần_mép={near_edge}")

                tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del track_meta[track_id]

            if should_update_display and display_frame is not None and display_only_detections:
                for det_box, det_conf, det_cls in display_only_detections:
                    x, y, w, h = [int(v) for v in det_box]
                    color = CLASS_COLORS.get(int(det_cls), (180, 180, 180))
                    label = f"{CLASS_NAMES.get(int(det_cls), det_cls)} {float(det_conf):.2f}"
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 1)
                    cv2.putText(display_frame, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            for track in primary_tracks:
                track_id = track.track_id
                l, t, r, b = _track_ltrb(track)
                det_cls = getattr(track, "det_class", None)
                det_conf = getattr(track, "det_conf", None)
                if det_cls is None:
                    det_cls = track_meta.get(track_id, {}).get("stable_cls", 2)

                if not _is_detect_cls(det_cls):
                    continue

                active_tracks += 1
                centroid = centroid_from_box((l, t, r, b))
                meta = track_meta.setdefault(track_id, create_track_meta(frame_id, int(det_cls)))
                previous_centroid = meta.get("last_centroid")
                previous_region = meta.get("current_region") or meta.get("stable_region")
                raw_region = get_direction_region(
                    centroid,
                    frame.shape[1],
                    frame.shape[0],
                    app.region_margin,
                    app.region_template,
                    previous_centroid=previous_centroid,
                    current_region=previous_region,
                )

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

                    if stable_region in valid_branches and _is_counted_cls(cls):
                        previous_fluid_region = meta.get("last_fluid_region")
                        if previous_fluid_region is None:
                            # Vùng ổn định đầu tiên của track này. Nếu nó xuất hiện
                            # trong center gần cổng trái bị khuất, tạo
                            # cạnh suy luận left->center cho fluid replay.
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
                                    from_region=HIDDEN_LEFT_INBOUND_REGION,
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
                    current_is_branch = stable_region in valid_branches and _is_counted_cls(cls)
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
                            print(f"[Flow] IN: nhánh={stable_region} cls={event_cls} track_id={track_id}")
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
                cls = meta.get("stable_cls", meta.get("cls", -1))
                if region in valid_branches and _is_counted_cls(cls):
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
                if async_display:
                    _put_display_frame(display_queue, display_frame, display_width, display_height, app.stop_event)
                else:
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    if display_width > 0 and display_height > 0:
                        rgb_frame = cv2.resize(rgb_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                    pil_image = Image.fromarray(rgb_frame)

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
                # Giữ trong worker_state để tương thích, dù app.py không
                # còn hiển thị cột veh/min theo nhánh.
                metric_updates[f"{branch}_in_flow"] = f"{in_flow:.1f}"
                metric_updates[f"{branch}_out_flow"] = f"{calc_veh_per_min(branch_event_windows[(branch, 'out')], app.flow_window):.1f}"
                metric_updates[f"{branch}_in_count"] = str(branch_count_total[(branch, "in")])
                metric_updates[f"{branch}_out_count"] = str(branch_count_total[(branch, "out")])

                for cls_id in DISPLAY_CLASS_IDS:
                    class_name = CLASS_NAMES[cls_id]
                    metric_updates[f"{branch}_{class_name}_in"] = str(branch_class_count_total[(branch, "in", cls_id)])
                    metric_updates[f"{branch}_{class_name}_out"] = str(branch_class_count_total[(branch, "out", cls_id)])

                total_current_pce += pce_now
                if branch in INBOUND_LANE_REGIONS:
                    # Tổng xe chỉ dùng event vào của các làn vào.
                    # Làn ra và center bị loại để tránh đếm
                    # cùng một xe lần nữa sau khi đi qua center.
                    total_in_count += branch_count_total[(branch, "in")]

            realtime_ratio = fps / target_process_fps if target_process_fps > 0 else 0.0
            source_label = "trực tiếp" if realtime_source else "file"
            downsample_label = f", src_fps={fps_input:.1f}->proc_fps={target_process_fps:.1f}"
            display_label = "headless" if headless_mode else f"hiển_thị/{display_every_n}"
            status_text = (
                f"Đang chạy ({source_label}, {performance_profile}, {device}, "
                f"{realtime_ratio:.2f}x tốc_độ_xử_lý{downsample_label}, detect/{detect_interval}, "
                f"{display_label}, max_det={max_det}, "
                f"bỏ_frame={'bật' if drop_frames_when_slow else 'tắt'}, q={buffer_size}, cpu={cpu_threads})"
            )

            with app.state_lock:
                if pil_image is not None:
                    app.latest_pil_image = pil_image
                app.worker_state["status"] = status_text
                app.worker_state["frame"] = f"{frame_id}/{source_frame_id}"
                app.worker_state["fps"] = f"{fps:.1f}"
                app.worker_state["active_tracks"] = str(active_tracks)
                app.worker_state["current_pce"] = f"{total_current_pce:.1f}"
                app.worker_state["flow_veh_pm"] = str(total_in_count)
                app.worker_state.update(metric_updates)

            if headless_mode:
                now_print = time.time()
                progress_interval = float(getattr(app, "headless_progress_interval", 10.0) or 10.0)
                if now_print - last_headless_progress_print >= progress_interval:
                    last_headless_progress_print = now_print
                    print(
                        f"[HEADLESS] video_t={current_time:9.1f}s "
                        f"frame={frame_id}/{source_frame_id} fps={fps:5.1f} "
                        f"tracks={active_tracks:3d} pce={total_current_pce:5.1f} "
                        f"total_in={total_in_count} export={getattr(app, 'export_output_dir', '') or '-'}"
                    )

            if (not realtime_source) and (not drop_frames_when_slow) and (not headless_mode):
                # Chế độ quality/offline giữ mọi frame đã lấy mẫu và căn replay
                # trong worker. Chế độ file realtime được reader điều tiết,
                # nên khi xử lý chậm sẽ bỏ frame cũ thay vì tích lũy độ trễ.
                expected_display = playback_start + current_time
                delay = expected_display - time.time()
                if delay > 0:
                    time.sleep(delay)

        # Khi file video kết thúc, đóng mọi lượt đã đếm nhưng còn mở để
        # báo cáo cuối cân bằng. Không làm việc này khi dừng thủ công
        # với stream live vì các xe đó có thể vẫn đang nằm trong vùng nhìn.
        if (not app.stop_event.is_set()) and (not realtime_source):
            final_time = last_source_time
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

                if branch in INBOUND_LANE_REGIONS:
                    total_in_count += branch_count_total[(branch, "in")]

            with app.state_lock:
                app.worker_state["active_tracks"] = "0"
                app.worker_state["current_pce"] = "0.0"
                app.worker_state["flow_veh_pm"] = str(total_in_count)
                app.worker_state.update(final_updates)

        if exporter is not None:
            exporter.write_region_state_snapshot(
                time_s=last_source_time,
                frame_id=frame_id,
                region_current_count={branch: 0 for branch in valid_branches},
                region_current_pce={branch: 0.0 for branch in valid_branches},
                force=True,
            )

        with app.state_lock:
            if exporter is not None:
                app.worker_state["status"] = ("Đã dừng" if app.stop_event.is_set() else "Hoàn tất") + f" | export: {exporter.output_dir}"
            else:
                app.worker_state["status"] = "Đã dừng" if app.stop_event.is_set() else "Hoàn tất"

    except Exception as exc:
        with app.state_lock:
            app.worker_state["status"] = f"Lỗi: {exc}"
    finally:
        if reader_thread is not None:
            try:
                reader_thread.join(timeout=1.0)
            except Exception:
                pass
        if display_queue is not None:
            try:
                display_queue.put_nowait(None)
            except Exception:
                pass
        if display_thread is not None:
            try:
                display_thread.join(timeout=1.0)
            except Exception:
                pass
        if exporter is not None:
            try:
                exporter.close()
            except Exception as export_exc:
                print(f"[FluidExport] Lỗi khi đóng exporter: {export_exc}")
        if cap is not None and cap.isOpened():
            cap.release()
