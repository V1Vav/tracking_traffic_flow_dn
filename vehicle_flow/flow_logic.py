"""Hàm hỗ trợ đếm, chống nhiễu vùng, PCE và flow theo cửa sổ trượt."""

from collections import Counter, deque

from .config import (
    CLASS_HISTORY_LEN,
    CLASS_LOCK_MIN_FRAMES,
    CLASS_STABILITY_RATIO,
    CLASS_WEIGHTS,
    COUNTED_CLASS_IDS,
    EVENT_COOLDOWN_FRAMES,
    FLUID_INFER_MISSING_CENTER,
    FLUID_INFERRED_CENTER_CONFIDENCE,
    FLUID_TRANSITION_COOLDOWN_FRAMES,
    INBOUND_LANE_REGIONS,
    OUTBOUND_LANE_REGIONS,
    MIN_CLASS_VOTES,
    CLASS_SWITCH_MIN_VOTES,
    CLASS_SWITCH_RATIO,
    REGION_HISTORY_LEN,
    STABLE_REGION_FRAMES,
)


def create_track_meta(frame_id, cls):
    return {
        "raw_history": deque(maxlen=REGION_HISTORY_LEN),
        "stable_region": None,
        "active_branch": None,
        "active_branch_counted": False,
        "active_branch_cls": None,
        "last_seen_frame": frame_id,
        "last_event_frame": {},
        "current_region": None,
        "last_box": None,
        "duplicate_of": None,
        "last_fluid_region": None,
        "last_fluid_region_time": None,
        "last_validated_transition_frame": -10**9,
        "fluid_origin_region": None,
        "fluid_passed_center": False,
        "fluid_done": False,
        "fluid_rejected_transition_count": 0,
        "weight": CLASS_WEIGHTS.get(cls, 0.0),
        "cls": cls,
        "stable_cls": cls,
        "stable_cls_age": 0,
        "class_history": deque(maxlen=CLASS_HISTORY_LEN),
    }


def update_stable_class(meta, det_cls, det_conf=None):
    """Làm mượt dự đoán class trên một track DeepSORT bằng hysteresis.

    Kết quả class của YOLO có thể nhảy theo từng frame, đặc biệt giữa xe máy
    và ô tô trong giao thông Việt Nam đông đúc. Class ổn định được dùng cho PCE
    và đếm theo loại xe. Sau khi class ổn định, nó được khóa trong vài frame và
    chỉ đổi khi class khác thắng với tỉ lệ vote mạnh hơn.
    """
    stable_cls = meta.get("stable_cls", meta.get("cls", 2))

    if det_cls is None:
        meta["stable_cls_age"] = meta.get("stable_cls_age", 0) + 1
        meta["cls"] = stable_cls
        meta["weight"] = CLASS_WEIGHTS.get(stable_cls, 0.0)
        return stable_cls

    det_cls = int(det_cls)
    conf = 1.0 if det_conf is None else max(float(det_conf), 0.01)
    meta["class_history"].append((det_cls, conf))

    if len(meta["class_history"]) < MIN_CLASS_VOTES:
        meta["stable_cls_age"] = meta.get("stable_cls_age", 0) + 1
        meta["cls"] = stable_cls
        meta["weight"] = CLASS_WEIGHTS.get(stable_cls, 0.0)
        return stable_cls

    weighted_votes = Counter()
    raw_votes = Counter()
    for cls, cls_conf in meta["class_history"]:
        weighted_votes[cls] += cls_conf
        raw_votes[cls] += 1

    best_cls, best_score = weighted_votes.most_common(1)[0]
    total_score = sum(weighted_votes.values())
    vote_ratio = best_score / total_score if total_score > 0 else 0.0
    stable_age = meta.get("stable_cls_age", 0)

    if best_cls == stable_cls:
        stable_cls = best_cls
    elif stable_age < CLASS_LOCK_MIN_FRAMES:
        # Giữ nhãn hiện tại trong giai đoạn khóa ban đầu.
        stable_cls = stable_cls
    elif (
        raw_votes[best_cls] >= CLASS_SWITCH_MIN_VOTES
        and vote_ratio >= CLASS_SWITCH_RATIO
    ):
        stable_cls = best_cls
        stable_age = 0
    elif (
        raw_votes[best_cls] >= MIN_CLASS_VOTES
        and vote_ratio >= CLASS_STABILITY_RATIO
        and meta.get("stable_cls") is None
    ):
        stable_cls = best_cls
        stable_age = 0

    meta["stable_cls"] = stable_cls
    meta["stable_cls_age"] = stable_age + 1
    meta["cls"] = stable_cls
    meta["weight"] = CLASS_WEIGHTS.get(stable_cls, 0.0)
    return stable_cls


def update_stable_region(meta, raw_region):
    """Chống nhiễu khi đổi vùng để tránh đếm sai ở mép polygon."""
    meta["raw_history"].append(raw_region)

    if len(meta["raw_history"]) < STABLE_REGION_FRAMES:
        return meta["stable_region"]

    recent = list(meta["raw_history"])[-STABLE_REGION_FRAMES:]
    if all(region == recent[0] for region in recent):
        meta["stable_region"] = recent[0]

    return meta["stable_region"]


def is_valid_fluid_transition(from_region, to_region):
    """Cạnh hợp lệ trong mô hình 8 làn: inbound->center hoặc center->outbound."""
    return (
        from_region in INBOUND_LANE_REGIONS and to_region == "center"
    ) or (
        from_region == "center" and to_region in OUTBOUND_LANE_REGIONS
    )


def _transition_on_cooldown(meta, frame_id):
    last_frame = meta.get("last_validated_transition_frame", -10**9)
    return frame_id - last_frame < FLUID_TRANSITION_COOLDOWN_FRAMES


def _mark_transition_accepted(meta, frame_id):
    meta["last_validated_transition_frame"] = frame_id


def resolve_validated_fluid_transitions(meta, from_region, to_region, frame_id, current_time):
    """Lọc và chuẩn hóa transition trước khi ghi CSV flow/OD.

    YOLO/DeepSORT có thể làm bbox rung ở mép polygon, mất vài frame trong center
    hoặc tạo hướng ngược như center->inbound, outbound->center. Hàm này chỉ cho
    qua luồng hợp lệ inbound->center->outbound. Nếu track nhảy trực tiếp từ
    inbound sang outbound, có thể suy luận center bị bỏ lỡ và tách thành hai
    cạnh hợp lệ để replay/RL không còn nhận lane_to_lane_direct.
    """
    if not from_region or not to_region or from_region == to_region:
        return []

    if meta.get("fluid_done"):
        meta["fluid_rejected_transition_count"] = meta.get("fluid_rejected_transition_count", 0) + 1
        return []

    if _transition_on_cooldown(meta, frame_id):
        meta["fluid_rejected_transition_count"] = meta.get("fluid_rejected_transition_count", 0) + 1
        return []

    if from_region in INBOUND_LANE_REGIONS and to_region == "center":
        meta["fluid_origin_region"] = from_region
        meta["fluid_passed_center"] = True
        _mark_transition_accepted(meta, frame_id)
        return [{
            "from_region": from_region,
            "to_region": to_region,
            "time_s": current_time,
            "source": "observed",
            "confidence": 1.0,
            "reason": "validated_inbound_to_center",
        }]

    if from_region == "center" and to_region in OUTBOUND_LANE_REGIONS:
        meta["fluid_done"] = True
        _mark_transition_accepted(meta, frame_id)
        return [{
            "from_region": from_region,
            "to_region": to_region,
            "time_s": current_time,
            "source": "observed",
            "confidence": 1.0,
            "reason": "validated_center_to_outbound",
        }]

    if (
        FLUID_INFER_MISSING_CENTER
        and from_region in INBOUND_LANE_REGIONS
        and to_region in OUTBOUND_LANE_REGIONS
    ):
        last_time = meta.get("last_fluid_region_time")
        if last_time is None:
            last_time = current_time
        mid_time = max(0.0, (float(last_time) + float(current_time)) * 0.5)
        meta["fluid_origin_region"] = from_region
        meta["fluid_passed_center"] = True
        meta["fluid_done"] = True
        _mark_transition_accepted(meta, frame_id)
        return [
            {
                "from_region": from_region,
                "to_region": "center",
                "time_s": mid_time,
                "source": "inferred",
                "confidence": FLUID_INFERRED_CENTER_CONFIDENCE,
                "reason": "inferred_center_missing_part1",
            },
            {
                "from_region": "center",
                "to_region": to_region,
                "time_s": current_time,
                "source": "inferred",
                "confidence": FLUID_INFERRED_CENTER_CONFIDENCE,
                "reason": "inferred_center_missing_part2",
            },
        ]

    meta["fluid_rejected_transition_count"] = meta.get("fluid_rejected_transition_count", 0) + 1
    return []


def can_emit_event(meta, branch, direction, frame_id):
    key = (branch, direction)
    last_frame = meta["last_event_frame"].get(key, -10**9)
    if frame_id - last_frame < EVENT_COOLDOWN_FRAMES:
        return False

    meta["last_event_frame"][key] = frame_id
    return True


def add_flow_event(branch_event_windows, branch, direction, current_time, frame_id, track_id, cls):
    branch_event_windows[(branch, direction)].append({
        "time": current_time,
        "frame": frame_id,
        "track_id": track_id,
        "cls": cls,
    })


def cleanup_flow_windows(branch_event_windows, current_time, window_seconds):
    for events in branch_event_windows.values():
        while events and current_time - events[0]["time"] > window_seconds:
            events.popleft()


def calc_veh_per_min(events, window_seconds):
    if window_seconds <= 0:
        return 0.0
    return len(events) * 60.0 / window_seconds



def mark_branch_enter(meta, branch, counted=False, event_cls=None):
    """Ghi nhớ track này hiện đang nằm trong một vùng/nhánh.

    ``counted`` cho biết event IN có thật sự được logic cooldown chấp nhận hay không.
    OUT chỉ được đếm cho lượt đã có IN được đếm.
    ``event_cls`` khóa loại phương tiện cho toàn bộ lượt đi, tránh lỗi thường gặp
    khi xe máy được đếm là Moto In nhưng nhãn đã làm mượt lại đổi trước OUT.
    """
    meta["active_branch"] = branch
    meta["active_branch_counted"] = bool(counted)
    meta["active_branch_cls"] = event_cls


def mark_branch_exit(meta, branch):
    """Xóa vùng đang hoạt động nếu track rời khỏi vùng đó."""
    if meta.get("active_branch") == branch:
        meta["active_branch"] = None
        meta["active_branch_counted"] = False
        meta["active_branch_cls"] = None


def emit_branch_event(
    branch,
    direction,
    meta,
    frame_id,
    current_time,
    track_id,
    branch_count_total,
    branch_class_count_total,
    branch_event_windows,
    event_cls=None,
    force=False,
):
    """Cập nhật bộ đếm tổng và event flow cửa sổ trượt cho một event nhánh."""
    if not force and not can_emit_event(meta, branch, direction, frame_id):
        return False
    if force:
        meta["last_event_frame"][(branch, direction)] = frame_id

    cls = event_cls if event_cls is not None else meta.get("stable_cls", meta.get("cls", 2))
    cls = int(cls)
    if cls not in COUNTED_CLASS_IDS:
        return False
    branch_count_total[(branch, direction)] += 1
    branch_class_count_total[(branch, direction, cls)] += 1
    add_flow_event(branch_event_windows, branch, direction, current_time, frame_id, track_id, cls)
    return True
