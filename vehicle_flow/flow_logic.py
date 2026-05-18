"""Counting, debounce, PCE, and sliding-window flow helpers."""

from collections import Counter, deque

from .config import (
    CLASS_HISTORY_LEN,
    CLASS_STABILITY_RATIO,
    CLASS_WEIGHTS,
    EVENT_COOLDOWN_FRAMES,
    MIN_CLASS_VOTES,
    REGION_HISTORY_LEN,
    STABLE_REGION_FRAMES,
)


def create_track_meta(frame_id, cls):
    return {
        "raw_history": deque(maxlen=REGION_HISTORY_LEN),
        "stable_region": None,
        "active_branch": None,
        "last_seen_frame": frame_id,
        "last_event_frame": {},
        "current_region": None,
        "weight": CLASS_WEIGHTS.get(cls, 1.0),
        "cls": cls,
        "stable_cls": cls,
        "class_history": deque(maxlen=CLASS_HISTORY_LEN),
    }


def update_stable_class(meta, det_cls, det_conf=None):
    """Smooth class predictions across one DeepSORT track.

    YOLO class output can flicker frame-by-frame, especially between motorcycle
    and car in dense Vietnamese traffic. The stable class is used for PCE and
    type counts, while raw detections are only treated as votes.
    """
    if det_cls is None:
        return meta.get("stable_cls", meta.get("cls", 2))

    det_cls = int(det_cls)
    conf = 1.0 if det_conf is None else max(float(det_conf), 0.01)
    meta["class_history"].append((det_cls, conf))

    if len(meta["class_history"]) < MIN_CLASS_VOTES:
        stable_cls = meta.get("stable_cls", det_cls)
        meta["cls"] = stable_cls
        meta["weight"] = CLASS_WEIGHTS.get(stable_cls, 1.0)
        return stable_cls

    weighted_votes = Counter()
    raw_votes = Counter()
    for cls, cls_conf in meta["class_history"]:
        weighted_votes[cls] += cls_conf
        raw_votes[cls] += 1

    best_cls, best_score = weighted_votes.most_common(1)[0]
    total_score = sum(weighted_votes.values())
    stable_cls = meta.get("stable_cls", best_cls)

    if (
        raw_votes[best_cls] >= MIN_CLASS_VOTES
        and total_score > 0
        and best_score / total_score >= CLASS_STABILITY_RATIO
    ):
        stable_cls = best_cls

    meta["stable_cls"] = stable_cls
    meta["cls"] = stable_cls
    meta["weight"] = CLASS_WEIGHTS.get(stable_cls, 1.0)
    return stable_cls


def update_stable_region(meta, raw_region):
    """Debounce region changes to avoid edge jitter count noise."""
    meta["raw_history"].append(raw_region)

    if len(meta["raw_history"]) < STABLE_REGION_FRAMES:
        return meta["stable_region"]

    recent = list(meta["raw_history"])[-STABLE_REGION_FRAMES:]
    if all(region == recent[0] for region in recent):
        meta["stable_region"] = recent[0]

    return meta["stable_region"]


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
):
    """Update total counters and sliding-window flow events for one branch event."""
    if not can_emit_event(meta, branch, direction, frame_id):
        return False

    cls = meta.get("stable_cls", meta.get("cls", 2))
    branch_count_total[(branch, direction)] += 1
    branch_class_count_total[(branch, direction, cls)] += 1
    add_flow_event(branch_event_windows, branch, direction, current_time, frame_id, track_id, cls)
    return True
