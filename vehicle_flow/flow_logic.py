"""Counting, debounce, PCE, and sliding-window flow helpers."""

from collections import Counter, deque

from .config import (
    CLASS_HISTORY_LEN,
    CLASS_LOCK_MIN_FRAMES,
    CLASS_STABILITY_RATIO,
    CLASS_WEIGHTS,
    EVENT_COOLDOWN_FRAMES,
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
        "weight": CLASS_WEIGHTS.get(cls, 1.0),
        "cls": cls,
        "stable_cls": cls,
        "stable_cls_age": 0,
        "class_history": deque(maxlen=CLASS_HISTORY_LEN),
    }


def update_stable_class(meta, det_cls, det_conf=None):
    """Smooth class predictions across one DeepSORT track with hysteresis.

    YOLO class output can flicker frame-by-frame, especially between motorcycle
    and car in dense Vietnamese traffic. The stable class is used for PCE and
    type counts. After a class becomes stable, it is locked for a few frames and
    only changes when another class wins by a stronger vote ratio.
    """
    stable_cls = meta.get("stable_cls", meta.get("cls", 2))

    if det_cls is None:
        meta["stable_cls_age"] = meta.get("stable_cls_age", 0) + 1
        meta["cls"] = stable_cls
        meta["weight"] = CLASS_WEIGHTS.get(stable_cls, 1.0)
        return stable_cls

    det_cls = int(det_cls)
    conf = 1.0 if det_conf is None else max(float(det_conf), 0.01)
    meta["class_history"].append((det_cls, conf))

    if len(meta["class_history"]) < MIN_CLASS_VOTES:
        meta["stable_cls_age"] = meta.get("stable_cls_age", 0) + 1
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
    vote_ratio = best_score / total_score if total_score > 0 else 0.0
    stable_age = meta.get("stable_cls_age", 0)

    if best_cls == stable_cls:
        stable_cls = best_cls
    elif stable_age < CLASS_LOCK_MIN_FRAMES:
        # Keep the current label during the initial lock period.
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



def mark_branch_enter(meta, branch, counted=False, event_cls=None):
    """Remember that this track is currently inside one region/branch.

    ``counted`` tells whether the IN event was actually accepted by the
    cooldown logic. OUT will only be counted for a visit that had a counted IN.
    ``event_cls`` freezes the vehicle type for the whole visit, preventing the
    common mismatch where a motorcycle is counted as Moto In but its smoothed
    label switches before OUT.
    """
    meta["active_branch"] = branch
    meta["active_branch_counted"] = bool(counted)
    meta["active_branch_cls"] = event_cls


def mark_branch_exit(meta, branch):
    """Clear active region if the track is leaving that region."""
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
    """Update total counters and sliding-window flow events for one branch event."""
    if not force and not can_emit_event(meta, branch, direction, frame_id):
        return False
    if force:
        meta["last_event_frame"][(branch, direction)] = frame_id

    cls = event_cls if event_cls is not None else meta.get("stable_cls", meta.get("cls", 2))
    cls = int(cls)
    branch_count_total[(branch, direction)] += 1
    branch_class_count_total[(branch, direction, cls)] += 1
    add_flow_event(branch_event_windows, branch, direction, current_time, frame_id, track_id, cls)
    return True
