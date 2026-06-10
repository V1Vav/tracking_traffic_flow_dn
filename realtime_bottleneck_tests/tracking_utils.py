from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    tid: int
    bbox: Tuple[float, float, float, float]
    cls: int
    conf: float
    age: int = 0


class MiniCentroidTracker:
    """Very small fallback tracker used only when DeepSORT is not installed.
    It is NOT a replacement for your app tracker. It helps estimate non-YOLO overhead.
    """

    def __init__(self, max_age: int = 20, iou_threshold: float = 0.2) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Tuple[Tuple[float, float, float, float], float, int]]) -> List[Track]:
        unmatched = set(range(len(detections)))
        used_tracks = set()
        assignments = []
        for tid, tr in list(self.tracks.items()):
            best_i = -1
            best = 0.0
            for i in unmatched:
                box, conf, cls = detections[i]
                score = iou_xyxy(tr.bbox, box)
                if score > best:
                    best = score
                    best_i = i
            if best_i >= 0 and best >= self.iou_threshold:
                assignments.append((tid, best_i))
                unmatched.remove(best_i)
                used_tracks.add(tid)

        for tid, i in assignments:
            box, conf, cls = detections[i]
            self.tracks[tid] = Track(tid=tid, bbox=box, cls=cls, conf=conf, age=0)

        for i in list(unmatched):
            box, conf, cls = detections[i]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid=tid, bbox=box, cls=cls, conf=conf, age=0)
            used_tracks.add(tid)

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                tr = self.tracks[tid]
                tr.age += 1
                if tr.age > self.max_age:
                    del self.tracks[tid]
        return list(self.tracks.values())
