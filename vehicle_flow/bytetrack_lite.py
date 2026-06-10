"""ByteTrack-style tracker nhẹ cho realtime.

Tracker này không dùng ReID/CNN embedding như DeepSORT. Nó chỉ dùng IoU +
chuyển động tuyến tính đơn giản để giữ ID. Mục tiêu là giảm bottleneck realtime
trong bài toán giao thông, nơi camera thường cố định và đối tượng di chuyển liên
tục qua các vùng.

Input tương thích với deep-sort-realtime trong project:
    detections = [([left, top, width, height], conf, cls), ...]
    tracks = tracker.update_tracks(detections, frame=frame)

Track output có các field/method đang được video_worker dùng:
    track_id, det_class, det_conf, time_since_update, is_confirmed(), to_ltrb()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

LTRB = Tuple[float, float, float, float]
Detection = Tuple[Sequence[float], float, int]


def _xywh_to_ltrb(xywh: Sequence[float]) -> LTRB:
    x, y, w, h = [float(v) for v in xywh]
    return x, y, x + max(0.0, w), y + max(0.0, h)


def _area(box: LTRB) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: LTRB, b: LTRB) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _center(box: LTRB) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _size(box: LTRB) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def _center_distance_ratio(a: LTRB, b: LTRB) -> float:
    acx, acy = _center(a)
    bcx, bcy = _center(b)
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    aw, ah = _size(a)
    bw, bh = _size(b)
    diag = max(min((aw * aw + ah * ah) ** 0.5, (bw * bw + bh * bh) ** 0.5), 1.0)
    return dist / diag


def _size_similarity(a: LTRB, b: LTRB) -> float:
    aw, ah = _size(a)
    bw, bh = _size(b)
    wr = min(aw, bw) / max(aw, bw)
    hr = min(ah, bh) / max(ah, bh)
    return min(wr, hr)


def _translate_box(box: LTRB, dx: float, dy: float) -> LTRB:
    x1, y1, x2, y2 = box
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


@dataclass
class _LiteDetection:
    ltrb: LTRB
    conf: float
    cls: int


class ByteTrackLiteTrack:
    """Track object tương thích phần tối thiểu với deep-sort-realtime Track."""

    def __init__(self, track_id: int, det: _LiteDetection, n_init: int = 1):
        self.track_id = int(track_id)
        self.det_class = int(det.cls)
        self.det_conf = float(det.conf)
        self.time_since_update = 0
        self.age = 1
        self.hits = 1
        self._n_init = max(1, int(n_init))
        self._ltrb: LTRB = tuple(det.ltrb)
        self._orig_ltrb: LTRB | None = tuple(det.ltrb)
        self._vx = 0.0
        self._vy = 0.0

    def is_confirmed(self) -> bool:
        return self.hits >= self._n_init

    def to_ltrb(self, orig: bool = False, orig_strict: bool = False):
        if orig:
            if self._orig_ltrb is not None:
                return np.asarray(self._orig_ltrb, dtype=float)
            if orig_strict:
                return None
        return np.asarray(self._ltrb, dtype=float)

    def predicted_ltrb(self) -> LTRB:
        # Dùng dự đoán chuyển động ngay cả khi track vừa được cập nhật ở frame trước.
        # Nếu chỉ dùng bbox cũ, xe chạy nhanh sẽ có IoU thấp với detection mới,
        # tracker dễ tạo ID mới và để lại một bbox cũ phía sau.
        return _translate_box(self._ltrb, self._vx, self._vy)

    def update(self, det: _LiteDetection, velocity_alpha: float = 0.70) -> None:
        old_cx, old_cy = _center(self._ltrb)
        new_cx, new_cy = _center(det.ltrb)
        dx = new_cx - old_cx
        dy = new_cy - old_cy
        alpha = min(max(float(velocity_alpha), 0.0), 1.0)
        self._vx = alpha * dx + (1.0 - alpha) * self._vx
        self._vy = alpha * dy + (1.0 - alpha) * self._vy
        self._ltrb = tuple(det.ltrb)
        self._orig_ltrb = tuple(det.ltrb)
        self.det_class = int(det.cls)
        self.det_conf = float(det.conf)
        self.time_since_update = 0
        self.age += 1
        self.hits += 1

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.age += 1
        self._orig_ltrb = None
        # Dự đoán vị trí nhẹ để khi mất detection 1-2 frame, bbox không đứng im.
        if self.time_since_update <= 3:
            self._ltrb = _translate_box(self._ltrb, self._vx, self._vy)


class ByteTrackLite:
    """Tracker IoU hai ngưỡng theo tinh thần ByteTrack, không dùng ReID.

    - High-score detections được match trước và có thể tạo track mới.
    - Low-score detections chỉ dùng để cứu track đã tồn tại, hạn chế tạo ID giả.
    - Matching class-aware mặc định để car/motorbike/truck không nuốt ID lẫn nhau.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 1,
        track_high_thresh: float = 0.45,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.45,
        match_thresh: float = 0.30,
        low_match_thresh: float = 0.20,
        class_aware: bool = True,
        velocity_alpha: float = 0.70,
        center_match_ratio: float = 0.75,
        min_iou_for_center_match: float = 0.02,
        min_size_similarity: float = 0.45,
    ):
        self.max_age = max(1, int(max_age))
        self.n_init = max(1, int(n_init))
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_thresh = float(match_thresh)
        self.low_match_thresh = float(low_match_thresh)
        self.class_aware = bool(class_aware)
        self.velocity_alpha = float(velocity_alpha)
        self.center_match_ratio = float(center_match_ratio)
        self.min_iou_for_center_match = float(min_iou_for_center_match)
        self.min_size_similarity = float(min_size_similarity)
        self._next_id = 1
        self._tracks: List[ByteTrackLiteTrack] = []

    def _new_track(self, det: _LiteDetection) -> ByteTrackLiteTrack:
        track = ByteTrackLiteTrack(self._next_id, det, n_init=self.n_init)
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _parse_detections(self, detections: Iterable[Detection]) -> List[_LiteDetection]:
        parsed: List[_LiteDetection] = []
        for det in detections or []:
            if len(det) < 3:
                continue
            xywh, conf, cls = det[0], det[1], det[2]
            ltrb = _xywh_to_ltrb(xywh)
            if _area(ltrb) <= 0.0:
                continue
            parsed.append(_LiteDetection(ltrb=ltrb, conf=float(conf), cls=int(cls)))
        return parsed

    def _match(self, track_indices: List[int], dets: List[_LiteDetection], det_indices: List[int], iou_thresh: float):
        candidates = []
        for ti in track_indices:
            track = self._tracks[ti]
            track_box = track.predicted_ltrb()
            for di in det_indices:
                det = dets[di]
                if self.class_aware and int(track.det_class) != int(det.cls):
                    continue

                iou_score = _iou(track_box, det.ltrb)
                center_ratio = _center_distance_ratio(track_box, det.ltrb)
                size_sim = _size_similarity(track_box, det.ltrb)

                # Match chuẩn theo IoU.
                accepted = iou_score >= iou_thresh

                # Fallback cho xe chạy nhanh/camera FPS thấp: bbox frame trước và frame mới
                # có thể ít chồng lấn, nhưng tâm vẫn gần theo đường chuyển động và kích thước
                # tương tự. Điều này giảm lỗi tạo ID mới + để lại ghost bbox phía sau.
                if not accepted:
                    accepted = (
                        iou_score >= self.min_iou_for_center_match
                        and center_ratio <= self.center_match_ratio
                        and size_sim >= self.min_size_similarity
                    )

                if accepted:
                    # Ưu tiên IoU, nhưng thêm tín hiệu tâm/kích thước để chọn đúng khi IoU thấp.
                    score = iou_score + max(0.0, 1.0 - center_ratio) * 0.35 + size_sim * 0.15
                    candidates.append((score, ti, di))

        candidates.sort(key=lambda item: item[0], reverse=True)
        matched_tracks = set()
        matched_dets = set()
        matches = []
        for score, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            matches.append((ti, di, score))
        return matches, matched_tracks, matched_dets

    def update_tracks(self, detections: Iterable[Detection], frame=None):
        dets = self._parse_detections(detections)
        if not self._tracks and not dets:
            return []

        high_det_indices = [i for i, det in enumerate(dets) if det.conf >= self.track_high_thresh]
        low_det_indices = [
            i for i, det in enumerate(dets)
            if self.track_low_thresh <= det.conf < self.track_high_thresh
        ]

        all_track_indices = list(range(len(self._tracks)))

        # Stage 1: match high-score detections.
        matches1, matched_tracks1, matched_dets1 = self._match(
            all_track_indices,
            dets,
            high_det_indices,
            self.match_thresh,
        )
        for ti, di, _score in matches1:
            self._tracks[ti].update(dets[di], velocity_alpha=self.velocity_alpha)

        unmatched_track_indices = [ti for ti in all_track_indices if ti not in matched_tracks1]
        unmatched_high_det_indices = [di for di in high_det_indices if di not in matched_dets1]

        # Stage 2: low-score detections chỉ dùng để nối track cũ, không tạo track mới.
        matches2, matched_tracks2, matched_dets2 = self._match(
            unmatched_track_indices,
            dets,
            low_det_indices,
            self.low_match_thresh,
        )
        for ti, di, _score in matches2:
            self._tracks[ti].update(dets[di], velocity_alpha=self.velocity_alpha)

        matched_tracks = matched_tracks1 | matched_tracks2

        # Track không match thì tăng lost age.
        for ti in all_track_indices:
            if ti not in matched_tracks:
                self._tracks[ti].mark_missed()

        # Chỉ high-score detection mới được tạo track mới.
        for di in unmatched_high_det_indices:
            if dets[di].conf >= self.new_track_thresh:
                self._new_track(dets[di])

        # Prune track quá cũ.
        self._tracks = [track for track in self._tracks if track.time_since_update <= self.max_age]
        return list(self._tracks)
