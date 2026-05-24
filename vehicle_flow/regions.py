"""Region template loading, region detection, and region drawing helpers."""

import csv
import math
import os

import cv2
import numpy as np

from .config import (
    APPROACH_TO_INBOUND_REGION,
    APPROACH_TO_OUTBOUND_REGION,
    INBOUND_LANE_REGIONS,
    LANE_REGION_ORDER,
    OUTBOUND_LANE_REGIONS,
    REGION_NAME_MAP,
    REGION_SHORT_LABELS,
    REGION_TO_APPROACH,
    VALID_BRANCHES,
)

# OpenCV uses BGR colors. Colors are intentionally soft so the video remains readable.
REGION_COLORS = {
    # OpenCV uses BGR. The palette is grouped by lane meaning:
    #   * Lane 1 / IN  : cooler colors
    #   * Lane 2 / OUT : warmer colors
    # Paired lanes such as t1/t2 still use clearly different hues, while the
    # fill alpha remains low so vehicles and tracking boxes stay readable.
    "t1": (222, 168, 78),    # blue - top inbound
    "t2": (97, 162, 244),    # amber - top outbound
    "l1": (136, 183, 82),    # green - left inbound
    "l2": (107, 107, 255),   # coral - left outbound
    "r1": (229, 93, 155),    # violet - right inbound
    "r2": (0, 127, 247),     # orange - right outbound
    "b1": (216, 180, 0),     # teal - bottom inbound
    "b2": (106, 196, 233),   # yellow - bottom outbound
    "center": (184, 163, 148),
    "outside": (120, 128, 140),
}
REGION_LABELS = {
    "t1": "T1 IN",
    "t2": "T2 OUT",
    "l1": "L1 IN",
    "l2": "L2 OUT",
    "r1": "R1 IN",
    "r2": "R2 OUT",
    "b1": "B1 IN",
    "b2": "B2 OUT",
    "center": "CENTER",
    "outside": "OUTSIDE",
}

_APPROACH_LANES = {
    "top": ("t1", "t2"),
    "left": ("l1", "l2"),
    "right": ("r1", "r2"),
    "bottom": ("b1", "b2"),
}


class RegionTemplate:
    def __init__(self, mapping_path):
        self.mapping_path = mapping_path
        self.regions = {}
        self.resolution = None
        self.loaded = False
        # Cache scaled points/contours per frame size. With 8 lane polygons,
        # this avoids rescaling every polygon for every tracked object.
        self._scaled_cache = {}
        self._load_mapping()

    def _parse_point_row(self, row):
        """Parse any row formatted as name,x1,y1,x2,y2,...

        Old templates used 4 points. The parser accepts any polygon with at
        least 4 points, so center or lane-change polygons can be more detailed.
        Empty cells are ignored to keep CSV editing simple.
        """
        values = [cell.strip() for cell in row[1:] if cell.strip() != ""]
        if len(values) < 8 or len(values) % 2 != 0:
            return []

        points = []
        for i in range(0, len(values), 2):
            points.append((int(float(values[i])), int(float(values[i + 1]))))
        return points

    def _load_mapping(self):
        if not self.mapping_path or not os.path.exists(self.mapping_path):
            return

        try:
            with open(self.mapping_path, newline="", encoding="utf-8") as csvfile:
                rows = list(csv.reader(csvfile))

            if not rows:
                return

            if len(rows[0]) >= 2:
                self.resolution = (int(float(rows[0][0])), int(float(rows[0][1])))

            for row in rows[1:]:
                if len(row) < 9:
                    continue

                region_name = row[0].strip().lower()
                region = REGION_NAME_MAP.get(region_name, region_name)
                if region is None:
                    continue

                # In the 8-lane layout, reject old 4-region names or typos
                # instead of silently loading them. Accepted final names are:
                # t1,t2,l1,l2,r1,r2,b1,b2,center.
                if region not in VALID_BRANCHES:
                    print(
                        f"Ignoring unsupported region '{region_name}' in template. "
                        "Use t1,t2,l1,l2,r1,r2,b1,b2,center."
                    )
                    continue

                points = self._parse_point_row(row)
                if len(points) >= 4:
                    self.regions[region] = points
                    print(f"Loaded region {region}: {points}")

            self.loaded = bool(self.regions)
            print(f"RegionTemplate loaded: {self.loaded}, regions: {list(self.regions.keys())}")
        except Exception as exc:
            print(f"Error loading mapping: {exc}")
            self.regions = {}
            self.loaded = False

    def _scale_points(self, points, frame_width, frame_height):
        if not self.resolution:
            return points
        if frame_width == self.resolution[0] and frame_height == self.resolution[1]:
            return points

        scale_x = frame_width / self.resolution[0]
        scale_y = frame_height / self.resolution[1]
        return [(int(x * scale_x), int(y * scale_y)) for x, y in points]

    def _get_scaled_cache(self, frame_width, frame_height):
        key = (int(frame_width), int(frame_height))
        cached = self._scaled_cache.get(key)
        if cached is not None:
            return cached

        data = {}
        for region_name, points in self.regions.items():
            scaled = self._scale_points(points, frame_width, frame_height)
            if len(scaled) < 3:
                continue
            contour = np.array(scaled, dtype=np.int32)
            data[region_name] = {
                "points": scaled,
                "contour": contour,
                "center": _safe_label_center(contour, frame_width, frame_height),
            }

        # Keep a tiny cache because the processing frame size can differ from
        # display size, but it normally only has 1-2 entries.
        if len(self._scaled_cache) >= 4:
            self._scaled_cache.clear()
        self._scaled_cache[key] = data
        return data

    def _scaled_region_points(self, region_name, frame_width, frame_height):
        entry = self._get_scaled_cache(frame_width, frame_height).get(region_name)
        return entry["points"] if entry else []

    def _center_point(self, frame_width, frame_height):
        entry = self._get_scaled_cache(frame_width, frame_height).get("center")
        if entry:
            return entry["center"]
        return frame_width * 0.5, frame_height * 0.5

    def get_regions(self, centroid, width, height):
        """Return all regions containing centroid, preserving template order.

        Overlapping lane polygons are allowed. The final single region is chosen
        by get_region(), which uses motion direction to disambiguate lane 1/2.
        """
        if not self.loaded:
            return []

        x, y = centroid
        if x < 0 or y < 0 or x >= width or y >= height:
            return []

        # Center is tested first because it is the semantic junction node.
        region_order = ["center"] if "center" in self.regions else []
        region_order += [name for name in LANE_REGION_ORDER if name in self.regions]
        region_order += [name for name in self.regions.keys() if name not in set(region_order)]

        matches = []
        scaled_cache = self._get_scaled_cache(width, height)
        for region_name in region_order:
            entry = scaled_cache.get(region_name)
            if not entry:
                continue
            if cv2.pointPolygonTest(entry["contour"], (x, y), False) >= 0:
                matches.append(region_name)
        return matches

    def get_region(self, centroid, width, height, previous_centroid=None, current_region=None):
        if not self.loaded:
            return None
        candidates = self.get_regions(centroid, width, height)
        return choose_region_from_candidates(
            candidates,
            centroid=centroid,
            previous_centroid=previous_centroid,
            current_region=current_region,
            center_point=self._center_point(width, height),
        )

    def overlay(self, frame):
        if not self.loaded:
            return

        height, width = frame.shape[:2]
        polygons = []
        scaled_cache = self._get_scaled_cache(width, height)
        added = set()
        # Draw lanes before center so center outline stays readable.
        for region_name in LANE_REGION_ORDER + ("center",):
            entry = scaled_cache.get(region_name)
            if entry:
                polygons.append((region_name, entry["points"]))
                added.add(region_name)
        for region_name, entry in scaled_cache.items():
            if region_name not in added:
                polygons.append((region_name, entry["points"]))
        draw_region_polygons(frame, polygons)


def _dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def choose_region_from_candidates(candidates, *, centroid, previous_centroid=None, current_region=None, center_point=None):
    """Choose one region when polygons overlap.

    If a point lies in both lane 1 and lane 2 of the same approach, use movement
    direction relative to the center: moving closer to center => lane 1; moving
    away from center => lane 2. This makes lane-changing overlap usable instead
    of causing random label flicker.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    if "center" in candidates:
        # Center is the intersection node. Prefer it over lane overlap so route
        # export gets explicit lane->center and center->lane transitions.
        return "center"

    candidate_set = set(candidates)
    if current_region in candidate_set:
        keep_current = current_region
    else:
        keep_current = None

    center_point = center_point or centroid
    moving_to_center = None
    if previous_centroid is not None:
        previous_distance = _dist(previous_centroid, center_point)
        current_distance = _dist(centroid, center_point)
        delta = current_distance - previous_distance
        if abs(delta) >= 2.0:
            moving_to_center = delta < 0

    for approach, (in_region, out_region) in _APPROACH_LANES.items():
        if in_region in candidate_set and out_region in candidate_set:
            if moving_to_center is True:
                return in_region
            if moving_to_center is False:
                return out_region
            if keep_current in (in_region, out_region):
                return keep_current
            return in_region

    # If overlap is between different approaches, keep previous region when
    # possible; otherwise fall back to deterministic display/order priority.
    if keep_current:
        return keep_current
    for region_name in LANE_REGION_ORDER:
        if region_name in candidate_set:
            return region_name
    return candidates[0]


def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_direction_region(centroid, width, height, margin_fraction, template=None, previous_centroid=None, current_region=None):
    """Return raw region from polygon template or fallback 8-lane regions."""
    if template and template.loaded:
        region = template.get_region(
            centroid,
            width,
            height,
            previous_centroid=previous_centroid,
            current_region=current_region,
        )
        return region if region is not None else "center"

    x, y = centroid
    left_margin = int(width * margin_fraction)
    right_margin = int(width * (1.0 - margin_fraction))
    top_margin = int(height * margin_fraction)
    bottom_margin = int(height * (1.0 - margin_fraction))

    if left_margin <= x <= right_margin and top_margin <= y <= bottom_margin:
        return "center"

    # Fallback split: each outer approach is divided into lane 1/lane 2. This is
    # only for quick testing; template.csv should be used for accurate geometry.
    if y < top_margin:
        return "t1" if x < width * 0.5 else "t2"
    if y > bottom_margin:
        return "b1" if x > width * 0.5 else "b2"
    if x < left_margin:
        return "l1" if y > height * 0.5 else "l2"
    if x > right_margin:
        return "r1" if y < height * 0.5 else "r2"
    return "outside"


def _safe_label_center(contour, frame_width, frame_height):
    moment = cv2.moments(contour)
    if moment["m00"] != 0:
        cx = int(moment["m10"] / moment["m00"])
        cy = int(moment["m01"] / moment["m00"])
    else:
        pts = contour.reshape(-1, 2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))

    cx = max(12, min(frame_width - 12, cx))
    cy = max(24, min(frame_height - 12, cy))
    return cx, cy


def _draw_readable_label(frame, text, center, color):
    """Draw a readable but subtle label pill."""
    x, y = center
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50 if len(text) > 6 else 0.56
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    tw, th = text_size
    pad_x = 8
    pad_y = 5

    x1 = int(x - tw / 2 - pad_x)
    y1 = int(y - th / 2 - pad_y)
    x2 = int(x + tw / 2 + pad_x)
    y2 = int(y + th / 2 + pad_y + baseline)

    x1 = max(4, x1)
    y1 = max(4, y1)
    x2 = min(frame.shape[1] - 4, x2)
    y2 = min(frame.shape[0] - 4, y2)

    label_overlay = frame.copy()
    cv2.rectangle(label_overlay, (x1, y1), (x2, y2), (20, 24, 31), -1)
    cv2.addWeighted(label_overlay, 0.66, frame, 0.34, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    tx = int((x1 + x2 - tw) / 2)
    ty = int((y1 + y2 + th) / 2) - baseline
    cv2.putText(frame, text, (tx + 1, ty + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, scale, (245, 248, 252), thickness, cv2.LINE_AA)




def draw_region_polygons(frame, polygons, alpha=0.065):
    """Draw soft region fills, high-contrast outlines, and compact labels.

    No direction arrows are drawn here. With 8 lane regions, arrows made the
    overlay noisy; the lane identity is communicated by distinct lane colors
    plus the label text such as T1 IN / T2 OUT.
    """
    if not polygons:
        return

    height, width = frame.shape[:2]
    overlay = frame.copy()
    prepared = []

    for region_name, points in polygons:
        if len(points) < 3:
            continue
        contour = np.array(points, dtype=np.int32)
        color = REGION_COLORS.get(region_name, (148, 163, 184))
        cv2.fillPoly(overlay, [contour], color)
        center = _safe_label_center(contour, width, height)
        prepared.append((region_name, contour, color, center))

    # Slightly stronger than before so t1/t2 are distinguishable, still low
    # enough to keep vehicles and boxes visible.
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Draw a dark outline first, then a colored outline. This separates
    # overlapping lane polygons without requiring arrows or strong fill color.
    for region_name, contour, color, center in prepared:
        cv2.polylines(frame, [contour], True, (10, 16, 28), 3, cv2.LINE_AA)
        cv2.polylines(frame, [contour], True, color, 2, cv2.LINE_AA)

    # Draw compact labels last so names remain readable on bright frames.
    for region_name, contour, color, center in prepared:
        label = REGION_LABELS.get(region_name, REGION_SHORT_LABELS.get(region_name, region_name.upper()))
        _draw_readable_label(frame, label, center, color)


def draw_region_overlay(frame, margin_fraction, template=None):
    """
    Draw region overlay only when explicitly called.
    If template is unavailable, draw fallback 8-lane margin regions.
    """
    if template and template.loaded:
        template.overlay(frame)
        return

    height, width = frame.shape[:2]
    left_margin = int(width * margin_fraction)
    right_margin = int(width * (1.0 - margin_fraction))
    top_margin = int(height * margin_fraction)
    bottom_margin = int(height * (1.0 - margin_fraction))
    mid_x = width // 2
    mid_y = height // 2

    polygons = [
        ("t1", [(0, 0), (mid_x, 0), (mid_x, top_margin), (left_margin, top_margin)]),
        ("t2", [(mid_x, 0), (width - 1, 0), (right_margin, top_margin), (mid_x, top_margin)]),
        ("l1", [(0, mid_y), (left_margin, mid_y), (left_margin, bottom_margin), (0, height - 1)]),
        ("l2", [(0, 0), (left_margin, top_margin), (left_margin, mid_y), (0, mid_y)]),
        ("r1", [(right_margin, top_margin), (width - 1, 0), (width - 1, mid_y), (right_margin, mid_y)]),
        ("r2", [(right_margin, mid_y), (width - 1, mid_y), (width - 1, height - 1), (right_margin, bottom_margin)]),
        ("b1", [(mid_x, bottom_margin), (right_margin, bottom_margin), (width - 1, height - 1), (mid_x, height - 1)]),
        ("b2", [(0, height - 1), (mid_x, height - 1), (mid_x, bottom_margin), (left_margin, bottom_margin)]),
        ("center", [(left_margin, top_margin), (right_margin, top_margin), (right_margin, bottom_margin), (left_margin, bottom_margin)]),
    ]
    draw_region_polygons(frame, polygons)
