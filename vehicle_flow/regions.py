"""Region template loading, region detection, and region drawing helpers."""

import csv
import os

import cv2
import numpy as np

from .config import REGION_NAME_MAP

# OpenCV uses BGR colors.
REGION_COLORS = {
    # Soft BGR colors: enough contrast for region recognition, but less harsh on video.
    "top": (172, 129, 67),
    "left": (156, 106, 147),
    "right": (106, 151, 105),
    "bottom": (153, 119, 92),
    "center": (117, 139, 151),
    "outside": (120, 128, 140),
}

REGION_LABELS = {
    "top": "TOP",
    "left": "LEFT",
    "right": "RIGHT",
    "bottom": "BOTTOM",
    "center": "CENTER",
    "outside": "OUTSIDE",
}


class RegionTemplate:
    def __init__(self, mapping_path):
        self.mapping_path = mapping_path
        self.regions = {}
        self.resolution = None
        self.loaded = False
        self._load_mapping()

    def _parse_point_row(self, row):
        """Parse any row formatted as name,x1,y1,x2,y2,...

        Older templates used 4 points per region. The center region can now use
        8 points, so this parser accepts any polygon with at least 4 points.
        Empty cells are ignored to keep old CSV files compatible.
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

    def get_region(self, centroid, width, height):
        if not self.loaded:
            return None

        x, y = centroid
        if x < 0 or y < 0 or x >= width or y >= height:
            return None

        # Check center first when it is explicitly present. This prevents a
        # center point that lies on a shared border from being captured by one
        # of the outer regions before the center polygon is tested.
        region_order = ["center"] if "center" in self.regions else []
        region_order += [name for name in self.regions.keys() if name != "center"]

        for region_name in region_order:
            points = self.regions[region_name]
            scaled = self._scale_points(points, width, height)
            contour = np.array(scaled, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                return region_name

        return None

    def overlay(self, frame):
        if not self.loaded:
            return

        height, width = frame.shape[:2]
        polygons = []
        for region_name, points in self.regions.items():
            scaled = self._scale_points(points, width, height)
            polygons.append((region_name, scaled))
        draw_region_polygons(frame, polygons)


def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_direction_region(centroid, width, height, margin_fraction, template=None):
    """Return raw region from polygon template or fallback margin-based regions."""
    if template and template.loaded:
        region = template.get_region(centroid, width, height)
        return region if region is not None else "center"

    x, y = centroid
    left_margin = int(width * margin_fraction)
    right_margin = int(width * (1.0 - margin_fraction))
    top_margin = int(height * margin_fraction)
    bottom_margin = int(height * (1.0 - margin_fraction))

    if left_margin <= x <= right_margin and top_margin <= y <= bottom_margin:
        return "center"
    if y < top_margin:
        return "top"
    if y > bottom_margin:
        return "bottom"
    if x < left_margin:
        return "left"
    if x > right_margin:
        return "right"
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
    scale = 0.58
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    tw, th = text_size
    pad_x = 10
    pad_y = 6

    x1 = int(x - tw / 2 - pad_x)
    y1 = int(y - th / 2 - pad_y)
    x2 = int(x + tw / 2 + pad_x)
    y2 = int(y + th / 2 + pad_y + baseline)

    x1 = max(4, x1)
    y1 = max(4, y1)
    x2 = min(frame.shape[1] - 4, x2)
    y2 = min(frame.shape[0] - 4, y2)

    # Dark neutral pill with a thin colored outline. This is easier to read
    # than a fully saturated region-color label on traffic footage.
    label_overlay = frame.copy()
    cv2.rectangle(label_overlay, (x1, y1), (x2, y2), (20, 24, 31), -1)
    cv2.addWeighted(label_overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    tx = int((x1 + x2 - tw) / 2)
    ty = int((y1 + y2 + th) / 2) - baseline
    cv2.putText(frame, text, (tx + 1, ty + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, scale, (245, 248, 252), thickness, cv2.LINE_AA)

def draw_region_polygons(frame, polygons, alpha=0.11):
    """Draw semi-transparent regions, thick outlines, and readable labels."""
    if not polygons:
        return

    height, width = frame.shape[:2]
    overlay = frame.copy()
    prepared = []

    # Fill first so overlapping outlines remain crisp afterwards.
    for region_name, points in polygons:
        if len(points) < 3:
            continue
        contour = np.array(points, dtype=np.int32)
        color = REGION_COLORS.get(region_name, (148, 163, 184))
        cv2.fillPoly(overlay, [contour], color)
        prepared.append((region_name, contour, color))

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    for region_name, contour, color in prepared:
        # Black underlay improves visibility on bright road markings.
        cv2.polylines(frame, [contour], True, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(frame, [contour], True, color, 2, cv2.LINE_AA)

        # Small vertex marks help users see the template shape without making
        # the video too busy.
        for point in contour.reshape(-1, 2):
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame, (px, py), 2, color, -1, cv2.LINE_AA)

        label = REGION_LABELS.get(region_name, region_name.upper())
        center = _safe_label_center(contour, width, height)
        _draw_readable_label(frame, label, center, color)


def draw_region_overlay(frame, margin_fraction, template=None):
    """
    Draw region overlay only when explicitly called.
    If template is unavailable, draw fallback margin regions.
    """
    if template and template.loaded:
        template.overlay(frame)
        return

    height, width = frame.shape[:2]
    left_margin = int(width * margin_fraction)
    right_margin = int(width * (1.0 - margin_fraction))
    top_margin = int(height * margin_fraction)
    bottom_margin = int(height * (1.0 - margin_fraction))

    polygons = [
        ("top", [(0, 0), (width - 1, 0), (right_margin, top_margin), (left_margin, top_margin)]),
        ("bottom", [(left_margin, bottom_margin), (right_margin, bottom_margin), (width - 1, height - 1), (0, height - 1)]),
        ("left", [(0, 0), (left_margin, top_margin), (left_margin, bottom_margin), (0, height - 1)]),
        ("right", [(right_margin, top_margin), (width - 1, 0), (width - 1, height - 1), (right_margin, bottom_margin)]),
        ("center", [(left_margin, top_margin), (right_margin, top_margin), (right_margin, bottom_margin), (left_margin, bottom_margin)]),
    ]
    draw_region_polygons(frame, polygons)
