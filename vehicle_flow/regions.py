"""Region template loading, region detection, and region drawing helpers."""

import csv
import os

import cv2
import numpy as np

from .config import REGION_NAME_MAP


class RegionTemplate:
    def __init__(self, mapping_path):
        self.mapping_path = mapping_path
        self.regions = {}
        self.resolution = None
        self.loaded = False
        self._load_mapping()

    def _load_mapping(self):
        if not self.mapping_path or not os.path.exists(self.mapping_path):
            return

        try:
            with open(self.mapping_path, newline="", encoding="utf-8") as csvfile:
                rows = list(csv.reader(csvfile))

            if not rows:
                return

            if len(rows[0]) >= 2:
                self.resolution = (int(rows[0][0]), int(rows[0][1]))

            for row in rows[1:]:
                if len(row) < 9:
                    continue

                region_name = row[0].strip().lower()
                region = REGION_NAME_MAP.get(region_name, region_name)
                if region is None:
                    continue

                points = []
                for i in range(1, 9, 2):
                    points.append((int(row[i].strip()), int(row[i + 1].strip())))

                if len(points) == 4:
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

        for region_name, points in self.regions.items():
            scaled = self._scale_points(points, width, height)
            contour = np.array(scaled, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                return region_name

        return None

    def overlay(self, frame):
        if not self.loaded:
            return

        height, width = frame.shape[:2]
        for region_name, points in self.regions.items():
            scaled = self._scale_points(points, width, height)
            contour = np.array(scaled, dtype=np.int32)
            cv2.polylines(frame, [contour], True, (0, 255, 0), 3)

            moment = cv2.moments(contour)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                cv2.putText(
                    frame,
                    region_name.upper(),
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                )


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

    cv2.line(frame, (left_margin, 0), (left_margin, height), (0, 255, 0), 2)
    cv2.line(frame, (right_margin, 0), (right_margin, height), (0, 255, 0), 2)
    cv2.line(frame, (0, top_margin), (width, top_margin), (0, 255, 0), 2)
    cv2.line(frame, (0, bottom_margin), (width, bottom_margin), (0, 255, 0), 2)
