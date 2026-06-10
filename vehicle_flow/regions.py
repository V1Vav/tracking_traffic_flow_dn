"""Hàm hỗ trợ tải template vùng, xác định vùng và vẽ vùng."""

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

# OpenCV dùng màu BGR. Màu được chọn dịu để video vẫn dễ quan sát.
REGION_COLORS = {
    # OpenCV dùng BGR. Bảng màu được nhóm theo ý nghĩa làn:
    #   * Làn 1 / VÀO : nhóm màu lạnh hơn
    #   * Làn 2 / RA  : nhóm màu ấm hơn
    # Các cặp làn như t1/t2 vẫn dùng sắc độ khác nhau rõ ràng, trong khi
    # alpha fill được giữ thấp để xe và box tracking vẫn dễ nhìn.
    "t1": (222, 168, 78),    # xanh lam - làn trên đi vào
    "t2": (97, 162, 244),    # vàng hổ phách - làn trên đi ra
    "l1": (136, 183, 82),    # xanh lá - làn trái đi vào
    "l2": (107, 107, 255),   # cam san hô - làn trái đi ra
    "r1": (229, 93, 155),    # tím - làn phải đi vào
    "r2": (0, 127, 247),     # cam - làn phải đi ra
    "b1": (216, 180, 0),     # xanh ngọc - làn dưới đi vào
    "b2": (106, 196, 233),   # vàng - làn dưới đi ra
    "center": (184, 163, 148),
    "outside": (120, 128, 140),
}
REGION_LABELS = {
    "t1": "T1 VAO",
    "t2": "T2 RA",
    "l1": "L1 VAO",
    "l2": "L2 RA",
    "r1": "R1 VAO",
    "r2": "R2 RA",
    "b1": "B1 VAO",
    "b2": "B2 RA",
    "center": "CENTER",
    "outside": "NGOÀI",
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
        # Cache điểm/contour đã scale theo kích thước frame. Với 8 polygon làn,
        # cách này tránh scale lại mọi polygon cho từng object được track.
        self._scaled_cache = {}
        self._load_mapping()

    def _parse_point_row(self, row):
        """Đọc một dòng có dạng name,x1,y1,x2,y2,...

        Template cũ dùng 4 điểm. Bộ đọc này nhận mọi polygon có ít nhất 4 điểm,
        nên vùng center hoặc vùng chuyển làn có thể chi tiết hơn.
        Ô rỗng được bỏ qua để chỉnh CSV dễ hơn.
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

                # Trong bố cục 8 làn, từ chối tên 4 vùng cũ hoặc tên gõ sai
                # thay vì âm thầm load. Các tên hợp lệ là:
                # t1,t2,l1,l2,r1,r2,b1,b2,center.
                if region not in VALID_BRANCHES:
                    print(
                        f"Bỏ qua vùng không hỗ trợ '{region_name}' trong template. "
                        "Hãy dùng t1,t2,l1,l2,r1,r2,b1,b2,center."
                    )
                    continue

                points = self._parse_point_row(row)
                if len(points) >= 4:
                    self.regions[region] = points
                    print(f"Đã tải vùng {region}: {points}")

            self.loaded = bool(self.regions)
            print(f"RegionTemplate đã tải: {self.loaded}, vùng: {list(self.regions.keys())}")
        except Exception as exc:
            print(f"Lỗi khi tải mapping: {exc}")
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

        # Giữ cache nhỏ vì kích thước frame xử lý có thể khác
        # kích thước hiển thị, nhưng thường chỉ có 1-2 mục.
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
        """Trả về tất cả vùng chứa centroid, giữ thứ tự trong template.

        Cho phép các polygon làn chồng lên nhau. Vùng cuối cùng sẽ được chọn
        bởi get_region(), có dùng hướng di chuyển để phân biệt làn 1/2.
        """
        if not self.loaded:
            return []

        x, y = centroid
        if x < 0 or y < 0 or x >= width or y >= height:
            return []

        # Kiểm tra các làn trước, center sau cùng.
        # Center chỉ dùng như vùng nút giao nội bộ; khi polygon center chồng lên
        # b1/t1/... thì làn đường phải được ưu tiên hơn center.
        region_order = [name for name in LANE_REGION_ORDER if name in self.regions]
        added = set(region_order)
        region_order += [
            name for name in self.regions.keys()
            if name not in added and name != "center"
        ]
        if "center" in self.regions:
            region_order.append("center")

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

    def overlay(self, frame, label_scale=1.0):
        if not self.loaded:
            return

        height, width = frame.shape[:2]
        polygons = []
        scaled_cache = self._get_scaled_cache(width, height)
        added = set()
        # Chỉ vẽ các làn. Center vẫn được load để xét logic flow nhưng
        # không hiển thị khi bật Show region để overlay đỡ rối.
        for region_name in LANE_REGION_ORDER:
            entry = scaled_cache.get(region_name)
            if entry:
                polygons.append((region_name, entry["points"]))
                added.add(region_name)
        for region_name, entry in scaled_cache.items():
            if region_name == "center" or region_name in added:
                continue
            polygons.append((region_name, entry["points"]))
        draw_region_polygons(frame, polygons, label_scale=label_scale)


def _dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def choose_region_from_candidates(candidates, *, centroid, previous_centroid=None, current_region=None, center_point=None):
    """Chọn một vùng khi các polygon chồng lên nhau.

    Nếu một điểm nằm trong cả làn 1 và làn 2 của cùng một nhánh, dùng hướng
    di chuyển so với center: tiến gần center => làn 1; đi xa center => làn 2.
    Nhờ đó vùng chồng lấn do lấn/chuyển làn dùng được, thay vì label nhảy ngẫu nhiên.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    candidate_set = set(candidates)

    if "center" in candidate_set:
        # Center có độ ưu tiên thấp hơn tất cả vùng làn.
        # Ví dụ điểm nằm trong cả b1 và center thì kết quả phải là b1.
        non_center_candidates = [region for region in candidates if region != "center"]
        if non_center_candidates:
            return choose_region_from_candidates(
                non_center_candidates,
                centroid=centroid,
                previous_centroid=previous_centroid,
                current_region=current_region,
                center_point=center_point,
            )
        return "center"

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

    # Nếu vùng chồng lấn thuộc các nhánh khác nhau, giữ vùng trước đó khi
    # có thể; nếu không thì dùng thứ tự ưu tiên cố định.
    if keep_current:
        return keep_current
    for region_name in LANE_REGION_ORDER:
        if region_name in candidate_set:
            return region_name
    return candidates[0]


def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bottom_center_from_box(box):
    """Điểm giữa cạnh dưới bbox, phù hợp hơn để gán xe vào polygon mặt đường."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def region_point_from_box(box, use_bottom_center=True):
    """Trả về điểm dùng cho nhận diện region.

    Với video giao thông góc nhìn chéo, tâm bbox thường nằm trên thân xe và có
    thể lệch sang polygon lân cận. Điểm đáy bbox gần vị trí tiếp xúc mặt đường
    hơn nên ổn định hơn cho đếm vùng/flow.
    """
    if use_bottom_center:
        return bottom_center_from_box(box)
    return centroid_from_box(box)


def get_direction_region(centroid, width, height, margin_fraction, template=None, previous_centroid=None, current_region=None):
    """Trả về vùng thô từ template polygon hoặc vùng 8 làn mặc định."""
    if template and template.loaded:
        region = template.get_region(
            centroid,
            width,
            height,
            previous_centroid=previous_centroid,
            current_region=current_region,
        )
        # Không tự gán center khi điểm không nằm trong polygon nào.
        # Nếu template có vùng center, template.get_region() sẽ trả về center
        # chỉ khi điểm thật sự nằm trong polygon center.
        return region

    x, y = centroid
    left_margin = int(width * margin_fraction)
    right_margin = int(width * (1.0 - margin_fraction))
    top_margin = int(height * margin_fraction)
    bottom_margin = int(height * (1.0 - margin_fraction))

    if left_margin <= x <= right_margin and top_margin <= y <= bottom_margin:
        return "center"

    # Chia fallback: mỗi nhánh ngoài được chia thành làn 1/làn 2. Cách này
    # chỉ dùng để test nhanh; nên dùng template.csv để có hình học chính xác.
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


def _draw_readable_label(frame, text, center, color, label_scale=1.0):
    """Vẽ label dạng pill dễ đọc nhưng không quá chói.

    label_scale dùng để giảm kích thước chữ khi chạy realtime ở phân giải thấp,
    tránh che mất xe/box nhưng vẫn giữ tên vùng đủ đọc.
    """
    x, y = center
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = max(0.45, min(1.15, float(label_scale or 1.0)))
    base_scale = 0.50 if len(text) > 6 else 0.56
    scale = base_scale * label_scale
    thickness = 1 if label_scale < 0.85 else 2
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    tw, th = text_size
    pad_x = max(4, int(8 * label_scale))
    pad_y = max(3, int(5 * label_scale))

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




def draw_region_polygons(frame, polygons, alpha=0.065, label_scale=1.0):
    """Vẽ nền vùng nhẹ, viền tương phản và label gọn.

    Không vẽ mũi tên hướng ở đây. Với 8 vùng làn, mũi tên làm overlay rối;
    danh tính làn được thể hiện bằng màu riêng và label như T1 VÀO / T2 RA.
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

    # Đậm hơn một chút để t1/t2 dễ phân biệt, nhưng vẫn đủ nhẹ
    # để xe và box vẫn thấy rõ.
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Vẽ viền tối trước, rồi viền màu sau. Cách này tách
    # các polygon làn chồng nhau mà không cần mũi tên hoặc fill quá mạnh.
    for region_name, contour, color, center in prepared:
        cv2.polylines(frame, [contour], True, (10, 16, 28), 3, cv2.LINE_AA)
        cv2.polylines(frame, [contour], True, color, 2, cv2.LINE_AA)

    # Vẽ label gọn sau cùng để tên vùng vẫn đọc được trên frame sáng.
    for region_name, contour, color, center in prepared:
        label = REGION_LABELS.get(region_name, REGION_SHORT_LABELS.get(region_name, region_name.upper()))
        _draw_readable_label(frame, label, center, color, label_scale=label_scale)


def draw_region_overlay(frame, margin_fraction, template=None, label_scale=1.0):
    """
    Chỉ vẽ overlay vùng khi được gọi rõ ràng.
    Nếu không có template, vẽ vùng biên 8 làn mặc định.
    """
    if template and template.loaded:
        template.overlay(frame, label_scale=label_scale)
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
    ]
    draw_region_polygons(frame, polygons, label_scale=label_scale)
