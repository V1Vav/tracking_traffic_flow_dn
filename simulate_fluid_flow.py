"""Mô phỏng PCE thật 8 làn từ region_state_timeseries.csv.

Nguyên tắc hiển thị:
- PCE tại vùng/làn là dữ liệu chính, lấy từ region_state_timeseries.csv.
- Mỗi làn hiển thị PCE hiện tại bằng dải xanh chiếm toàn bộ lane.
- Center hiển thị PCE hiện tại bằng vòng tròn tăng/giảm bán kính.
- flow_edges_real.csv tạo lớp chuyển vùng động, mô phỏng dòng chảy từ vùng nguồn sang vùng đích.
- Không nội suy và không giả lập kịch bản không đèn.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

# -----------------------------------------------------------------------------
# Bố cục
# -----------------------------------------------------------------------------

CANVAS_W = 1280
CANVAS_H = 720
CENTER = (640, 370)

LANES = ("t1", "t2", "l1", "l2", "r1", "r2", "b1", "b2")
INBOUND = {"t1", "l1", "r1", "b1"}
OUTBOUND = {"t2", "l2", "r2", "b2"}
IN_TO_OUT = {"t1": "t2", "l1": "l2", "r1": "r2", "b1": "b2"}
OUT_TO_IN = {"t2": "t1", "l2": "l1", "r2": "r1", "b2": "b1"}
REGIONS = LANES + ("center",)

REGION_LABELS = {
    "t1": "T1 vào", "t2": "T2 ra",
    "l1": "L1 vào", "l2": "L2 ra",
    "r1": "R1 vào", "r2": "R2 ra",
    "b1": "B1 vào", "b2": "B2 ra",
    "center": "Center",
}

# BGR. Palette tối hiện đại, giảm chói và tách rõ PCE với flow chuyển vùng.
BG = (24, 14, 8)
BG_TOP = (42, 23, 15)
BG_BOTTOM = (18, 10, 5)
GRID = (52, 36, 24)
PANEL = (45, 28, 17)
PANEL_INNER = (58, 41, 28)
PANEL_EDGE = (108, 83, 55)
PANEL_EDGE_SOFT = (78, 58, 38)
CARD_SHADOW = (0, 0, 0)
PILL_BG = (31, 21, 13)
ROAD = (64, 50, 35)
ROAD_EDGE = (105, 82, 55)
ROAD_HIGHLIGHT = (84, 66, 45)
LANE_GUIDE = (122, 102, 76)
CENTER_FILL = (54, 39, 24)
CENTER_RING = (205, 220, 218)
TEXT = (248, 244, 239)
MUTED = (194, 178, 166)
DIM = (134, 114, 101)
OK = (118, 237, 163)
WARN = (25, 178, 246)
CRITICAL = (82, 88, 239)

# PCE hiện tại dùng xanh lá; flow chuyển vùng dùng cyan để không lẫn với occupancy.
PCE_COLOR = (118, 237, 163)
PCE_COLOR_DIM = (44, 145, 96)
TRANSITION_COLOR = (238, 211, 34)
TRANSITION_COLOR_DIM = (128, 104, 20)
FLOW_COLOR = PCE_COLOR
FLOW_COLOR_DIM = PCE_COLOR_DIM
FLOW_ARROW = (14, 47, 34)
FLOW_TRANSFER_ARROW = (72, 62, 16)


@dataclass
class Event:
    time_s: float
    frame: int
    track_id: str
    from_region: str
    to_region: str
    pce: float
    class_name: str
    source: str
    edge_type: str
    valid_8lane: bool


# -----------------------------------------------------------------------------
# Font tiếng Việt
# -----------------------------------------------------------------------------

_FONT_CACHE: Dict[int, object] = {}
_FONT_PATH = None


def _find_font_path() -> str | None:
    env_path = os.environ.get("FLOW_SIM_FONT", "").strip()
    candidates = [
        env_path,
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _get_font(size: int):
    global _FONT_PATH
    if ImageFont is None:
        return None
    if _FONT_PATH is None:
        _FONT_PATH = _find_font_path() or ""
    key = max(10, int(size))
    if key not in _FONT_CACHE:
        if _FONT_PATH:
            _FONT_CACHE[key] = ImageFont.truetype(_FONT_PATH, key)
        else:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _bgr_to_rgb(color):
    return int(color[2]), int(color[1]), int(color[0])


def draw_text(frame, text, pos, size=16, color=TEXT, bg=None):
    if Image is None or ImageDraw is None or ImageFont is None:
        cv2.putText(frame, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color, 1, cv2.LINE_AA)
        return

    font = _get_font(size)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    x, y = int(pos[0]), int(pos[1])
    if bg is not None:
        bbox = draw.textbbox((x, y), str(text), font=font)
        pad = 4
        draw.rounded_rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            radius=5,
            fill=_bgr_to_rgb(bg),
        )
    draw.text((x, y), str(text), font=font, fill=_bgr_to_rgb(color))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------------
# Hình học / màu
# -----------------------------------------------------------------------------


def float_value(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def int_value(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def dim_color(color, factor=0.45):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def blend_color(c1, c2, alpha):
    alpha = max(0.0, min(1.0, float(alpha)))
    return tuple(int(c1[i] * (1.0 - alpha) + c2[i] * alpha) for i in range(3))


def center_color(center_pce: float, args):
    density = center_pce / max(args.center_size * args.center_size, 1e-6)
    if density < args.center_density_warning:
        return blend_color(PCE_COLOR, WARN, density / max(args.center_density_warning, 1e-6) * 0.30)
    if density < args.center_density_critical:
        t = (density - args.center_density_warning) / max(args.center_density_critical - args.center_density_warning, 1e-6)
        return blend_color(WARN, CRITICAL, t)
    return CRITICAL


def draw_gradient_background(frame):
    h, w = frame.shape[:2]
    top = np.array(BG_TOP, dtype=np.float32)
    bottom = np.array(BG_BOTTOM, dtype=np.float32)
    for y in range(h):
        t = y / max(h - 1, 1)
        frame[y, :] = (top * (1.0 - t) + bottom * t).astype(np.uint8)

    # Lưới nền mảnh, đủ tạo chiều sâu nhưng không làm rối flow.
    for x in range(0, w, 80):
        cv2.line(frame, (x, 0), (x, h), GRID, 1, cv2.LINE_AA)
    for y in range(0, h, 80):
        cv2.line(frame, (0, y), (w, y), GRID, 1, cv2.LINE_AA)

    # Glow nhẹ tại vùng giao nhau để layout có điểm nhấn.
    overlay = frame.copy()
    cv2.circle(overlay, CENTER, 245, (18, 28, 38), -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0)


def draw_alpha_rect(frame, pt1, pt2, color, alpha=0.6, radius=18, border=None):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    overlay = frame.copy()
    if radius <= 0:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    else:
        r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    if border is not None:
        if radius <= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
        else:
            # Viền đơn giản bằng polylines + cung không cần quá chính xác.
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), border, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), border, 1, cv2.LINE_AA)


def draw_card(frame, pt1, pt2, alpha=0.78, radius=20):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    draw_alpha_rect(frame, (x1 + 7, y1 + 9), (x2 + 7, y2 + 9), CARD_SHADOW, 0.18, radius=radius, border=None)
    draw_alpha_rect(frame, (x1, y1), (x2, y2), PANEL, alpha, radius=radius, border=PANEL_EDGE_SOFT)
    cv2.line(frame, (x1 + radius, y1 + 1), (x2 - radius, y1 + 1), PANEL_EDGE, 1, cv2.LINE_AA)


def draw_progress_bar(frame, x, y, w, value, color, thickness=7):
    value = max(0.0, min(1.0, float(value)))
    cv2.line(frame, (int(x), int(y)), (int(x + w), int(y)), PANEL_INNER, thickness, cv2.LINE_AA)
    if value > 0:
        cv2.line(frame, (int(x), int(y)), (int(x + w * value), int(y)), color, thickness, cv2.LINE_AA)


def point_on_segment(p1, p2, t):
    return (int(p1[0] + (p2[0] - p1[0]) * t), int(p1[1] + (p2[1] - p1[1]) * t))


def polyline_length(pts: List[Tuple[int, int]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:]))


def point_on_polyline(pts: List[Tuple[int, int]], t: float) -> Tuple[int, int]:
    if not pts:
        return CENTER
    if len(pts) == 1:
        return pts[0]
    lengths = []
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        lengths.append(length)
        total += length
    if total <= 1e-6:
        return pts[-1]
    target = max(0.0, min(1.0, t)) * total
    acc = 0.0
    for (a, b), length in zip(zip(pts[:-1], pts[1:]), lengths):
        if acc + length >= target:
            local_t = (target - acc) / max(length, 1e-6)
            return point_on_segment(a, b, local_t)
        acc += length
    return pts[-1]


def road_frame_width(args) -> int:
    road_w = int(args.road_width * args.cell_px * args.road_frame_scale)
    return max(70, min(132, road_w))


def lane_center_gap(args) -> int:
    # lane_gap_px là khoảng lệch từ trục đường đến tâm mỗi làn.
    # Giới hạn theo bề rộng đường để flow luôn nằm giữa làn, không dính mép đường.
    road_w = road_frame_width(args)
    return int(max(12, min(float(args.lane_gap_px), road_w * 0.28)))


def center_anchor_radius(args) -> int:
    return int(args.cell_px * args.center_size * 0.88)


def center_radius_from_pce(center_pce: float, args) -> int:
    capacity = max(args.center_size * args.center_size, 1e-6)
    density = max(0.0, center_pce / capacity)
    r_min = int(args.cell_px * 0.70)
    r_max = int(args.center_size * args.cell_px * args.center_radius_scale)
    level = min(density / max(args.center_density_critical, 1e-6), 1.0)
    return int(r_min + (r_max - r_min) * level)


def effective_lane_length(args) -> int:
    """Độ dài mỗi làn trong UI.

    Tất cả 8 làn dùng cùng một độ dài để layout cân bằng. Giá trị mặc
    định được chọn theo dạng khung vuông quanh center; nếu người dùng truyền
    quá lớn thì giới hạn lại để không tràn khỏi canvas.
    """
    r = center_anchor_radius(args)
    cx, cy = CENTER
    margin = int(args.square_layout_margin_px)
    max_len = min(
        cx - r - margin,
        CANVAS_W - cx - r - margin - int(args.right_panel_reserve_px),
        cy - r - margin,
        CANVAS_H - cy - r - margin,
    )
    requested = int(args.lane_length_px)
    return max(120, min(requested, int(max_len)))


def lane_points(args) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
    # 8 làn bằng kích cỡ: mỗi làn bắt đầu từ mép center và kéo ra ngoài
    # cùng một độ dài. Không dùng margin riêng top/bottom/left/right nữa.
    r = center_anchor_radius(args)
    gap = lane_center_gap(args)
    length = effective_lane_length(args)
    cx, cy = CENTER

    return {
        "t1": ((cx - gap, cy - r - length), (cx - gap, cy - r)),
        "t2": ((cx + gap, cy - r - length), (cx + gap, cy - r)),
        "l1": ((cx - r - length, cy + gap), (cx - r, cy + gap)),
        "l2": ((cx - r - length, cy - gap), (cx - r, cy - gap)),
        "r1": ((cx + r + length, cy - gap), (cx + r, cy - gap)),
        "r2": ((cx + r + length, cy + gap), (cx + r, cy + gap)),
        "b1": ((cx + gap, cy + r + length), (cx + gap, cy + r)),
        "b2": ((cx - gap, cy + r + length), (cx - gap, cy + r)),
    }


def edge_points(edge: Tuple[str, str], args) -> List[Tuple[int, int]]:
    a, b = edge
    points = lane_points(args)
    if a in LANES and b == "center":
        outer, anchor = points[a]
        return [outer, anchor, CENTER]
    if a == "center" and b in LANES:
        outer, anchor = points[b]
        return [CENTER, anchor, outer]
    if a in LANES and b in LANES:
        a_outer, a_anchor = points[a]
        b_outer, b_anchor = points[b]
        return [a_outer, a_anchor, CENTER, b_anchor, b_outer]
    return [CENTER, CENTER]


# -----------------------------------------------------------------------------
# Đọc dữ liệu
# -----------------------------------------------------------------------------


def is_valid_8lane_edge(from_region: str, to_region: str) -> bool:
    return (from_region in INBOUND and to_region == "center") or (from_region == "center" and to_region in OUTBOUND)


def edge_type(from_region: str, to_region: str) -> str:
    if from_region in INBOUND and to_region == "center":
        return "inbound_to_center"
    if from_region == "center" and to_region in OUTBOUND:
        return "center_to_outbound"
    if from_region in OUTBOUND and to_region == "center":
        return "outbound_to_center_unexpected"
    if from_region == "center" and to_region in INBOUND:
        return "center_to_inbound_unexpected"
    if from_region in LANES and to_region in LANES:
        return "lane_to_lane_direct"
    return "other"


def display_edge(edge: Tuple[str, str], args) -> Tuple[Tuple[str, str] | None, bool]:
    """Trả về cạnh dùng để vẽ.

    Dữ liệu thực tế đôi khi có cạnh không hợp lệ, ví dụ center->t1 hoặc t2->center.
    Mặc định ẩn các cạnh này để tránh UI gây hiểu nhầm hướng làn.
    Có thể dùng --reverse-flow-policy show/normalize khi cần debug.
    """
    a, b = edge
    if is_valid_8lane_edge(a, b):
        return edge, False
    policy = getattr(args, "reverse_flow_policy", "hide")
    if policy == "show":
        return edge, True
    if policy == "normalize":
        if a == "center" and b in INBOUND:
            return ("center", IN_TO_OUT[b]), True
        if a in OUTBOUND and b == "center":
            return (OUT_TO_IN[a], "center"), True
    return None, True


def find_csv(run_dir: str, name: str) -> str:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thiếu {name} trong {run_dir}")
    return path


def load_events(run_dir: str, args) -> List[Event]:
    path = os.path.join(run_dir, "flow_edges_real.csv")
    events: List[Event] = []
    if not os.path.exists(path):
        return events
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = row.get("from_region", "")
            to = row.get("to_region", "")
            if fr not in REGIONS or to not in REGIONS:
                continue
            valid = str(row.get("is_valid_8lane_edge", "")).strip()
            valid_bool = valid in {"1", "true", "True"} if valid != "" else is_valid_8lane_edge(fr, to)
            if args.valid_edges_only and not valid_bool:
                continue
            events.append(Event(
                time_s=float_value(row.get("time_s")),
                frame=int_value(row.get("frame")),
                track_id=str(row.get("track_id", "")),
                from_region=fr,
                to_region=to,
                pce=float_value(row.get("pce")),
                class_name=row.get("class_name", ""),
                source=row.get("source", "observed") or "observed",
                edge_type=row.get("edge_type", "") or edge_type(fr, to),
                valid_8lane=valid_bool,
            ))
    events.sort(key=lambda e: e.time_s)
    return events


def load_state_rows(run_dir: str) -> Dict[str, List[dict]]:
    path = os.path.join(run_dir, "region_state_timeseries.csv")
    rows_by_region: Dict[str, List[dict]] = {r: [] for r in REGIONS}
    if not os.path.exists(path):
        return rows_by_region
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row.get("region", "")
            if region not in rows_by_region:
                continue
            rows_by_region[region].append({
                "time_s": float_value(row.get("time_s")),
                "vehicle_count_now": float_value(row.get("vehicle_count_now")),
                "pce_now": float_value(row.get("pce_now")),
                "queue_estimate_pce": float_value(row.get("queue_estimate_pce"), float_value(row.get("pce_now"))),
            })
    for rows in rows_by_region.values():
        rows.sort(key=lambda r: r["time_s"])
    return rows_by_region


def state_at_or_before(rows_by_region: Dict[str, List[dict]], sim_time: float) -> Dict[str, dict]:
    out = {}
    for region, rows in rows_by_region.items():
        best = None
        for row in rows:
            if row["time_s"] <= sim_time:
                best = row
            else:
                break
        out[region] = best or {"time_s": 0.0, "vehicle_count_now": 0.0, "pce_now": 0.0, "queue_estimate_pce": 0.0}
    return out


def aggregate_edges(events: List[Event], sim_time: float, args) -> Dict[Tuple[str, str], dict]:
    start = max(0.0, sim_time - args.window_seconds)
    agg = defaultdict(lambda: {"pce": 0.0, "count": 0, "sources": Counter(), "classes": Counter(), "valid": True, "edge_type": ""})
    for e in events:
        if e.time_s < start:
            continue
        if e.time_s > sim_time:
            break
        original_key = (e.from_region, e.to_region)
        key, is_reverse = display_edge(original_key, args)
        if key is None:
            # Lưu thống kê cạnh không hợp lệ nếu cần debug, không vẽ lên lane chính.
            rev = agg[("__reverse__", "__reverse__")]
            rev["pce"] += e.pce
            rev["count"] += 1
            rev["sources"][e.source] += 1
            rev["classes"][e.class_name] += 1
            rev["valid"] = False
            rev["edge_type"] = e.edge_type
            rev.setdefault("reverse_edges", Counter())[f"{e.from_region}->{e.to_region}"] += 1
            continue
        item = agg[key]
        item["pce"] += e.pce
        item["count"] += 1
        item["sources"][e.source] += 1
        item["classes"][e.class_name] += 1
        item["valid"] = item["valid"] and e.valid_8lane and not is_reverse
        item["edge_type"] = e.edge_type
        if is_reverse:
            item.setdefault("reverse_edges", Counter())[f"{e.from_region}->{e.to_region}"] += 1
    for item in agg.values():
        pce_per_s = item["pce"] / max(args.window_seconds, 1e-6)
        item["pce_per_s"] = pce_per_s
        item["width_units"] = pce_per_s / max(args.lane_capacity_pceps, 1e-6)
    return dict(agg)


# -----------------------------------------------------------------------------
# Vẽ UI
# -----------------------------------------------------------------------------


def draw_base(frame, args):
    draw_gradient_background(frame)

    road_w = road_frame_width(args)
    cx, cy = CENTER
    r = center_anchor_radius(args)
    length = effective_lane_length(args)
    left_x = cx - r - length - road_w // 2
    right_x = cx + r + length + road_w // 2
    top_y = cy - r - length - road_w // 2
    bottom_y = cy + r + length + road_w // 2

    # Mảng đường nền: tối, ít viền, có highlight nhẹ để hiện đại hơn.
    draw_alpha_rect(frame, (left_x, cy - road_w // 2), (right_x, cy + road_w // 2), ROAD, 0.88, radius=18, border=ROAD_EDGE)
    draw_alpha_rect(frame, (cx - road_w // 2, top_y), (cx + road_w // 2, bottom_y), ROAD, 0.88, radius=18, border=ROAD_EDGE)
    draw_alpha_rect(frame, (cx - road_w // 2 - 5, cy - road_w // 2 - 5), (cx + road_w // 2 + 5, cy + road_w // 2 + 5), ROAD, 0.96, radius=20, border=None)

    # Vạch phân làn mảnh, màu trung tính để không cạnh tranh với flow.
    pts = lane_points(args)
    for lane, (outer, anchor) in pts.items():
        cv2.line(frame, outer, anchor, ROAD_HIGHLIGHT, 1, cv2.LINE_AA)

    # Nhãn lane dạng pill nhỏ, phân biệt làn vào/ra bằng màu chữ.
    label_positions = {
        "t1": (-52, 16), "t2": (10, 16),
        "b1": (10, -30), "b2": (-52, -30),
        "l1": (18, 12), "l2": (18, -26),
        "r1": (-76, -26), "r2": (-76, 12),
    }
    for lane, (outer, anchor) in pts.items():
        dx, dy = label_positions.get(lane, (0, 0))
        color = PCE_COLOR if lane in INBOUND else TRANSITION_COLOR
        draw_text(frame, REGION_LABELS[lane], (outer[0] + dx, outer[1] + dy), 12, color, bg=PILL_BG)


def draw_polyline(frame, pts: List[Tuple[int, int]], color, thickness):
    if len(pts) < 2:
        return
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def draw_flow_arrows(frame, pts: List[Tuple[int, int]], color, thickness, sim_time: float, speed_units: float):
    # Flow chính: line xanh rộng + mũi tên di chuyển theo hướng flow.
    # Viền/glow chỉ lớn hơn core một ít để tránh cảm giác có khung đen bao quanh.
    halo = max(thickness + 1, int(thickness * 1.04))
    core = max(5, int(thickness))
    draw_polyline(frame, pts, dim_color(color, 0.52), halo)
    draw_polyline(frame, pts, color, core)

    length = polyline_length(pts)
    if length <= 1e-6:
        return

    arrow_gap = max(88.0, 140.0 - min(core, 50) * 0.7)
    arrow_count = max(2, int(length / arrow_gap))
    phase = (sim_time * (0.20 + min(speed_units, 4.0) * 0.08)) % 1.0
    arrow_len_ratio = min(0.034, max(0.016, 15.0 / max(length, 1.0)))
    arrow_thick = max(1, min(3, int(core * 0.16)))

    for i in range(arrow_count):
        t_mid = ((i / arrow_count) + phase) % 1.0
        t0 = max(0.0, t_mid - arrow_len_ratio)
        t1 = min(1.0, t_mid + arrow_len_ratio)
        if t1 <= t0 + 1e-4:
            continue
        tail = point_on_polyline(pts, t0)
        head = point_on_polyline(pts, t1)
        cv2.arrowedLine(
            frame,
            tail,
            head,
            FLOW_ARROW,
            arrow_thick,
            cv2.LINE_AA,
            tipLength=0.30,
        )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def polyline_subsegment(pts: List[Tuple[int, int]], t0: float, t1: float, samples: int = 12) -> List[Tuple[int, int]]:
    t0 = clamp01(t0)
    t1 = clamp01(t1)
    if t1 <= t0:
        t1 = min(1.0, t0 + 0.01)
    samples = max(2, int(samples))
    return [point_on_polyline(pts, t0 + (t1 - t0) * i / (samples - 1)) for i in range(samples)]


def draw_moving_flow_segment(frame, pts: List[Tuple[int, int]], t_tail: float, t_head: float, color, thickness: int):
    segment = polyline_subsegment(pts, t_tail, t_head, samples=14)
    if len(segment) < 2:
        return

    halo = max(thickness + 5, int(thickness * 1.55))
    core = max(5, int(thickness))
    draw_polyline(frame, segment, dim_color(color, 0.30), halo)
    draw_polyline(frame, segment, dim_color(color, 0.70), max(core + 2, int(core * 1.12)))
    draw_polyline(frame, segment, color, core)

    arrow_tail = point_on_polyline(pts, max(t_tail, t_head - 0.055))
    arrow_head = point_on_polyline(pts, t_head)
    arrow_thick = max(2, min(5, int(core * 0.22)))
    cv2.arrowedLine(frame, arrow_tail, arrow_head, FLOW_TRANSFER_ARROW, arrow_thick, cv2.LINE_AA, tipLength=0.38)
    cv2.circle(frame, arrow_head, max(4, core // 3), color, -1, cv2.LINE_AA)


def draw_transition_events(frame, events: List[Event], args, sim_time: float):
    if args.hide_transition_flow:
        return

    lifetime = max(0.25, float(args.transition_flow_seconds))
    t_min = sim_time - lifetime
    active: List[Tuple[Event, Tuple[str, str]]] = []

    for event in events:
        if event.time_s > sim_time:
            break
        if event.time_s < t_min:
            continue
        edge, _ = display_edge((event.from_region, event.to_region), args)
        if edge is None:
            continue
        active.append((event, edge))

    if not active:
        return

    active = active[-max(1, int(args.max_transition_events)):]
    max_thick = max(int(args.flow_min_px), int(args.road_width * args.cell_px * args.flow_max_scale))
    trail = max(0.08, min(0.65, float(args.transition_trail_ratio)))

    for event, edge in active:
        age = max(0.0, sim_time - event.time_s)
        progress = clamp01(age / lifetime)
        # Ease-out: đầu dòng đi nhanh lúc đầu rồi chậm lại nhẹ khi tới vùng đích.
        head_t = 1.0 - (1.0 - progress) * (1.0 - progress)
        tail_t = max(0.0, head_t - trail)
        pts = edge_points(edge, args)
        pce = max(0.4, float_value(event.pce, 1.0))
        thickness = max(
            int(args.flow_min_px),
            min(max_thick, int(args.flow_base_px + pce * args.transition_width_scale)),
        )
        draw_moving_flow_segment(frame, pts, tail_t, head_t, TRANSITION_COLOR, thickness)

        if args.show_flow_values and head_t > 0.18:
            label_pos = point_on_polyline(pts, min(0.92, head_t))
            draw_text(
                frame,
                f"{event.from_region.upper()}→{event.to_region.upper()}  {event.pce:.1f} PCE",
                (label_pos[0] + 10, label_pos[1] - 18),
                11,
                TEXT,
                bg=PILL_BG,
            )

def draw_flow(frame, edges: Dict[Tuple[str, str], dict], args, sim_time: float):
    # Vẽ outbound trước, inbound sau để dòng vào center nổi bật hơn khi giao nhau.
    ordered_edges = sorted(edges.items(), key=lambda kv: 0 if kv[0][0] == "center" else 1)
    for edge, item in ordered_edges:
        if edge[0].startswith("__"):
            continue
        width_units = float(item.get("width_units", 0.0))
        if width_units <= 0:
            continue
        pts = edge_points(edge, args)
        max_thick = int(args.road_width * args.cell_px * args.flow_max_scale)
        thickness = max(
            int(args.flow_min_px),
            min(max_thick, int(args.flow_base_px + width_units * args.cell_px * args.flow_width_scale)),
        )
        color = TRANSITION_COLOR
        draw_flow_arrows(frame, pts, color, thickness, sim_time, width_units)
        if args.show_flow_values or width_units >= args.flow_note_threshold:
            p = point_on_polyline(pts, 0.58)
            draw_text(frame, f"{item['pce_per_s']:.2f} PCE/s", (p[0] + 10, p[1] - 16), 12, TEXT, bg=PILL_BG)


def draw_lane_pce_arrows(frame, start, end, width, arrow_color):
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length <= 1e-6:
        return

    # 1-2 mũi tên nhỏ nằm trên vạch PCE, dùng màu tối để nổi trên nền xanh.
    arrow_count = 2 if length >= 230 else 1
    positions = (0.42, 0.72) if arrow_count == 2 else (0.58,)
    arrow_len = min(48.0, max(26.0, length * 0.16))
    arrow_t = arrow_len / length
    arrow_thick = max(1, min(3, int(width * 0.16)))

    for t_mid in positions:
        t0 = max(0.0, t_mid - arrow_t * 0.5)
        t1 = min(1.0, t_mid + arrow_t * 0.5)
        tail = point_on_segment(start, end, t0)
        head = point_on_segment(start, end, t1)
        cv2.arrowedLine(frame, tail, head, arrow_color, arrow_thick, cv2.LINE_AA, tipLength=0.32)


def draw_occupancy(frame, states: Dict[str, dict], args):
    # Dải PCE tại làn: xanh lá mềm, có nền xám để thấy hướng/làn kể cả khi PCE thấp.
    pts = lane_points(args)
    max_width = max(8, int(road_frame_width(args) * args.lane_pce_max_width_scale))

    for lane in LANES:
        pce = float_value(states.get(lane, {}).get("pce_now"))
        outer, anchor = pts[lane]
        if lane in INBOUND:
            start = outer
            end = anchor
        else:
            start = anchor
            end = outer

        # Track nền mảnh cho từng làn giúp giao diện gọn và thống nhất.
        cv2.line(frame, start, end, PANEL_INNER, 5, cv2.LINE_AA)
        if pce < args.lane_pce_draw_threshold:
            continue

        width = max(
            int(args.lane_pce_min_px),
            min(max_width, int(args.lane_pce_base_px + pce * args.lane_pce_width_scale)),
        )

        overlay = frame.copy()
        cv2.line(overlay, start, end, dim_color(PCE_COLOR, 0.52), max(width + 4, int(width * 1.22)), cv2.LINE_AA)
        cv2.line(overlay, start, end, PCE_COLOR, width, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(overlay, args.lane_pce_alpha, frame, 1.0 - args.lane_pce_alpha, 0)
        draw_lane_pce_arrows(frame, start, end, width, FLOW_ARROW)


def draw_center(frame, states: Dict[str, dict], args):
    center_pce = float_value(states.get("center", {}).get("pce_now"))
    radius = center_radius_from_pce(center_pce, args)
    color = center_color(center_pce, args)

    # Halo nhiều lớp nhưng alpha thấp để center nổi bật mà không lấn flow.
    overlay = frame.copy()
    cv2.circle(overlay, CENTER, radius + int(args.center_border_px * 3.2), dim_color(color, 0.25), -1, cv2.LINE_AA)
    cv2.circle(overlay, CENTER, radius + int(args.center_border_px * 1.5), dim_color(color, 0.45), -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)

    fill = blend_color(CENTER_FILL, color, 0.16)
    cv2.circle(frame, CENTER, radius, fill, -1, cv2.LINE_AA)
    cv2.circle(frame, CENTER, radius, color, max(4, int(args.center_border_px * 0.55)), cv2.LINE_AA)
    cv2.circle(frame, CENTER, max(1, radius - int(args.center_border_px * 1.10)), PANEL_EDGE, 1, cv2.LINE_AA)

    density = center_pce / max(args.center_size * args.center_size, 1e-6)
    label_color = CRITICAL if density >= args.center_density_critical else (WARN if density >= args.center_density_warning else TEXT)

    # Nhãn center dạng compact card nhỏ đặt trong vòng tròn.
    cx, cy = CENTER
    draw_alpha_rect(frame, (cx - 62, cy - 36), (cx + 62, cy + 46), PILL_BG, 0.62, radius=14, border=PANEL_EDGE_SOFT)
    draw_text(frame, "CENTER", (cx - 34, cy - 28), 12, MUTED)
    draw_text(frame, f"{center_pce:.1f} PCE", (cx - 48, cy - 6), 19, label_color)
    draw_text(frame, f"mật độ {density:.2f}", (cx - 42, cy + 24), 11, MUTED)


def analyze_warnings(edges: Dict[Tuple[str, str], dict], states: Dict[str, dict], args) -> List[dict]:
    warnings = []

    # Cảnh báo chính dựa trên PCE hiện tại từng vùng, không dựa vào event chuyển vùng.
    heavy_lanes = []
    for lane in INBOUND:
        pce = float_value(states.get(lane, {}).get("pce_now"))
        if pce >= args.large_lane_pce:
            heavy_lanes.append((lane, pce))

    if len(heavy_lanes) >= 2:
        lanes = ", ".join(f"{lane.upper()}={pce:.1f}" for lane, pce in sorted(heavy_lanes, key=lambda x: -x[1])[:3])
        warnings.append({
            "level": "cảnh báo",
            "type": "many_inbound_pce",
            "edge": "center",
            "message": f"nhiều PCE ở làn vào: {lanes}",
        })

    center_pce = float_value(states.get("center", {}).get("pce_now"))
    density = center_pce / max(args.center_size * args.center_size, 1e-6)
    if density >= args.center_density_critical:
        warnings.append({
            "level": "nghiêm trọng",
            "type": "center_density",
            "edge": "center",
            "message": f"center quá tải, mật độ {density:.2f}",
        })
    elif density >= args.center_density_warning:
        warnings.append({
            "level": "cảnh báo",
            "type": "center_density",
            "edge": "center",
            "message": f"center bắt đầu dày, mật độ {density:.2f}",
        })

    # Dấu hiệu ùn tắc: làn vào đang có PCE lớn, center cũng đang cao.
    if density >= args.center_density_warning:
        for lane, pce in sorted(heavy_lanes, key=lambda x: -x[1])[:2]:
            warnings.append({
                "level": "cảnh báo",
                "type": "inbound_queue_to_dense_center",
                "edge": f"{lane}->center",
                "message": f"{lane.upper()} còn {pce:.1f} PCE khi center đang dày",
            })

    # Cảnh báo phụ theo event chuyển vùng, chỉ tắt khi ẩn transition flow.
    if not args.hide_transition_flow:
        inbound_heavy = []
        for lane in INBOUND:
            item = edges.get((lane, "center"))
            if item and item.get("width_units", 0.0) >= args.large_inflow_width:
                inbound_heavy.append((lane, item.get("width_units", 0.0)))
        if len(inbound_heavy) >= 2:
            lanes = ", ".join(lane.upper() for lane, _ in sorted(inbound_heavy, key=lambda x: -x[1])[:3])
            warnings.append({
                "level": "cảnh báo",
                "type": "two_large_transition_flows",
                "edge": "center",
                "message": f"{len(inbound_heavy)} chuyển vùng lớn vào center: {lanes}",
            })

    return warnings

def draw_pce_table(frame, states: Dict[str, dict], args):
    """Bảng PCE hiện tại của từng vùng, dùng pce_now từ region_state_timeseries.csv."""
    if getattr(args, "hide_pce_table", False):
        return

    x, y = 920, 126
    w, h = 332, 316
    draw_card(frame, (x, y), (x + w, y + h), alpha=0.80, radius=20)
    draw_text(frame, "PCE vùng", (x + 18, y + 14), 18, TEXT)
    draw_text(frame, "theo pce_now", (x + 18, y + 39), 11, DIM)

    rows = [
        ("T1 vào", "t1"), ("T2 ra", "t2"),
        ("L1 vào", "l1"), ("L2 ra", "l2"),
        ("R1 vào", "r1"), ("R2 ra", "r2"),
        ("B1 vào", "b1"), ("B2 ra", "b2"),
        ("Center", "center"),
    ]

    max_pce = max(float_value(states.get(region, {}).get("pce_now")) for _, region in rows)
    max_pce = max(max_pce, 3.0)
    draw_text(frame, f"max {max_pce:.1f}", (x + 254, y + 21), 11, MUTED, bg=PILL_BG)
    cv2.line(frame, (x + 18, y + 66), (x + w - 18, y + 66), PANEL_EDGE_SOFT, 1, cv2.LINE_AA)

    row_y = y + 78
    for i, (label, region) in enumerate(rows):
        pce = float_value(states.get(region, {}).get("pce_now"))
        yy = row_y + i * 25
        is_center = region == "center"
        color = center_color(pce, args) if is_center else (PCE_COLOR if region in INBOUND else TRANSITION_COLOR)
        text_color = TEXT if is_center else MUTED
        bar_x = x + 112
        bar_w = 122
        fill_w = int(bar_w * min(1.0, pce / max_pce))

        cv2.circle(frame, (x + 24, yy + 9), 3, color, -1, cv2.LINE_AA)
        draw_text(frame, label, (x + 34, yy), 11, text_color)
        cv2.line(frame, (bar_x, yy + 10), (bar_x + bar_w, yy + 10), PANEL_INNER, 7, cv2.LINE_AA)
        if fill_w > 0:
            cv2.line(frame, (bar_x, yy + 10), (bar_x + fill_w, yy + 10), color, 7, cv2.LINE_AA)
        draw_text(frame, f"{pce:4.1f}", (x + 260, yy), 11, TEXT)


def draw_warnings(frame, warnings: List[dict], args):
    x, y = 920, 466
    w, h = 332, 126
    draw_card(frame, (x, y), (x + w, y + h), alpha=0.80, radius=20)
    draw_text(frame, "Ghi chú", (x + 18, y + 14), 17, TEXT)
    if not warnings:
        cv2.circle(frame, (x + 28, y + 56), 5, OK, -1, cv2.LINE_AA)
        draw_text(frame, "Không có dấu hiệu bất thường", (x + 42, y + 47), 12, MUTED)
        return
    for i, warn in enumerate(warnings[:args.max_warnings_on_screen]):
        color = CRITICAL if warn.get("level") == "nghiêm trọng" else WARN
        msg = warn.get("message", "cảnh báo")
        if len(msg) > 42:
            msg = msg[:39] + "..."
        yy = y + 45 + i * 24
        cv2.circle(frame, (x + 24, yy + 9), 4, color, -1, cv2.LINE_AA)
        draw_text(frame, msg, (x + 38, yy), 11, color)


def draw_hud(frame, sim_time, duration, args, source_name, paused):
    # Header gọn hơn: title + trạng thái + progress thời gian.
    x, y = 24, 22
    w, h = 404, 92
    draw_card(frame, (x, y), (x + w, y + h), alpha=0.76, radius=20)
    draw_text(frame, "Mô phỏng lưu lượng", (x + 18, y + 14), 21, TEXT)
    status = "TẠM DỪNG" if paused else "ĐANG CHẠY"
    status_color = WARN if paused else OK
    draw_text(frame, status, (x + 264, y + 18), 11, status_color, bg=PILL_BG)
    draw_text(frame, f"t={sim_time:6.2f}/{duration:.1f}s", (x + 18, y + 45), 12, MUTED)
    draw_text(frame, f"{args.speed:.2f}x", (x + 160, y + 45), 12, TEXT, bg=PILL_BG)
    draw_text(frame, f"cửa sổ {args.window_seconds:g}s", (x + 214, y + 45), 12, MUTED, bg=PILL_BG)
    draw_text(frame, os.path.basename(source_name), (x + 18, y + 68), 11, DIM)
    draw_progress_bar(frame, x + 18, y + h - 8, w - 36, sim_time / max(duration, 1e-6), OK, thickness=5)

    # Chú giải hiện đại, tách PCE và transition flow bằng hai màu khác nhau.
    x, y = 920, 24
    w, h = 332, 84
    draw_card(frame, (x, y), (x + w, y + h), alpha=0.76, radius=20)
    draw_text(frame, "Chú giải", (x + 18, y + 13), 16, TEXT)
    cv2.line(frame, (x + 22, y + 47), (x + 94, y + 47), PCE_COLOR, 10, cv2.LINE_AA)
    draw_text(frame, "PCE hiện tại", (x + 106, y + 38), 11, MUTED)
    cv2.line(frame, (x + 210, y + 47), (x + 282, y + 47), TRANSITION_COLOR, 10, cv2.LINE_AA)
    cv2.arrowedLine(frame, (x + 238, y + 47), (x + 268, y + 47), FLOW_TRANSFER_ARROW, 2, cv2.LINE_AA, tipLength=0.32)
    draw_text(frame, "flow", (x + 286, y + 38), 11, MUTED)
    extra = "chuyển vùng: tắt" if args.hide_transition_flow else "chuyển vùng: động"
    draw_text(frame, extra, (x + 22, y + 62), 10, DIM)


def setup_log(args):
    if args.no_warning_log:
        args._warning_log_path = ""
        args._log_seen = set()
        return
    path = args.warning_log or os.path.join(args.run_dir, "fluid_replay_warnings.log")
    args._warning_log_path = path
    args._log_seen = set()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Cảnh báo mô phỏng flow thật 8 làn\n")
        f.write("# Ghi đè mỗi lần chạy\n")
        f.write("# time_s\tmức\tloại\tcạnh\tnội_dung\n")


def log_warnings(args, sim_time: float, warnings: List[dict]):
    path = getattr(args, "_warning_log_path", "")
    if not path:
        return
    bucket = int(sim_time / max(args.log_interval, 1e-6))
    lines = []
    for w in warnings:
        key = (bucket, w.get("level"), w.get("type"), w.get("edge"), w.get("message"))
        if key in args._log_seen:
            continue
        args._log_seen.add(key)
        lines.append(f"{sim_time:.3f}\t{w.get('level','')}\t{w.get('type','')}\t{w.get('edge','')}\t{w.get('message','')}\n")
    if lines:
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(lines)


def draw_frame(events, states_by_region, sim_time, duration, args, paused=False):
    frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    states = state_at_or_before(states_by_region, sim_time)
    edges = aggregate_edges(events, sim_time, args)
    warnings = analyze_warnings(edges, states, args)
    log_warnings(args, sim_time, warnings)

    draw_base(frame, args)
    draw_occupancy(frame, states, args)
    if not args.hide_transition_flow:
        if args.show_flow_values:
            draw_flow(frame, edges, args, sim_time)
        draw_transition_events(frame, events, args, sim_time)
    draw_center(frame, states, args)
    draw_pce_table(frame, states, args)
    draw_warnings(frame, warnings, args)
    draw_hud(frame, sim_time, duration, args, os.path.basename(args.run_dir), paused)
    return frame


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Mô phỏng PCE 8 làn bằng region_state_timeseries.csv.")
    parser.add_argument("run_dir", help="Thư mục flow_exports/<run_dir> chứa flow_edges_real.csv")
    parser.add_argument("--speed", type=float, default=1.0, help="Tốc độ phát. 1.0 = thời gian thực.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS hiển thị/video.")
    parser.add_argument("--window-seconds", "--bin-seconds", dest="window_seconds", type=float, default=1.0, help="Cửa sổ gom event thật gần nhất, tính bằng giây.")
    parser.add_argument("--road-width", type=float, default=3.0, help="Độ rộng đường/làn theo đơn vị hiển thị. >= 3 là nghiêm trọng.")
    parser.add_argument("--center-size", type=float, default=3.0, help="Sức chứa center theo đơn vị ô. 3 nghĩa là center 3x3.")
    parser.add_argument("--cell-px", type=float, default=40.0)
    parser.add_argument("--lane-gap-px", type=float, default=24.0)
    parser.add_argument("--lane-length-px", type=float, default=230.0, help="Độ dài UI của mỗi làn; 8 làn dùng cùng giá trị này.")
    parser.add_argument("--square-layout-margin-px", type=float, default=34.0, help="Khoảng chừa mép cho layout vuông cân bằng.")
    parser.add_argument("--right-panel-reserve-px", type=float, default=250.0, help="Khoảng chừa bên phải cho bảng PCE/ghi chú.")
    parser.add_argument("--road-frame-scale", type=float, default=0.38, help="Bề rộng nền đường.")
    parser.add_argument("--center-border-px", type=float, default=10.0)
    parser.add_argument("--center-radius-scale", type=float, default=0.72, help="Scale bán kính center theo PCE, giảm để center ít phình hơn.")
    parser.add_argument("--flow-width-scale", type=float, default=0.72, help="Tỉ lệ bề rộng dòng flow theo PCE/s.")
    parser.add_argument("--flow-base-px", type=float, default=4.0)
    parser.add_argument("--flow-min-px", type=float, default=5.0)
    parser.add_argument("--flow-max-scale", type=float, default=0.56)
    parser.add_argument("--reverse-flow-policy", choices=("hide", "show", "normalize"), default="hide", help="Xử lý cạnh không hợp lệ: hide=ẩn, show=vẽ đúng cạnh, normalize=đưa về cạnh hợp lệ tương ứng.")
    parser.add_argument("--lane-capacity-pceps", type=float, default=1.0)
    parser.add_argument("--hide-transition-flow", action="store_true", help="Ẩn lớp flow động từ flow_edges_real.csv.")
    parser.add_argument("--transition-flow-seconds", type=float, default=1.55, help="Thời gian một event chuyển vùng chảy từ nguồn đến đích.")
    parser.add_argument("--transition-trail-ratio", type=float, default=0.34, help="Độ dài vệt chảy theo tỉ lệ chiều dài đường đi.")
    parser.add_argument("--transition-width-scale", type=float, default=3.8, help="Scale bề rộng vệt flow theo PCE của event.")
    parser.add_argument("--max-transition-events", type=int, default=80, help="Số event chuyển vùng tối đa được vẽ đồng thời.")
    parser.add_argument("--large-lane-pce", type=float, default=1.8, help="Ngưỡng PCE lớn tại làn vào để ghi chú ùn tắc.")
    parser.add_argument("--center-density-warning", type=float, default=0.65)
    parser.add_argument("--center-density-critical", type=float, default=0.95)
    parser.add_argument("--large-inflow-width", type=float, default=1.15)
    parser.add_argument("--flow-note-threshold", type=float, default=2.0)
    parser.add_argument("--lane-pce-draw-threshold", type=float, default=0.08, help="PCE tối thiểu để vẽ dải PCE tại làn.")
    parser.add_argument("--lane-pce-base-px", type=float, default=6.0)
    parser.add_argument("--lane-pce-min-px", type=float, default=7.0)
    parser.add_argument("--lane-pce-width-scale", type=float, default=3.6, help="Scale bề rộng dải PCE theo pce_now của vùng.")
    parser.add_argument("--lane-pce-max-width-scale", type=float, default=0.30, help="Giới hạn bề rộng dải PCE theo bề rộng nền đường.")
    parser.add_argument("--lane-pce-alpha", type=float, default=0.66)
    parser.add_argument("--hide-pce-table", action="store_true", help="Ẩn bảng PCE bên phải để khung mô phỏng rộng hơn.")
    parser.add_argument("--max-warnings-on-screen", type=int, default=3)
    parser.add_argument("--show-flow-values", action="store_true")
    parser.add_argument("--valid-edges-only", action="store_true")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--save", default="")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--warning-log", default="")
    parser.add_argument("--no-warning-log", action="store_true")
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.run_dir = os.path.abspath(args.run_dir)
    args.window_seconds = max(args.window_seconds, 1e-3)
    args.fps = max(args.fps, 1.0)
    args.speed = max(args.speed, 1e-6)
    args.transition_flow_seconds = max(args.transition_flow_seconds, 0.25)
    args.transition_trail_ratio = max(0.08, min(args.transition_trail_ratio, 0.65))
    args.max_transition_events = max(1, args.max_transition_events)

    events = load_events(args.run_dir, args)
    states_by_region = load_state_rows(args.run_dir)
    has_state = any(bool(rows) for rows in states_by_region.values())
    if not has_state:
        raise RuntimeError("Không tìm thấy dữ liệu PCE trong region_state_timeseries.csv")

    duration = max((e.time_s for e in events), default=0.0)
    for rows in states_by_region.values():
        if rows:
            duration = max(duration, max(r["time_s"] for r in rows))
    if args.end_time is not None:
        duration = min(duration, args.end_time)

    setup_log(args)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (CANVAS_W, CANVAS_H))
        if not writer.isOpened():
            raise RuntimeError(f"Không thể mở bộ ghi video: {args.save}")

    window_name = "Mô phỏng lưu lượng"
    if not args.no_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    sim_time = max(0.0, args.start_time)
    paused = False
    frame_dt = 1.0 / args.fps
    last_wall = time.time()

    try:
        while sim_time <= duration:
            now = time.time()
            elapsed = now - last_wall
            last_wall = now
            if not paused:
                sim_time += elapsed * args.speed

            frame = draw_frame(events, states_by_region, sim_time, duration, args, paused)
            if writer is not None:
                writer.write(frame)

            if not args.no_window:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord(" "):
                    paused = not paused
                if key == ord("r"):
                    sim_time = max(0.0, args.start_time)
            else:
                time.sleep(frame_dt)
    finally:
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    log_path = getattr(args, "_warning_log_path", "")
    if log_path:
        print(f"Log cảnh báo: {log_path}")


if __name__ == "__main__":
    main()
