"""Mô phỏng lại dữ liệu flow thật 8 làn được xuất từ ứng dụng chính.

Chương trình này chỉ dùng dữ liệu xuất thật:
- flow_edges_real.csv cho các sự kiện chuyển vùng thật
- region_state_timeseries.csv cho trạng thái PCE/mật độ quan sát được

Không nội suy, không tạo kịch bản giả định không đèn, không mô phỏng hành vi
đèn giao thông. Các tham số runtime chỉ thay đổi cách hiển thị/gom cửa sổ,
không thay đổi dữ liệu gốc.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

# ----------------------------- Hằng số bố cục -----------------------------

CANVAS_W = 1120
CANVAS_H = 780
CENTER = (560, 390)
LANES = ("t1", "t2", "l1", "l2", "r1", "r2", "b1", "b2")
INBOUND = {"t1", "l1", "r1", "b1"}
OUTBOUND = {"t2", "l2", "r2", "b2"}
REGIONS = LANES + ("center",)

REGION_LABELS = {
    "t1": "T1 VÀO", "t2": "T2 RA",
    "l1": "L1 VÀO", "l2": "L2 RA",
    "r1": "R1 VÀO", "r2": "R2 RA",
    "b1": "B1 VÀO", "b2": "B2 RA",
    "center": "CENTER",
}

# Màu BGR
BG = (15, 23, 42)
PANEL = (30, 41, 59)
ROAD = (62, 74, 96)
ROAD_EDGE = (100, 116, 139)
CENTER_FILL = (51, 65, 85)
CENTER_EDGE = (148, 163, 184)
TEXT = (241, 245, 249)
MUTED = (148, 163, 184)
FLOW_OK = (94, 234, 212)
FLOW_INFERRED = (45, 212, 191)
FLOW_UNKNOWN = (156, 163, 175)
FLOW_WARNING = (0, 191, 255)
FLOW_CRITICAL = (0, 85, 255)
QUEUE_COLOR = (250, 204, 21)
UNEXPECTED = (147, 51, 234)

LANE_COLORS = {
    "t1": (222, 168, 78), "t2": (97, 162, 244),
    "l1": (136, 183, 82), "l2": (107, 107, 255),
    "r1": (229, 93, 155), "r2": (0, 127, 247),
    "b1": (216, 180, 0), "b2": (106, 196, 233),
}


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


# ----------------------------- Hàm hỗ trợ hình học -----------------------------

def center_side(args) -> int:
    return int(args.center_size * args.cell_px)


def lane_points(args) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Trả về điểm ngoài làn và điểm neo sát center cho từng vùng 8 làn."""
    side = center_side(args)
    half = side // 2
    lane_sep = max(28, int(args.cell_px * 1.20))
    cx, cy = CENTER

    return {
        # Nhánh trên: t1 đi vào center, t2 đi ra phía trên.
        "t1": ((cx - lane_sep, 80), (cx - lane_sep, cy - half)),
        "t2": ((cx + lane_sep, 80), (cx + lane_sep, cy - half)),
        # Nhánh trái: l1 đi vào center, l2 đi ra phía trái.
        "l1": ((90, cy + lane_sep), (cx - half, cy + lane_sep)),
        "l2": ((90, cy - lane_sep), (cx - half, cy - lane_sep)),
        # Nhánh phải: r1 đi vào center, r2 đi ra phía phải.
        "r1": ((CANVAS_W - 90, cy - lane_sep), (cx + half, cy - lane_sep)),
        "r2": ((CANVAS_W - 90, cy + lane_sep), (cx + half, cy + lane_sep)),
        # Nhánh dưới: b1 đi vào center, b2 đi ra phía dưới.
        "b1": ((cx + lane_sep, CANVAS_H - 70), (cx + lane_sep, cy + half)),
        "b2": ((cx - lane_sep, CANVAS_H - 70), (cx - lane_sep, cy + half)),
    }


def edge_points(edge: Tuple[str, str], args) -> List[Tuple[int, int]]:
    a, b = edge
    points = lane_points(args)
    if a in LANES and b == "center":
        outer, anchor = points[a]
        return [outer, anchor]
    if a == "center" and b in LANES:
        outer, anchor = points[b]
        return [anchor, outer]
    if a in LANES and b in LANES:
        a_outer, a_anchor = points[a]
        b_outer, b_anchor = points[b]
        return [a_outer, a_anchor, b_anchor, b_outer]
    if a == "center" and b == "center":
        return [CENTER, CENTER]
    return [CENTER, CENTER]


def point_on_segment(p1, p2, t):
    return (int(p1[0] + (p2[0] - p1[0]) * t), int(p1[1] + (p2[1] - p1[1]) * t))


# ------------------------------- Đọc dữ liệu --------------------------------

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


def find_csv(run_dir: str, name: str) -> str:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thiếu {name} trong {run_dir}")
    return path


def load_events(run_dir: str, args) -> List[Event]:
    path = find_csv(run_dir, "flow_edges_real.csv")
    events: List[Event] = []
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
    # Không nội suy: dùng snapshot thật gần nhất tại hoặc trước sim_time.
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


def events_in_window(events: List[Event], start: float, end: float) -> List[Event]:
    return [e for e in events if start <= e.time_s <= end]


def aggregate_edges(events: List[Event], sim_time: float, args) -> Dict[Tuple[str, str], dict]:
    start = max(0.0, sim_time - args.window_seconds)
    win = events_in_window(events, start, sim_time)
    agg = defaultdict(lambda: {"pce": 0.0, "count": 0, "sources": Counter(), "classes": Counter(), "valid": True, "edge_type": ""})
    for e in win:
        key = (e.from_region, e.to_region)
        item = agg[key]
        item["pce"] += e.pce
        item["count"] += 1
        item["sources"][e.source] += 1
        item["classes"][e.class_name] += 1
        item["valid"] = item["valid"] and e.valid_8lane
        item["edge_type"] = e.edge_type
    for key, item in agg.items():
        pce_per_s = item["pce"] / max(args.window_seconds, 1e-6)
        item["pce_per_s"] = pce_per_s
        item["width_units"] = pce_per_s / max(args.lane_capacity_pceps, 1e-6)
    return dict(agg)


# ----------------------------------- Vẽ hiển thị ---------------------------------

def draw_text(frame, text, pos, scale=0.48, color=TEXT, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_base(frame, args):
    frame[:] = BG
    side = center_side(args)
    half = side // 2
    road_thick = int(args.road_width * args.cell_px)
    points = lane_points(args)

    # Vẽ mỗi làn như một dải đường thật.
    for lane in LANES:
        outer, anchor = points[lane]
        cv2.line(frame, outer, anchor, ROAD, road_thick, cv2.LINE_AA)
        cv2.line(frame, outer, anchor, ROAD_EDGE, 2, cv2.LINE_AA)

    # Vùng center là ô vuông trơn, không vẽ lưới 3x3.
    x1, y1 = CENTER[0] - half, CENTER[1] - half
    x2, y2 = CENTER[0] + half, CENTER[1] + half
    cv2.rectangle(frame, (x1, y1), (x2, y2), CENTER_FILL, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), CENTER_EDGE, 2, cv2.LINE_AA)

    # Nhãn làn.
    for lane, (outer, anchor) in points.items():
        label = REGION_LABELS[lane]
        p = point_on_segment(outer, anchor, 0.18)
        color = LANE_COLORS.get(lane, TEXT)
        draw_text(frame, label, (p[0] - 28, p[1] + 5), 0.44, color, 1)
    draw_text(frame, "CENTER", (CENTER[0] - 36, CENTER[1] + 5), 0.52, TEXT, 1)


def flow_color(item):
    if item.get("width_units", 0) >= 3.0:
        return FLOW_CRITICAL
    if not item.get("valid", True):
        return UNEXPECTED
    sources = item.get("sources", {})
    if sources and sources.get("unknown", 0):
        return FLOW_UNKNOWN
    if sources and sources.get("inferred", 0):
        return FLOW_INFERRED
    return FLOW_OK


def draw_polyline(frame, pts: List[Tuple[int, int]], color, thickness):
    if len(pts) < 2:
        return
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def draw_flow(frame, edges: Dict[Tuple[str, str], dict], args):
    for edge, item in edges.items():
        width_units = float(item.get("width_units", 0.0))
        if width_units <= 0:
            continue
        pts = edge_points(edge, args)
        max_thick = int(args.road_width * args.cell_px * 1.15)
        thickness = max(3, min(max_thick, int(width_units * args.cell_px)))
        color = flow_color(item)
        draw_polyline(frame, pts, color, thickness)
        # đầu mũi tên nhỏ ở gần cuối hướng đi
        if len(pts) >= 2:
            p1, p2 = pts[-2], pts[-1]
            cv2.arrowedLine(frame, point_on_segment(p1, p2, 0.55), point_on_segment(p1, p2, 0.82), color, max(2, thickness // 8), cv2.LINE_AA, tipLength=0.28)
            label_pos = point_on_segment(p1, p2, 0.48)
            draw_text(frame, f"{item['pce_per_s']:.2f} PCE/s", (label_pos[0] + 6, label_pos[1] - 6), 0.40, TEXT, 1)


def draw_occupancy(frame, states: Dict[str, dict], args):
    points = lane_points(args)
    for lane in LANES:
        pce = float_value(states.get(lane, {}).get("queue_estimate_pce"))
        if pce <= 0.01:
            continue
        outer, anchor = points[lane]
        t = min(0.55, 0.08 + pce * 0.04)
        q_end = point_on_segment(outer, anchor, t)
        q_thick = max(5, min(int(args.road_width * args.cell_px), int(5 + pce * args.cell_px * 0.15)))
        cv2.line(frame, outer, q_end, QUEUE_COLOR, q_thick, cv2.LINE_AA)
        label_at = point_on_segment(outer, anchor, min(0.35, t + 0.08))
        draw_text(frame, f"mật độ {pce:.1f}", (label_at[0] - 22, label_at[1] - 8), 0.38, QUEUE_COLOR, 1)

    center_pce = float_value(states.get("center", {}).get("pce_now"))
    capacity = max(args.center_size * args.center_size, 1e-6)
    density = center_pce / capacity
    x = CENTER[0] - center_side(args) // 2
    y = CENTER[1] + center_side(args) // 2 + 24
    color = FLOW_CRITICAL if density >= args.center_density_critical else (FLOW_WARNING if density >= args.center_density_warning else TEXT)
    draw_text(frame, f"center={center_pce:.2f} PCE  mật độ={density:.2f}", (x, y), 0.47, color, 1)


def analyze_warnings(edges: Dict[Tuple[str, str], dict], states: Dict[str, dict], events: List[Event], sim_time: float, args) -> List[dict]:
    warnings = []
    for (a, b), item in edges.items():
        w = float(item.get("width_units", 0.0))
        if not item.get("valid", True):
            warnings.append({"level": "cảnh báo", "type": "unexpected_edge", "edge": f"{a}->{b}", "message": f"cạnh 8 làn bất thường {a}->{b}"})
        if w >= args.road_width:
            warnings.append({"level": "nghiêm trọng", "type": "do_rong_flow", "edge": f"{a}->{b}", "message": f"{a}->{b} rộng {w:.2f} >= đường {args.road_width:g}"})
        elif w >= args.road_width * args.flow_warning_ratio:
            warnings.append({"level": "cảnh báo", "type": "do_rong_flow", "edge": f"{a}->{b}", "message": f"flow cao {a}->{b}: {w:.2f}/{args.road_width:g}"})

    center_pce = float_value(states.get("center", {}).get("pce_now"))
    density = center_pce / max(args.center_size * args.center_size, 1e-6)
    if density >= args.center_density_critical:
        warnings.append({"level": "nghiêm trọng", "type": "center_density", "edge": "center", "message": f"mật độ center {density:.2f} nghiêm trọng"})
    elif density >= args.center_density_warning:
        warnings.append({"level": "cảnh báo", "type": "center_density", "edge": "center", "message": f"mật độ center {density:.2f}"})

    if args.imbalance_seconds > 0:
        win = events_in_window(events, max(0.0, sim_time - args.imbalance_seconds), sim_time)
        in_pce = sum(e.pce for e in win if e.to_region == "center" and e.from_region in INBOUND)
        out_pce = sum(e.pce for e in win if e.from_region == "center" and e.to_region in OUTBOUND)
        diff = (in_pce - out_pce) / max(args.imbalance_seconds, 1e-6)
        if abs(diff) >= args.imbalance_threshold_pceps:
            direction = "vào > ra" if diff > 0 else "ra > vào"
            warnings.append({"level": "cảnh báo", "type": "long_imbalance", "edge": "center", "message": f"mất cân bằng kéo dài {direction}: {diff:+.2f} PCE/s"})
    return warnings


def warning_label_pos(edge: str, idx: int, args):
    if edge == "center" or "->" not in edge:
        return CENTER[0] + center_side(args) // 2 + 14, CENTER[1] - 18 + idx * 18
    a, b = edge.split("->", 1)
    pts = edge_points((a, b), args)
    if len(pts) >= 2:
        base = point_on_segment(pts[0], pts[-1], 0.72)
        return base[0] + 12, base[1] + idx * 18
    return 830, 120 + idx * 18


def draw_warnings(frame, warnings: List[dict], args):
    counts = Counter()
    for w in warnings:
        edge = w.get("edge", "center")
        idx = counts[edge]
        counts[edge] += 1
        x, y = warning_label_pos(edge, idx, args)
        color = FLOW_CRITICAL if w.get("level") == "nghiêm trọng" else FLOW_WARNING
        msg = w.get("message", "cảnh báo")
        if len(msg) > 48:
            msg = msg[:45] + "..."
        draw_text(frame, msg, (x, y), 0.39, color, 1)


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
        f.write("# file được ghi đè mỗi lần chạy\n")
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


def draw_hud(frame, sim_time, duration, args, source_name, paused):
    draw_text(frame, "Mô phỏng flow thật 8 làn", (24, 34), 0.82, TEXT, 2)
    draw_text(frame, f"t={sim_time:7.2f}/{duration:.2f}s | tốc độ={args.speed:.2f}x | cửa sổ={args.window_seconds:g}s | dữ liệu={source_name}" + (" | TẠM DỪNG" if paused else ""), (24, 62), 0.47, MUTED, 1)
    draw_text(frame, f"rộng đường={args.road_width:g} ô | center={args.center_size:g}x{args.center_size:g} ô | không nội suy, không giả định không đèn", (24, 88), 0.45, MUTED, 1)
    draw_text(frame, "Độ rộng dòng = PCE/s thật trong cửa sổ gần nhất. >= độ rộng đường là nghiêm trọng.", (24, 113), 0.43, MUTED, 1)

    x, y = 840, 32
    legend = (("quan sát", FLOW_OK), ("suy luận", FLOW_INFERRED), ("không rõ", FLOW_UNKNOWN), ("cạnh bất thường", UNEXPECTED), ("cảnh báo", FLOW_WARNING), ("nghiêm trọng", FLOW_CRITICAL))
    for i, (name, color) in enumerate(legend):
        yy = y + i * 22
        cv2.line(frame, (x, yy), (x + 38, yy), color, 7, cv2.LINE_AA)
        draw_text(frame, name, (x + 50, yy + 5), 0.42, TEXT, 1)


def draw_frame(events, states_by_region, sim_time, duration, args, paused=False):
    frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    states = state_at_or_before(states_by_region, sim_time)
    edges = aggregate_edges(events, sim_time, args)
    warnings = analyze_warnings(edges, states, events, sim_time, args)
    log_warnings(args, sim_time, warnings)

    draw_base(frame, args)
    draw_occupancy(frame, states, args)
    draw_flow(frame, edges, args)
    draw_warnings(frame, warnings, args)
    draw_hud(frame, sim_time, duration, args, os.path.basename(args.run_dir), paused)
    return frame


# ------------------------------------ Chương trình chính -----------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Mô phỏng lại dữ liệu flow_edges_real.csv 8 làn thật dưới dạng sơ đồ đường/chất lỏng.")
    parser.add_argument("run_dir", help="thư mục flow_exports/<run_dir> chứa flow_edges_real.csv")
    parser.add_argument("--speed", type=float, default=1.0, help="Tốc độ phát. 1.0 = thời gian thực, 0.5 = chậm một nửa.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS hiển thị/video.")
    parser.add_argument("--window-seconds", "--bin-seconds", dest="window_seconds", type=float, default=1.0, help="Cửa sổ gom dữ liệu thật gần nhất, tính bằng giây.")
    parser.add_argument("--road-width", type=float, default=3.0, help="Độ rộng sức chứa hiển thị của đường/làn, tính theo đơn vị ô làn.")
    parser.add_argument("--center-size", type=float, default=3.0, help="Kích thước ô center theo đơn vị ô làn. 3 nghĩa là 3x3.")
    parser.add_argument("--cell-px", type=float, default=42.0, help="Số pixel cho mỗi đơn vị ô làn.")
    parser.add_argument("--lane-capacity-pceps", type=float, default=1.0, help="PCE/s tương ứng với một ô độ rộng flow.")
    parser.add_argument("--flow-warning-ratio", type=float, default=0.75, help="Cảnh báo khi độ rộng flow đạt tỉ lệ này so với độ rộng đường.")
    parser.add_argument("--center-density-warning", type=float, default=0.65)
    parser.add_argument("--center-density-critical", type=float, default=0.95)
    parser.add_argument("--imbalance-seconds", type=float, default=20.0)
    parser.add_argument("--imbalance-threshold-pceps", type=float, default=0.8)
    parser.add_argument("--valid-edges-only", action="store_true", help="Ẩn các chuyển vùng thật nhưng không chuẩn như center->làn vào hoặc làn ra->center.")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--save", default="", help="Đường dẫn .mp4 đầu ra nếu muốn lưu video.")
    parser.add_argument("--no-window", action="store_true", help="Không mở cửa sổ hiển thị OpenCV.")
    parser.add_argument("--warning-log", default="", help="Đường dẫn file log cảnh báo. Mặc định: <run_dir>/fluid_replay_warnings.log")
    parser.add_argument("--no-warning-log", action="store_true")
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.run_dir = os.path.abspath(args.run_dir)
    args.window_seconds = max(args.window_seconds, 1e-3)
    args.fps = max(args.fps, 1.0)
    args.speed = max(args.speed, 1e-6)

    events = load_events(args.run_dir, args)
    states_by_region = load_state_rows(args.run_dir)

    if not events:
        raise RuntimeError("Không tìm thấy event hợp lệ trong flow_edges_real.csv")

    duration = max(e.time_s for e in events)
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

    window_name = "Mô phỏng flow thật 8 làn"
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
