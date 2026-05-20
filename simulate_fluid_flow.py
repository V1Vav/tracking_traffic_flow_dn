"""Replay exported traffic flow as a fluid-like road diagram.

This simulator reads real event-level flow data by default:

    flow_edges_real.csv

The export step does not interpolate, smooth, or resample that file. Runtime
parameters here decide how to aggregate the raw events, so the same recording can
be viewed as 0.5 s, 1 s, 5 s, or RL-step demand without re-running detection.

Examples:
    python simulate_fluid_flow.py flow_exports/<run_dir>
    python simulate_fluid_flow.py flow_exports/<run_dir> --speed 0.5
    python simulate_fluid_flow.py flow_exports/<run_dir> --bin-seconds 3 --smooth-seconds 6
    python simulate_fluid_flow.py flow_exports/<run_dir> --road-width 3 --center-size 3
    python simulate_fluid_flow.py flow_exports/<run_dir> --flow-layout center
    python simulate_fluid_flow.py flow_exports/<run_dir> --flow-layout right_left
    python simulate_fluid_flow.py flow_exports/<run_dir> --scenario observed
    python simulate_fluid_flow.py flow_exports/<run_dir> --scenario no_signal
    python simulate_fluid_flow.py flow_exports/<run_dir> --scenario compare
    python simulate_fluid_flow.py flow_exports/<run_dir> --save fluid_replay.mp4 --no-window

Window controls:
    Space: pause/resume
    q/Esc: quit
    ] / [: faster / slower
"""

import argparse
import bisect
import csv
import math
import os
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

ROAD_BRANCHES = ("top", "right", "bottom", "left")
REGIONS = ("top", "left", "right", "bottom", "center")
ADJACENT_INFLOW_PAIRS = (("top", "right"), ("right", "bottom"), ("bottom", "left"), ("left", "top"))
CANVAS_SIZE = (1120, 820)
SCENARIO_CHOICES = ("observed", "no_signal", "compare")
SCENARIO_DISPLAY = {
    "observed": "OBSERVED REAL FLOW",
    "no_signal": "HYPOTHETICAL NO-SIGNAL",
    "compare": "COMPARE",
}
CENTER = (560, 410)
NODE_POS = {
    "top": (560, 70),
    "right": (1050, 410),
    "bottom": (560, 750),
    "left": (70, 410),
    "center": CENTER,
}

# OpenCV uses BGR.
BACKGROUND = (20, 22, 26)
ROAD_COLOR = (50, 54, 62)
ROAD_EDGE = (83, 88, 96)
CENTER_COLOR = (42, 46, 54)
GRID_COLOR = (86, 90, 100)
TEXT = (235, 238, 242)
MUTED = (160, 166, 174)
FLOW_OK = (70, 205, 120)
FLOW_INFERRED = (70, 170, 255)
FLOW_UNKNOWN = (150, 150, 150)
FLOW_WARNING = (30, 200, 255)
FLOW_CRITICAL = (40, 70, 245)
QUEUE_COLOR = (75, 80, 185)
COLLISION_COLOR = (30, 30, 240)

SOURCE_COLOR = {
    "observed": FLOW_OK,
    "inferred": FLOW_INFERRED,
    "unknown": FLOW_UNKNOWN,
    "none": (80, 80, 80),
}


def _float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


class RawFlowData:
    def __init__(self, events):
        self.events = sorted(events, key=lambda e: e["time_s"])
        self.times = [e["time_s"] for e in self.events]
        self.duration = self.times[-1] if self.times else 0.0

    def window(self, start_t, end_t):
        if not self.events:
            return []
        lo = bisect.bisect_left(self.times, start_t)
        hi = bisect.bisect_right(self.times, end_t)
        return self.events[lo:hi]


def read_raw_edges(run_dir):
    """Read real edge events. Prefer flow_edges_real.csv, fallback to transition CSV."""
    candidates = [
        os.path.join(run_dir, "flow_edges_real.csv"),
        os.path.join(run_dir, "region_transitions.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), "")
    if not path:
        raise FileNotFoundError("Missing flow_edges_real.csv or region_transitions.csv")

    events = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_region = row.get("from_region", "")
            to_region = row.get("to_region", "")
            if from_region not in REGIONS or to_region not in REGIONS:
                continue
            events.append({
                "time_s": _float(row.get("time_s")),
                "frame": _int(row.get("frame")),
                "track_id": row.get("track_id", ""),
                "from_region": from_region,
                "to_region": to_region,
                "edge": row.get("edge") or f"{from_region}->{to_region}",
                "pce": _float(row.get("pce"), _float(row.get("pce_sum"), 0.0)),
                "vehicle_count": _int(row.get("vehicle_count"), 1),
                "class_name": row.get("class_name", ""),
                "source": "inferred" if row.get("source", "unknown") == "mixed" else row.get("source", "unknown"),
                "confidence": _float(row.get("confidence"), _float(row.get("mean_confidence"), 0.0)),
                "reason": row.get("reason", ""),
            })
    return RawFlowData(events), path


def read_region_states(run_dir):
    path = os.path.join(run_dir, "region_state_timeseries.csv")
    rows_by_region = defaultdict(list)
    all_times = set()
    if not os.path.exists(path):
        return [], rows_by_region

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = round(_float(row.get("time_s")), 3)
            region = row.get("region", "")
            if region not in REGIONS:
                continue
            item = {
                "time_s": t,
                "pce_now": _float(row.get("pce_now")),
                "vehicle_count_now": _float(row.get("vehicle_count_now")),
                "queue_estimate_pce": _float(row.get("queue_estimate_pce")),
                "source": row.get("source", "observed"),
            }
            rows_by_region[region].append(item)
            all_times.add(t)
    for region in rows_by_region:
        rows_by_region[region].sort(key=lambda r: r["time_s"])
    return sorted(all_times), rows_by_region


def _merge_source_names(*sources):
    """Collapse source names without creating a visual 'mixed' class."""
    clean = [s for s in sources if s and s != "none"]
    if not clean:
        return "none"
    if "unknown" in clean:
        return "unknown"
    if "inferred" in clean or "mixed" in clean:
        return "inferred"
    return "observed"


def _state_at_region(region_rows, t, interpolate=False):
    if not region_rows:
        return {"pce_now": 0.0, "vehicle_count_now": 0.0, "queue_estimate_pce": 0.0, "source": "none"}
    times = [r["time_s"] for r in region_rows]
    idx = bisect.bisect_right(times, t) - 1
    if idx < 0:
        return region_rows[0]
    if not interpolate or idx >= len(region_rows) - 1:
        return region_rows[idx]
    a = region_rows[idx]
    b = region_rows[idx + 1]
    span = max(b["time_s"] - a["time_s"], 1e-6)
    alpha = min(max((t - a["time_s"]) / span, 0.0), 1.0)
    return {
        "time_s": t,
        "pce_now": a["pce_now"] + (b["pce_now"] - a["pce_now"]) * alpha,
        "vehicle_count_now": a["vehicle_count_now"] + (b["vehicle_count_now"] - a["vehicle_count_now"]) * alpha,
        "queue_estimate_pce": a["queue_estimate_pce"] + (b["queue_estimate_pce"] - a["queue_estimate_pce"]) * alpha,
        "source": _merge_source_names(a.get("source", "observed"), b.get("source", "observed")),
    }


def get_state_at_time(rows_by_region, t, interpolate=False):
    return {region: _state_at_region(rows_by_region.get(region, []), t, interpolate) for region in REGIONS}


def _combine_sources(counter):
    if not counter:
        return "none"
    nonzero = [k for k, v in counter.items() if v > 0]
    if not nonzero:
        return "none"
    return _merge_source_names(*nonzero)


def aggregate_events(raw_data, sim_time, args):
    """Aggregate raw events at runtime using CLI parameters."""
    bin_s = max(args.bin_seconds, 1e-3)
    smooth_s = max(args.smooth_seconds, 0.0)
    # For visualization and RL stepping, smoothing is a runtime decision. A value
    # of 0 means use only bin_seconds.
    window_s = max(bin_s, smooth_s if smooth_s > 0 else bin_s)
    start_t = max(0.0, sim_time - window_s)
    events = raw_data.window(start_t, sim_time)

    acc = defaultdict(lambda: {
        "pce_sum": 0.0,
        "vehicle_count": 0,
        "source_counter": Counter(),
        "confidence_sum": 0.0,
        "confidence_count": 0,
    })

    for e in events:
        key = (e["from_region"], e["to_region"])
        item = acc[key]
        item["pce_sum"] += e["pce"]
        item["vehicle_count"] += e.get("vehicle_count", 1)
        item["source_counter"][e.get("source", "unknown")] += 1
        item["confidence_sum"] += e.get("confidence", 0.0)
        item["confidence_count"] += 1

    edges = {}
    for key, item in acc.items():
        mean_conf = item["confidence_sum"] / max(item["confidence_count"], 1)
        pce_per_s = item["pce_sum"] / window_s
        width_units = pce_per_s / max(args.lane_capacity_pceps, 1e-6)
        edges[key] = {
            "pce_sum": item["pce_sum"],
            "pce_per_s": pce_per_s,
            "pce_per_min": pce_per_s * 60.0,
            "vehicle_count": item["vehicle_count"],
            "source": _combine_sources(item["source_counter"]),
            "confidence": mean_conf,
            "width_units": width_units,
        }
    return edges


def _point_on_segment(p1, p2, alpha):
    return (
        int(p1[0] + (p2[0] - p1[0]) * alpha),
        int(p1[1] + (p2[1] - p1[1]) * alpha),
    )


def _offset_points(p1, p2, offset):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = max(math.hypot(dx, dy), 1.0)
    ox = -dy / length * offset
    oy = dx / length * offset
    return (int(x1 + ox), int(y1 + oy)), (int(x2 + ox), int(y2 + oy))


def _direction_offset(branch, direction, lane_px, flow_layout="center"):
    """Return a lateral offset for the visual flow line.

    flow_layout="center": draw both directions in the middle of the road pipe.
    flow_layout="right_left": vehicles entering center use the right side of
    their movement direction, vehicles leaving center use the left side. This is
    only a visualization convention and can be unstable when the observed event
    direction itself is uncertain.
    """
    if flow_layout == "center":
        return 0.0
    if flow_layout == "right_left":
        return 0.34 * lane_px if direction == "in" else -0.34 * lane_px
    return 0.0


def draw_center_grid(frame, args, warnings):
    cell = int(args.cell_px)
    side = int(args.center_size * cell)
    x0 = CENTER[0] - side // 2
    y0 = CENTER[1] - side // 2
    x1 = x0 + side
    y1 = y0 + side

    cv2.rectangle(frame, (x0, y0), (x1, y1), CENTER_COLOR, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x1, y1), ROAD_EDGE, 2, cv2.LINE_AA)

    if any(w.get("type") == "center_density" and w.get("level") == "critical" for w in warnings):
        cv2.rectangle(frame, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), COLLISION_COLOR, 4, cv2.LINE_AA)

    cv2.putText(frame, f"CENTER size={args.center_size:g}", (x0 + 8, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT, 1, cv2.LINE_AA)


def draw_base(frame, args, warnings):
    road_px = int(args.road_width * args.cell_px)
    for branch in ROAD_BRANCHES:
        p1, p2 = NODE_POS[branch], CENTER
        cv2.line(frame, p1, p2, ROAD_COLOR, road_px, cv2.LINE_AA)
        cv2.line(frame, p1, p2, ROAD_EDGE, 2, cv2.LINE_AA)

    draw_center_grid(frame, args, warnings)

    label_offsets = {
        "top": (-36, -24),
        "right": (-78, -24),
        "bottom": (-58, 42),
        "left": (10, -24),
    }
    for name in ROAD_BRANCHES:
        pos = NODE_POS[name]
        dx, dy = label_offsets.get(name, (0, 0))
        cv2.putText(frame, name.upper(), (pos[0] + dx, pos[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT, 2, cv2.LINE_AA)


def flow_color(edge_data, args):
    source = edge_data.get("source", "unknown")
    width_units = edge_data.get("width_units", 0.0)
    if width_units >= args.road_width:
        return FLOW_CRITICAL
    if width_units >= args.road_width * args.flow_warning_ratio:
        return FLOW_WARNING
    return SOURCE_COLOR.get(source, FLOW_UNKNOWN)


def draw_flow_edge(frame, branch, direction, edge_data, sim_time, args):
    if not edge_data or edge_data.get("pce_per_s", 0.0) <= 0:
        return

    if direction == "in":
        p1, p2 = NODE_POS[branch], CENTER
    else:
        p1, p2 = CENTER, NODE_POS[branch]

    offset = _direction_offset(branch, direction, args.cell_px, args.flow_layout)
    p1, p2 = _offset_points(p1, p2, offset)

    width_units = edge_data.get("width_units", 0.0)
    visible_units = min(width_units, args.road_width)
    thickness = max(2, int(visible_units * args.cell_px))
    color = flow_color(edge_data, args)

    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    if width_units >= args.road_width:
        cv2.line(frame, p1, p2, COLLISION_COLOR, max(2, thickness + 6), cv2.LINE_AA)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    # White particles show continuous "always flowing" behavior. They are only
    # visualization particles; no vehicle-level simulation is added here.
    particle_count = max(1, min(14, int(2 + visible_units * 3)))
    particle_speed = 0.25 + min(edge_data.get("pce_per_s", 0.0) * 0.10, 0.85)
    for i in range(particle_count):
        alpha = ((sim_time * particle_speed) + i / particle_count) % 1.0
        pt = _point_on_segment(p1, p2, alpha)
        radius = max(3, min(10, int(thickness * 0.13)))
        cv2.circle(frame, pt, radius, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius + 1, color, 1, cv2.LINE_AA)

    label_pt = _point_on_segment(p1, p2, 0.55)
    label = f"{edge_data['pce_per_s']:.2f} PCE/s | w={width_units:.2f}/{args.road_width:g}"
    cv2.putText(frame, label, (label_pt[0] + 8, label_pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA)


def draw_flows(frame, edges, sim_time, args):
    for branch in ROAD_BRANCHES:
        draw_flow_edge(frame, branch, "in", edges.get((branch, "center")), sim_time, args)
        draw_flow_edge(frame, branch, "out", edges.get(("center", branch)), sim_time, args)


def draw_queue_and_density(frame, states, args):
    for branch in ROAD_BRANCHES:
        pce = _float(states.get(branch, {}).get("queue_estimate_pce"), 0.0)
        if pce <= 0:
            continue
        p1, p2 = NODE_POS[branch], CENTER
        alpha = min(0.50, 0.08 + pce * 0.025)
        q_end = _point_on_segment(p1, p2, alpha)
        q_thick = max(6, min(int(args.road_width * args.cell_px), int(6 + pce * args.cell_px * 0.16)))
        cv2.line(frame, p1, q_end, QUEUE_COLOR, q_thick, cv2.LINE_AA)
        cv2.putText(frame, f"occ {pce:.1f}", _point_on_segment(p1, p2, 0.12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA)

    center_pce = _float(states.get("center", {}).get("pce_now"), 0.0)
    capacity = max(args.center_size * args.center_size, 1e-6)
    density = center_pce / capacity
    side = int(args.center_size * args.cell_px)
    x0 = CENTER[0] - side // 2
    y1 = CENTER[1] + side // 2
    cv2.putText(frame, f"center mass={center_pce:.2f} PCE density={density:.2f}", (x0, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT, 1, cv2.LINE_AA)


def analyze_warnings(edges, states, raw_data, sim_time, args, scenario="observed"):
    warnings = []

    # 1) Flow width overload. Width units >= road_width is the severe threshold.
    for (a, b), data in edges.items():
        w = data.get("width_units", 0.0)
        if w >= args.road_width:
            warnings.append({"type": "flow_width", "level": "critical", "edge": (a, b), "message": f"CRITICAL: {a}->{b} width {w:.2f} >= road width {args.road_width:g}"})
        elif w >= args.road_width * args.flow_warning_ratio:
            warnings.append({"type": "flow_width", "level": "warning", "edge": (a, b), "message": f"High flow: {a}->{b} width {w:.2f}/{args.road_width:g}"})

    # 2) Center density/collision risk.
    center_pce = _float(states.get("center", {}).get("pce_now"), 0.0)
    center_capacity = max(args.center_size * args.center_size, 1e-6)
    center_density = center_pce / center_capacity
    if center_density >= args.center_density_critical:
        warnings.append({"type": "center_density", "level": "critical", "edge": ("center", "center"), "message": f"COLLISION RISK: center density {center_density:.2f} >= {args.center_density_critical:g}"})
    elif center_density >= args.center_density_warning:
        warnings.append({"type": "center_density", "level": "warning", "edge": ("center", "center"), "message": f"Dense center: density {center_density:.2f}"})

    # 3) Adjacent inflows both high -> crossing/conflict risk.
    # This rule is only enabled in the hypothetical no-signal simulation.
    # In observed mode, adjacent inflows are still real data, but we do not
    # assume they must collide because a real traffic phase, priority rule, or
    # driver behavior may have already resolved the conflict.
    if scenario == "no_signal":
        inflow_units = {branch: edges.get((branch, "center"), {}).get("width_units", 0.0) for branch in ROAD_BRANCHES}
        for a, b in ADJACENT_INFLOW_PAIRS:
            if inflow_units[a] >= args.collision_flow_width and inflow_units[b] >= args.collision_flow_width:
                level = "critical" if center_density >= args.center_density_warning else "warning"
                warnings.append({"type": "adjacent_inflow", "level": level, "edge": (a, b), "message": f"No-signal conflict risk: adjacent inflows {a}+{b} high ({inflow_units[a]:.2f}, {inflow_units[b]:.2f})"})

    # 4) Long imbalance: inflow to center vs outflow from center over a longer window.
    if args.imbalance_seconds > 0:
        long_events = raw_data.window(max(0.0, sim_time - args.imbalance_seconds), sim_time)
        in_pce = sum(e["pce"] for e in long_events if e["to_region"] == "center" and e["from_region"] in ROAD_BRANCHES)
        out_pce = sum(e["pce"] for e in long_events if e["from_region"] == "center" and e["to_region"] in ROAD_BRANCHES)
        rate_diff = (in_pce - out_pce) / max(args.imbalance_seconds, 1e-6)
        if abs(rate_diff) >= args.imbalance_threshold_pceps:
            direction = "inflow > outflow" if rate_diff > 0 else "outflow > inflow"
            prefix = "Observed" if scenario == "observed" else "No-signal"
            warnings.append({"type": "long_imbalance", "level": "warning", "edge": ("center", "center"), "message": f"{prefix} long imbalance {direction}: {rate_diff:+.2f} PCE/s over {args.imbalance_seconds:g}s"})

    return warnings


def _edge_label_point(edge, args, index=0):
    """Place warning labels beside the lane/center instead of in a bottom panel."""
    a, b = edge if edge else ("center", "center")
    if a == "center" and b == "center":
        return (CENTER[0] + int(args.center_size * args.cell_px * 0.65), CENTER[1] - 18 + index * 18)

    if a in ROAD_BRANCHES and b == "center":
        p1, p2 = NODE_POS[a], CENTER
    elif a == "center" and b in ROAD_BRANCHES:
        p1, p2 = CENTER, NODE_POS[b]
    elif a in ROAD_BRANCHES and b in ROAD_BRANCHES:
        # Adjacent-inflow warning: put it near the center between the two arms.
        p1 = _point_on_segment(NODE_POS[a], CENTER, 0.78)
        p2 = _point_on_segment(NODE_POS[b], CENTER, 0.78)
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 + index * 18)
    else:
        return (820, 690 + index * 18)

    base = _point_on_segment(p1, p2, 0.70)
    # Put label outside the pipe, not on top of the flow.
    p_label, _ = _offset_points(base, p2, args.road_width * args.cell_px * 0.72)
    return (p_label[0], p_label[1] + index * 18)


def draw_warnings(frame, warnings, args):
    if not warnings:
        return

    per_edge_count = Counter()
    for w in warnings:
        edge = tuple(w.get("edge", ("center", "center")))
        idx = per_edge_count[edge]
        per_edge_count[edge] += 1
        x, y = _edge_label_point(edge, args, idx)
        color = COLLISION_COLOR if w.get("level") == "critical" else FLOW_WARNING
        short = w.get("message", "warning")
        if len(short) > 54:
            short = short[:51] + "..."
        cv2.putText(frame, short, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, color, 1, cv2.LINE_AA)


def setup_warning_log(args):
    if args.no_warning_log:
        args._warning_log_path = ""
        args._warning_log_seen = set()
        return
    path = args.warning_log.strip()
    if not path:
        path = os.path.join(args.run_dir, "fluid_replay_warnings.log")
    args._warning_log_path = path
    args._warning_log_seen = set()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fluid flow warning log\n")
        f.write("# overwritten on each run\n")
        f.write("# time_s\tscenario\tlevel\ttype\tedge\tmessage\n")


def log_warnings(args, scenario, sim_time, warnings):
    path = getattr(args, "_warning_log_path", "")
    if not path or not warnings:
        return
    interval = max(float(getattr(args, "log_interval", 1.0)), 1e-3)
    bucket = int(sim_time / interval)
    lines = []
    seen = getattr(args, "_warning_log_seen", set())
    for w in warnings:
        edge = w.get("edge", ("", ""))
        edge_text = "->".join(edge) if isinstance(edge, (list, tuple)) else str(edge)
        key = (scenario, bucket, w.get("level"), w.get("type"), edge_text, w.get("message"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{sim_time:.3f}\t{scenario}\t{w.get('level','')}\t{w.get('type','')}\t{edge_text}\t{w.get('message','')}\n")
    args._warning_log_seen = seen
    if lines:
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(lines)


def draw_hud(frame, sim_time, duration, args, source_path, paused, scenario="observed"):
    title = f"Fluid Flow Replay - {SCENARIO_DISPLAY.get(scenario, scenario).replace('_', ' ')}"
    cv2.putText(frame, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.84, TEXT, 2, cv2.LINE_AA)
    if scenario == "observed":
        scenario_note = "observed real-flow baseline"
    elif scenario == "no_signal":
        scenario_note = "hypothetical no-signal / always-flow conflict check"
    else:
        scenario_note = scenario
    cv2.putText(
        frame,
        f"t={sim_time:7.2f}/{duration:.2f}s | speed={args.speed:.2f}x | bin={args.bin_seconds:g}s smooth={args.smooth_seconds:g}s | {scenario_note}" + (" | PAUSED" if paused else ""),
        (24, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(frame, f"road width={args.road_width:g} cells | center={args.center_size:g}x{args.center_size:g} cells | data={os.path.basename(source_path)}", (24, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.46, MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Pipe width = PCE/s converted to road cells | layout={args.flow_layout}. >= road width is severe.", (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.44, MUTED, 1, cv2.LINE_AA)

    x, y = 812, 34
    legend = (("observed", FLOW_OK), ("inferred", FLOW_INFERRED), ("unknown", FLOW_UNKNOWN), ("warning", FLOW_WARNING), ("critical", FLOW_CRITICAL))
    for i, (name, color) in enumerate(legend):
        yy = y + i * 23
        cv2.line(frame, (x, yy), (x + 42, yy), color, 7, cv2.LINE_AA)
        cv2.putText(frame, name, (x + 54, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)


def draw_single_frame(raw_data, rows_by_region, source_path, sim_time, duration, args, paused=False, scenario="observed"):
    states = get_state_at_time(rows_by_region, sim_time, args.interpolate_state)
    edges = aggregate_events(raw_data, sim_time, args)
    warnings = analyze_warnings(edges, states, raw_data, sim_time, args, scenario=scenario)
    log_warnings(args, scenario, sim_time, warnings)

    frame = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    frame[:] = BACKGROUND
    draw_base(frame, args, warnings)
    draw_queue_and_density(frame, states, args)
    draw_flows(frame, edges, sim_time, args)
    draw_hud(frame, sim_time, duration, args, source_path, paused, scenario=scenario)
    draw_warnings(frame, warnings, args)
    return frame


def draw_frame(raw_data, rows_by_region, source_path, sim_time, duration, args, paused=False):
    if args.scenario == "compare":
        observed = draw_single_frame(raw_data, rows_by_region, source_path, sim_time, duration, args, paused, scenario="observed")
        no_signal = draw_single_frame(raw_data, rows_by_region, source_path, sim_time, duration, args, paused, scenario="no_signal")
        divider = np.full((CANVAS_SIZE[1], 6, 3), (10, 10, 12), dtype=np.uint8)
        return cv2.hconcat([observed, divider, no_signal])
    return draw_single_frame(raw_data, rows_by_region, source_path, sim_time, duration, args, paused, scenario=args.scenario)


def output_size(args):
    if args.scenario == "compare":
        return (CANVAS_SIZE[0] * 2 + 6, CANVAS_SIZE[1])
    return CANVAS_SIZE


def replay_window(args, raw_data, rows_by_region, source_path, duration):
    paused = False
    sim_time = 0.0
    last_wall = time.time()
    render_delay_ms = max(1, int(1000 / max(args.fps, 1.0)))

    while sim_time <= duration + 1e-6:
        now = time.time()
        elapsed = now - last_wall
        last_wall = now
        if not paused:
            sim_time += elapsed * max(args.speed, 0.01)

        frame = draw_frame(raw_data, rows_by_region, source_path, sim_time, duration, args, paused)
        cv2.imshow("Fluid Flow Replay", frame)
        key = cv2.waitKey(render_delay_ms) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("]"):
            args.speed = min(args.speed * 1.25, 16.0)
        if key == ord("["):
            args.speed = max(args.speed / 1.25, 0.05)
    cv2.destroyAllWindows()


def save_video(args, raw_data, rows_by_region, source_path, duration):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.save, fourcc, max(args.fps, 1.0), output_size(args))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {args.save}")

    render_fps = max(args.fps, 1.0)
    sim_step = max(args.speed, 0.01) / render_fps
    sim_time = 0.0
    while sim_time <= duration + 1e-6:
        frame = draw_frame(raw_data, rows_by_region, source_path, sim_time, duration, args, paused=False)
        writer.write(frame)
        sim_time += sim_step
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Fluid-like replay using real edge flow events.")
    parser.add_argument("run_dir", help="Directory containing flow_edges_real.csv")
    parser.add_argument("--save", default="", help="Optional output .mp4 path")
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="observed", help="observed = real-flow baseline, no_signal = hypothetical always-flow conflict check, compare = side-by-side")
    parser.add_argument("--fps", type=float, default=30.0, help="Render FPS for window/video")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed. 1.0 = real time, 0.25 = 4x slower")
    parser.add_argument("--bin-seconds", type=float, default=1.0, help="Runtime aggregation window in seconds")
    parser.add_argument("--smooth-seconds", type=float, default=0.0, help="Runtime smoothing window. 0 = use only bin-seconds")
    parser.add_argument("--road-width", type=float, default=3.0, help="Road width in car/lane cells. Default 3")
    parser.add_argument("--center-size", type=float, default=3.0, help="Center square side in car cells. Default 3 means 3x3")
    parser.add_argument("--cell-px", type=float, default=42.0, help="Pixels per road/center cell")
    parser.add_argument("--flow-layout", choices=("center", "right_left"), default="center", help="Flow drawing layout. center = one line in the middle of each road. right_left = entering center on right side, exiting on left side.")
    parser.add_argument("--lane-capacity-pceps", type=float, default=1.0, help="PCE/s represented by one width cell")
    parser.add_argument("--flow-warning-ratio", type=float, default=0.75, help="Warn when flow width exceeds this fraction of road width")
    parser.add_argument("--center-density-warning", type=float, default=0.65, help="Warn when center PCE/(center_size^2) exceeds this")
    parser.add_argument("--center-density-critical", type=float, default=0.90, help="Critical collision risk density threshold")
    parser.add_argument("--collision-flow-width", type=float, default=1.0, help="Adjacent inflow conflict threshold in width cells")
    parser.add_argument("--imbalance-seconds", type=float, default=20.0, help="Long-window imbalance check duration")
    parser.add_argument("--imbalance-threshold-pceps", type=float, default=0.8, help="Warn if |inflow-outflow| exceeds this PCE/s over imbalance window")
    parser.add_argument("--interpolate-state", action="store_true", help="Interpolate region_state_timeseries.csv at runtime")
    parser.add_argument("--warning-log", default="", help="Warning .log output path. Default: <run_dir>/fluid_replay_warnings.log")
    parser.add_argument("--log-interval", type=float, default=1.0, help="Minimum seconds between repeated warning log entries")
    parser.add_argument("--no-warning-log", action="store_true", help="Disable warning log output")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    setup_warning_log(args)

    raw_data, source_path = read_raw_edges(args.run_dir)
    _, rows_by_region = read_region_states(args.run_dir)
    if not raw_data.events:
        raise RuntimeError("No raw edge events found in flow_edges_real.csv/region_transitions.csv")

    duration = raw_data.duration
    if args.save:
        save_video(args, raw_data, rows_by_region, source_path, duration)
        print(f"Saved fluid replay to {args.save}")

    if not args.no_window:
        replay_window(args, raw_data, rows_by_region, source_path, duration)


if __name__ == "__main__":
    main()
