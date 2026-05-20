"""Fluid-flow export utilities.

This module exports data at two levels:

1. Validation/replay files
   - track_replay.csv
   - region_transitions.csv
   - region_state_timeseries.csv
   - flow_edges_timeseries.csv

2. RL-oriented files
   - od_routes.csv
   - rl_demand_timeseries.csv
   - rl_state_timeseries.csv

The RL files are still macroscopic. They are not an RL environment by
themselves; they provide demand/observation time series that can feed a future
traffic-signal RL environment.
"""

import csv
import json
import math
import os
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from .config import CLASS_NAMES, CLASS_WEIGHTS, FLUID_REGIONS, ROAD_BRANCHES

UNKNOWN_REGION = "unknown"


def _safe_stem(value):
    text = str(value or "source").strip()
    if not text:
        return "source"
    if text.isdigit():
        return f"camera_{text}"
    if "://" in text:
        text = text.split("://", 1)[1]
    stem = Path(text).stem or text.replace("/", "_").replace("\\", "_")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe[:80] or "source"


def _box_fields(box):
    if not box:
        return "", "", "", "", "", ""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return x1, y1, x2, y2, cx, cy


def _centroid_fields(centroid):
    if centroid is None:
        return "", ""
    return int(centroid[0]), int(centroid[1])


def _source_label(observed_count, inferred_count, unknown_count):
    used = []
    if observed_count:
        used.append("observed")
    if inferred_count:
        used.append("inferred")
    if unknown_count:
        used.append("unknown")
    if not used:
        return "none"
    if len(used) == 1:
        return used[0]
    return "mixed"


def _combine_source(a, b):
    if a == b:
        return a
    if "unknown" in (a, b):
        return "unknown"
    return "mixed"


class FluidFlowExporter:
    """Export raw tracking and aggregated macroscopic traffic-flow CSV files."""

    def __init__(
        self,
        *,
        root_dir,
        source_path,
        model_path,
        fps,
        valid_regions,
        template_path="",
        bin_seconds=1.0,
        smooth_seconds=5.0,
        track_sample_seconds=0.25,
        region_state_sample_seconds=1.0,
        hidden_left_enabled=True,
    ):
        self.root_dir = root_dir or "flow_exports"
        self.source_path = str(source_path)
        self.model_path = str(model_path)
        self.fps = float(fps or 0.0)
        self.valid_regions = [r for r in FLUID_REGIONS if r in set(valid_regions)]
        self.template_path = str(template_path or "")
        self.bin_seconds = max(float(bin_seconds or 1.0), 0.001)
        self.smooth_seconds = max(float(smooth_seconds or 0.0), 0.0)
        self.track_sample_seconds = max(float(track_sample_seconds or 0.25), 0.0)
        self.region_state_sample_seconds = max(float(region_state_sample_seconds or 1.0), 0.001)
        self.hidden_left_enabled = bool(hidden_left_enabled)

        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_stem = _safe_stem(source_path)
        self.output_dir = os.path.abspath(os.path.join(self.root_dir, f"{source_stem}_{run_time}"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.track_replay_path = os.path.join(self.output_dir, "track_replay.csv")
        self.transition_path = os.path.join(self.output_dir, "region_transitions.csv")
        self.region_state_path = os.path.join(self.output_dir, "region_state_timeseries.csv")
        # Raw, event-level edge flow. This is the preferred file for replay/RL data
        # because it preserves real timestamps and does not apply binning, smoothing,
        # or interpolation. The simulator decides how to aggregate it at runtime.
        self.edge_real_path = os.path.join(self.output_dir, "flow_edges_real.csv")
        # Backward-compatible aggregated file. It is still written for quick CSV
        # inspection, but the replay script defaults to flow_edges_real.csv.
        self.edge_timeseries_path = os.path.join(self.output_dir, "flow_edges_timeseries.csv")
        self.od_routes_path = os.path.join(self.output_dir, "od_routes.csv")
        self.rl_demand_path = os.path.join(self.output_dir, "rl_demand_timeseries.csv")
        self.rl_state_path = os.path.join(self.output_dir, "rl_state_timeseries.csv")
        self.metadata_path = os.path.join(self.output_dir, "metadata.json")

        self._transition_seq = 0
        self._route_seq = 0
        self._last_track_sample_time = {}
        self._last_region_state_time = None
        self._transition_events = []
        self._route_events = []
        self._pending_origin_by_track = {}
        self._state_times = []
        self._state_snapshots = {}
        self._closed = False

        self._track_file = open(self.track_replay_path, "w", newline="", encoding="utf-8")
        self._track_writer = csv.DictWriter(self._track_file, fieldnames=self._track_header())
        self._track_writer.writeheader()

        self._transition_file = open(self.transition_path, "w", newline="", encoding="utf-8")
        self._transition_writer = csv.DictWriter(self._transition_file, fieldnames=self._transition_header())
        self._transition_writer.writeheader()

        self._state_file = open(self.region_state_path, "w", newline="", encoding="utf-8")
        self._state_writer = csv.DictWriter(self._state_file, fieldnames=self._state_header())
        self._state_writer.writeheader()

        self._edge_real_file = open(self.edge_real_path, "w", newline="", encoding="utf-8")
        self._edge_real_writer = csv.DictWriter(self._edge_real_file, fieldnames=self._edge_real_header())
        self._edge_real_writer.writeheader()

        self._route_file = open(self.od_routes_path, "w", newline="", encoding="utf-8")
        self._route_writer = csv.DictWriter(self._route_file, fieldnames=self._route_header())
        self._route_writer.writeheader()

        self._write_metadata(extra_files_written=False)

    @property
    def output_files(self):
        return {
            "output_dir": self.output_dir,
            "track_replay": self.track_replay_path,
            "region_transitions": self.transition_path,
            "region_state_timeseries": self.region_state_path,
            "flow_edges_real": self.edge_real_path,
            "flow_edges_timeseries": self.edge_timeseries_path,
            "od_routes": self.od_routes_path,
            "rl_demand_timeseries": self.rl_demand_path,
            "rl_state_timeseries": self.rl_state_path,
            "metadata": self.metadata_path,
        }

    def _track_header(self):
        return [
            "time_s", "frame", "track_id", "class_id", "class_name", "pce",
            "raw_region", "stable_region", "active_branch",
            "x1", "y1", "x2", "y2", "cx", "cy", "source",
        ]

    def _transition_header(self):
        return [
            "transition_id", "time_s", "frame", "track_id",
            "from_region", "to_region", "edge",
            "class_id", "class_name", "pce", "source", "confidence", "reason",
            "x1", "y1", "x2", "y2", "cx", "cy",
        ]

    def _state_header(self):
        return [
            "time_s", "dt_s", "frame", "region",
            "vehicle_count_now", "pce_now", "queue_estimate_pce", "source",
        ]

    def _edge_real_header(self):
        return [
            "event_id", "time_s", "frame", "track_id",
            "from_region", "to_region", "edge",
            "vehicle_count", "pce",
            "class_id", "class_name",
            "source", "confidence", "reason",
            "x1", "y1", "x2", "y2", "cx", "cy",
        ]

    def _edge_header(self):
        class_ids = sorted(CLASS_NAMES.keys())
        header = [
            "time_s", "dt_s", "from_region", "to_region", "edge",
            "vehicle_count", "pce_sum",
            "flow_pce_per_s", "flow_pce_per_min",
            "smooth_pce_per_s", "smooth_pce_per_min",
            "observed_count", "inferred_count", "unknown_count",
            "source", "mean_confidence",
        ]
        for cls_id in class_ids:
            name = CLASS_NAMES[cls_id]
            header.append(f"{name}_count")
            header.append(f"{name}_pce")
        return header

    def _route_header(self):
        return [
            "route_id", "time_s", "entry_time_s", "travel_time_s", "frame", "track_id",
            "origin", "destination", "od", "class_id", "class_name", "pce",
            "source", "confidence", "entry_source", "exit_source", "reason",
        ]

    def _rl_demand_header(self):
        return [
            "time_s", "dt_s", "origin", "destination", "od",
            "vehicle_count", "pce_sum", "demand_pce_per_s", "demand_pce_per_min",
            "observed_count", "inferred_count", "unknown_count", "source", "mean_confidence",
        ]

    def _rl_state_header(self):
        header = ["time_s", "dt_s", "action_phase", "reward_proxy", "total_queue_pce", "throughput_pce"]
        for branch in ROAD_BRANCHES:
            header.extend([
                f"queue_{branch}_pce",
                f"queue_{branch}_veh",
                f"demand_{branch}_pce",
                f"served_{branch}_pce",
                f"exit_to_{branch}_pce",
            ])
        header.extend([
            "queue_center_pce",
            "queue_center_veh",
            "ns_queue_pce",
            "ew_queue_pce",
            "ns_demand_pce",
            "ew_demand_pce",
        ])
        return header

    def _write_metadata(self, *, extra_files_written):
        files = {
            "track_replay": os.path.basename(self.track_replay_path),
            "region_transitions": os.path.basename(self.transition_path),
            "region_state_timeseries": os.path.basename(self.region_state_path),
            "flow_edges_real": os.path.basename(self.edge_real_path),
            "od_routes": os.path.basename(self.od_routes_path),
            "metadata": os.path.basename(self.metadata_path),
        }
        if extra_files_written:
            files.update({
                "flow_edges_timeseries": os.path.basename(self.edge_timeseries_path),
                "rl_demand_timeseries": os.path.basename(self.rl_demand_path),
                "rl_state_timeseries": os.path.basename(self.rl_state_path),
            })

        metadata = {
            "source_path": self.source_path,
            "model_path": self.model_path,
            "template_path": self.template_path,
            "fps_input": self.fps,
            "valid_regions": self.valid_regions,
            "road_branches": ROAD_BRANCHES,
            "center_region": "center",
            "bin_seconds": self.bin_seconds,
            "smooth_seconds": self.smooth_seconds,
            "track_sample_seconds": self.track_sample_seconds,
            "region_state_sample_seconds": self.region_state_sample_seconds,
            "hidden_left_inference_enabled": self.hidden_left_enabled,
            "class_names": CLASS_NAMES,
            "class_weights_pce": CLASS_WEIGHTS,
            "files": files,
            "concept": {
                "pce": "Passenger-car-equivalent / lane-occupation weight.",
                "flow_edges_real": "Raw edge events with real timestamps. Preferred for replay/RL; no binning, smoothing, or interpolation is applied during export.",
                "flow_edges_timeseries": "Backward-compatible binned pipe flow between regions. Quick to inspect, but less faithful than flow_edges_real.csv.",
                "region_state_timeseries": "Observed mass inside each region. Use as occupancy/queue proxy.",
                "od_routes": "Origin-destination route events inferred from branch->center then center->branch transitions.",
                "rl_demand_timeseries": "Aggregated OD demand by time bin. Use this as input demand for an RL traffic environment.",
                "rl_state_timeseries": "State-like observation table with queues/demand/served/throughput. action_phase is -1 because the source video has no controller action label.",
            },
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def should_sample_track(self, track_id, time_s):
        if self.track_sample_seconds <= 0:
            return True
        last = self._last_track_sample_time.get(track_id)
        if last is None or float(time_s) - last >= self.track_sample_seconds:
            self._last_track_sample_time[track_id] = float(time_s)
            return True
        return False

    def log_track_sample(self, *, time_s, frame_id, track_id, cls, raw_region, stable_region, active_branch, box=None, source="observed"):
        if not self.should_sample_track(track_id, time_s):
            return
        cls = int(cls)
        x1, y1, x2, y2, cx, cy = _box_fields(box)
        self._track_writer.writerow({
            "time_s": f"{float(time_s):.3f}",
            "frame": int(frame_id),
            "track_id": track_id,
            "class_id": cls,
            "class_name": CLASS_NAMES.get(cls, str(cls)),
            "pce": f"{CLASS_WEIGHTS.get(cls, 1.0):.3f}",
            "raw_region": raw_region or "",
            "stable_region": stable_region or "",
            "active_branch": active_branch or "",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy,
            "source": source,
        })

    def _register_route_from_transition(self, event, *, reason):
        track_id = event["track_id"]
        from_region = event["from_region"]
        to_region = event["to_region"]

        # Entry into the junction. Store the current origin for this track.
        if to_region == "center" and from_region in ROAD_BRANCHES:
            self._pending_origin_by_track[track_id] = {
                "origin": from_region,
                "entry_time_s": event["time_s"],
                "frame": event["frame"],
                "class_id": event["class_id"],
                "pce": event["pce"],
                "source": event["source"],
                "confidence": event["confidence"],
            }
            return

        # Exit from the junction. Pair with previous origin if available.
        if from_region == "center" and to_region in ROAD_BRANCHES:
            pending = self._pending_origin_by_track.pop(track_id, None)
            if pending is None:
                self._write_route_event(
                    time_s=event["time_s"],
                    entry_time_s=event["time_s"],
                    frame=event["frame"],
                    track_id=track_id,
                    origin=UNKNOWN_REGION,
                    destination=to_region,
                    cls=event["class_id"],
                    pce=event["pce"],
                    source="unknown",
                    confidence=event["confidence"],
                    entry_source="unknown",
                    exit_source=event["source"],
                    reason="exit_without_observed_entry",
                )
                return

            origin = pending["origin"]
            destination = to_region
            if origin == destination:
                # U-turn/noise can still be useful for debugging but is less useful
                # for traffic-light RL demand. Keep it in od_routes.
                route_reason = "u_turn_or_region_noise"
            else:
                route_reason = reason
            confidence = min(float(pending["confidence"]), float(event["confidence"]))
            source = _combine_source(pending["source"], event["source"])
            self._write_route_event(
                time_s=event["time_s"],
                entry_time_s=pending["entry_time_s"],
                frame=event["frame"],
                track_id=track_id,
                origin=origin,
                destination=destination,
                cls=pending["class_id"],
                pce=pending["pce"],
                source=source,
                confidence=confidence,
                entry_source=pending["source"],
                exit_source=event["source"],
                reason=route_reason,
            )
            return

        # Direct branch->branch transition can happen if center is missing/too small.
        if from_region in ROAD_BRANCHES and to_region in ROAD_BRANCHES and from_region != to_region:
            self._write_route_event(
                time_s=event["time_s"],
                entry_time_s=event["time_s"],
                frame=event["frame"],
                track_id=track_id,
                origin=from_region,
                destination=to_region,
                cls=event["class_id"],
                pce=event["pce"],
                source=event["source"],
                confidence=event["confidence"],
                entry_source=event["source"],
                exit_source=event["source"],
                reason="direct_branch_to_branch_transition",
            )

    def _write_route_event(self, *, time_s, entry_time_s, frame, track_id, origin, destination, cls, pce, source, confidence, entry_source, exit_source, reason):
        self._route_seq += 1
        travel_time = max(0.0, float(time_s) - float(entry_time_s))
        cls = int(cls)
        row = {
            "route_id": self._route_seq,
            "time_s": f"{float(time_s):.3f}",
            "entry_time_s": f"{float(entry_time_s):.3f}",
            "travel_time_s": f"{travel_time:.3f}",
            "frame": int(frame),
            "track_id": track_id,
            "origin": origin,
            "destination": destination,
            "od": f"{origin}->{destination}",
            "class_id": cls,
            "class_name": CLASS_NAMES.get(cls, str(cls)),
            "pce": f"{float(pce):.3f}",
            "source": source,
            "confidence": f"{float(confidence):.3f}",
            "entry_source": entry_source,
            "exit_source": exit_source,
            "reason": reason,
        }
        self._route_writer.writerow(row)
        self._route_events.append({
            "time_s": float(time_s),
            "entry_time_s": float(entry_time_s),
            "travel_time_s": travel_time,
            "frame": int(frame),
            "track_id": track_id,
            "origin": origin,
            "destination": destination,
            "class_id": cls,
            "pce": float(pce),
            "source": source,
            "confidence": float(confidence),
        })

    def log_transition(self, *, time_s, frame_id, track_id, from_region, to_region, cls, box=None, centroid=None, source="observed", confidence=1.0, reason="region_change"):
        if not from_region or not to_region or from_region == to_region:
            return False
        if from_region not in FLUID_REGIONS or to_region not in FLUID_REGIONS:
            return False

        cls = int(cls)
        pce = float(CLASS_WEIGHTS.get(cls, 1.0))
        self._transition_seq += 1
        x1, y1, x2, y2, box_cx, box_cy = _box_fields(box)
        cx, cy = _centroid_fields(centroid)
        if cx == "" and box_cx != "":
            cx, cy = box_cx, box_cy

        row = {
            "transition_id": self._transition_seq,
            "time_s": f"{float(time_s):.3f}",
            "frame": int(frame_id),
            "track_id": track_id,
            "from_region": from_region,
            "to_region": to_region,
            "edge": f"{from_region}->{to_region}",
            "class_id": cls,
            "class_name": CLASS_NAMES.get(cls, str(cls)),
            "pce": f"{pce:.3f}",
            "source": source,
            "confidence": f"{float(confidence):.3f}",
            "reason": reason,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy,
        }
        self._transition_writer.writerow(row)

        # Write the raw flow edge immediately. This file is intentionally not
        # resampled. Runtime replay/RL scripts choose bin width, smoothing, and
        # interpolation policies through CLI parameters.
        self._edge_real_writer.writerow({
            "event_id": self._transition_seq,
            "time_s": f"{float(time_s):.3f}",
            "frame": int(frame_id),
            "track_id": track_id,
            "from_region": from_region,
            "to_region": to_region,
            "edge": f"{from_region}->{to_region}",
            "vehicle_count": 1,
            "pce": f"{pce:.3f}",
            "class_id": cls,
            "class_name": CLASS_NAMES.get(cls, str(cls)),
            "source": source,
            "confidence": f"{float(confidence):.3f}",
            "reason": reason,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy,
        })
        event = {
            "time_s": float(time_s),
            "frame": int(frame_id),
            "track_id": track_id,
            "from_region": from_region,
            "to_region": to_region,
            "edge": f"{from_region}->{to_region}",
            "class_id": cls,
            "pce": pce,
            "source": source,
            "confidence": float(confidence),
        }
        self._transition_events.append(event)
        self._register_route_from_transition(event, reason=reason)
        return True

    def should_write_region_state(self, time_s):
        if self._last_region_state_time is None:
            return True
        return float(time_s) - self._last_region_state_time >= self.region_state_sample_seconds

    def write_region_state_snapshot(self, *, time_s, frame_id, region_current_count, region_current_pce, force=False):
        if not force and not self.should_write_region_state(time_s):
            return False

        time_s = float(time_s)
        dt_s = 0.0 if self._last_region_state_time is None else max(0.0, time_s - self._last_region_state_time)
        self._last_region_state_time = time_s
        self._state_times.append(time_s)
        snapshot = {}

        for region in FLUID_REGIONS:
            count_now = int(region_current_count.get(region, 0))
            pce_now = float(region_current_pce.get(region, 0.0))
            source = "observed" if region in self.valid_regions else "unavailable"
            if region == "left" and count_now == 0 and self.hidden_left_enabled:
                source = "hidden_or_unobserved"
            self._state_writer.writerow({
                "time_s": f"{time_s:.3f}",
                "dt_s": f"{dt_s:.3f}",
                "frame": int(frame_id),
                "region": region,
                "vehicle_count_now": count_now,
                "pce_now": f"{pce_now:.3f}",
                "queue_estimate_pce": f"{pce_now:.3f}",
                "source": source,
            })
            snapshot[region] = {
                "vehicle_count_now": count_now,
                "pce_now": pce_now,
                "queue_estimate_pce": pce_now,
                "source": source,
            }
        self._state_snapshots[round(time_s, 3)] = snapshot
        return True

    def _default_edges(self):
        edges = []
        for branch in ROAD_BRANCHES:
            edges.append((branch, "center"))
            edges.append(("center", branch))
        return edges

    def _time_range(self):
        max_time = 0.0
        if self._transition_events:
            max_time = max(max_time, max(e["time_s"] for e in self._transition_events))
        if self._route_events:
            max_time = max(max_time, max(e["time_s"] for e in self._route_events))
        if self._state_times:
            max_time = max(max_time, max(self._state_times))
        return max_time

    def write_edge_timeseries(self):
        class_ids = sorted(CLASS_NAMES.keys())
        bin_s = self.bin_seconds
        max_time = self._time_range()
        n_bins = int(math.floor(max_time / bin_s)) + 1

        all_edges = set(self._default_edges())
        for event in self._transition_events:
            all_edges.add((event["from_region"], event["to_region"]))
        ordered_edges = self._default_edges() + sorted(edge for edge in all_edges if edge not in set(self._default_edges()))

        aggregate = defaultdict(lambda: {
            "vehicle_count": 0,
            "pce_sum": 0.0,
            "observed_count": 0,
            "inferred_count": 0,
            "unknown_count": 0,
            "confidence_sum": 0.0,
            "class_count": defaultdict(int),
            "class_pce": defaultdict(float),
        })

        for event in self._transition_events:
            bin_idx = int(math.floor(event["time_s"] / bin_s))
            key = (bin_idx, event["from_region"], event["to_region"])
            item = aggregate[key]
            item["vehicle_count"] += 1
            item["pce_sum"] += event["pce"]
            source = event.get("source", "unknown")
            if source == "observed":
                item["observed_count"] += 1
            elif source == "inferred":
                item["inferred_count"] += 1
            else:
                item["unknown_count"] += 1
            item["confidence_sum"] += float(event.get("confidence", 0.0))
            cls_id = int(event["class_id"])
            item["class_count"][cls_id] += 1
            item["class_pce"][cls_id] += event["pce"]

        rolling = {edge: deque() for edge in ordered_edges}
        rolling_pce = {edge: 0.0 for edge in ordered_edges}
        smooth_window = max(self.smooth_seconds, bin_s)

        with open(self.edge_timeseries_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._edge_header())
            writer.writeheader()

            for bin_idx in range(n_bins):
                time_s = bin_idx * bin_s
                for from_region, to_region in ordered_edges:
                    edge = (from_region, to_region)
                    item = aggregate[(bin_idx, from_region, to_region)]
                    pce_sum = float(item["pce_sum"])
                    vehicle_count = int(item["vehicle_count"])

                    rolling[edge].append((time_s, pce_sum))
                    rolling_pce[edge] += pce_sum
                    while rolling[edge] and time_s - rolling[edge][0][0] >= smooth_window:
                        _, old_pce = rolling[edge].popleft()
                        rolling_pce[edge] -= old_pce

                    confidence_count = item["observed_count"] + item["inferred_count"] + item["unknown_count"]
                    mean_conf = item["confidence_sum"] / confidence_count if confidence_count else 0.0
                    source = _source_label(item["observed_count"], item["inferred_count"], item["unknown_count"])

                    row = {
                        "time_s": f"{time_s:.3f}",
                        "dt_s": f"{bin_s:.3f}",
                        "from_region": from_region,
                        "to_region": to_region,
                        "edge": f"{from_region}->{to_region}",
                        "vehicle_count": vehicle_count,
                        "pce_sum": f"{pce_sum:.3f}",
                        "flow_pce_per_s": f"{pce_sum / bin_s:.3f}",
                        "flow_pce_per_min": f"{pce_sum * 60.0 / bin_s:.3f}",
                        "smooth_pce_per_s": f"{rolling_pce[edge] / smooth_window:.3f}",
                        "smooth_pce_per_min": f"{rolling_pce[edge] * 60.0 / smooth_window:.3f}",
                        "observed_count": int(item["observed_count"]),
                        "inferred_count": int(item["inferred_count"]),
                        "unknown_count": int(item["unknown_count"]),
                        "source": source,
                        "mean_confidence": f"{mean_conf:.3f}",
                    }
                    for cls_id in class_ids:
                        name = CLASS_NAMES[cls_id]
                        row[f"{name}_count"] = int(item["class_count"][cls_id])
                        row[f"{name}_pce"] = f"{item['class_pce'][cls_id]:.3f}"
                    writer.writerow(row)

    def _aggregate_edge_bins(self):
        bin_s = self.bin_seconds
        aggregate = defaultdict(float)
        for event in self._transition_events:
            bin_idx = int(math.floor(event["time_s"] / bin_s))
            aggregate[(bin_idx, event["from_region"], event["to_region"])] += event["pce"]
        return aggregate

    def _nearest_state_snapshot(self, time_s):
        if not self._state_snapshots:
            return {}
        keys = sorted(self._state_snapshots.keys())
        # previous snapshot is causal and best for RL observations
        best = keys[0]
        for key in keys:
            if key <= time_s:
                best = key
            else:
                break
        return self._state_snapshots.get(best, {})

    def write_rl_demand_timeseries(self):
        bin_s = self.bin_seconds
        max_time = self._time_range()
        n_bins = int(math.floor(max_time / bin_s)) + 1

        ods = [(o, d) for o in ROAD_BRANCHES for d in ROAD_BRANCHES if o != d]
        # Include unknown origins/destinations for auditing, not for default RL env.
        for event in self._route_events:
            pair = (event["origin"], event["destination"])
            if pair not in ods:
                ods.append(pair)

        aggregate = defaultdict(lambda: {
            "vehicle_count": 0,
            "pce_sum": 0.0,
            "observed_count": 0,
            "inferred_count": 0,
            "unknown_count": 0,
            "confidence_sum": 0.0,
        })

        for event in self._route_events:
            bin_idx = int(math.floor(event["entry_time_s"] / bin_s))
            key = (bin_idx, event["origin"], event["destination"])
            item = aggregate[key]
            item["vehicle_count"] += 1
            item["pce_sum"] += event["pce"]
            source = event.get("source", "unknown")
            if source == "observed":
                item["observed_count"] += 1
            elif source in ("inferred", "mixed"):
                item["inferred_count"] += 1
            else:
                item["unknown_count"] += 1
            item["confidence_sum"] += float(event.get("confidence", 0.0))

        with open(self.rl_demand_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._rl_demand_header())
            writer.writeheader()
            for bin_idx in range(n_bins):
                time_s = bin_idx * bin_s
                for origin, destination in ods:
                    item = aggregate[(bin_idx, origin, destination)]
                    total_count = item["observed_count"] + item["inferred_count"] + item["unknown_count"]
                    mean_conf = item["confidence_sum"] / total_count if total_count else 0.0
                    source = _source_label(item["observed_count"], item["inferred_count"], item["unknown_count"])
                    pce_sum = item["pce_sum"]
                    writer.writerow({
                        "time_s": f"{time_s:.3f}",
                        "dt_s": f"{bin_s:.3f}",
                        "origin": origin,
                        "destination": destination,
                        "od": f"{origin}->{destination}",
                        "vehicle_count": int(item["vehicle_count"]),
                        "pce_sum": f"{pce_sum:.3f}",
                        "demand_pce_per_s": f"{pce_sum / bin_s:.3f}",
                        "demand_pce_per_min": f"{pce_sum * 60.0 / bin_s:.3f}",
                        "observed_count": int(item["observed_count"]),
                        "inferred_count": int(item["inferred_count"]),
                        "unknown_count": int(item["unknown_count"]),
                        "source": source,
                        "mean_confidence": f"{mean_conf:.3f}",
                    })

    def write_rl_state_timeseries(self):
        bin_s = self.bin_seconds
        max_time = self._time_range()
        n_bins = int(math.floor(max_time / bin_s)) + 1
        edge_bins = self._aggregate_edge_bins()

        with open(self.rl_state_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._rl_state_header())
            writer.writeheader()
            for bin_idx in range(n_bins):
                time_s = bin_idx * bin_s
                snapshot = self._nearest_state_snapshot(time_s)
                row = {
                    "time_s": f"{time_s:.3f}",
                    "dt_s": f"{bin_s:.3f}",
                    # -1 means this row is only observation/demand from video;
                    # there was no RL action taken in the source recording.
                    "action_phase": -1,
                    "reward_proxy": "0.000",
                    "total_queue_pce": "0.000",
                    "throughput_pce": "0.000",
                }

                total_queue = 0.0
                throughput = 0.0
                ns_demand = 0.0
                ew_demand = 0.0

                for branch in ROAD_BRANCHES:
                    state = snapshot.get(branch, {})
                    q_pce = float(state.get("queue_estimate_pce", 0.0))
                    q_veh = float(state.get("vehicle_count_now", 0.0))
                    demand = edge_bins[(bin_idx, branch, "center")]
                    served = edge_bins[(bin_idx, branch, "center")]
                    exit_to = edge_bins[(bin_idx, "center", branch)]
                    row[f"queue_{branch}_pce"] = f"{q_pce:.3f}"
                    row[f"queue_{branch}_veh"] = f"{q_veh:.3f}"
                    row[f"demand_{branch}_pce"] = f"{demand:.3f}"
                    row[f"served_{branch}_pce"] = f"{served:.3f}"
                    row[f"exit_to_{branch}_pce"] = f"{exit_to:.3f}"
                    total_queue += q_pce
                    throughput += served
                    if branch in ("top", "bottom"):
                        ns_demand += demand
                    elif branch in ("left", "right"):
                        ew_demand += demand

                center_state = snapshot.get("center", {})
                center_q = float(center_state.get("queue_estimate_pce", 0.0))
                center_veh = float(center_state.get("vehicle_count_now", 0.0))
                row["queue_center_pce"] = f"{center_q:.3f}"
                row["queue_center_veh"] = f"{center_veh:.3f}"
                row["ns_queue_pce"] = f"{float(row['queue_top_pce']) + float(row['queue_bottom_pce']):.3f}"
                row["ew_queue_pce"] = f"{float(row['queue_left_pce']) + float(row['queue_right_pce']):.3f}"
                row["ns_demand_pce"] = f"{ns_demand:.3f}"
                row["ew_demand_pce"] = f"{ew_demand:.3f}"
                row["total_queue_pce"] = f"{total_queue + center_q:.3f}"
                row["throughput_pce"] = f"{throughput:.3f}"
                # Reward proxy is only a diagnostic. In true RL, reward should
                # be computed after the environment applies an action.
                row["reward_proxy"] = f"{throughput - 0.05 * (total_queue + center_q):.3f}"
                writer.writerow(row)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for f in (self._track_file, self._transition_file, self._edge_real_file, self._state_file, self._route_file):
            if f is not None and not f.closed:
                f.flush()
                f.close()
        self.write_edge_timeseries()
        self.write_rl_demand_timeseries()
        self.write_rl_state_timeseries()
        self._write_metadata(extra_files_written=True)
