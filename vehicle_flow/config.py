"""Shared constants for the vehicle flow application."""

# Class mapping for the current Roboflow traffic_count_vn data.yaml:
#   0=bicycle, 1=bus, 2=car, 3=motorbike, 4=person.
#
# ``person`` is intentionally kept in detection/tracking/display so the UI can
# show it for auditing, but it is excluded from COUNTED_CLASS_IDS and has no PCE
# weight. Therefore people are detected and drawn, but never contribute to
# PCE/count/flow/route/RL export.
CLASS_NAMES = {
    0: "bicycle",
    1: "bus",
    2: "car",
    3: "motorbike",
    4: "person",
}

CLASS_WEIGHTS = {
    0: 0.2,   # bicycle
    1: 2.0,   # bus
    2: 1.0,   # car
    3: 0.3,   # motorbike
    # 4=person is deliberately excluded from flow counting.
}

# Classes sent to YOLO/DeepSORT. Keep person here so it is still detected and
# visible on the video overlay.
DETECT_CLASS_IDS = tuple(CLASS_NAMES.keys())

# Classes that contribute to all counting/flow/PCE/RL files.
COUNTED_CLASS_IDS = tuple(CLASS_WEIGHTS.keys())
IGNORED_CLASS_IDS = tuple(sorted(set(DETECT_CLASS_IDS) - set(COUNTED_CLASS_IDS)))
EXPECTED_MODEL_NAMES = CLASS_NAMES.copy()

# Region/lane layout. Each road approach is split into two lane regions:
#   t/l/r/b + 1 = inbound lane, moving toward center
#   t/l/r/b + 2 = outbound lane, moving away from center
# The polygons of lane 1 and lane 2 may overlap slightly to cover lane-changing.
INBOUND_LANE_REGIONS = ("t1", "l1", "r1", "b1")
OUTBOUND_LANE_REGIONS = ("t2", "l2", "r2", "b2")
LANE_REGION_ORDER = ("t1", "t2", "l1", "l2", "r1", "r2", "b1", "b2")
BRANCH_ORDER = LANE_REGION_ORDER + ("center",)
VALID_BRANCHES = set(BRANCH_ORDER)
DIRECTIONS = ("in", "out")
DISPLAY_CLASS_IDS = (2, 3)  # car, motorbike

REGION_DISPLAY_NAMES = {
    "t1": "T1 In",
    "t2": "T2 Out",
    "l1": "L1 In",
    "l2": "L2 Out",
    "r1": "R1 In",
    "r2": "R2 Out",
    "b1": "B1 In",
    "b2": "B2 Out",
    "center": "Center",
}

REGION_SHORT_LABELS = {
    "t1": "T1",
    "t2": "T2",
    "l1": "L1",
    "l2": "L2",
    "r1": "R1",
    "r2": "R2",
    "b1": "B1",
    "b2": "B2",
    "center": "CENTER",
}

REGION_TO_APPROACH = {
    "t1": "top", "t2": "top",
    "l1": "left", "l2": "left",
    "r1": "right", "r2": "right",
    "b1": "bottom", "b2": "bottom",
    "center": "center",
}
APPROACH_TO_INBOUND_REGION = {"top": "t1", "left": "l1", "right": "r1", "bottom": "b1"}
APPROACH_TO_OUTBOUND_REGION = {"top": "t2", "left": "l2", "right": "r2", "bottom": "b2"}
INBOUND_REGION_BY_OUTBOUND_REGION = {"t2": "t1", "l2": "l1", "r2": "r1", "b2": "b1"}
OUTBOUND_REGION_BY_INBOUND_REGION = {"t1": "t2", "l1": "l2", "r1": "r2", "b1": "b2"}
REGION_DIRECTION = {region: "inbound" for region in INBOUND_LANE_REGIONS}
REGION_DIRECTION.update({region: "outbound" for region in OUTBOUND_LANE_REGIONS})
REGION_DIRECTION["center"] = "center"

REGION_NAME_MAP = {
    # Direct lane names
    "t1": "t1", "t2": "t2",
    "l1": "l1", "l2": "l2",
    "r1": "r1", "r2": "r2",
    "b1": "b1", "b2": "b2",

    # Friendly aliases for template.csv
    "top1": "t1", "top_in": "t1", "in_top": "t1", "top_inbound": "t1",
    "top2": "t2", "top_out": "t2", "out_top": "t2", "top_outbound": "t2",
    "left1": "l1", "left_in": "l1", "in_left": "l1", "left_inbound": "l1",
    "left2": "l2", "left_out": "l2", "out_left": "l2", "left_outbound": "l2",
    "right1": "r1", "right_in": "r1", "in_right": "r1", "right_inbound": "r1",
    "right2": "r2", "right_out": "r2", "out_right": "r2", "right_outbound": "r2",
    "bottom1": "b1", "bottom_in": "b1", "in_bottom": "b1", "bottom_inbound": "b1",
    "bottom2": "b2", "bottom_out": "b2", "out_bottom": "b2", "bottom_outbound": "b2",

    # Current 8-lane templates should use only t1/t2/l1/l2/r1/r2/b1/b2/center
    # or the explicit lane aliases above. Old one-region names such as
    # top/left/right/bottom are intentionally not accepted anymore, because
    # they can make an outdated 4-region template look valid in the 8-lane app.
    "center": "center",
    "none": None,
}

# Default template path shown in the UI. Old 4-region templates are still
# rejected by RegionTemplate, so keeping this default is safe for the 8-lane app.
DEFAULT_TEMPLATE_MAPPING = "template.csv"
DEFAULT_MODEL_PATH = "models/tuning_200.pt"
DEFAULT_AVAILABLE_MODELS = [
    "models/tuning_200.pt",
    "models/tuning_50.pt",
    # "models/tuning.pt",
    # "models/yolov8n.pt",
    # "models/yolov8s.pt",
    # "models/yolov8m.pt",
    # "models/yolov8l.pt",
    # "models/yolov8x.pt",
]

# Region-counting stability parameters.
# Increase STABLE_REGION_FRAMES if vehicles jitter around polygon edges.
# Decrease it if vehicles pass through small regions too fast.
STABLE_REGION_FRAMES = 5
REGION_HISTORY_LEN = 12
EVENT_COOLDOWN_FRAMES = 12

# Keep a branch active for a short occlusion period before emitting OUT.
# This prevents one lost detection from immediately changing count/branch state.
LOST_OUT_FRAMES = 120

# If a track disappears near the frame border, it probably left the camera view.
# Emit OUT faster in that case, while still keeping a long timeout for occlusions
# in the middle of the frame.
EDGE_LOST_OUT_FRAMES = 24
FRAME_EXIT_MARGIN_RATIO = 0.025

# Inference tuning.
# The global YOLO confidence is intentionally low, then each class is filtered
# by CLASS_CONF_THRESHOLDS below. This is more flexible than one global conf.
MODEL_IMGSZ = 1280
MODEL_CONF = 0.10
MODEL_IOU = 0.50

# Class-specific confidence thresholds.
# Raise car threshold when motorcycles/bicycles are often predicted as car.
# Lower motorcycle/bicycle thresholds if small vehicles are missed too often.
CLASS_CONF_THRESHOLDS = {
    0: 0.2,  # bicycle
    1: 0.65,  # bus
    2: 0.6,  # car
    3: 0.2,  # motorbike
    4: 0.30,  # person, detected/displayed only
}

# Basic geometric filters to reject unstable / impossible boxes.
# Aspect ratio is width / height. Keep these loose because traffic-camera
# perspective changes object shape depending on direction.
MIN_BOX_AREA_RATIO = 0.00008
MIN_BOX_WH = {
    0: (8, 8),    # bicycle
    1: (20, 20),  # bus
    2: (16, 16),  # car
    3: (8, 8),    # motorbike
    4: (8, 14),   # person, detected/displayed only
}
CLASS_ASPECT_RATIO_LIMITS = {
    0: (0.18, 5.50),  # bicycle
    1: (0.30, 7.00),  # bus
    2: (0.30, 5.50),  # car
    3: (0.18, 5.50),  # motorbike
    4: (0.15, 2.80),  # person, detected/displayed only
}

# Suppress duplicate detections before DeepSORT. This handles the common case
# where YOLO returns several boxes for the same motorcycle/car, which then
# becomes several DeepSORT IDs.
DETECTION_DUPLICATE_IOU = 0.35
DETECTION_DUPLICATE_CONTAINMENT = 0.72
DETECTION_DUPLICATE_CENTER_RATIO = 0.32

# Suppress duplicate DeepSORT tracks after update. This prevents old ghost IDs
# from being drawn/counted together with the new ID for the same vehicle.
TRACK_DUPLICATE_IOU = 0.30
TRACK_DUPLICATE_CONTAINMENT = 0.68
TRACK_DUPLICATE_CENTER_RATIO = 0.35
TRACK_STALE_MERGE_FRAMES = 180

# DeepSORT tracking stability. Internal max_age is high so the tracker can keep
# identity through short occlusions, but stale tracks are not drawn for long.
TRACK_MAX_AGE = 300
TRACK_N_INIT = 4
TRACK_MAX_COSINE_DISTANCE = 0.22
TRACK_NN_BUDGET = 200
TRACK_DISPLAY_MAX_AGE = 4
TRACK_COUNT_HOLD_FRAMES = 18

# Real-time source support. You can type 0/1 for webcam, or RTSP/HTTP URL.
REALTIME_SOURCE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")
REALTIME_QUEUE_SIZE = 2
FILE_QUEUE_SECONDS = 10

# Printing every track every frame can make real-time mode much slower.
DEBUG_TRACK_LOGS = False

# Track-level class smoothing. A class must be repeatedly observed before it
# becomes stable. Person can become a stable label for display, but non-counted
# classes still have zero PCE and are blocked from all counting/flow export.
CLASS_HISTORY_LEN = 28
MIN_CLASS_VOTES = 6
CLASS_STABILITY_RATIO = 0.66
CLASS_LOCK_MIN_FRAMES = 14
CLASS_SWITCH_MIN_VOTES = 10
CLASS_SWITCH_RATIO = 0.80

CLASS_COLORS = {
    0: (0, 200, 80),      # bicycle
    1: (0, 190, 255),     # bus
    2: (0, 0, 255),       # car
    3: (255, 80, 0),      # motorbike
    4: (160, 160, 160),   # person, ignored by flow
}

# Performance presets.
# quality: maximum accuracy, heavier inference.
# balanced: good default for recorded video.
# realtime: lower latency; drops visual update work and uses smaller inference size.
DEFAULT_PERFORMANCE_PROFILE = "balanced"
PERFORMANCE_PROFILES = {
    "quality": {
        "model_imgsz": 1280,
        "process_width": 0,        # 0 = keep original frame size
        "detect_interval": 1,
        "display_every_n": 1,
        "half_cuda": True,
        "max_det": 300,
    },
    "balanced": {
        "model_imgsz": 960,
        "process_width": 1280,
        "detect_interval": 1,
        "display_every_n": 2,
        "half_cuda": True,
        "max_det": 250,
    },
    "realtime": {
        "model_imgsz": 736,
        "process_width": 960,
        "detect_interval": 1,
        "display_every_n": 3,
        "half_cuda": True,
        "max_det": 200,
    },
}

# Optional OpenCV CPU threading. 0 means leave OpenCV default unchanged.
CV2_NUM_THREADS = 0

# Warm up YOLO once before the processing loop so the first visible frames are
# not slowed by CUDA kernel initialization / autotuning.
YOLO_WARMUP = True

# Fluid-flow export/replay configuration.
# ROAD_BRANCHES are lane regions connected to the center node.
# Inbound lanes normally produce lane->center edges; outbound lanes normally
# produce center->lane edges. Keeping all 8 lanes here preserves the real data
# and still allows lane-changing overlaps to be exported.
ROAD_BRANCHES = LANE_REGION_ORDER
FLUID_REGIONS = LANE_REGION_ORDER + ("center",)
DEFAULT_EXPORT_ROOT = "flow_exports"
DEFAULT_FLUID_BIN_SECONDS = 1.0
DEFAULT_FLUID_SMOOTH_SECONDS = 5.0
DEFAULT_TRACK_SAMPLE_SECONDS = 0.25
DEFAULT_REGION_STATE_SAMPLE_SECONDS = 1.0
DEFAULT_INFER_HIDDEN_LEFT = True
# If a track appears/disappears in center with x <= width * LEFT_GATE_X_RATIO,
# treat it as an inferred movement from/to the hidden-left lane pair.
DEFAULT_LEFT_GATE_X_RATIO = 0.30
HIDDEN_LEFT_INBOUND_REGION = "l1"
HIDDEN_LEFT_OUTBOUND_REGION = "l2"
