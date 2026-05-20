"""Shared constants for the vehicle flow application."""

# This class mapping assumes your custom model uses:
# 0=bicycle, 1=motorcycle, 2=car, 3=bus, 4=truck.
# If you use official COCO YOLOv8 weights, the class IDs are different.
CLASS_WEIGHTS = {
    0: 0.2,   # bicycle
    1: 0.3,   # motorcycle
    2: 1.0,   # car
    3: 2.0,   # bus
    4: 2.0,   # truck
}

CLASS_NAMES = {
    0: "bicycle",
    1: "motorcycle",
    2: "car",
    3: "bus",
    4: "truck",
}

EXPECTED_MODEL_NAMES = CLASS_NAMES.copy()

BRANCH_ORDER = ("top", "left", "right", "bottom", "center")
VALID_BRANCHES = set(BRANCH_ORDER)
DIRECTIONS = ("in", "out")
DISPLAY_CLASS_IDS = (2, 1)  # car, motorcycle

REGION_NAME_MAP = {
    "up": "top",
    "down": "bottom",
    "left": "left",
    "right": "right",
    "center": "center",
    "none": None,
}

DEFAULT_TEMPLATE_MAPPING = "template.csv"
DEFAULT_MODEL_PATH = "models/tuning_200.pt"
DEFAULT_AVAILABLE_MODELS = [
    "models/tuning_200.pt",
    "models/tuning_50.pt",
    "models/tuning.pt",
    "models/yolov8n.pt",
    "models/yolov8s.pt",
    "models/yolov8m.pt",
    "models/yolov8l.pt",
    "models/yolov8x.pt",
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
    0: 0.22,  # bicycle
    1: 0.22,  # motorcycle
    2: 0.7,  # car
    3: 0.45,  # bus
    4: 0.45,  # truck
}

# Basic geometric filters to reject unstable / impossible boxes.
# Aspect ratio is width / height. Keep these loose because traffic-camera
# perspective changes object shape depending on direction.
MIN_BOX_AREA_RATIO = 0.00008
MIN_BOX_WH = {
    0: (8, 8),
    1: (8, 8),
    2: (16, 16),
    3: (20, 20),
    4: (20, 20),
}
CLASS_ASPECT_RATIO_LIMITS = {
    0: (0.18, 5.50),
    1: (0.18, 5.50),
    2: (0.30, 5.50),
    3: (0.30, 7.00),
    4: (0.30, 7.00),
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

# Track-level class smoothing. A vehicle class must be repeatedly observed
# before it becomes the stable class used for labels, PCE, and type counts.
CLASS_HISTORY_LEN = 28
MIN_CLASS_VOTES = 6
CLASS_STABILITY_RATIO = 0.66
CLASS_LOCK_MIN_FRAMES = 14
CLASS_SWITCH_MIN_VOTES = 10
CLASS_SWITCH_RATIO = 0.80

CLASS_COLORS = {
    0: (0, 255, 0),      # bicycle
    1: (255, 0, 0),      # motorcycle
    2: (0, 0, 255),      # car
    3: (0, 255, 255),    # bus
    4: (0, 165, 255),    # truck
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
# ROAD_BRANCHES are real road approaches connected to the center node.
ROAD_BRANCHES = ("top", "left", "right", "bottom")
FLUID_REGIONS = ("top", "left", "right", "bottom", "center")
DEFAULT_EXPORT_ROOT = "flow_exports"
DEFAULT_FLUID_BIN_SECONDS = 1.0
DEFAULT_FLUID_SMOOTH_SECONDS = 5.0
DEFAULT_TRACK_SAMPLE_SECONDS = 0.25
DEFAULT_REGION_STATE_SAMPLE_SECONDS = 1.0
DEFAULT_INFER_HIDDEN_LEFT = True
# If a track appears/disappears in center with x <= width * LEFT_GATE_X_RATIO,
# treat it as an inferred movement from/to the hidden left branch.
DEFAULT_LEFT_GATE_X_RATIO = 0.30
