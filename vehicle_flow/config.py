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

BRANCH_ORDER = ("top", "left", "right", "bottom")
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

# Region-counting stability parameters.
# Increase STABLE_REGION_FRAMES if vehicles jitter around polygon edges.
# Decrease it if vehicles pass through small regions too fast.
STABLE_REGION_FRAMES = 4
REGION_HISTORY_LEN = 8
EVENT_COOLDOWN_FRAMES = 10
LOST_OUT_FRAMES = 15

# Inference tuning.
# The global YOLO confidence is intentionally low, then each class is filtered
# by CLASS_CONF_THRESHOLDS below. This is more flexible than one global conf.
MODEL_IMGSZ = 1280
MODEL_CONF = 0.10
MODEL_IOU = 0.55

# Class-specific confidence thresholds.
# Raise car threshold when motorcycles/bicycles are often predicted as car.
# Lower motorcycle/bicycle thresholds if small vehicles are missed too often.
CLASS_CONF_THRESHOLDS = {
    0: 0.22,  # bicycle
    1: 0.22,  # motorcycle
    2: 0.45,  # car
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

# Track-level class smoothing. A vehicle class must be repeatedly observed
# before it becomes the stable class used for labels, PCE, and type counts.
CLASS_HISTORY_LEN = 12
MIN_CLASS_VOTES = 3
CLASS_STABILITY_RATIO = 0.58

DEFAULT_AVAILABLE_MODELS = [
    "models/yolov8n.pt",
    "models/yolov8s.pt",
    "models/yolov8m.pt",
    "models/yolov8l.pt",
    "models/yolov8x.pt",
    "models/tuning.pt",
]

CLASS_COLORS = {
    0: (0, 255, 0),      # bicycle
    1: (255, 0, 0),      # motorcycle
    2: (0, 0, 255),      # car
    3: (0, 255, 255),    # bus
    4: (0, 165, 255),    # truck
}
