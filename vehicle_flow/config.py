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
