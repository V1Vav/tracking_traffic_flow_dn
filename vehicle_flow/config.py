"""Các hằng số dùng chung cho ứng dụng phân tích lưu lượng phương tiện."""

# Mapping class cho file data.yaml traffic_count_vn hiện tại:
#   0=bicycle, 1=bus, 2=car, 3=motorbike, 4=person.
#
# ``person`` vẫn được giữ trong detection/tracking/hiển thị để UI có thể
# hiển thị phục vụ kiểm tra, nhưng bị loại khỏi COUNTED_CLASS_IDS và không có trọng số PCE.
# Vì vậy người vẫn được detect/vẽ box, nhưng không bao giờ đóng góp Vao
# PCE/count/flow/route/file xuất cho RL.
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
    # 4=person được cố ý loại khỏi phần đếm flow.
}

# Các class gửi Vao YOLO/DeepSORT. Giữ person ở đây để vẫn detect và
# hiển thị được trên video overlay.
DETECT_CLASS_IDS = tuple(CLASS_NAMES.keys())

# Các class được tính Vao count/flow/PCE/file cho RL.
COUNTED_CLASS_IDS = tuple(CLASS_WEIGHTS.keys())
IGNORED_CLASS_IDS = tuple(sorted(set(DETECT_CLASS_IDS) - set(COUNTED_CLASS_IDS)))
EXPECTED_MODEL_NAMES = CLASS_NAMES.copy()

# Bố cục vùng/làn. Mỗi nhánh đường được chia thành hai vùng làn:
#   t/l/r/b + 1 = làn Vao, đi về phía center
#   t/l/r/b + 2 = làn ra, đi từ center ra ngoài
# Polygon của làn 1 và làn 2 có thể chồng nhẹ để bao phủ trường hợp lấn/chuyển làn.
INBOUND_LANE_REGIONS = ("t1", "l1", "r1", "b1")
OUTBOUND_LANE_REGIONS = ("t2", "l2", "r2", "b2")
LANE_REGION_ORDER = ("t1", "t2", "l1", "l2", "r1", "r2", "b1", "b2")
BRANCH_ORDER = LANE_REGION_ORDER + ("center",)
VALID_BRANCHES = set(BRANCH_ORDER)
DIRECTIONS = ("in", "out")
DISPLAY_CLASS_IDS = (2, 3)  # car, motorbike

REGION_DISPLAY_NAMES = {
    "t1": "T1 Vao",
    "t2": "T2 Ra",
    "l1": "L1 Vao",
    "l2": "L2 Ra",
    "r1": "R1 Vao",
    "r2": "R2 Ra",
    "b1": "B1 Vao",
    "b2": "B2 Ra",
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
    # Tên làn trực tiếp
    "t1": "t1", "t2": "t2",
    "l1": "l1", "l2": "l2",
    "r1": "r1", "r2": "r2",
    "b1": "b1", "b2": "b2",

    # Tên thay thế dễ đọc cho template.csv
    "top1": "t1", "top_in": "t1", "in_top": "t1", "top_inbound": "t1",
    "top2": "t2", "top_out": "t2", "out_top": "t2", "top_outbound": "t2",
    "left1": "l1", "left_in": "l1", "in_left": "l1", "left_inbound": "l1",
    "left2": "l2", "left_out": "l2", "out_left": "l2", "left_outbound": "l2",
    "right1": "r1", "right_in": "r1", "in_right": "r1", "right_inbound": "r1",
    "right2": "r2", "right_out": "r2", "out_right": "r2", "right_outbound": "r2",
    "bottom1": "b1", "bottom_in": "b1", "in_bottom": "b1", "bottom_inbound": "b1",
    "bottom2": "b2", "bottom_out": "b2", "out_bottom": "b2", "bottom_outbound": "b2",

    # Template 8 làn hiện tại chỉ nên dùng t1/t2/l1/l2/r1/r2/b1/b2/center
    # hoặc các alias làn rõ ràng ở trên. Các tên vùng cũ kiểu
    # top/left/right/bottom được cố ý không chấp nhận nữa, vì
    # chúng có thể khiến template 4 vùng cũ bị hiểu nhầm là hợp lệ trong app 8 làn.
    "center": "center",
    "none": None,
}

# Đường dẫn template mặc định hiển thị trên UI. Template 4 vùng cũ vẫn sẽ
# bị RegionTemplate từ chối, nên giữ mặc định này vẫn an toàn cho app 8 làn.
DEFAULT_TEMPLATE_MAPPING = "template.csv"
DEFAULT_MODEL_PATH = "models/tuning_50.pt"
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

# Tham số ổn định khi đếm theo vùng.
# Tăng STABLE_REGION_FRAMES nếu xe bị nhiễu quanh mép polygon.
# Giảm tham số này nếu xe đi qua vùng nhỏ quá nhanh.
STABLE_REGION_FRAMES = 5
REGION_HISTORY_LEN = 12
EVENT_COOLDOWN_FRAMES = 12

# Giữ branch đang hoạt động thêm một thời gian ngắn khi bị che khuất trước khi phát OUT.
# Điều này tránh việc mất detection một frame làm thay đổi count/trạng thái branch ngay lập tức.
LOST_OUT_FRAMES = 120

# Nếu track biến mất gần mép frame, khả năng cao xe đã rời khỏi vùng camera.
# Khi đó phát OUT sớm hơn, nhưng vẫn giữ timeout dài cho che khuất
# ở giữa frame.
EDGE_LOST_OUT_FRAMES = 24
FRAME_EXIT_MARGIN_RATIO = 0.025

# Tinh chỉnh inference.
# Ngưỡng confidence YOLO toàn cục được để thấp, sau đó từng class được lọc
# bằng CLASS_CONF_THRESHOLDS bên dưới. Cách này linh hoạt hơn một ngưỡng chung.
MODEL_IMGSZ = 1280
MODEL_CONF = 0.10
MODEL_IOU = 0.50

# Ngưỡng confidence riêng theo từng class.
# Tăng ngưỡng car nếu xe máy/xe đạp thường bị đoán nhầm thành car.
# Giảm ngưỡng motorbike/bicycle nếu xe nhỏ bị bỏ sót nhiều.
CLASS_CONF_THRESHOLDS = {
    0: 0.22,  # bicycle
    1: 0.8,  # bus
    2: 0.5,  # car
    3: 0.22,  # motorbike
    4: 0.30,  # person, chỉ detect/hiển thị
}

# Bộ lọc hình học cơ bản để loại box nhiễu/không hợp lý.
# Tỉ lệ aspect là width / height. Giữ ngưỡng khá rộng vì camera giao thông
# làm hình dạng vật thể thay đổi theo hướng nhìn.
MIN_BOX_AREA_RATIO = 0.00008
MIN_BOX_WH = {
    0: (8, 8),    # bicycle
    1: (20, 20),  # bus
    2: (16, 16),  # car
    3: (8, 8),    # motorbike
    4: (8, 14),   # person, chỉ detect/hiển thị
}
CLASS_ASPECT_RATIO_LIMITS = {
    0: (0.18, 5.50),  # bicycle
    1: (0.30, 7.00),  # bus
    2: (0.30, 5.50),  # car
    3: (0.18, 5.50),  # motorbike
    4: (0.15, 2.80),  # person, chỉ detect/hiển thị
}

# Khử detection trùng trước DeepSORT. Xử lý trường hợp thường gặp
# khi YOLO trả nhiều box cho cùng một xe máy/ô tô, sau đó
# bị DeepSORT tách thành nhiều ID.
DETECTION_DUPLICATE_IOU = 0.35
DETECTION_DUPLICATE_CONTAINMENT = 0.72
DETECTION_DUPLICATE_CENTER_RATIO = 0.32

# Khử track DeepSORT trùng sau khi update. Tránh các ghost ID cũ
# bị vẽ/đếm cùng ID mới của cùng một phương tiện.
TRACK_DUPLICATE_IOU = 0.30
TRACK_DUPLICATE_CONTAINMENT = 0.68
TRACK_DUPLICATE_CENTER_RATIO = 0.35
TRACK_STALE_MERGE_FRAMES = 180

# Độ ổn định tracking DeepSORT. max_age nội bộ được để cao để tracker giữ
# danh tính qua che khuất ngắn, nhưng track cũ không được vẽ quá lâu.
TRACK_MAX_AGE = 300
TRACK_N_INIT = 4
TRACK_MAX_COSINE_DISTANCE = 0.22
TRACK_NN_BUDGET = 200
TRACK_DISPLAY_MAX_AGE = 4
TRACK_COUNT_HOLD_FRAMES = 18


# Giảm FPS nguồn. Video FPS cao (60/120 FPS) có thể làm quá tải
# pipeline dù cảnh giao thông không cần xử lý mọi frame. Khi bật,
# video sẽ được lấy mẫu xuống FPS này trước YOLO/DeepSORT. Đặt 0 hoặc
# None để xử lý mọi frame nguồn. Nguồn live cũng dùng tham số này như giới hạn mềm
# khi có thể.
ENABLE_SOURCE_FPS_DOWNSAMPLE = True
TARGET_PROCESS_FPS = 15.0

# Hỗ trợ nguồn thời gian thực. Có thể nhập 0/1 cho webcam hoặc URL RTSP/HTTP.
REALTIME_SOURCE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")
REALTIME_QUEUE_SIZE = 2
FILE_QUEUE_SECONDS = 10

# In log mọi track ở mọi frame có thể làm chế độ realtime chậm hơn nhiều.
DEBUG_TRACK_LOGS = False

# Làm mượt class theo track. Một class phải được quan sát lặp lại trước khi
# trở thành ổn định. Person có thể ổn định để hiển thị, nhưng class không đếm
# vẫn có PCE bằng 0 và bị chặn khỏi mọi count/flow export.
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
    4: (160, 160, 160),   # person, bỏ qua trong flow
}

# Các preset hiệu năng.
# quality: chế độ offline/kiểm chứng, giữ nhiều detection hơn và không bỏ frame file.
# balanced: chế độ realtime mặc định. Dùng cấu hình realtime trước đó và
#           bỏ frame file/live đã cũ khi xử lý không theo kịp.
# realtime: chế độ xem nhanh hơn cho cảnh đông, không còn preset dense/ultra.
DEFAULT_PERFORMANCE_PROFILE = "realtime"
PERFORMANCE_PROFILES = {
    "quality": {
        # Chế độ offline/kiểm chứng: giữ nhiều chi tiết ảnh hơn và không bỏ frame
        # đã lấy mẫu. Dùng khi độ chính xác export quan trọng hơn cảm giác realtime.
        "model_imgsz": 1280,
        "process_width": 0,        # 0 = giữ kích thước frame gốc
        "detect_interval": 1,
        "display_every_n": 1,
        "half_cuda": True,
        "max_det": 300,
        "track_ignored_classes": True,   # person cũng có ID DeepSORT
        "display_width": 880,
        "display_height": 620,
        "target_process_fps": TARGET_PROCESS_FPS,
        "drop_frames_when_slow": False,
        "realtime_queue_size": 3,
    },
    "balanced": {
        # Chế độ realtime mượt. Thay vì cố xử lý 30 frame nặng mỗi
        # giây rồi phải bỏ frame theo cụm, chế độ này lấy mẫu nguồn xuống 20 FPS và
        # hiển thị mọi frame đã xử lý. Kết quả thường ít giật hơn.
        "model_imgsz": 736,
        "process_width": 960,
        "detect_interval": 1,
        "display_every_n": 1,
        "half_cuda": True,
        "max_det": 160,
        "track_ignored_classes": False,  # detect/hiển thị person, nhưng không gửi person Vao DeepSORT
        "display_width": 820,
        "display_height": 580,
        "target_process_fps": 20.0,
        "drop_frames_when_slow": True,
        "realtime_queue_size": 2,
    },
    "realtime": {
        # Chế độ realtime độ trễ thấp cho cảnh đông. FPS thấp là chủ ý:
        # hiển thị ổn định 15 FPS dễ quan sát hơn pipeline 30 FPS nhưng
        # liên tục bỏ frame và nhảy thời gian.
        "model_imgsz": 640,
        "process_width": 832,
        "detect_interval": 1,
        "display_every_n": 2,
        "half_cuda": True,
        "max_det": 130,
        "track_ignored_classes": False,
        "display_width": 780,
        "display_height": 552,
        "target_process_fps": 15.0,
        "drop_frames_when_slow": True,
        "realtime_queue_size": 1,
    },
}

# Tùy chọn dùng luồng CPU. 0 nghĩa là tự động. Tự động chừa 1-2 core cho UI/OS.
# Hỗ trợ OpenCV decode/resize/vẽ, DeepSORT trên CPU và các tác vụ PyTorch CPU.
USE_CPU_THREADS = True
CPU_THREAD_COUNT = 0
TORCH_INTEROP_THREADS = 2

# Override OpenCV để tương thích ngược. Nếu >0, ưu tiên hơn CPU_THREAD_COUNT cho riêng OpenCV.
CV2_NUM_THREADS = 0

# Chuyển frame OpenCV sang đầu Vao PIL/ImageTk trong một luồng worker riêng.
# Tách BGR->RGB + resize + tạo PIL khỏi vòng lặp detection/tracking.
ASYNC_DISPLAY_CONVERSION = True
DISPLAY_CONVERSION_QUEUE_SIZE = 1

# Warm up YOLO một lần trước vòng xử lý để các frame đầu hiển thị
# không bị chậm do khởi tạo CUDA kernel / autotuning.
YOLO_WARMUP = True

# Cấu hình xuất/replay flow dạng chất lỏng.
# ROAD_BRANCHES là các vùng làn nối với nút center.
# Làn Vao thường tạo cạnh lane->center; làn ra thường
# tạo cạnh center->lane. Giữ đủ 8 làn ở đây để bảo toàn dữ liệu thật
# và vẫn cho phép export các vùng chồng lấn do chuyển/lấn làn.
ROAD_BRANCHES = LANE_REGION_ORDER
FLUID_REGIONS = LANE_REGION_ORDER + ("center",)
DEFAULT_EXPORT_ROOT = "flow_exports"
DEFAULT_FLUID_BIN_SECONDS = 1.0
DEFAULT_FLUID_SMOOTH_SECONDS = 5.0
DEFAULT_TRACK_SAMPLE_SECONDS = 0.25
DEFAULT_REGION_STATE_SAMPLE_SECONDS = 1.0
DEFAULT_INFER_HIDDEN_LEFT = False
# Nếu track xuất hiện/biến mất ở center với x <= width * LEFT_GATE_X_RATIO,
# xem đó là chuyển động suy luận từ/đến cặp làn trái bị khuất.
DEFAULT_LEFT_GATE_X_RATIO = 0.30
HIDDEN_LEFT_INBOUND_REGION = "l1"
HIDDEN_LEFT_OUTBOUND_REGION = "l2"
