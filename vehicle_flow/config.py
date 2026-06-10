"""Các hằng số dùng chung cho ứng dụng phân tích lưu lượng phương tiện."""

# Mapping class cho file data.yaml sau khi đã bỏ license plate:
#   0=bus, 1=car, 2=motorbike, 3=pedestrian, 4=truck.
#
# Trong pipeline realtime, truck được gộp về class 0 để tăng ổn định tracker
# cho nhóm xe lớn. Model YOLO vẫn có thể trả truck=4, nhưng sau YOLO và trước
# tracker/count, class 4 sẽ được đổi thành class 0. Vì vậy class 0 được hiển thị
# là Xe buýt và dùng chung trọng số PCE cho xe lớn.
#
# ``pedestrian`` vẫn được giữ trong detection/hiển thị để UI có thể kiểm tra,
# nhưng bị loại khỏi COUNTED_CLASS_IDS và không có trọng số PCE.
CLASS_NAMES = {
    0: "bus",
    1: "car",
    2: "motorbike",
    3: "pedestrian",
    4: "truck",  # class gốc của model, sẽ merge về 0 trước tracker/count
}

# Gộp class sau YOLO, trước filter/tracker/count.
# 4=truck -> 0=bus để tránh cùng một xe lớn lúc là bus, lúc là truck.
CLASS_MERGE_MAP = {
    4: 0,
}

# Tên hiển thị cho UI/log. Class 4 truck được merge về 0 nên nhóm xe lớn
# vẫn hiển thị là "Xe buýt" theo yêu cầu giao diện.
CLASS_DISPLAY_NAMES = {
    0: "Xe buýt",
    1: "Ô tô",
    2: "Xe máy",
    3: "Người đi bộ",
    4: "Xe tải",
}

CLASS_DISPLAY_NAME_OVERRIDE = {
    0: "Xe buýt",
}

CLASS_WEIGHTS = {
    0: 2.0,   # bus/heavy vehicle, truck đã merge về bus
    1: 1.0,   # car
    2: 0.3,   # motorbike
    # 3=pedestrian được cố ý loại khỏi phần đếm flow.
    # 4=truck đã được merge về 0 trước khi count/PCE.
}

# Các class gửi vào YOLO/DeepSORT. Giữ pedestrian ở đây để vẫn detect và
# hiển thị được trên video overlay.
DETECT_CLASS_IDS = tuple(CLASS_NAMES.keys())

# Các class được tính vào count/flow/PCE/file cho RL.
COUNTED_CLASS_IDS = tuple(CLASS_WEIGHTS.keys())
IGNORED_CLASS_IDS = tuple(sorted(set(DETECT_CLASS_IDS) - set(COUNTED_CLASS_IDS)))
# Profile realtime vẫn detect đủ class, bao gồm pedestrian. Pedestrian được track
# bằng ByteTrack để hiển thị ID/bbox nhưng không thuộc COUNTED_CLASS_IDS nên
# không tham gia PCE/count/flow.
REALTIME_DETECT_CLASS_IDS = DETECT_CLASS_IDS
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
DISPLAY_CLASS_IDS = (0, 1, 2)  # Xe buýt, Ô tô, Xe máy
COLOR_LEGEND_CLASS_IDS = (0, 1, 2, 3)  # thêm Người đi bộ vào chú thích màu, không đưa vào bảng đếm

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
DEFAULT_MODEL_PATH = "models/best.pt"
DEFAULT_AVAILABLE_MODELS = [
    "models/tuning_200.pt",
    "models/tuning_50.pt",
    "models/best.pt",
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

# Hậu xử lý region/transition cho dữ liệu flow.
# Dùng điểm giữa cạnh dưới bbox để xác định vùng xe đang đứng trên mặt đường,
# thay vì dùng tâm bbox. Cách này giảm sai lệch khi camera nhìn chéo.
USE_BOTTOM_CENTER_FOR_REGION = True

# Cooldown riêng cho transition xuất ra file fluid replay/OD. Giữ nhỏ để không
# làm mất cạnh center->outbound ở xe chạy nhanh; hướng ngược vẫn bị FSM loại.
FLUID_TRANSITION_COOLDOWN_FRAMES = 3

# Nếu một track nhảy trực tiếp từ làn vào sang làn ra mà thiếu center, có thể
# suy luận center bị bỏ lỡ giữa hai frame và tách thành lane->center->lane.
# Tắt nếu muốn chỉ giữ cạnh quan sát tuyệt đối.
FLUID_INFER_MISSING_CENTER = True
FLUID_INFERRED_CENTER_CONFIDENCE = 0.80

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
MODEL_IOU = 0.60

# Ngưỡng confidence riêng theo từng class.
# Tăng ngưỡng car/truck nếu xe máy thường bị đoán nhầm thành xe lớn.
# Giảm ngưỡng motorbike nếu xe nhỏ bị bỏ sót nhiều.
CLASS_CONF_THRESHOLDS = {
    0: 0.75,  # bus
    1: 0.86,  # car
    2: 0.36,  # motorbike
    3: 0.2,  # pedestrian, chỉ detect/hiển thị
    4: 0.8,  # truck
}

# Bộ lọc hình học cơ bản để loại box nhiễu/không hợp lý.
# Tỉ lệ aspect là width / height. Giữ ngưỡng khá rộng vì camera giao thông
# làm hình dạng vật thể thay đổi theo hướng nhìn.
MIN_BOX_AREA_RATIO = 0.00008
MIN_BOX_WH = {
    0: (20, 20),  # bus
    1: (16, 16),  # car
    2: (8, 8),    # motorbike
    3: (10, 22),  # pedestrian, chỉ detect/hiển thị
    4: (20, 20),  # truck
}

# Các ngưỡng pixel như MIN_BOX_WH được khai báo theo mốc tham chiếu này.
# Khi đổi profile sang 736/960/1280, video_worker sẽ tự scale ngưỡng theo
# frame_width / FILTER_REFERENCE_WIDTH. Nhờ vậy filter không bị quá lỏng khi
# tăng phân giải và không quá gắt khi giảm phân giải realtime.
FILTER_REFERENCE_WIDTH = 960
CLASS_ASPECT_RATIO_LIMITS = {
    0: (0.30, 7.00),  # bus/heavy vehicle
    1: (0.30, 5.50),  # car
    2: (0.18, 5.50),  # motorbike
    3: (0.18, 1.80),  # pedestrian, chỉ detect/hiển thị
    4: (0.30, 7.00),  # truck gốc, thường đã merge về 0
}

# Lọc kích thước bbox theo phối cảnh camera cố định.
# Đây không phải distance/depth thật, mà là ước lượng theo vị trí đáy bbox:
# - y2/frame_height nhỏ: vật thể xa camera, cho phép bbox nhỏ hơn.
# - y2/frame_height lớn: vật thể gần camera, yêu cầu bbox lớn hơn để loại nhiễu.
PERSPECTIVE_SIZE_FILTER_ENABLED = True

# class_id: (far_min_area_ratio, near_min_area_ratio)
# area_ratio = bbox_area / frame_area.
# Vì truck đã merge về 0, nhóm xe lớn dùng rule của class 0.
PERSPECTIVE_MIN_AREA_RATIO = {
    0: (0.00045, 0.00350),  # bus/heavy vehicle
    1: (0.00030, 0.00220),  # car
    2: (0.00018, 0.00120),  # motorbike
    3: (0.00018, 0.00110),  # pedestrian, nếu còn hiển thị
}

# >1 nghĩa là vùng gần camera bị siết mạnh hơn, vùng xa vẫn thoáng.
PERSPECTIVE_AREA_GAMMA = 1.35

# Scale riêng cho phần xa/gần trong perspective filter.
# near-only giúp tăng lọc nhiễu ở gần camera mà không làm mất xe máy nhỏ ở xa.
PERSPECTIVE_FAR_MIN_AREA_SCALE = 1.0
PERSPECTIVE_NEAR_MIN_AREA_SCALE = 1.0

# Mở rộng bbox theo class sau YOLO và trước tracker.
# Dùng cho trường hợp YOLO chỉ bắt phần thân xe máy, còn kính/ghi-đông phía trên
# bị detect thành một motorbike nhỏ khác. Tuple là (left, top, right, bottom)
# theo tỉ lệ của bbox gốc.
CLASS_BBOX_EXPAND_RATIO = {
    2: (0.08, 0.35, 0.08, 0.08),  # motorbike: mở rộng nhiều lên phía trên
}

# Sau khi mở rộng bbox motorbike, xóa các box motorbike nhỏ nằm phần lớn
# trong một box motorbike lớn hơn. Mục tiêu là bỏ box phụ ở kính/ghi-đông
# nhưng vẫn hạn chế xóa nhầm xe thật đứng cạnh nhau.
MOTORBIKE_PART_SUPPRESSION_ENABLED = True
MOTORBIKE_PART_CLASS_ID = 2
MOTORBIKE_PART_MAX_AREA_RATIO = 0.55   # box nhỏ < 55% diện tích box lớn mới bị xét là part
MOTORBIKE_PART_MIN_IOA = 0.70          # phần giao / diện tích box nhỏ


# Khử detection trùng trước DeepSORT. Xử lý trường hợp thường gặp
# khi YOLO trả nhiều box cho cùng một xe máy/ô tô, sau đó
# bị DeepSORT tách thành nhiều ID.
DETECTION_DUPLICATE_IOU = 0.45
DETECTION_DUPLICATE_CONTAINMENT = 0.82
DETECTION_DUPLICATE_CENTER_RATIO = 0.30

# Khử track DeepSORT trùng sau khi update. Tránh các ghost ID cũ
# bị vẽ/đếm cùng ID mới của cùng một phương tiện.
TRACK_DUPLICATE_IOU = 0.42
TRACK_DUPLICATE_CONTAINMENT = 0.78
TRACK_DUPLICATE_CENTER_RATIO = 0.32
TRACK_STALE_MERGE_FRAMES = 180

# Độ ổn định tracking DeepSORT. max_age nội bộ được để cao để tracker giữ
# danh tính qua che khuất ngắn, nhưng track cũ không được vẽ quá lâu.
TRACK_MAX_AGE = 300
TRACK_N_INIT = 4
TRACK_MAX_COSINE_DISTANCE = 0.22
TRACK_NN_BUDGET = 200
TRACK_DISPLAY_MAX_AGE = 4
# Riêng realtime/ByteTrack nên chỉ vẽ track vừa match detection.
# Nếu vẽ track mất match 3-4 frame, xe chạy nhanh sẽ xuất hiện 1 bbox thật quanh xe
# và 1 bbox ghost bị kéo phía sau.
REALTIME_TRACK_DISPLAY_MAX_AGE = 0
TRACK_COUNT_HOLD_FRAMES = 18


# Giảm FPS nguồn. Video FPS cao (60/120 FPS) có thể làm quá tải
# pipeline dù cảnh giao thông không cần xử lý mọi frame. Khi bật,
# video sẽ được lấy mẫu xuống FPS này trước YOLO/DeepSORT. Đặt 0 hoặc
# None để xử lý mọi frame nguồn. Nguồn live cũng dùng tham số này như giới hạn mềm
# khi có thể.
ENABLE_SOURCE_FPS_DOWNSAMPLE = True
TARGET_PROCESS_FPS = 15.0

# Khóa FPS realtime theo nguồn: nếu video < 30 FPS thì giữ đúng FPS video,
# nếu video > 30 FPS thì lấy mẫu/hiển thị tối đa 30 FPS.
# Giá trị này chỉ có hiệu lực với profile bật lock_to_source_fps.
REALTIME_FPS_LOCK_MAX = 30.0

# Hỗ trợ nguồn thời gian thực. Có thể nhập 0/1 cho webcam hoặc URL RTSP/HTTP.
REALTIME_SOURCE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")
REALTIME_QUEUE_SIZE = 2
FILE_QUEUE_SECONDS = 10

# In log mọi track ở mọi frame có thể làm chế độ realtime chậm hơn nhiều.
DEBUG_TRACK_LOGS = False

# Làm mượt class theo track. Một class phải được quan sát lặp lại trước khi
# trở thành ổn định. Pedestrian có thể ổn định để hiển thị, nhưng class không đếm
# vẫn có PCE bằng 0 và bị chặn khỏi mọi count/flow export.
CLASS_HISTORY_LEN = 28
MIN_CLASS_VOTES = 6
CLASS_STABILITY_RATIO = 0.66
CLASS_LOCK_MIN_FRAMES = 14
CLASS_SWITCH_MIN_VOTES = 10
CLASS_SWITCH_RATIO = 0.80

CLASS_COLORS = {
    0: (0, 190, 255),     # bus
    1: (0, 0, 255),       # car
    2: (255, 80, 0),      # motorbike
    3: (255, 0, 255),     # pedestrian, màu tím/hồng để không trùng màu đường/vùng
    4: (255, 180, 0),     # truck
}

# Các preset hiệu năng.
# quality: chế độ offline/kiểm chứng, giữ nhiều detection hơn và không bỏ frame file.
# balanced: chế độ realtime mặc định. Dùng cấu hình realtime trước đó và
#           bỏ frame file/live đã cũ khi xử lý không theo kịp.
# realtime: chế độ xem nhanh hơn cho cảnh đông, không còn preset dense/ultra.
DEFAULT_PERFORMANCE_PROFILE = "realtime"
PERFORMANCE_PROFILES = {
    "quality": {
        # 1280p xử lý: dùng để kiểm chứng/offline/export khi ưu tiên độ chính xác.
        # Filter pixel sẽ tự scale theo FILTER_REFERENCE_WIDTH nên không bị lỏng
        # khi tăng phân giải từ 960 lên 1280.
        "profile_resolution_name": "1280p quality",
        "model_imgsz": 1280,
        "process_width": 1280,
        "detect_interval": 1,
        "display_every_n": 1,
        "half_cuda": True,
        "max_det": 300,
        "track_ignored_classes": True,
        "tracker_type": "deepsort",
        "detection_conf_scale": 1.00,
        # Quality giữ filter nền cho mọi class.
        # Phần siết thêm chỉ áp dụng cho motorbike để giảm false detection xe máy
        # ở 1280p mà không làm mất car/bus/pedestrian thật.
        "filter_min_box_wh_scale": 1.00,
        "filter_min_box_area_scale": 1.00,
        "perspective_min_area_scale": 1.00,
        "filter_min_box_wh_scale_by_class": {2: 1.},
        "filter_min_box_area_scale_by_class": {2: 1.55},
        # Chỉ siết mạnh motorbike ở vùng gần camera; vùng xa giữ scale 1.0.
        "perspective_far_min_area_scale_by_class": {2: 1.00},
        "perspective_near_min_area_scale_by_class": {2: 5.00, 3: 5.00},
        "display_width": 960,
        "display_height": 680,
        "target_process_fps": TARGET_PROCESS_FPS,
        "drop_frames_when_slow": False,
        "realtime_queue_size": 3,
        "bbox_thickness": 3,
        "bbox_center_radius": 3,
        "draw_track_labels": True,
        "draw_track_class_name": False,
        "track_label_font_scale": 0.45,
        "track_label_thickness": 1,
        "bbox_smoothing_enabled": True,
        "bbox_smooth_center_alpha": 0.15,
        "bbox_smooth_size_alpha": 0.45,
        "bbox_smooth_reset_iou": 0.08,
        "bbox_smooth_reset_center_ratio": 1.40,
        "draw_detection_labels": False,
        "region_label_scale": 1.00,
    },
    "balanced": {
        # 960p xử lý: mốc cân bằng để kiểm tra kết quả và vẫn giữ tốc độ khá tốt.
        "profile_resolution_name": "960p balanced",
        "model_imgsz": 960,
        "process_width": 960,
        "detect_interval": 1,
        "display_every_n": 1,
        "half_cuda": True,
        "max_det": 200,
        "track_ignored_classes": True,
        "tracker_type": "bytetrack_lite",
        "bytetrack_high_thresh": 0.45,
        "bytetrack_low_thresh": 0.10,
        "bytetrack_new_track_thresh": 0.50,
        "bytetrack_match_thresh": 0.18,
        "bytetrack_low_match_thresh": 0.10,
        "bytetrack_max_age": 30,
        "bytetrack_n_init": 1,
        "bytetrack_class_aware": True,
        "bytetrack_velocity_alpha": 0.85,
        "bytetrack_center_match_ratio": 0.85,
        "bytetrack_min_iou_for_center_match": 0.01,
        "bytetrack_min_size_similarity": 0.40,
        "track_display_max_age": 1,
        "detection_conf_scale": 1.00,
        # Balanced giữ filter nền cho mọi class.
        # Phần siết thêm chỉ áp dụng cho motorbike; các class khác dùng scale 1.0.
        "filter_min_box_wh_scale": 1.00,
        "filter_min_box_area_scale": 1.00,
        "perspective_min_area_scale": 1.00,
        "filter_min_box_wh_scale_by_class": {2: 1.15},
        "filter_min_box_area_scale_by_class": {2: 1.25},
        # Chỉ siết motorbike ở vùng gần camera; vùng xa giữ scale 1.0.
        "perspective_far_min_area_scale_by_class": {2: 1.00},
        "perspective_near_min_area_scale_by_class": {2: 5.00, 3: 5.00},
        "display_width": 900,
        "display_height": 640,
        "target_process_fps": 25.0,
        "drop_frames_when_slow": True,
        "realtime_queue_size": 2,
        "bbox_thickness": 2,
        "bbox_center_radius": 3,
        "draw_track_labels": True,
        "draw_track_class_name": False,
        "track_label_font_scale": 0.38,
        "track_label_thickness": 1,
        "bbox_smoothing_enabled": True,
        "bbox_smooth_center_alpha": 0.20,
        "bbox_smooth_size_alpha": 0.60,
        "bbox_smooth_reset_iou": 0.08,
        "bbox_smooth_reset_center_ratio": 1.40,
        "draw_detection_labels": False,
        "region_label_scale": 0.85,
    },
    "realtime": {
        # 736p xử lý: mốc realtime cho RTX 3050 Laptop + i7 gen 11.
        # Giảm process_width nên bbox cũng mảnh hơn để tránh che hình.
        "profile_resolution_name": "736p realtime",
        "model_imgsz": 736,
        "process_width": 736,
        "detect_interval": 1,
        "display_every_n": 1,
        "half_cuda": True,
        "max_det": 160,
        "track_ignored_classes": True,
        "detect_class_ids": REALTIME_DETECT_CLASS_IDS,
        "tracker_type": "bytetrack_lite",
        "bytetrack_high_thresh": 0.45,
        "bytetrack_low_thresh": 0.10,
        "bytetrack_new_track_thresh": 0.50,
        "bytetrack_match_thresh": 0.18,
        "bytetrack_low_match_thresh": 0.10,
        "bytetrack_max_age": 30,
        "bytetrack_n_init": 1,
        "bytetrack_class_aware": True,
        "bytetrack_velocity_alpha": 0.85,
        "bytetrack_center_match_ratio": 0.85,
        "bytetrack_min_iou_for_center_match": 0.01,
        "bytetrack_min_size_similarity": 0.40,
        "track_display_max_age": 1,
        "detection_conf_scale": 1.00,
        # Realtime đang ổn nên giữ filter scale = 1.0 để không làm mất xe xa.
        "filter_min_box_wh_scale": 1.00,
        "filter_min_box_area_scale": 1.00,
        "perspective_min_area_scale": 1.00,
        # Realtime chỉ tăng rất nhẹ filter motorbike ở gần để tránh nhiễu đáy ảnh.
        "perspective_far_min_area_scale_by_class": {2: 1.00},
        "perspective_near_min_area_scale_by_class": {2: 1.25},
        "tracker_embedder": "mobilenet",
        "display_width": 900,
        "display_height": 620,
        # Khóa realtime theo min(video_fps, 30). Nếu video 24/25 FPS thì chạy đúng
        # 24/25; nếu video 50/60 FPS thì chỉ lấy mẫu và phát tối đa 30 FPS.
        "target_process_fps": REALTIME_FPS_LOCK_MAX,
        "lock_to_source_fps": True,
        "fps_lock_max": REALTIME_FPS_LOCK_MAX,
        "strict_fps_lock": True,
        "drop_frames_when_slow": True,
        "realtime_queue_size": 1,
        "cpu_thread_count": 6,
        "bbox_thickness": 2,
        "bbox_center_radius": 2,
        "draw_track_labels": True,
        "draw_track_class_name": False,
        "track_label_font_scale": 0.32,
        "track_label_thickness": 1,
        "bbox_smoothing_enabled": True,
        "bbox_smooth_center_alpha": 0.25,
        "bbox_smooth_size_alpha": 0.70,
        "bbox_smooth_reset_iou": 0.08,
        "bbox_smooth_reset_center_ratio": 1.40,
        "draw_detection_labels": False,
        "region_label_scale": 0.65,
    },
}

# Tùy chọn dùng luồng CPU. 0 nghĩa là tự động. Tự động chừa 1-2 core cho UI/OS.
# Hỗ trợ OpenCV decode/resize/vẽ, DeepSORT trên CPU và các tác vụ PyTorch CPU.
USE_CPU_THREADS = True
CPU_THREAD_COUNT = 0
TORCH_INTEROP_THREADS = 2

# Override OpenCV để tương thích ngược. Nếu >0, ưu tiên hơn CPU_THREAD_COUNT cho riêng OpenCV.
CV2_NUM_THREADS = 0

# Chuyển frame OpenCV sang đầu vào PIL/ImageTk trong một luồng worker riêng.
# Tách BGR->RGB + resize + tạo PIL khỏi vòng lặp detection/tracking.
ASYNC_DISPLAY_CONVERSION = True
DISPLAY_CONVERSION_QUEUE_SIZE = 1

# Warm up YOLO một lần trước vòng xử lý để các frame đầu hiển thị
# không bị chậm do khởi tạo CUDA kernel / autotuning.
YOLO_WARMUP = True

# Cấu hình xuất/replay flow dạng chất lỏng.
# ROAD_BRANCHES là các vùng làn nối với nút center.
# Làn vào thường tạo cạnh lane->center; làn ra thường
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
