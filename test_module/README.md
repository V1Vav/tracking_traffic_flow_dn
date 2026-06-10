# test_module - kiểm tra bottleneck realtime

Thư mục này chứa các script benchmark độc lập với GUI để xác định module nào đang làm pipeline không chạy realtime.

## Chạy nhanh toàn bộ

Từ thư mục gốc project:

```bash
python test_module/run_all_tests.py \
  --video path/to/video.mp4 \
  --model models/best.pt \
  --template template.csv \
  --profile realtime \
  --frames 300
```

Kết quả được lưu trong `test_results/` dưới dạng `.csv` và `.json`.

## Chạy từng phần

### 0. Kiểm tra môi trường

```bash
python test_module/test_00_env.py --model models/best.pt --profile realtime
```

Dùng để kiểm tra CUDA có bật không, GPU là gì, OpenCV/PyTorch/Ultralytics/DeepSORT version, `model.names` và cấu hình profile.

### 1. Video I/O + resize

```bash
python test_module/test_01_video_io.py --video path/to/video.mp4 --profile realtime --frames 300
```

Nếu `read_avg_ms` hoặc `read_p95_ms` cao, block nằm ở decode video/camera/RTSP. Nếu `resize_avg_ms` cao, nên giảm `process_width`.

### 2. YOLO inference

```bash
python test_module/test_02_yolo_inference.py \
  --video path/to/video.mp4 \
  --model models/best.pt \
  --profile realtime \
  --frames 300
```

Nếu `yolo_avg_ms` lớn hơn 33 ms thì khó đạt 30 FPS. Nếu lớn hơn 50 ms thì khó đạt 20 FPS. Khi nghi ngờ lọc class sai, để `--classes all` như mặc định.

### 3. YOLO + DeepSORT

```bash
python test_module/test_03_deepsort_update.py \
  --video path/to/video.mp4 \
  --model models/best.pt \
  --profile realtime \
  --frames 300
```

So sánh `yolo_avg_ms` và `deepsort_avg_ms`:

- YOLO cao: giảm `imgsz`, `process_width`, `max_det`.
- DeepSORT cao: số object quá nhiều hoặc ReID crop/embedding đang nặng; thử profile `realtime` dùng ByteTrackLite, không track person, giảm `max_det`, tăng ngưỡng confidence.
- `raw_boxes` nhiều nhưng `mapped_boxes` thấp: mapping class/model.names đang sai.

### 4. Region logic

```bash
python test_module/test_04_region_logic.py \
  --video path/to/video.mp4 \
  --template template.csv \
  --frames 300
```

Nếu region logic cao bất thường, template polygon quá phức tạp hoặc cache scale bị miss liên tục.

### 5. Display / overlay

```bash
python test_module/test_05_display_overlay.py \
  --video path/to/video.mp4 \
  --template template.csv \
  --profile realtime \
  --frames 150
```

Nếu `draw_avg_ms` hoặc `convert_resize_rgb_avg_ms` cao, GUI/vẽ overlay đang là bottleneck. Thử tắt region overlay, giảm `display_width/display_height`, hoặc tăng `display_every_n`.

### 6. Pipeline tổng hợp

```bash
python test_module/test_06_full_pipeline_profile.py \
  --video path/to/video.mp4 \
  --model models/best.pt \
  --template template.csv \
  --profile realtime \
  --frames 300
```

Đây là script quan trọng nhất. Nó đo trung bình từng stage:

- `read_avg_ms`
- `resize_avg_ms`
- `yolo_avg_ms`
- `tracker_avg_ms`
- `region_avg_ms`
- `draw_avg_ms`
- `display_convert_avg_ms`
- `total_avg_ms` và `total_p95_ms`

Ngưỡng tham khảo:

- 30 FPS: tổng nên dưới khoảng **33 ms/frame**.
- 20 FPS: tổng nên dưới khoảng **50 ms/frame**.
- Nếu `total_avg_ms` ổn nhưng `total_p95_ms` cao, pipeline sẽ vẫn giật do spike.


### 8. So sánh DeepSORT và ByteTrackLite

```bash
python test_module/test_08_tracker_compare.py \
  --video path/to/video.mp4 \
  --model models/best.pt \
  --profile realtime \
  --frames 300
```

Dùng khi nghi ngờ bottleneck nằm ở DeepSORT. Script này chạy cùng detection YOLO nhưng cập nhật hai tracker riêng:

- `deepsort_ms`: thời gian DeepSORT + ReID embedding.
- `bytetrack_lite_ms`: thời gian tracker nhẹ không dùng ReID.
- `tracker_speedup_ratio`: DeepSORT chậm hơn ByteTrackLite bao nhiêu lần.

Nếu ByteTrackLite nhanh hơn rõ rệt và ID vẫn đủ ổn cho đếm vùng, nên dùng profile `realtime` mặc định mới.

### 7. Queue latency

```bash
python test_module/test_07_queue_latency.py \
  --video path/to/video.mp4 \
  --profile realtime \
  --frames 300 \
  --process-ms 45 \
  --drop-old
```

Dùng để hiểu vì sao realtime bị delay tích lũy. Với realtime, queue size nhỏ và drop frame cũ thường tốt hơn giữ tất cả frame.

### 9. Kiểm tra khóa FPS realtime

```bash
python test_module/test_09_fps_lock.py \
  --video path/to/video.mp4 \
  --profile realtime \
  --frames 300
```

Script này kiểm tra rule mới của profile `realtime`:

```text
target_process_fps = min(video_fps, 30)
```

Ví dụ: video 24/25 FPS sẽ chạy 24/25 FPS; video 50/60 FPS sẽ lấy mẫu xuống 30 FPS. Kết quả nằm ở `test_results/09_fps_lock.json`.

## Cách đọc kết quả để tìm bottleneck

Mở `test_results/06_full_pipeline_profile.json`, xem `stage_avg_ms` và `diagnosis`.

Ví dụ:

```json
"stage_avg_ms": {
  "read": 2.1,
  "resize": 1.4,
  "yolo": 18.6,
  "tracker": 27.2,
  "region": 0.5,
  "draw": 4.3,
  "display_convert": 2.2
}
```

Trong ví dụ này, tracker là bottleneck chính, không phải YOLO. Nếu `tracker_type=deepsort` thì nên thử ByteTrackLite.

## Gợi ý chỉnh theo kết quả

- `read` cao: đổi codec video, cắt video nhẹ hơn, dùng file local thay vì network/RTSP, giảm FPS nguồn.
- `resize` cao: giảm `process_width` hoặc tránh resize nhiều lần.
- `yolo` cao: giảm `model_imgsz` 736 -> 640, giảm `process_width` 960 -> 832, giảm `max_det`.
- `deepsort` cao: giảm số box đưa vào tracker, không track pedestrian/person, giảm `max_det`, tăng confidence từng class.
- `region` cao: kiểm tra template, hạn chế polygon quá nhiều điểm.
- `draw/display` cao: tăng `display_every_n`, giảm kích thước preview, tắt overlay khi benchmark realtime.
- `queue_delay` cao: dùng queue size 1 và bỏ frame cũ khi chậm.
