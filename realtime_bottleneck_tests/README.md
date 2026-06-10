# Realtime Bottleneck Tests

Thư mục này dùng để tách realtime pipeline thành nhiều phase nhỏ để biết nghẽn ở đâu: đọc video, preprocess, YOLO, DeepSORT/tracker, region/flow logic, render/UI và pipeline tổng.

## Cách đặt thư mục

Chép nguyên thư mục `realtime_bottleneck_tests/` vào root project, cùng cấp với:

```text
main.py
vehicle_flow/
realtime_bottleneck_tests/
```

Chạy bằng đúng Python environment đang chạy app realtime, ví dụ environment có `ultralytics`, `torch`, `opencv-python`.

## Chạy nhanh toàn bộ

Windows PowerShell:

```powershell
python realtime_bottleneck_tests\run_all_bottleneck_tests.py `
  --video "E:\DATN\videos\test.mp4" `
  --model "E:\DATN\runs\detect\train\weights\best.pt" `
  --template "E:\DATN\template.csv" `
  --device 0 `
  --imgsz 640 `
  --half `
  --max-frames 500 `
  --target-fps 30 `
  --reports bottleneck_reports
```

Linux/macOS:

```bash
python realtime_bottleneck_tests/run_all_bottleneck_tests.py \
  --video ./videos/test.mp4 \
  --model ./runs/detect/train/weights/best.pt \
  --template ./template.csv \
  --device 0 \
  --imgsz 640 \
  --half \
  --max-frames 500 \
  --target-fps 30 \
  --reports bottleneck_reports
```

Nếu muốn đo cả chi phí vẽ overlay trong full pipeline thì thêm `--render`.

## Chạy từng phase

### 00 - Môi trường

```bash
python realtime_bottleneck_tests/00_collect_env.py --video ./videos/test.mp4 --model ./runs/detect/train/weights/best.pt --reports bottleneck_reports
```

Dùng để kiểm tra CUDA, torch, OpenCV, FPS/resolution video, `nvidia-smi`.

### 01 - Video IO/decode

```bash
python realtime_bottleneck_tests/01_video_io_decode.py --video ./videos/test.mp4 --max-frames 500 --bgr2rgb --reports bottleneck_reports
```

Nếu phase này đã cao, bottleneck nằm ở đọc/giải mã video hoặc resize/conversion, chưa liên quan YOLO.

### 02 - Preprocess sang tensor

```bash
python realtime_bottleneck_tests/02_preprocess_to_tensor.py --video ./videos/test.mp4 --imgsz 640 --device 0 --half --reports bottleneck_reports
```

Đo letterbox, transpose, tạo tensor và copy CPU→GPU.

### 03 - YOLO detect

```bash
python realtime_bottleneck_tests/03_yolo_detect_speed.py --video ./videos/test.mp4 --model ./runs/detect/train/weights/best.pt --device 0 --imgsz 640 --half --save-detections --reports bottleneck_reports
```

Đo `preprocess`, `inference`, `postprocess` do Ultralytics trả về và tổng thời gian gọi `model.predict`.

### 04 - YOLO + Tracker/DeepSORT

```bash
python realtime_bottleneck_tests/04_yolo_plus_tracker_speed.py --video ./videos/test.mp4 --model ./runs/detect/train/weights/best.pt --tracker auto --device 0 --imgsz 640 --half --save-tracks --reports bottleneck_reports
```

Nếu có `deep_sort_realtime`, script sẽ dùng DeepSORT. Nếu không có, script tự fallback sang tracker nhẹ `centroid` để vẫn đo được phần overhead không phải YOLO.

### 05 - Region/flow logic

```bash
python realtime_bottleneck_tests/05_region_flow_logic_speed.py --template ./template.csv --video ./videos/test.mp4 --tracks-csv bottleneck_reports/04_tracks.csv --reports bottleneck_reports
```

Đo chi phí xác định vùng theo tâm bbox và cập nhật chuyển vùng. `center` được đặt ưu tiên thấp hơn các vùng lane.

### 06 - Render/overlay/UI-like conversion

```bash
python realtime_bottleneck_tests/06_render_overlay_speed.py --video ./videos/test.mp4 --template ./template.csv --tracks-csv bottleneck_reports/04_tracks.csv --pil-convert --reports bottleneck_reports
```

Đo chi phí vẽ bbox, text, polygon vùng và chuyển frame BGR→RGB→PIL giống đường đi thường gặp khi hiển thị trên Tkinter.

### 07 - Pipeline tổng

```bash
python realtime_bottleneck_tests/07_full_pipeline_probe.py --video ./videos/test.mp4 --model ./runs/detect/train/weights/best.pt --template ./template.csv --tracker auto --device 0 --imgsz 640 --half --render --reports bottleneck_reports
```

Đo một vòng pipeline: YOLO → tracker → region → render. Đây là script quan trọng nhất để so với mốc 30 FPS.

### 99 - Phân tích report

```bash
python realtime_bottleneck_tests/99_analyze_reports.py --reports bottleneck_reports --target-fps 30
```

Script sẽ xếp hạng các phase theo P95 ms. Với 30 FPS, ngân sách mỗi frame khoảng `33.33 ms`. Phase nào P95 vượt hoặc chiếm phần lớn ngân sách là ứng viên bottleneck chính.

## Cách đọc kết quả

Các file CSV/JSON nằm trong thư mục `bottleneck_reports/`.

Mốc tham khảo cho 30 FPS:

| Phase | Nếu cao hơn mức này thì đáng nghi |
|---|---:|
| Video read/decode | > 5-8 ms |
| Preprocess/tensor | > 3-5 ms |
| YOLO total | > 20-25 ms |
| Tracker | > 5-8 ms |
| Region logic | > 1-2 ms |
| Render/UI conversion | > 5-10 ms |
| Total pipeline | > 33.33 ms |

Ưu tiên nhìn `p95_ms`, không chỉ nhìn `mean_ms`, vì realtime hay bị giật do spike.

## Gợi ý tối ưu theo bottleneck

- **Video IO cao**: giảm resolution nguồn, dùng file nằm trên SSD, thử convert video sang codec nhẹ hơn, tách thread đọc frame.
- **Preprocess cao**: tránh resize nhiều lần, thống nhất `imgsz`, dùng `half` trên CUDA, hạn chế copy không cần thiết.
- **YOLO inference cao**: giảm `imgsz` 640 → 512/416, dùng YOLOv8n, TensorRT/ONNX, bật `half`, tăng `conf` để giảm postprocess.
- **Postprocess cao**: tăng `conf`, giảm class cần detect, giảm số object nhỏ/nhiễu.
- **Tracker cao**: DeepSORT embedder có thể nặng; giảm detection đầu vào, tăng `conf`, giảm `max_age`, hoặc dùng tracker nhẹ hơn nếu chỉ cần ID ngắn hạn.
- **Region logic cao**: cache polygon/bounding box, kiểm tra bbox vùng trước `point_in_polygon`, không duyệt `center` trước lane.
- **Render/UI cao**: giảm số text/polygon, update UI mỗi 2 frame, giữ đúng aspect ratio canvas, không vẽ vùng center, tránh tạo `ImageTk` quá nhiều lần.
