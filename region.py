import cv2
import numpy as np

VIDEO_PATH = "data_dn/split_videos/output_000.mp4"

points = []

# ===== Callback chuột =====
def mouse_callback(event, x, y, flags, param):
    global frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        print(f"Điểm {len(points)}: ({x}, {y})")

        # vẽ điểm
        cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)

        # hiện tọa độ
        cv2.putText(
            frame_display,
            f"{x},{y}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

# ===== Đọc frame đầu tiên =====
cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

if not ret:
    print("Không đọc được video")
    exit()

cap.release()

# resize về 1280x720
frame = cv2.resize(frame, (1280, 720))

frame_display = frame.copy()

# ===== Cửa sổ =====
cv2.namedWindow("Chọn điểm")

cv2.setMouseCallback("Chọn điểm", mouse_callback)

print("Click chuột trái để chọn điểm")
print("Nhấn 'q' để kết thúc")

while True:

    temp = frame_display.copy()

    # nối các điểm
    if len(points) > 1:
        cv2.polylines(
            temp,
            [np.array(points, dtype=np.int32)],
            False,
            (255, 0, 0),
            2
        )

    cv2.imshow("Chọn điểm", temp)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("\nCác điểm đã chọn:")
print(points)