import cv2
import numpy as np

VIDEO_PATH = "data_dn/split_videos/output_000.mp4"

points = []

# ===== Mouse callback =====
def mouse_callback(event, x, y, flags, param):
    global frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        print(f"Point {len(points)}: ({x}, {y})")

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

# ===== Read first frame =====
cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

if not ret:
    print("Cannot read video")
    exit()

cap.release()

# resize về 1280x720
frame = cv2.resize(frame, (1280, 720))

frame_display = frame.copy()

# ===== Window =====
cv2.namedWindow("Select Points")

cv2.setMouseCallback("Select Points", mouse_callback)

print("Left click to select points")
print("Press 'q' to finish")

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

    cv2.imshow("Select Points", temp)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("\nSelected points:")
print(points)