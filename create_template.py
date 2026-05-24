# create_template.py
import argparse
import csv
import os
import cv2
import numpy as np


WINDOW_NAME = "Template Creator"


class TemplateCreator:
    def __init__(self, video_path, output_path="template.csv", max_display_width=1280):
        self.video_path = video_path
        self.output_path = output_path
        self.max_display_width = max_display_width

        self.frame = None
        self.display_frame = None
        self.scale = 1.0

        self.points = []
        self.saved_regions = []

    def load_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {self.video_path}")

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError("Không đọc được frame đầu tiên.")

        self.frame = frame
        h, w = frame.shape[:2]

        if w > self.max_display_width:
            self.scale = self.max_display_width / w
            self.display_frame = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))
        else:
            self.scale = 1.0
            self.display_frame = frame.copy()

    def display_to_original_point(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def original_to_display_point(self, x, y):
        return int(x * self.scale), int(y * self.scale)

    def draw_polygon(self, canvas, points, color, label=None):
        if len(points) < 2:
            return

        display_points = [self.original_to_display_point(x, y) for x, y in points]

        for i in range(len(display_points) - 1):
            cv2.line(canvas, display_points[i], display_points[i + 1], color, 2)

        if len(display_points) == 4:
            cv2.line(canvas, display_points[3], display_points[0], color, 2)

        for idx, p in enumerate(display_points):
            cv2.circle(canvas, p, 5, color, -1)
            cv2.putText(
                canvas,
                str(idx + 1),
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        if label and len(display_points) == 4:
            cx = sum(p[0] for p in display_points) // 4
            cy = sum(p[1] for p in display_points) // 4

            cv2.rectangle(canvas, (cx - 50, cy - 24), (cx + 50, cy + 8), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                label.upper(),
                (cx - 42, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    def draw(self):
        canvas = self.display_frame.copy()

        overlay = canvas.copy()

        for region_name, pts in self.saved_regions:
            dpts = np.array([self.original_to_display_point(x, y) for x, y in pts], dtype=np.int32)
            cv2.fillPoly(overlay, [dpts], (80, 180, 80))

        canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

        for region_name, pts in self.saved_regions:
            self.draw_polygon(canvas, pts, (60, 180, 60), region_name)

        self.draw_polygon(canvas, self.points, (0, 0, 255))

        help_lines = [
            "Left click: choose point",
            "s: save 4 points",
            "u: undo | r: reset current",
            "q: quit",
        ]

        y = 28
        for line in help_lines:
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3)
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1)
            y += 26

        cv2.imshow(WINDOW_NAME, canvas)

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.points) >= 4:
            print("Đã đủ 4 điểm. Nhấn 's' để lưu hoặc 'r' để chọn lại.")
            return

        ox, oy = self.display_to_original_point(x, y)
        self.points.append((ox, oy))
        print(f"Point {len(self.points)} = ({ox}, {oy})")
        self.draw()

    def ensure_csv_header(self):
        h, w = self.frame.shape[:2]

        if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
            return

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([w, h])

    def save_current_region(self):
        if len(self.points) != 4:
            print("Bạn cần chọn đúng 4 điểm.")
            return

        region_name = input("Nhập tên region top/left/right/bottom/center: ").strip().lower()

        if not region_name:
            print("Tên region rỗng, bỏ qua.")
            return

        self.ensure_csv_header()

        row = [region_name]
        for x, y in self.points:
            row.extend([x, y])

        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.saved_regions.append((region_name, self.points.copy()))
        print(f"Đã lưu: {row}")

        self.points.clear()
        self.draw()

    def run(self):
        self.load_first_frame()

        print("Click 4 điểm theo thứ tự quanh vùng, ví dụ thuận chiều kim đồng hồ.")
        print("Phím: s=lưu, u=undo, r=reset, q=thoát.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        self.draw()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                self.save_current_region()
            elif key == ord("u"):
                if self.points:
                    print("Undo:", self.points.pop())
                    self.draw()
            elif key == ord("r"):
                self.points.clear()
                print("Reset current region.")
                self.draw()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Đường dẫn video")
    parser.add_argument("--out", default="template.csv", help="File template CSV output")
    parser.add_argument("--max-display-width", type=int, default=1280)
    args = parser.parse_args()

    app = TemplateCreator(args.video, args.out, args.max_display_width)
    app.run()


if __name__ == "__main__":
    main()