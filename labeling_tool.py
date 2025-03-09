import sys
import json
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QMessageBox, QScrollArea, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtCore import Qt, QPoint, QPointF

ANNOTATIONS_DIR = "annotations"
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.polygons = []       # 확정된 폴리곤 목록 (각각 "points"만 저장)
        self.current_polygon = []  # 현재 그리고 있는 폴리곤의 점들
        self.mouse_position = None

    def setImage(self, cv_image):
        """OpenCV(RGB) 이미지를 QPixmap으로 변환하여 QLabel에 표시"""
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_image))

        # 폴리곤 정보 초기화
        self.polygons.clear()
        self.current_polygon.clear()
        self.mouse_position = None
        self.update()

    def mousePressEvent(self, event):
        """마우스 클릭 시 현재 폴리곤에 점 추가"""
        if self.pixmap() is not None and event.button() == Qt.LeftButton:
            x, y = event.pos().x(), event.pos().y()
            # 이미지 범위 안에서만 점 추가
            if 0 <= x < self.pixmap().width() and 0 <= y < self.pixmap().height():
                self.current_polygon.append((x, y))
                self.update()

    def mouseMoveEvent(self, event):
        """마우스 이동 시 마지막 점과 연결되는 점선 표시를 위해 현재 좌표 갱신"""
        if self.pixmap() is not None:
            x, y = event.pos().x(), event.pos().y()
            if 0 <= x < self.pixmap().width() and 0 <= y < self.pixmap().height():
                self.mouse_position = (x, y)
                self.update()

    def paintEvent(self, event):
        """확정된 폴리곤과 그리고 있는 폴리곤을 화면에 그려줌"""
        super().paintEvent(event)
        if self.pixmap() is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 확정된 폴리곤 그리기
        pen_confirmed = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen_confirmed)
        for poly_data in self.polygons:
            points = poly_data["points"]

            if len(points) > 2:
                polygon_q = QPolygonF([QPointF(x, y) for x, y in points])
                painter.setBrush(Qt.red)
                painter.setOpacity(0.3)
                painter.drawPolygon(polygon_q)
                painter.setOpacity(1.0)

                for i in range(len(points)):
                    p1 = points[i]
                    p2 = points[(i+1) % len(points)]
                    painter.drawLine(QPoint(*p1), QPoint(*p2))
            else:
                for i in range(len(points) - 1):
                    painter.drawLine(QPoint(*points[i]), QPoint(*points[i+1]))

        # 현재 그리고 있는 폴리곤 그리기
        if self.current_polygon:
            pen_current = QPen(Qt.blue, 2, Qt.SolidLine)
            painter.setPen(pen_current)
            for i in range(len(self.current_polygon) - 1):
                painter.drawLine(QPoint(*self.current_polygon[i]), QPoint(*self.current_polygon[i+1]))

            if self.mouse_position:
                dashed_pen = QPen(Qt.blue, 2, Qt.DashLine)
                painter.setPen(dashed_pen)
                painter.drawLine(QPoint(*self.current_polygon[-1]), QPoint(*self.mouse_position))

class SegmentationTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Nail Segmentation Tool")
        self.setGeometry(100, 100, 900, 700)

        # 이미지 목록
        self.image_files = []
        self.current_image_index = 0
        self.current_image_path = None

        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)

        self.btn_open = QPushButton("이미지 폴더 열기")
        self.btn_open.clicked.connect(self.load_images)

        self.btn_next = QPushButton("다음 이미지")
        self.btn_next.clicked.connect(self.load_next_image)
        self.btn_next.setEnabled(False)

        self.btn_save = QPushButton("JSON 저장")
        self.btn_save.clicked.connect(self.save_json)
        self.btn_save.setEnabled(False)

        self.btn_undo_point = QPushButton("되돌리기 (마지막 점)")
        self.btn_undo_point.clicked.connect(self.undo_last_point)

        self.btn_undo_polygon = QPushButton("마지막 폴리곤 삭제")
        self.btn_undo_polygon.clicked.connect(self.undo_last_polygon)

        self.btn_finalize_polygon = QPushButton("폴리곤 확정")
        self.btn_finalize_polygon.clicked.connect(self.finalize_polygon)

        layout_main = QVBoxLayout()
        layout_main.addWidget(self.scroll_area)

        layout_controls = QHBoxLayout()
        layout_controls.addWidget(self.btn_open)
        layout_controls.addWidget(self.btn_next)
        layout_controls.addWidget(self.btn_save)

        layout_polygon = QHBoxLayout()
        layout_polygon.addWidget(self.btn_undo_point)
        layout_polygon.addWidget(self.btn_undo_polygon)
        layout_polygon.addWidget(self.btn_finalize_polygon)

        layout_main.addLayout(layout_controls)
        layout_main.addLayout(layout_polygon)

        self.setLayout(layout_main)

    def load_images(self):
        """이미지 폴더 선택 후 이미지 파일 리스트 불러오기"""
        folder_path = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if folder_path:
            self.image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and 
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if self.image_files:
                self.current_image_index = 0
                self.load_image(self.image_files[self.current_image_index])
                self.btn_next.setEnabled(True)
                self.btn_save.setEnabled(True)
            else:
                QMessageBox.warning(self, "경고", "선택한 폴더에 이미지 파일이 없습니다.")

    def load_image(self, file_path):
        """단일 이미지 로드 및 화면에 맞게 리사이즈"""
        self.current_image_path = file_path
        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "오류", "이미지를 불러올 수 없습니다.")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        screen_width, screen_height = 1920, 1080
        height, width, _ = image.shape
        scale_factor = min(screen_width / width, screen_height / height, 1.0)
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        self.image_label.setImage(resized)

    def load_next_image(self):
        """다음 이미지 로드"""
        if self.current_image_index + 1 < len(self.image_files):
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])
        else:
            QMessageBox.information(self, "완료", "모든 이미지를 라벨링했습니다.")

    def undo_last_point(self):
        """현재 그리고 있는 폴리곤의 마지막 점 되돌리기"""
        if self.image_label.current_polygon:
            self.image_label.current_polygon.pop()
            self.image_label.update()

    def undo_last_polygon(self):
        """마지막으로 확정된 폴리곤 삭제"""
        if self.image_label.polygons:
            self.image_label.polygons.pop()
            self.image_label.update()

    def finalize_polygon(self):
        """현재 그리고 있는 폴리곤을 확정하여 목록에 저장"""
        if len(self.image_label.current_polygon) < 3:
            QMessageBox.warning(self, "폴리곤 오류", "폴리곤은 최소 3개 이상의 점이 필요합니다.")
            return

        polygon_data = {
            "points": self.image_label.current_polygon.copy()
        }
        self.image_label.polygons.append(polygon_data)
        self.image_label.current_polygon.clear()
        self.image_label.update()

    def save_json(self):
        """JSON 파일을 annotations 폴더에 저장"""
        if not self.current_image_path:
            QMessageBox.warning(self, "오류", "저장할 이미지가 없습니다.")
            return

        if not self.image_label.polygons:
            QMessageBox.warning(self, "오류", "저장할 폴리곤이 없습니다.")
            return

        data = {
            "image": self.current_image_path,
            "polygons": self.image_label.polygons
        }

        filename = os.path.basename(self.current_image_path)
        json_filename = os.path.splitext(filename)[0] + "_annotations.json"
        json_path = os.path.join(ANNOTATIONS_DIR, json_filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        QMessageBox.information(self, "저장 완료", f"JSON 파일이 저장되었습니다:\n{json_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tool = SegmentationTool()
    tool.show()
    sys.exit(app.exec_())
