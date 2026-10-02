from __future__ import annotations

import sys
from datetime import datetime

import cv2
import pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets
from pynput import keyboard

from .automation import V380Automation
from .config import AppConfig, Point, Rect, load_config, save_config
from .worker import CameraWorker, LazyDetector


APP_TITLE = "V380 Smart Recorder"


class CalibrationBridge(QtCore.QObject):
    captured = QtCore.pyqtSignal()


class CameraPanel(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal(str)
    stop_requested = QtCore.pyqtSignal(str)
    manual_record_requested = QtCore.pyqtSignal(str)
    manual_stop_requested = QtCore.pyqtSignal(str)
    calibrate_requested = QtCore.pyqtSignal(str)
    region_requested = QtCore.pyqtSignal(str)
    title_requested = QtCore.pyqtSignal(str)

    def __init__(self, camera_name: str) -> None:
        super().__init__()
        self.camera_name = camera_name

        self.preview = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background:#111;color:#ddd;border:1px solid #333;")
        self.preview.setText("Preview")

        self.status = QtWidgets.QLabel("Idle")
        self.timer = QtWidgets.QLabel("--:--")

        self.start_button = QtWidgets.QPushButton("Start Monitoring")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.manual_record = QtWidgets.QPushButton("Manual Record")
        self.manual_stop = QtWidgets.QPushButton("Manual Stop")
        self.calibrate = QtWidgets.QPushButton("Calibrate Record Button")
        self.region = QtWidgets.QPushButton("Set Capture Region")
        self.title = QtWidgets.QPushButton("Set V380 Title")

        controls = QtWidgets.QHBoxLayout()
        for button in (
            self.start_button,
            self.stop_button,
            self.manual_record,
            self.manual_stop,
            self.calibrate,
            self.region,
            self.title,
        ):
            controls.addWidget(button)

        meta = QtWidgets.QHBoxLayout()
        meta.addWidget(QtWidgets.QLabel("Status:"))
        meta.addWidget(self.status, 1)
        meta.addWidget(QtWidgets.QLabel("Cooldown:"))
        meta.addWidget(self.timer)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.preview)
        layout.addLayout(meta)
        layout.addLayout(controls)

        self.start_button.clicked.connect(lambda: self.start_requested.emit(camera_name))
        self.stop_button.clicked.connect(lambda: self.stop_requested.emit(camera_name))
        self.manual_record.clicked.connect(
            lambda: self.manual_record_requested.emit(camera_name)
        )
        self.manual_stop.clicked.connect(lambda: self.manual_stop_requested.emit(camera_name))
        self.calibrate.clicked.connect(lambda: self.calibrate_requested.emit(camera_name))
        self.region.clicked.connect(lambda: self.region_requested.emit(camera_name))
        self.title.clicked.connect(lambda: self.title_requested.emit(camera_name))

    def update_status(self, status: str, remaining: int) -> None:
        self.status.setText(status)
        minutes, seconds = divmod(max(0, remaining), 60)
        self.timer.setText(f"{minutes}:{seconds:02d}" if remaining else "--:--")

    def update_frame(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QtGui.QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QtGui.QImage.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            self.preview.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig, preloaded_model=None) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(980, 720)

        self.config = config
        self.detector = LazyDetector(preloaded_model)
        self.automation = V380Automation(APP_TITLE)
        self.workers: dict[str, CameraWorker] = {}
        self.panels: dict[str, CameraPanel] = {}

        self.tabs = QtWidgets.QTabWidget()
        for camera_name in self.config.cameras:
            panel = CameraPanel(camera_name)
            self.panels[camera_name] = panel
            self.tabs.addTab(panel, camera_name)
            panel.start_requested.connect(self.start_camera)
            panel.stop_requested.connect(self.stop_camera)
            panel.manual_record_requested.connect(
                lambda name: self.manual_click(name, "Manual Record")
            )
            panel.manual_stop_requested.connect(
                lambda name: self.manual_click(name, "Manual Stop")
            )
            panel.calibrate_requested.connect(self.calibrate_button)
            panel.region_requested.connect(self.set_capture_region)
            panel.title_requested.connect(self.set_window_title)

        self.threshold = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.threshold.setRange(30, 90)
        self.threshold.setValue(int(self.config.confidence_threshold * 100))
        self.threshold.valueChanged.connect(self.update_threshold)

        self.cooldown = QtWidgets.QSpinBox()
        self.cooldown.setRange(30, 600)
        self.cooldown.setValue(self.config.cooldown_seconds)
        self.cooldown.setSuffix(" sec")
        self.cooldown.valueChanged.connect(self.update_cooldown)

        settings = QtWidgets.QHBoxLayout()
        settings.addWidget(QtWidgets.QLabel("Confidence"))
        settings.addWidget(self.threshold, 1)
        settings.addWidget(QtWidgets.QLabel("Cooldown"))
        settings.addWidget(self.cooldown)

        self.log = QtWidgets.QTableWidget(0, 4)
        self.log.setHorizontalHeaderLabels(["Time", "Camera", "Event", "Detail"])
        self.log.horizontalHeader().setStretchLastSection(True)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.addLayout(settings)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.log)
        self.setCentralWidget(root)

    def start_camera(self, camera_name: str) -> None:
        if camera_name in self.workers:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.add_log(camera_name, "Loading YOLO", "first start can take a moment")
        try:
            self.detector.warm_up()
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.add_log(camera_name, "YOLO load failed", str(exc))
            self.panels[camera_name].update_status("YOLO load failed", 0)
            return
        finally:
            if QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()

        worker = CameraWorker(
            self.config,
            self.config.cameras[camera_name],
            self.detector,
            self.automation,
        )
        worker.frame_ready.connect(self.on_frame_ready)
        worker.status_changed.connect(self.on_status_changed)
        worker.log_event.connect(self.add_log)
        self.workers[camera_name] = worker
        worker.start()
        self.add_log(camera_name, "Monitoring started", "")

    def stop_camera(self, camera_name: str) -> None:
        worker = self.workers.pop(camera_name, None)
        if worker is None:
            return
        worker.stop()
        worker.wait(1500)
        self.panels[camera_name].update_status("Idle", 0)
        self.add_log(camera_name, "Monitoring stopped", "")

    def manual_click(self, camera_name: str, label: str) -> None:
        worker = self.workers.get(camera_name)
        if worker is not None:
            worker.force_click(label)
            return
        try:
            self.automation.click_record_button(self.config.cameras[camera_name])
            self.add_log(camera_name, label, "manual override")
        except Exception as exc:
            self.add_log(camera_name, f"{label} failed", str(exc))

    def calibrate_button(self, camera_name: str) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Calibration",
            "Hover over V380's Record button, then press F8.",
        )
        loop = QtCore.QEventLoop()

        def finish() -> None:
            pos = pyautogui.position()
            self.config.cameras[camera_name].record_button = Point(pos.x, pos.y)
            save_config(self.config)
            self.showNormal()
            self.activateWindow()
            self.add_log(camera_name, "Record button calibrated", f"x={pos.x}, y={pos.y}")
            loop.quit()

        bridge = CalibrationBridge(self)
        bridge.captured.connect(finish)
        self.showMinimized()

        def on_press(key) -> bool | None:
            if key == keyboard.Key.f8:
                bridge.captured.emit()
                return False
            return None

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        loop.exec_()
        listener.stop()
        bridge.deleteLater()

    def set_capture_region(self, camera_name: str) -> None:
        existing = self.config.cameras[camera_name].capture_region
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Capture Region",
            "Enter x,y,width,height for V380 video panel:",
            text=f"{existing.x},{existing.y},{existing.width},{existing.height}",
        )
        if not ok:
            return
        try:
            x, y, width, height = [int(part.strip()) for part in text.split(",")]
            self.config.cameras[camera_name].capture_region = Rect(x, y, width, height)
            save_config(self.config)
            self.add_log(camera_name, "Capture region saved", text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Region", "Use x,y,width,height.")

    def set_window_title(self, camera_name: str) -> None:
        existing = self.config.cameras[camera_name].v380_window_title
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "V380 Window Title",
            "Enter part of the V380 window title:",
            text=existing,
        )
        if not ok:
            return
        self.config.cameras[camera_name].v380_window_title = text.strip() or "Cloud service"
        save_config(self.config)
        self.add_log(camera_name, "V380 title saved", self.config.cameras[camera_name].v380_window_title)

    def update_threshold(self, value: int) -> None:
        self.config.confidence_threshold = value / 100
        save_config(self.config)

    def update_cooldown(self, value: int) -> None:
        self.config.cooldown_seconds = value
        save_config(self.config)

    def on_frame_ready(self, camera_name: str, frame_bgr) -> None:
        self.panels[camera_name].update_frame(frame_bgr)

    def on_status_changed(self, camera_name: str, status: str, remaining: int) -> None:
        self.panels[camera_name].update_status(status, remaining)

    def add_log(self, camera_name: str, event: str, detail: str) -> None:
        row = self.log.rowCount()
        self.log.insertRow(row)
        values = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            camera_name,
            event,
            detail,
        ]
        for column, value in enumerate(values):
            self.log.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self.log.scrollToBottom()

    def closeEvent(self, event) -> None:
        for camera_name in list(self.workers):
            self.stop_camera(camera_name)
        save_config(self.config)
        super().closeEvent(event)


def main(preloaded_model=None) -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(load_config(), preloaded_model)
    window.show()
    sys.exit(app.exec_())
