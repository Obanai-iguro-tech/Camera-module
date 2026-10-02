from __future__ import annotations

import time
import threading
from enum import Enum

import numpy as np
from PyQt5 import QtCore

from .automation import AutomationError, V380Automation
from .capture import ScreenCapture
from .config import AppConfig, CameraConfig
from .detection import Detection, Detector, draw_detections


class RecorderState(str, Enum):
    IDLE = "Idle"
    RECORDING = "Recording via V380"
    COOLDOWN = "Cooldown"
    WARNING = "Worker error"


class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(str, object)
    status_changed = QtCore.pyqtSignal(str, str, int)
    log_event = QtCore.pyqtSignal(str, str, str)

    def __init__(
        self,
        app_config: AppConfig,
        camera_config: CameraConfig,
        detector: Detector,
        automation: V380Automation,
    ) -> None:
        super().__init__()
        self.app_config = app_config
        self.camera_config = camera_config
        self.detector = detector
        self.automation = automation
        self._running = False
        self._state = RecorderState.IDLE
        self._last_person_seen = 0.0
        self._cooldown_started = 0.0
        self._frame_count = 0

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        capture = ScreenCapture(self.camera_config.capture_region)
        frame_delay = 1.0 / max(1, self.app_config.capture_fps)
        detections: list[Detection] = []

        try:
            while self._running:
                loop_started = time.monotonic()
                frame = capture.grab_bgr()
                self._frame_count += 1

                if self._frame_count % max(1, self.app_config.inference_every_n_frames) == 0:
                    detections = self.detector.detect(
                        frame,
                        self.app_config.confidence_threshold,
                    )
                    self._advance_state(detections)

                preview = draw_detections(frame, detections)
                self.frame_ready.emit(self.camera_config.name, preview)

                elapsed = time.monotonic() - loop_started
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
        except Exception as exc:
            self._set_status(RecorderState.WARNING, 0)
            self.log_event.emit(self.camera_config.name, "Worker stopped", str(exc))
        finally:
            capture.close()

    def force_click(self, label: str) -> None:
        try:
            self.automation.click_record_button(self.camera_config)
            self.log_event.emit(self.camera_config.name, label, "manual override")
        except AutomationError as exc:
            self.log_event.emit(self.camera_config.name, "Manual click failed", str(exc))

    def _advance_state(self, detections: list[Detection]) -> None:
        now = time.monotonic()
        person_present = bool(detections)
        best_confidence = max((d.confidence for d in detections), default=0.0)

        if person_present:
            self._last_person_seen = now
            if self._state == RecorderState.IDLE:
                if self._click_or_retry("Record started", best_confidence):
                    self._state = RecorderState.RECORDING
                else:
                    self._set_status(RecorderState.WARNING, 0)
                    return
            elif self._state == RecorderState.COOLDOWN:
                self._state = RecorderState.RECORDING
            self._set_status(RecorderState.RECORDING, self.app_config.cooldown_seconds)
            return

        person_absent_for = now - self._last_person_seen
        if self._state == RecorderState.RECORDING and person_absent_for >= 1.0:
            self._state = RecorderState.COOLDOWN
            self._cooldown_started = now

        if self._state == RecorderState.COOLDOWN:
            remaining = max(
                0,
                int(self.app_config.cooldown_seconds - (now - self._cooldown_started)),
            )
            self._set_status(RecorderState.COOLDOWN, remaining)
            if remaining <= 0:
                if self._click_or_retry("Record stopped", best_confidence):
                    self._state = RecorderState.IDLE
                    self._set_status(RecorderState.IDLE, 0)
            return

        self._set_status(RecorderState.IDLE, 0)

    def _click_or_retry(self, event: str, confidence: float) -> bool:
        try:
            self.automation.click_record_button(self.camera_config)
            detail = f"confidence={confidence:.0%}" if confidence else "cooldown expired"
            self.log_event.emit(self.camera_config.name, event, detail)
            return True
        except AutomationError as exc:
            self._set_status(RecorderState.WARNING, 0)
            self.log_event.emit(self.camera_config.name, event, str(exc))
            time.sleep(5)
            return False

    def _set_status(self, state: RecorderState, remaining: int) -> None:
        self.status_changed.emit(self.camera_config.name, state.value, remaining)


class LazyDetector:
    def __init__(self, preloaded_model=None) -> None:
        self._detector: Detector | None = None
        self._lock = threading.Lock()
        self._preloaded_model = preloaded_model

    def detect(self, frame_bgr: np.ndarray, threshold: float) -> list[Detection]:
        self.warm_up()
        with self._lock:
            assert self._detector is not None
            return self._detector.detect(frame_bgr, threshold)

    def warm_up(self) -> None:
        with self._lock:
            if self._detector is None:
                from .detection import YoloPersonDetector

                self._detector = YoloPersonDetector(model=self._preloaded_model)
                self._preloaded_model = None
