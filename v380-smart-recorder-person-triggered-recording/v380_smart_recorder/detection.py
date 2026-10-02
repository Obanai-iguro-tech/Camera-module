from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray, threshold: float) -> list[Detection]:
        ...


class YoloPersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", model=None) -> None:
        from ultralytics import YOLO

        self._model = model if model is not None else YOLO(model_name)
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass

    def detect(self, frame_bgr: np.ndarray, threshold: float) -> list[Detection]:
        results = self._model.predict(frame_bgr, verbose=False, conf=threshold)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                if class_id != PERSON_CLASS_ID or confidence < threshold:
                    continue
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detections.append(Detection(x1, y1, x2, y2, confidence))
        return detections


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    preview = frame_bgr.copy()
    for detection in detections:
        cv2.rectangle(
            preview,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 220, 80),
            2,
        )
        label = f"Person {detection.confidence:.0%}"
        cv2.putText(
            preview,
            label,
            (detection.x1, max(18, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )
    return preview
