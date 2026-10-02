from __future__ import annotations

import cv2
import mss
import numpy as np

from .config import Rect


class ScreenCapture:
    def __init__(self, region: Rect) -> None:
        self.region = region
        self._mss = mss.mss()

    def grab_bgr(self) -> np.ndarray:
        shot = self._mss.grab(self.region.to_mss())
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._mss.close()

