from __future__ import annotations

import time

import pyautogui
import pygetwindow as gw

from .config import CameraConfig


class AutomationError(RuntimeError):
    pass


class V380Automation:
    def __init__(self, extension_title: str = "V380 Smart Recorder") -> None:
        self.extension_title = extension_title

    def click_record_button(self, camera: CameraConfig) -> None:
        if not camera.record_button.configured:
            raise AutomationError(f"{camera.name} Record button is not calibrated")

        v380_window = self._find_window(camera.v380_window_title)
        if not self._window_is_usable(v380_window):
            raise AutomationError("click failed - V380 not reachable")

        try:
            v380_window.activate()
            time.sleep(0.15)
            pyautogui.click(camera.record_button.x, camera.record_button.y)
            time.sleep(0.30)
            extension_window = self._find_window(self.extension_title)
            if extension_window is not None:
                extension_window.activate()
        except Exception as exc:
            raise AutomationError(f"click failed - {exc}") from exc

    @staticmethod
    def window_available(camera: CameraConfig) -> bool:
        window = V380Automation._find_window(camera.v380_window_title)
        return V380Automation._window_is_usable(window)

    @staticmethod
    def _window_is_usable(window) -> bool:
        return window is not None and not window.isMinimized

    @staticmethod
    def _find_window(title_fragment: str):
        matches = [
            window
            for window in gw.getAllWindows()
            if title_fragment.lower() in window.title.lower()
        ]
        exact_matches = [
            window
            for window in matches
            if window.title.lower() == title_fragment.lower()
        ]
        if exact_matches:
            return exact_matches[0]
        matches.sort(key=lambda window: len(window.title))
        return matches[0] if matches else None
