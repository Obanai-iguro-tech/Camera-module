from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("config.json")
DEFAULT_CAMERAS = ("Staircase", "Balcony")


@dataclass
class Rect:
    x: int = 0
    y: int = 0
    width: int = 640
    height: int = 360

    def to_mss(self) -> dict[str, int]:
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Point:
    x: int = 0
    y: int = 0

    @property
    def configured(self) -> bool:
        return self.x > 0 and self.y > 0


@dataclass
class CameraConfig:
    name: str
    capture_region: Rect = field(default_factory=Rect)
    record_button: Point = field(default_factory=Point)
    v380_window_title: str = "Cloud service"


@dataclass
class AppConfig:
    confidence_threshold: float = 0.50
    cooldown_seconds: int = 120
    inference_every_n_frames: int = 2
    capture_fps: int = 15
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            name: CameraConfig(name=name) for name in DEFAULT_CAMERAS
        }
    )


def _rect_from(data: dict[str, Any] | None) -> Rect:
    data = data or {}
    return Rect(
        x=int(data.get("x", 0)),
        y=int(data.get("y", 0)),
        width=int(data.get("width", 640)),
        height=int(data.get("height", 360)),
    )


def _point_from(data: dict[str, Any] | None) -> Point:
    data = data or {}
    return Point(x=int(data.get("x", 0)), y=int(data.get("y", 0)))


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()

    raw = json.loads(path.read_text(encoding="utf-8"))
    cameras: dict[str, CameraConfig] = {}
    for name, camera_data in raw.get("cameras", {}).items():
        cameras[name] = CameraConfig(
            name=name,
            capture_region=_rect_from(camera_data.get("capture_region")),
            record_button=_point_from(camera_data.get("record_button")),
            v380_window_title=str(camera_data.get("v380_window_title", "Cloud service")),
        )

    for name in DEFAULT_CAMERAS:
        cameras.setdefault(name, CameraConfig(name=name))

    return AppConfig(
        confidence_threshold=float(raw.get("confidence_threshold", 0.50)),
        cooldown_seconds=int(raw.get("cooldown_seconds", 120)),
        inference_every_n_frames=int(raw.get("inference_every_n_frames", 2)),
        capture_fps=int(raw.get("capture_fps", 15)),
        cameras=cameras,
    )


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
