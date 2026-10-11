"""Stroke engine: structured strokes rendered onto a fixed-resolution canvas.

Shared by gesture and mouse input. Has no Qt dependency.
Images are NumPy arrays in RGB order, shape (height, width, 3), dtype uint8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Color = tuple[int, int, int]  # RGB
Point = tuple[int, int]

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
WHITE: Color = (255, 255, 255)
MIN_BRUSH_WIDTH = 1
MAX_BRUSH_WIDTH = 64


@dataclass
class Stroke:
    color: Color
    width: int
    points: list[Point] = field(default_factory=list)


def draw_stroke(image: np.ndarray, stroke: Stroke) -> None:
    """Draw a stroke with rounded ends and joins. A single point becomes a dot."""
    if not stroke.points:
        return
    color = tuple(int(c) for c in stroke.color)
    radius = max(1, stroke.width // 2)
    if len(stroke.points) == 1:
        cv2.circle(image, stroke.points[0], radius, color, -1, cv2.LINE_AA)
        return
    pts = np.asarray(stroke.points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], False, color, max(1, stroke.width), cv2.LINE_AA)
    # Round caps and joins regardless of OpenCV's thick-line implementation.
    if stroke.width > 2:
        for p in stroke.points:
            cv2.circle(image, p, radius, color, -1, cv2.LINE_AA)


class StrokeCanvas:
    """Holds strokes, supports undo/clear, renders to an RGB image.

    Memory is bounded:
    * at most ``max_points_per_stroke`` points per stroke (longer strokes are
      continued as a new stroke);
    * at most ``max_strokes`` undoable strokes; older strokes are baked into a
      base image and can no longer be undone individually.
    Undo re-renders stored strokes instead of keeping full-canvas snapshots.
    """

    def __init__(
        self,
        width: int = CANVAS_WIDTH,
        height: int = CANVAS_HEIGHT,
        background: Color = WHITE,
        max_strokes: int = 300,
        max_points_per_stroke: int = 4000,
        min_point_distance: float = 1.0,
    ) -> None:
        self.width = width
        self.height = height
        self.background = background
        self.max_strokes = max_strokes
        self.max_points_per_stroke = max_points_per_stroke
        self.min_point_distance = min_point_distance
        self.strokes: list[Stroke] = []
        self.active: Stroke | None = None
        self.version = 0
        self._base = self._blank()
        self._committed = self._base.copy()

    # ----------------------------------------------------------------- helpers
    def _blank(self) -> np.ndarray:
        img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.background
        return img

    def _clamp(self, point: tuple[float, float]) -> Point:
        x = int(round(min(max(point[0], 0), self.width - 1)))
        y = int(round(min(max(point[1], 0), self.height - 1)))
        return (x, y)

    def _rerender_committed(self) -> None:
        self._committed = self._base.copy()
        for stroke in self.strokes:
            draw_stroke(self._committed, stroke)

    def _touch(self) -> None:
        self.version += 1

    # --------------------------------------------------------------- strokes
    @property
    def is_drawing(self) -> bool:
        return self.active is not None

    def begin_stroke(self, point: tuple[float, float], color: Color, width: int) -> None:
        if self.active is not None:
            self.end_stroke()
        width = int(min(max(width, MIN_BRUSH_WIDTH), MAX_BRUSH_WIDTH))
        self.active = Stroke(color=tuple(int(c) for c in color), width=width, points=[self._clamp(point)])
        self._touch()

    def add_point(self, point: tuple[float, float]) -> bool:
        """Append a point to the active stroke. Returns True if it was added."""
        if self.active is None:
            return False
        p = self._clamp(point)
        last = self.active.points[-1]
        if (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2 < self.min_point_distance ** 2:
            return False
        if len(self.active.points) >= self.max_points_per_stroke:
            color, width = self.active.color, self.active.width
            self.end_stroke()
            self.active = Stroke(color=color, width=width, points=[last])
        self.active.points.append(p)
        self._touch()
        return True

    def end_stroke(self) -> None:
        if self.active is None:
            return
        stroke = self.active
        self.active = None
        self.strokes.append(stroke)
        draw_stroke(self._committed, stroke)
        if len(self.strokes) > self.max_strokes:
            oldest = self.strokes.pop(0)
            draw_stroke(self._base, oldest)
        self._touch()

    def set_brush(self, color: Color, width: int) -> None:
        """Apply new brush settings. An active stroke is ended and continued
        from its last point as a new stroke with the new settings."""
        if self.active is None:
            return
        last = self.active.points[-1]
        self.end_stroke()
        self.begin_stroke(last, color, width)

    def undo(self) -> bool:
        if self.active is not None:
            self.active = None
            self._touch()
            return True
        if not self.strokes:
            return False
        self.strokes.pop()
        self._rerender_committed()
        self._touch()
        return True

    def clear(self) -> None:
        self.active = None
        self.strokes.clear()
        self._base = self._blank()
        self._committed = self._base.copy()
        self._touch()

    @property
    def can_undo(self) -> bool:
        return self.active is not None or bool(self.strokes)

    # -------------------------------------------------------------- output
    def render(self) -> np.ndarray:
        """Return the full drawing (committed + active stroke), RGB."""
        if self.active is None:
            return self._committed.copy()
        image = self._committed.copy()
        draw_stroke(image, self.active)
        return image

    def encode_png(self) -> bytes:
        bgr = cv2.cvtColor(self.render(), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise OSError("Не удалось закодировать изображение в PNG")
        return buf.tobytes()

    def save_png(self, path: str | Path) -> Path:
        """Save the canvas only (no cursor, preview or UI). Works with
        non-ASCII paths, unlike cv2.imwrite on Windows."""
        path = Path(path)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        data = self.encode_png()
        path.write_bytes(data)
        return path
