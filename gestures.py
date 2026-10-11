"""Gesture logic: coordinate mapping, smoothing, pinch detection with hysteresis.

Coordinate convention used throughout AirCanvas
-----------------------------------------------
* MediaPipe returns landmarks normalized to the *raw* camera frame:
  x in [0, 1] from the left edge of the raw image, y in [0, 1] from the top.
* The preview is mirrored (like a mirror / selfie view). All landmarks are
  converted ONCE into "display coordinates" with ``to_display_landmarks``:
  x_display = 1 - x_raw, y_display = y_raw.
* From then on everything (preview overlay, cursor, canvas) uses display
  coordinates, so moving the hand to the user's right moves the cursor right.
* Canvas coordinates are pixels of the fixed internal canvas (e.g. 1280x720),
  origin at the top-left corner.

This module has no dependency on Qt, OpenCV or MediaPipe and is fully testable
with synthetic landmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

NormPoint = tuple[float, float]
CanvasPoint = tuple[float, float]

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
NUM_LANDMARKS = 21

DEFAULT_PRESS_THRESHOLD = 0.30
DEFAULT_RELEASE_THRESHOLD = 0.45
DEFAULT_SMOOTHING_ALPHA = 0.5
DEFAULT_MARGIN = 0.10
MIN_HAND_SIZE = 1e-4


@dataclass(frozen=True)
class HandObservation:
    """One detected hand in display (mirrored) normalized coordinates."""

    landmarks: tuple[NormPoint, ...]
    handedness: str | None = None
    score: float = 1.0
    image_aspect: float = 4 / 3  # frame width / frame height


@dataclass(frozen=True)
class GestureFrame:
    """Result of interpreting one processed camera frame."""

    hand_detected: bool
    cursor: CanvasPoint | None
    drawing: bool
    stroke_break: bool  # handle BEFORE ``drawing``: end the current stroke
    pinch_ratio: float | None = None
    raw_cursor: CanvasPoint | None = None  # before smoothing, for diagnostics


def to_display_landmarks(raw: Sequence[NormPoint], mirror: bool = True) -> tuple[NormPoint, ...]:
    """Convert raw-frame normalized landmarks into display coordinates."""
    if mirror:
        return tuple((1.0 - x, y) for x, y in raw)
    return tuple((x, y) for x, y in raw)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def map_to_canvas(
    point: NormPoint,
    canvas_width: int,
    canvas_height: int,
    margin: float = DEFAULT_MARGIN,
) -> CanvasPoint:
    """Map a display-normalized point to canvas pixels, clamped to bounds.

    ``margin`` defines an inner "active region" of the camera frame:
    the range [margin, 1 - margin] is stretched to the full canvas, so the user
    can reach the canvas edges without moving the fingertip out of the frame.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin должен быть в диапазоне [0, 0.5)")
    span = 1.0 - 2.0 * margin
    u = clamp((point[0] - margin) / span, 0.0, 1.0)
    v = clamp((point[1] - margin) / span, 0.0, 1.0)
    return (u * (canvas_width - 1), v * (canvas_height - 1))


def pinch_ratio(
    landmarks: Sequence[NormPoint],
    image_aspect: float = 1.0,
    min_hand_size: float = MIN_HAND_SIZE,
) -> float | None:
    """Thumb-tip/index-tip distance normalized by wrist/middle-MCP distance.

    x is multiplied by ``image_aspect`` so distances are isotropic in pixels.
    Returns ``None`` when the hand size is too small to normalize safely.
    """
    if len(landmarks) < NUM_LANDMARKS:
        return None

    def dist(a: NormPoint, b: NormPoint) -> float:
        return math.hypot((a[0] - b[0]) * image_aspect, a[1] - b[1])

    hand_size = dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if not math.isfinite(hand_size) or hand_size < min_hand_size:
        return None
    ratio = dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_size
    return ratio if math.isfinite(ratio) else None


class PinchDetector:
    """Pinch state machine with hysteresis.

    * Not pinched -> pinched when ratio < press_threshold.
    * Pinched -> not pinched when ratio > release_threshold.
    * Between the thresholds the previous state is retained.
    """

    def __init__(
        self,
        press_threshold: float = DEFAULT_PRESS_THRESHOLD,
        release_threshold: float = DEFAULT_RELEASE_THRESHOLD,
    ) -> None:
        self.set_thresholds(press_threshold, release_threshold)
        self.pinched = False

    def set_thresholds(self, press_threshold: float, release_threshold: float) -> None:
        if press_threshold <= 0 or release_threshold <= 0:
            raise ValueError("Пороги должны быть положительными")
        if press_threshold >= release_threshold:
            raise ValueError("Порог сведения должен быть меньше порога разведения")
        self.press_threshold = press_threshold
        self.release_threshold = release_threshold

    def update(self, ratio: float | None) -> bool:
        if ratio is None:
            self.pinched = False
        elif self.pinched:
            if ratio > self.release_threshold:
                self.pinched = False
        elif ratio < self.press_threshold:
            self.pinched = True
        return self.pinched

    def reset(self) -> None:
        self.pinched = False


class ExponentialSmoother:
    """Exponential moving average for 2D points.

    s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    alpha close to 1: little smoothing, minimal delay, more jitter.
    alpha close to 0: strong smoothing, stable cursor, noticeable lag.
    The time constant is about -1 / ln(1 - alpha) frames.
    """

    def __init__(self, alpha: float = DEFAULT_SMOOTHING_ALPHA) -> None:
        self.alpha = alpha
        self._state: CanvasPoint | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError("Коэффициент сглаживания должен быть в диапазоне (0, 1]")
        self._alpha = value

    @property
    def state(self) -> CanvasPoint | None:
        return self._state

    def update(self, point: CanvasPoint) -> CanvasPoint:
        if self._state is None:
            self._state = point
        else:
            a = self._alpha
            self._state = (
                a * point[0] + (1.0 - a) * self._state[0],
                a * point[1] + (1.0 - a) * self._state[1],
            )
        return self._state

    def reset(self) -> None:
        self._state = None


def smoothing_time_constant(alpha: float) -> float:
    """Number of frames for the EMA to cover ~63% of a step change."""
    if alpha >= 1.0:
        return 0.0
    return -1.0 / math.log(1.0 - alpha)


class GestureInterpreter:
    """Turns per-frame hand observations into cursor and drawing state.

    The stroke must be ended whenever ``stroke_break`` is True: on tracking
    loss, on uncertain hand identity, or when the pinch is released.
    """

    def __init__(
        self,
        canvas_width: int,
        canvas_height: int,
        alpha: float = DEFAULT_SMOOTHING_ALPHA,
        press_threshold: float = DEFAULT_PRESS_THRESHOLD,
        release_threshold: float = DEFAULT_RELEASE_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
        max_wrist_jump: float = 0.25,
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.margin = margin
        self.max_wrist_jump = max_wrist_jump
        self.smoother = ExponentialSmoother(alpha)
        self.pinch = PinchDetector(press_threshold, release_threshold)
        self._prev_wrist: NormPoint | None = None
        self._prev_handedness: str | None = None
        self._was_drawing = False

    def reset(self) -> None:
        self.smoother.reset()
        self.pinch.reset()
        self._prev_wrist = None
        self._prev_handedness = None
        self._was_drawing = False

    def _identity_changed(self, obs: HandObservation) -> bool:
        if self._prev_wrist is None:
            return False
        if (
            obs.handedness is not None
            and self._prev_handedness is not None
            and obs.handedness != self._prev_handedness
        ):
            return True
        wx, wy = obs.landmarks[WRIST]
        px, py = self._prev_wrist
        return math.hypot(wx - px, wy - py) > self.max_wrist_jump

    def update(self, obs: HandObservation | None) -> GestureFrame:
        if obs is None or len(obs.landmarks) < NUM_LANDMARKS:
            self.reset()
            return GestureFrame(False, None, False, stroke_break=True)

        identity_break = self._identity_changed(obs)
        if identity_break:
            self.smoother.reset()
            self.pinch.reset()

        self._prev_wrist = obs.landmarks[WRIST]
        self._prev_handedness = obs.handedness

        raw_cursor = map_to_canvas(
            obs.landmarks[INDEX_TIP], self.canvas_width, self.canvas_height, self.margin
        )
        cursor = self.smoother.update(raw_cursor)

        ratio = pinch_ratio(obs.landmarks, obs.image_aspect)
        drawing = self.pinch.update(ratio)
        stroke_break = identity_break or (self._was_drawing and not drawing)
        self._was_drawing = drawing
        return GestureFrame(True, cursor, drawing, stroke_break, ratio, raw_cursor)
