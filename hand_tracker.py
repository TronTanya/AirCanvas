"""MediaPipe Hand Landmarker wrapper (Tasks API, VIDEO running mode)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gestures import HandObservation, to_display_landmarks

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

SETUP_HINT = (
    "Модель отслеживания руки не найдена:\n{path}\n\n"
    "Загрузите её один раз (нужен интернет), затем приложение работает офлайн:\n"
    "    python download_model.py\n\n"
    "Пока модель не загружена, можно рисовать мышью."
)


class ModelNotFoundError(FileNotFoundError):
    def __init__(self, path: Path) -> None:
        super().__init__(SETUP_HINT.format(path=path))
        self.path = path


def model_available(path: Path = MODEL_PATH) -> bool:
    return path.is_file() and path.stat().st_size > 0


@dataclass(frozen=True)
class TrackingResult:
    observation: HandObservation | None  # display (mirrored) coordinates
    inference_ms: float


class HandTracker:
    """Detects one hand per frame. Not thread-safe: use from a single thread."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_available(model_path):
            raise ModelNotFoundError(model_path)

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        # Pass the model as bytes: robust to non-ASCII paths on Windows.
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def _next_timestamp_ms(self) -> int:
        ts = time.monotonic_ns() // 1_000_000
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def process(self, frame_rgb: np.ndarray, mirror: bool = True) -> TrackingResult:
        """Run inference on a raw (non-mirrored) RGB frame."""
        if self._landmarker is None:
            raise RuntimeError("HandTracker уже закрыт")
        height, width = frame_rgb.shape[:2]
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        ts = self._next_timestamp_ms()
        t0 = time.perf_counter()
        result = self._landmarker.detect_for_video(image, ts)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        if not result.hand_landmarks:
            return TrackingResult(None, inference_ms)

        raw = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
        handedness = None
        score = 1.0
        if result.handedness and result.handedness[0]:
            category = result.handedness[0][0]
            handedness = category.category_name
            score = float(category.score)
        obs = HandObservation(
            landmarks=to_display_landmarks(raw, mirror=mirror),
            handedness=handedness,
            score=score,
            image_aspect=width / height if height else 1.0,
        )
        return TrackingResult(obs, inference_ms)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
