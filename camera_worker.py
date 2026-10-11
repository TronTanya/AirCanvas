"""Webcam capture and hand inference off the GUI thread.

* ``FrameGrabber`` (plain thread) reads frames continuously and keeps only the
  most recent one, so stale frames never pile up.
* ``CameraWorker`` (QThread) takes the latest frame, runs the hand landmarker
  and emits a ``FrameResult``. It waits until the GUI acknowledges the
  previous result, so the Qt event queue never builds a backlog either.
* No frame is ever written to disk or sent anywhere.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from gestures import HandObservation
from hand_tracker import HandTracker, ModelNotFoundError

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

OPEN_TIMEOUT_S = 4.0
READ_FAILURE_TIMEOUT_S = 2.5


@dataclass(frozen=True)
class FrameResult:
    preview: QImage
    observation: HandObservation | None
    inference_ms: float
    timestamp_s: float
    frame_width: int
    frame_height: int


class FrameGrabber(threading.Thread):
    """Continuously reads the camera and keeps only the latest frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        super().__init__(name="AirCanvasFrameGrabber", daemon=True)
        self._capture = capture
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._running = True
        self.failed = False

    def run(self) -> None:
        last_ok = time.perf_counter()
        while self._running:
            ok, frame = self._capture.read()
            now = time.perf_counter()
            if ok and frame is not None:
                last_ok = now
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
            else:
                if now - last_ok > READ_FAILURE_TIMEOUT_S:
                    self.failed = True
                    with self._cond:
                        self._cond.notify_all()
                    return
                time.sleep(0.01)

    def latest(self, after_seq: int, timeout: float) -> tuple[np.ndarray | None, int]:
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq > after_seq or self.failed or not self._running,
                timeout=timeout,
            )
            if self._seq > after_seq and self._frame is not None:
                return self._frame, self._seq
            return None, after_seq

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    """Open a camera, preferring DirectShow on Windows (fast startup)."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        deadline = time.perf_counter() + OPEN_TIMEOUT_S
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.05)
        cap.release()
    return None


def draw_preview(frame_rgb: np.ndarray, obs: HandObservation | None, width: int) -> QImage:
    """Mirrored, downscaled preview with landmarks drawn in display coordinates."""
    mirrored = cv2.flip(frame_rgb, 1)
    h, w = mirrored.shape[:2]
    height = max(1, int(round(h * width / w)))
    small = cv2.resize(mirrored, (width, height), interpolation=cv2.INTER_AREA)
    if obs is not None:
        pts = [(int(x * width), int(y * height)) for x, y in obs.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(small, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(small, pts[a], pts[b], (49, 94, 251), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(small, p, 2, (49, 94, 251), -1, cv2.LINE_AA)
        cv2.circle(small, pts[8], 5, (229, 72, 77), -1, cv2.LINE_AA)
        cv2.circle(small, pts[4], 5, (229, 72, 77), 1, cv2.LINE_AA)
    small = np.ascontiguousarray(small)
    image = QImage(small.data, width, height, small.strides[0], QImage.Format.Format_RGB888)
    return image.copy()


class CameraWorker(QThread):
    camera_started = Signal(int, int)
    frame_processed = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        preview_width: int = 320,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self.req_width = width
        self.req_height = height
        self.preview_width = preview_width
        self._stop = threading.Event()
        self._ack = threading.Event()
        self._ack.set()

    def stop(self) -> None:
        self._stop.set()
        self._ack.set()

    def acknowledge(self) -> None:
        """Called by the GUI after it has handled the last FrameResult."""
        self._ack.set()

    def run(self) -> None:
        tracker: HandTracker | None = None
        cap: cv2.VideoCapture | None = None
        grabber: FrameGrabber | None = None
        try:
            try:
                tracker = HandTracker()
            except ModelNotFoundError as exc:
                self.error_occurred.emit(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - surface any model init failure
                self.error_occurred.emit(f"Не удалось загрузить модель отслеживания руки:\n{exc}")
                return
            if self._stop.is_set():
                return

            cap = open_camera(self.camera_index, self.req_width, self.req_height)
            if cap is None:
                self.error_occurred.emit(
                    f"Камера {self.camera_index} недоступна или занята другим приложением.\n"
                    "Закройте программы, использующие камеру, проверьте разрешения "
                    "и выберите другую камеру."
                )
                return
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_started.emit(actual_w, actual_h)

            grabber = FrameGrabber(cap)
            grabber.start()
            seq = 0
            while not self._stop.is_set():
                frame_bgr, seq = grabber.latest(seq, timeout=0.5)
                if frame_bgr is None:
                    if grabber.failed:
                        self.error_occurred.emit(
                            "Камера перестала передавать кадры. Возможно, она отключена "
                            "или занята другим приложением."
                        )
                        break
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = tracker.process(frame_rgb, mirror=True)
                preview = draw_preview(frame_rgb, result.observation, self.preview_width)
                h, w = frame_rgb.shape[:2]

                while not self._ack.wait(0.1):
                    if self._stop.is_set():
                        break
                if self._stop.is_set():
                    break
                self._ack.clear()
                self.frame_processed.emit(
                    FrameResult(preview, result.observation, result.inference_ms,
                                time.perf_counter(), w, h)
                )
        except Exception as exc:  # noqa: BLE001 - report instead of crashing the thread
            self.error_occurred.emit(f"Ошибка обработки видео: {exc}")
        finally:
            if grabber is not None:
                grabber.stop()
                grabber.join(timeout=2.0)
            if cap is not None:
                cap.release()
            if tracker is not None:
                tracker.close()
