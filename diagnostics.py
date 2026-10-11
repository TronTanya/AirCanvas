"""Performance diagnostics and stationary-hand trial measurements.

Only aggregated numbers are stored and exported - never video frames.
All values come from actual timing measurements on the current machine.
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path


@dataclass
class PerformanceSnapshot:
    fps: float
    avg_inference_ms: float
    detection_percent: float
    frames: int


class PerformanceMonitor:
    """Rolling statistics over the last ``window_s`` seconds plus session totals.

    ``inference_ms`` is the time spent inside the hand-landmark model call only;
    it is NOT end-to-end latency (capture, conversion and display are excluded).
    """

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._samples: deque[tuple[float, float, bool]] = deque()
        self.reset_session()

    def reset_session(self) -> None:
        self._samples.clear()
        self.session_frames = 0
        self.session_detected = 0
        self.session_inference_total_ms = 0.0
        self.session_start: float | None = None
        self.session_last: float | None = None

    def record(self, timestamp_s: float, inference_ms: float, hand_detected: bool) -> None:
        self._samples.append((timestamp_s, inference_ms, hand_detected))
        while self._samples and timestamp_s - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        if self.session_start is None:
            self.session_start = timestamp_s
        self.session_last = timestamp_s
        self.session_frames += 1
        self.session_detected += int(hand_detected)
        self.session_inference_total_ms += inference_ms

    def snapshot(self) -> PerformanceSnapshot:
        n = len(self._samples)
        if n == 0:
            return PerformanceSnapshot(0.0, 0.0, 0.0, 0)
        span = self._samples[-1][0] - self._samples[0][0]
        fps = (n - 1) / span if span > 0 else 0.0
        avg = sum(s[1] for s in self._samples) / n
        det = 100.0 * sum(1 for s in self._samples if s[2]) / n
        return PerformanceSnapshot(fps, avg, det, n)

    def session_summary(self) -> PerformanceSnapshot:
        n = self.session_frames
        if n == 0 or self.session_start is None or self.session_last is None:
            return PerformanceSnapshot(0.0, 0.0, 0.0, 0)
        span = self.session_last - self.session_start
        fps = (n - 1) / span if span > 0 else 0.0
        return PerformanceSnapshot(
            fps,
            self.session_inference_total_ms / n,
            100.0 * self.session_detected / n,
            n,
        )


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


@dataclass
class TrialResult:
    trial: int
    started_at: str
    alpha: float
    duration_s: float
    frames: int
    detected_frames: int
    detection_percent: float
    tracking_losses: int
    fps: float
    avg_inference_ms: float
    raw_std_x_norm: float
    raw_std_y_norm: float
    smoothed_std_x_norm: float
    smoothed_std_y_norm: float
    note: str = ""


class TrialRecorder:
    """Records a stationary-hand trial: cursor variation normalized by canvas size.

    Raw (unsmoothed) and smoothed cursor positions are recorded in the same
    trial so the effect of the smoothing coefficient can be compared directly.
    """

    def __init__(self, canvas_width: int, canvas_height: int) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.results: list[TrialResult] = []
        self.active = False
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._raw: list[tuple[float, float]] = []
        self._smooth: list[tuple[float, float]] = []
        self._frames = 0
        self._detected = 0
        self._losses = 0
        self._prev_detected = False
        self._inference_ms = 0.0
        self._t0: float | None = None
        self._t1: float | None = None
        self._alpha = 0.0
        self._started_at = ""

    def start(self, alpha: float) -> None:
        self._reset_buffers()
        self._alpha = alpha
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self.active = True

    def record(
        self,
        timestamp_s: float,
        inference_ms: float,
        raw_cursor: tuple[float, float] | None,
        smoothed_cursor: tuple[float, float] | None,
    ) -> None:
        if not self.active:
            return
        if self._t0 is None:
            self._t0 = timestamp_s
        self._t1 = timestamp_s
        self._frames += 1
        self._inference_ms += inference_ms
        detected = raw_cursor is not None
        if detected:
            self._detected += 1
            self._raw.append(raw_cursor)
            if smoothed_cursor is not None:
                self._smooth.append(smoothed_cursor)
        elif self._prev_detected:
            self._losses += 1
        self._prev_detected = detected

    def stop(self, note: str = "") -> TrialResult | None:
        if not self.active:
            return None
        self.active = False
        if self._frames == 0:
            return None
        duration = (self._t1 or 0.0) - (self._t0 or 0.0)
        w, h = float(self.canvas_width), float(self.canvas_height)
        result = TrialResult(
            trial=len(self.results) + 1,
            started_at=self._started_at,
            alpha=self._alpha,
            duration_s=round(duration, 3),
            frames=self._frames,
            detected_frames=self._detected,
            detection_percent=round(100.0 * self._detected / self._frames, 2),
            tracking_losses=self._losses,
            fps=round((self._frames - 1) / duration, 2) if duration > 0 else 0.0,
            avg_inference_ms=round(self._inference_ms / self._frames, 3),
            raw_std_x_norm=round(_std([p[0] for p in self._raw]) / w, 6),
            raw_std_y_norm=round(_std([p[1] for p in self._raw]) / h, 6),
            smoothed_std_x_norm=round(_std([p[0] for p in self._smooth]) / w, 6),
            smoothed_std_y_norm=round(_std([p[1] for p in self._smooth]) / h, 6),
            note=note,
        )
        self.results.append(result)
        return result


def export_csv(
    path: str | Path,
    session: PerformanceSnapshot,
    trials: list[TrialResult],
    settings: dict[str, object],
) -> Path:
    """Write aggregated session metrics, settings and trial results to CSV."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["section", "key", "value"])
        writer.writerow(["meta", "exported_at", datetime.now().isoformat(timespec="seconds")])
        writer.writerow(["meta", "clock", "time.perf_counter"])
        for key, value in settings.items():
            writer.writerow(["settings", key, value])
        writer.writerow(["session", "processed_frames", session.frames])
        writer.writerow(["session", "processed_fps", f"{session.fps:.2f}"])
        writer.writerow(["session", "avg_inference_ms", f"{session.avg_inference_ms:.3f}"])
        writer.writerow(["session", "detection_percent", f"{session.detection_percent:.2f}"])
        writer.writerow([])
        names = [f.name for f in fields(TrialResult)]
        writer.writerow(names)
        for trial in trials:
            row = asdict(trial)
            writer.writerow([row[n] for n in names])
    return path


def now_s() -> float:
    return time.perf_counter()
