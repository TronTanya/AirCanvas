"""Aggregated diagnostics: rolling FPS, trial variation and CSV export."""

import csv

import pytest

from diagnostics import PerformanceMonitor, TrialRecorder, export_csv


def test_fps_inference_and_detection_from_timestamps():
    m = PerformanceMonitor(window_s=10)
    for i in range(31):
        m.record(i / 30, inference_ms=5.0 if i % 2 else 7.0, hand_detected=i < 15)
    snap = m.snapshot()
    assert snap.fps == pytest.approx(30.0)
    assert snap.avg_inference_ms == pytest.approx(6.0, abs=0.1)
    assert snap.detection_percent == pytest.approx(100 * 15 / 31)


def test_trial_variation_normalized_by_canvas_and_losses():
    t = TrialRecorder(1000, 500)
    t.start(alpha=0.5)
    for i, x in enumerate([100, 110, 100, 110]):
        t.record(i * 0.1, 5.0, (x, 200), (105, 200))
    t.record(0.4, 5.0, None, None)
    t.record(0.5, 5.0, (100, 200), (105, 200))
    r = t.stop()
    assert r.frames == 6 and r.detected_frames == 5 and r.tracking_losses == 1
    assert r.raw_std_x_norm > 0 and r.raw_std_y_norm == 0
    assert r.smoothed_std_x_norm == 0
    assert r.fps == pytest.approx(10.0)


def test_csv_contains_aggregates_only(tmp_path):
    m = PerformanceMonitor()
    m.record(0.0, 5.0, True)
    m.record(0.1, 5.0, True)
    t = TrialRecorder(1280, 720)
    t.start(0.3)
    t.record(0.0, 5.0, (1, 1), (1, 1))
    t.stop()
    path = export_csv(tmp_path / "диагностика", m.session_summary(), t.results, {"alpha": 0.3})
    assert path.suffix == ".csv"
    rows = list(csv.reader(path.open(encoding="utf-8-sig"), delimiter=";"))
    assert ["session", "processed_frames", "2"] in rows
    header = next(r for r in rows if r and r[0] == "trial")
    assert "smoothed_std_x_norm" in header
    assert rows[-1][header.index("alpha")] == "0.3"
