"""Stroke engine: dots, undo, clear, brush changes, memory bounds, PNG output."""

import cv2
import numpy as np
import pytest

from canvas import StrokeCanvas

BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def test_stationary_pinch_creates_dot():
    c = StrokeCanvas(200, 100)
    c.begin_stroke((50, 50), BLACK, 10)
    for _ in range(20):
        c.add_point((50, 50))  # identical points are not stored
    c.end_stroke()
    assert c.strokes[0].points == [(50, 50)]
    img = c.render()
    assert tuple(img[50, 50]) == BLACK
    assert tuple(img[50, 53]) == BLACK  # radius 5
    assert tuple(img[50, 60]) == (255, 255, 255)


def test_repeated_identical_points_are_skipped():
    c = StrokeCanvas(200, 100, min_point_distance=2.0)
    c.begin_stroke((10, 10), BLACK, 4)
    assert c.add_point((10, 10)) is False
    assert c.add_point((11, 10)) is False
    assert c.add_point((20, 10)) is True
    assert len(c.active.points) == 2


def test_points_clamped_to_canvas():
    c = StrokeCanvas(100, 50)
    c.begin_stroke((-20, 500), BLACK, 4)
    assert c.active.points[0] == (0, 49)


def test_undo_removes_last_stroke_only():
    c = StrokeCanvas(200, 100)
    c.begin_stroke((10, 10), RED, 6)
    c.add_point((60, 10))
    c.end_stroke()
    c.begin_stroke((10, 80), BLUE, 6)
    c.add_point((60, 80))
    c.end_stroke()
    assert c.undo()
    assert len(c.strokes) == 1
    img = c.render()
    assert tuple(img[10, 30]) == RED
    assert tuple(img[80, 30]) == (255, 255, 255)
    assert c.undo()
    assert not c.undo()
    assert (c.render() == 255).all()


def test_undo_discards_active_stroke():
    c = StrokeCanvas(100, 100)
    c.begin_stroke((10, 10), RED, 6)
    c.add_point((50, 10))
    assert c.undo()
    assert not c.is_drawing and not c.strokes
    assert (c.render() == 255).all()


def test_clear():
    c = StrokeCanvas(100, 100)
    c.begin_stroke((10, 10), RED, 6)
    c.add_point((50, 50))
    c.end_stroke()
    c.begin_stroke((20, 20), RED, 6)
    c.clear()
    assert not c.strokes and not c.is_drawing and not c.can_undo
    assert (c.render() == 255).all()


def test_brush_change_during_drawing_starts_new_stroke():
    c = StrokeCanvas(300, 100)
    c.begin_stroke((10, 50), RED, 4)
    c.add_point((100, 50))
    c.set_brush(BLUE, 12)
    c.add_point((200, 50))
    c.end_stroke()
    assert len(c.strokes) == 2
    first, second = c.strokes
    assert (first.color, first.width) == (RED, 4)
    assert (second.color, second.width) == (BLUE, 12)
    assert second.points[0] == first.points[-1]
    img = c.render()
    assert tuple(img[50, 50]) == RED
    assert tuple(img[50, 150]) == BLUE


def test_brush_change_when_idle_does_not_create_strokes():
    c = StrokeCanvas(100, 100)
    c.set_brush(BLUE, 10)
    assert not c.strokes and not c.is_drawing


def test_max_points_per_stroke_splits_stroke():
    c = StrokeCanvas(1000, 100, max_points_per_stroke=10)
    c.begin_stroke((0, 50), BLACK, 2)
    for x in range(5, 200, 5):
        c.add_point((x, 50))
    c.end_stroke()
    assert all(len(s.points) <= 10 for s in c.strokes)
    assert len(c.strokes) > 1
    for a, b in zip(c.strokes, c.strokes[1:]):
        assert b.points[0] == a.points[-1]  # continuous, no gaps


def test_max_strokes_bakes_oldest_into_base():
    c = StrokeCanvas(200, 100, max_strokes=3)
    for i in range(5):
        c.begin_stroke((10 + 40 * i, 50), BLACK, 6)
        c.end_stroke()
    assert len(c.strokes) == 3
    while c.undo():
        pass
    img = c.render()
    assert tuple(img[50, 10]) == BLACK  # baked strokes remain
    assert tuple(img[50, 50]) == BLACK
    assert tuple(img[50, 90]) == (255, 255, 255)


def test_png_dimensions_and_content(tmp_path):
    c = StrokeCanvas(1280, 720)
    c.begin_stroke((100, 100), RED, 20)
    c.add_point((300, 100))
    c.end_stroke()
    path = c.save_png(tmp_path / "рисунок")  # non-ASCII name, suffix added
    assert path.suffix == ".png" and path.exists()
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    assert bgr.shape == (720, 1280, 3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    assert tuple(rgb[100, 200]) == RED
    assert tuple(rgb[600, 1000]) == (255, 255, 255)  # white background


def test_png_includes_active_stroke(tmp_path):
    c = StrokeCanvas(100, 100)
    c.begin_stroke((50, 50), BLUE, 10)
    path = c.save_png(tmp_path / "a.png")
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1), cv2.COLOR_BGR2RGB)
    assert tuple(rgb[50, 50]) == BLUE


def test_save_to_invalid_location_raises(tmp_path):
    c = StrokeCanvas(10, 10)
    with pytest.raises(OSError):
        c.save_png(tmp_path / "missing_dir" / "a.png")
