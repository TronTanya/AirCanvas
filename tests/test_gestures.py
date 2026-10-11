"""Pinch detection, hysteresis, normalization and tracking-loss behaviour."""

import math

import pytest

from canvas import StrokeCanvas
from gestures import (
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    THUMB_TIP,
    WRIST,
    GestureInterpreter,
    HandObservation,
    PinchDetector,
    pinch_ratio,
)


def make_hand(
    index=(0.5, 0.4),
    pinch_gap=0.2,
    hand_size=0.2,
    wrist=None,
    handedness="Right",
) -> HandObservation:
    """Synthetic hand: wrist below index tip; thumb tip ``pinch_gap * hand_size`` away."""
    pts = [(0.5, 0.5)] * NUM_LANDMARKS
    pts = list(pts)
    wx, wy = wrist if wrist is not None else (index[0], index[1] + 0.3)
    pts[WRIST] = (wx, wy)
    pts[MIDDLE_MCP] = (wx, wy - hand_size)
    pts[INDEX_TIP] = index
    pts[THUMB_TIP] = (index[0] + pinch_gap * hand_size, index[1])
    return HandObservation(tuple(pts), handedness=handedness, image_aspect=1.0)


def test_pinch_ratio_is_scale_invariant():
    small = make_hand(pinch_gap=0.3, hand_size=0.1)
    large = make_hand(pinch_gap=0.3, hand_size=0.3)
    assert pinch_ratio(small.landmarks) == pytest.approx(0.3)
    assert pinch_ratio(large.landmarks) == pytest.approx(0.3)


def test_pinch_ratio_uses_image_aspect():
    hand = make_hand(pinch_gap=0.3, hand_size=0.2)  # thumb offset is horizontal
    assert pinch_ratio(hand.landmarks, image_aspect=2.0) == pytest.approx(0.6)


def test_near_zero_hand_size_is_safe():
    pts = [(0.5, 0.5)] * NUM_LANDMARKS  # wrist == middle MCP
    assert pinch_ratio(pts) is None
    tiny = make_hand(hand_size=1e-7)
    assert pinch_ratio(tiny.landmarks) is None


def test_too_few_landmarks_returns_none():
    assert pinch_ratio([(0.0, 0.0)] * 5) is None


def test_pinch_activation_and_release():
    d = PinchDetector(0.3, 0.45)
    assert d.update(0.8) is False
    assert d.update(0.25) is True
    assert d.update(0.6) is False


def test_hysteresis_retains_state_between_thresholds():
    d = PinchDetector(0.3, 0.45)
    assert d.update(0.38) is False  # between thresholds, was released
    assert d.update(0.2) is True
    assert d.update(0.38) is True  # between thresholds, stays pinched
    assert d.update(0.44) is True
    assert d.update(0.46) is False
    assert d.update(0.35) is False


def test_degenerate_ratio_releases_pinch():
    d = PinchDetector(0.3, 0.45)
    d.update(0.1)
    assert d.update(None) is False


@pytest.mark.parametrize("press,release", [(0.5, 0.5), (0.6, 0.4), (0.0, 0.4), (-0.1, 0.3)])
def test_invalid_thresholds_rejected(press, release):
    with pytest.raises(ValueError):
        PinchDetector(press, release)


def test_interpreter_tracking_loss_resets_and_breaks_stroke():
    gi = GestureInterpreter(1280, 720, alpha=0.5, margin=0.0)
    f1 = gi.update(make_hand(pinch_gap=0.1))
    assert f1.hand_detected and f1.drawing
    lost = gi.update(None)
    assert not lost.hand_detected and not lost.drawing and lost.stroke_break
    assert lost.cursor is None
    assert gi.smoother.state is None
    assert gi.pinch.pinched is False


def test_no_connecting_stroke_after_tracking_loss():
    """Feed interpreter output into the canvas exactly like the window does."""
    gi = GestureInterpreter(1000, 1000, alpha=1.0, margin=0.0)
    canvas = StrokeCanvas(1000, 1000)

    def feed(obs):
        f = gi.update(obs)
        if f.stroke_break and canvas.is_drawing:
            canvas.end_stroke()
        if f.drawing and f.cursor is not None:
            if canvas.is_drawing:
                canvas.add_point(f.cursor)
            else:
                canvas.begin_stroke(f.cursor, (0, 0, 0), 4)

    feed(make_hand(index=(0.1, 0.1), pinch_gap=0.1))
    feed(make_hand(index=(0.12, 0.1), pinch_gap=0.1))
    feed(None)
    feed(make_hand(index=(0.8, 0.1), pinch_gap=0.1))
    feed(make_hand(index=(0.82, 0.1), pinch_gap=0.1))
    canvas.end_stroke()

    assert len(canvas.strokes) == 2
    first, second = canvas.strokes
    assert max(p[0] for p in first.points) < 200
    assert min(p[0] for p in second.points) > 700
    # The gap between the strokes stays unpainted.
    img = canvas.render()
    assert (img[100, 400:600] == 255).all()


def test_identity_jump_breaks_stroke_and_resets_smoothing():
    gi = GestureInterpreter(1000, 1000, alpha=0.2, margin=0.0)
    gi.update(make_hand(index=(0.2, 0.3), pinch_gap=0.1))
    f = gi.update(make_hand(index=(0.8, 0.3), pinch_gap=0.1))  # wrist jumps 0.6
    assert f.stroke_break
    assert f.cursor[0] == pytest.approx(0.8 * 999)  # no smoothing towards old hand


def test_handedness_change_breaks_stroke():
    gi = GestureInterpreter(1000, 1000)
    gi.update(make_hand(pinch_gap=0.1, handedness="Right"))
    f = gi.update(make_hand(pinch_gap=0.1, handedness="Left"))
    assert f.stroke_break


def test_release_produces_stroke_break():
    gi = GestureInterpreter(1000, 1000)
    gi.update(make_hand(pinch_gap=0.1))
    f = gi.update(make_hand(pinch_gap=0.9))
    assert not f.drawing and f.stroke_break
    assert math.isfinite(f.pinch_ratio)
