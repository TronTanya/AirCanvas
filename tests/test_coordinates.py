"""Coordinate mapping, mirroring, clamping and smoothing."""

import pytest

from gestures import (
    ExponentialSmoother,
    map_to_canvas,
    smoothing_time_constant,
    to_display_landmarks,
)

W, H = 1280, 720


def test_mirroring_flips_x_only():
    assert to_display_landmarks([(0.2, 0.3)]) == ((0.8, 0.3),)
    assert to_display_landmarks([(0.2, 0.3)], mirror=False) == ((0.2, 0.3),)


def test_mirrored_movement_direction_matches_preview():
    """In the raw frame the user's rightward movement decreases x.
    After mirroring, the cursor must move right on the canvas."""
    before_raw, after_raw = (0.6, 0.5), (0.4, 0.5)
    (before,), (after,) = to_display_landmarks([before_raw]), to_display_landmarks([after_raw])
    assert map_to_canvas(after, W, H)[0] > map_to_canvas(before, W, H)[0]


def test_mapping_corners_and_center():
    assert map_to_canvas((0.0, 0.0), W, H, margin=0.0) == (0.0, 0.0)
    assert map_to_canvas((1.0, 1.0), W, H, margin=0.0) == (W - 1, H - 1)
    x, y = map_to_canvas((0.5, 0.5), W, H, margin=0.1)
    assert x == pytest.approx((W - 1) / 2)
    assert y == pytest.approx((H - 1) / 2)


def test_margin_stretches_active_region():
    assert map_to_canvas((0.1, 0.1), W, H, margin=0.1) == (0.0, 0.0)
    assert map_to_canvas((0.9, 0.9), W, H, margin=0.1) == pytest.approx((W - 1, H - 1))


@pytest.mark.parametrize("point", [(-0.5, 0.5), (1.7, 0.5), (0.5, -2.0), (0.5, 3.0), (-1, 5)])
def test_clamping_to_canvas_bounds(point):
    x, y = map_to_canvas(point, W, H)
    assert 0 <= x <= W - 1
    assert 0 <= y <= H - 1


def test_invalid_margin_rejected():
    with pytest.raises(ValueError):
        map_to_canvas((0.5, 0.5), W, H, margin=0.5)


def test_smoothing_first_sample_passes_through():
    s = ExponentialSmoother(0.3)
    assert s.update((100.0, 50.0)) == (100.0, 50.0)


def test_smoothing_formula():
    s = ExponentialSmoother(0.25)
    s.update((0.0, 0.0))
    assert s.update((100.0, 200.0)) == pytest.approx((25.0, 50.0))
    assert s.update((100.0, 200.0)) == pytest.approx((43.75, 87.5))


def test_alpha_one_disables_smoothing():
    s = ExponentialSmoother(1.0)
    s.update((0.0, 0.0))
    assert s.update((10.0, 20.0)) == (10.0, 20.0)


def test_smoothing_reduces_jitter():
    jitter = [(100 + (5 if i % 2 else -5), 100.0) for i in range(50)]
    s = ExponentialSmoother(0.2)
    out = [s.update(p) for p in jitter][10:]
    assert max(p[0] for p in out) - min(p[0] for p in out) < 10 * 0.5


def test_smoothing_reset_after_tracking_loss():
    s = ExponentialSmoother(0.1)
    s.update((0.0, 0.0))
    s.reset()
    assert s.state is None
    assert s.update((500.0, 300.0)) == (500.0, 300.0)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        ExponentialSmoother(alpha)


def test_time_constant():
    assert smoothing_time_constant(1.0) == 0.0
    assert smoothing_time_constant(0.5) == pytest.approx(1.4427, rel=1e-3)
    assert smoothing_time_constant(0.1) > smoothing_time_constant(0.5)
