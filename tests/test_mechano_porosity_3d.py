"""Analytic geometry and complete-network checks for the 3D void example."""
import numpy as np
import pytest

from examples.mechano_porosity_3d import _cylinder_union_volume, run_example


BOX = np.array([[-.7, -.7, -.7], [.7, .7, .7]])


def _volume(p, q, r, axis):
    return _cylinder_union_volume(p, q, r, BOX, samples=768, axis=axis, phase=.27)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_oblique_coaxial_overlap_is_not_counted_twice(axis):
    p = np.array([-.31, -.31, -.33])
    v = np.array([.24, .23, .27])
    r = .093
    x = np.array([p, p + .45 * v])
    y = np.array([p + v, p + 1.34 * v])
    exact = np.pi * r**2 * 1.34 * np.linalg.norm(v)
    assert _volume(x, y, [r, r], axis) == pytest.approx(exact, rel=5.e-4)
    assert _volume(y[::-1], x[::-1], [r, r], axis) == pytest.approx(
        _volume(x, y, [r, r], axis), abs=8.e-13)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_box_clips_cylinder_longitudinally(axis):
    r = .15
    exact = np.pi * r**2 * 1.4
    assert _volume([[-1., 0., .12]], [[1., 0., .12]], [r], axis) == pytest.approx(
        exact, rel=2.e-3)


def test_disjoint_union_and_rigid_reflection():
    p = np.array([[-.35, -.35, -.15], [-.15, .25, .22]])
    q = np.array([[.25, -.38, -.22], [.25, .33, .26]])
    r = np.array([.09, .07])
    mirror = np.array([-1., 1., 1.])
    one = sum(_volume([x], [y], [a], 0) for x, y, a in zip(p, q, r))
    assert _volume(p, q, r, 0) == pytest.approx(one, abs=8.e-13)
    assert _volume(p * mirror, q * mirror, r, 0) == pytest.approx(one, abs=8.e-13)


def test_degenerate_segment_is_rejected():
    with pytest.raises(ValueError, match="positive finite lengths"):
        _volume([[0., 0., 0.]], [[0., 0., 0.]], [.08], 0)


def test_complete_network_geometry_before_and_after_loading():
    result = run_example(samples=384)
    assert (result["nodes"], result["rods"]) == (91, 360)
    assert result["strain_magnitude"] == .005
    assert result["box_mm3"] == pytest.approx(411.6)
    for item in result["estimates"]:
        base = item["void_mm3"]["reference"]
        assert base == pytest.approx(391.627, abs=.01)
        ratio = item["relative_change"]
        assert 3.e-5 < ratio["compression"] < 9.e-5
        assert -9.e-5 < ratio["tension"] < -3.e-5
