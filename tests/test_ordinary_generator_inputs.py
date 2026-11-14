"""Input admission for the public ordinary FiberNetwork generators."""
import numpy as np
import pytest
from fibernet.gen import disordered, ordered


@pytest.mark.parametrize("name", ["fiber_length", "radius", "box_size"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0, -1, True, np.bool_(True)])
def test_random_invalid_scalar_rejected_before_rng(name, value, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("RNG constructed before input admission")
    monkeypatch.setattr(disordered.np.random, "RandomState", forbidden)
    options = dict(num_fibers=1, fiber_length=1., radius=.1, box_size=1., ensure_connected=False)
    options[name] = value
    with pytest.raises(ValueError, match=name):
        disordered.random_straight_2d(**options)


@pytest.mark.parametrize("box", [[1., float("nan")], [float("inf")], [1., -1.], np.array(1.), [[1.], [2.]]])
def test_random_invalid_effective_box_rejected(box):
    with pytest.raises(ValueError, match="box_size"):
        disordered.random_straight_2d(num_fibers=1, box_size=box)


@pytest.mark.parametrize("count", [-1, 1.5, 1., True, np.bool_(False)])
def test_invalid_counts_rejected(count):
    with pytest.raises(ValueError, match="num_fibers"):
        disordered.random_straight_2d(num_fibers=count, box_size=1.)
    with pytest.raises(ValueError, match="grid_size"):
        ordered.square_lattice_2d(grid_size=(count, 2))


@pytest.mark.parametrize("name", ["spacing", "radius"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0, -1, True, np.bool_(True)])
def test_square_invalid_scalar_rejected_before_builder(name, value, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("graph constructed before input admission")
    monkeypatch.setattr(ordered, "FiberGraph", forbidden)
    with pytest.raises(ValueError, match=name):
        ordered.square_lattice_2d(grid_size=(1, 1), **{name: value})


@pytest.mark.parametrize("grid", [None, 1, (1,), (1, 2, 3)])
def test_square_grid_shape_error(grid):
    with pytest.raises(ValueError, match="grid_size"):
        ordered.square_lattice_2d(grid_size=grid)


def test_zero_counts_preserved():
    assert disordered.random_straight_2d(num_fibers=0, seed=0).num_fibers == 0
    assert ordered.square_lattice_2d(grid_size=(0, 0)).num_fibers == 0
    assert ordered.square_lattice_2d(grid_size=(0, 2)).num_fibers == 2


def test_numpy_numeric_types_remain_supported():
    net = disordered.random_straight_2d(num_fibers=np.int64(1), fiber_length=np.float32(1.),
                                      radius=np.float64(.1), box_size=np.array([2., 3.]), seed=0,
                                      ensure_connected=False)
    assert net.num_fibers == 1
    assert np.isfinite(net.fibers[0].centerline).all()
    grid = ordered.square_lattice_2d(grid_size=(np.int64(1), np.int32(1)), spacing=np.float32(2.))
    assert grid.num_fibers == 4


def test_automatic_box_overflow_is_clear_error():
    with pytest.raises(ValueError, match="box_size"):
        disordered.random_straight_2d(num_fibers=1, fiber_length=1e308)
