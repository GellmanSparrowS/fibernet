"""Exact scalar intersection oracle for the public 2D stick generator."""

import numpy as np
import pytest

from fibernet.gen import disordered


def _record(i, j, t, u, point):
    values = np.array([t, u, *point], dtype=np.float64).view(np.uint64)
    return i, j, *values


def _scalar_crossings(fibers):
    records = []
    for i, fi in enumerate(fibers):
        p = fi.centerline[0]
        d = fi.centerline[-1] - p
        for j in range(i + 1, len(fibers)):
            q = fibers[j].centerline[0]
            e = fibers[j].centerline[-1] - q
            denominator = d[0] * e[1] - d[1] * e[0]
            if abs(denominator) < 1e-10:
                continue
            offset = q - p
            t = (offset[0] * e[1] - offset[1] * e[0]) / denominator
            u = (offset[0] * d[1] - offset[1] * d[0]) / denominator
            if 0 <= t <= 1 and 0 <= u <= 1:
                records.append(_record(i, j, t, u, p + t * d))
    return records


def _assert_crossings(network):
    for link in network.crosslinks:
        assert type(link.fiber_i) is int and type(link.fiber_j) is int
        assert link.crosslink_type == 'welded'
    assert [
        _record(link.fiber_i, link.fiber_j, link.param_i, link.param_j, link.position)
        for link in network.crosslinks
    ] == _scalar_crossings(network.fibers)


@pytest.mark.parametrize('count', [0, 1, 120])
def test_random_sticks_use_scalar_order_and_bits(count):
    network = disordered.random_straight_2d(
        count, fiber_length=5., box_size=(15., 12.), seed=31, ensure_connected=False,
    )
    _assert_crossings(network)
    if count > 1:
        assert len(network.crosslinks) > 100


def test_endpoints_parallel_and_determinant_cutoff(monkeypatch):
    ends = iter([
        ((0., 0.), (1., 0.)),
        ((1., -1.), (1., 0.)),
        ((.5, -1.), (.5, 1.)),
        ((.25, 0.), (.75, 0.)),
        ((0., 5e-11), (1., -5e-11)),
        ((0., 2e-11), (1., -2e-11)),
        ((.5, 0.), (.5, 0.)),
    ])
    straight = disordered.Fiber.straight

    def synthetic_straight(start, end, **kwargs):
        a, b = next(ends)
        return straight([*a, 0.], [*b, 0.], **kwargs)

    monkeypatch.setattr(disordered.Fiber, 'straight', staticmethod(synthetic_straight))
    with np.errstate(divide='raise', invalid='raise'):
        network = disordered.random_straight_2d(7, seed=5, ensure_connected=False)
        _assert_crossings(network)
    from_first = {
        link.fiber_j: (link.param_i, link.param_j) for link in network.crosslinks if link.fiber_i == 0
    }
    assert from_first == {1: (1., 1.), 2: (.5, .5), 4: (.5, .5)}


def test_default_repair_uses_the_same_generator_and_rng():
    options = dict(num_fibers=80, fiber_length=2., box_size=20., seed=42)
    prior_state = np.random.get_state()
    try:
        np.random.seed(1)
        raw = disordered.random_straight_2d(**options, ensure_connected=False)
        _assert_crossings(raw)
        expected = disordered._ensure_connected(raw)
        expected_rng = np.random.bytes(32)

        np.random.seed(1)
        actual = disordered.random_straight_2d(**options)
        assert actual.to_dict() == expected.to_dict()
        assert np.random.bytes(32) == expected_rng
        assert actual.num_fibers > options['num_fibers']
    finally:
        np.random.set_state(prior_state)
