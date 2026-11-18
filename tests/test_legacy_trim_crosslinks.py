"""Crosslink preservation in the legacy sampled-vertex trim operation."""

from copy import deepcopy
import inspect

import numpy as np
import pytest

from fibernet.core.fiber import Fiber
from fibernet.core.network import Crosslink, FiberNetwork
from fibernet.core.transform import trim_to_box

LOW = np.array([0.0, -1.0, -1.0])
HIGH = np.array([3.0, 1.0, 1.0])


def fiber(xs, y=0.0):
    return Fiber(np.array([[x, y, 0.0] for x in xs]))


def link(i, j, pi=0.5, pj=0.5, position=None):
    return Crosslink(
        i,
        j,
        pi,
        pj,
        np.array([1.5, 0.0, 0.0] if position is None else position),
        crosslink_type="sliding",
        strength=7.0,
        stiffness=11.0,
    )


def network():
    net = FiberNetwork(dimension=2, periodic=True)
    for f in [
        fiber([-4, -3]),
        fiber([0, 1, 2, 3]),
        fiber([0, 1, 2, 3], 0.5),
        fiber([8, 9]),
    ]:
        net.add_fiber(f)
    # Crosslink endpoints are list indices, never the mutable fiber_id labels.
    for i, f in enumerate(net.fibers):
        f.fiber_id = 100 + 3 * i
    shared = link(1, 2)
    shared.extra = {"nested": [1]}
    net.crosslinks = [link(0, 1), shared, link(2, 1, 0.75, 0.25), shared, link(2, 3)]
    return net


@pytest.mark.parametrize("remove_outside", [True, False])
def test_keep_order_multiplicity_remap_indices_and_preserve_fields(remove_outside):
    net = network()
    before = deepcopy(net.to_dict())
    out = trim_to_box(net, LOW, HIGH, remove_outside=remove_outside)
    assert out.num_fibers == 2
    assert [f.fiber_id for f in out.fibers] == [0, 1]
    assert [(c.fiber_i, c.fiber_j) for c in out.crosslinks] == [(0, 1), (1, 0), (0, 1)]
    assert [(c.param_i, c.param_j) for c in out.crosslinks] == [
        (0.5, 0.5),
        (0.75, 0.25),
        (0.5, 0.5),
    ]
    for c in out.crosslinks:
        assert (c.crosslink_type, c.strength, c.stiffness) == ("sliding", 7.0, 11.0)
        np.testing.assert_array_equal(c.position, [1.5, 0, 0])
    np.testing.assert_array_equal(out.connectivity_matrix(), [[0, 3], [3, 0]])
    np.testing.assert_array_equal(out.degree_distribution(), [3, 3])
    assert out.dimension == 2 and out.periodic
    np.testing.assert_array_equal(out.box_size, HIGH - LOW)
    assert net.to_dict() == before
    assert out.crosslinks[0].extra == {"nested": [1]}
    out.crosslinks[0].position[0] = 99
    out.crosslinks[0].extra["nested"].append(2)
    assert net.to_dict() == before
    assert net.crosslinks[1].extra == {"nested": [1]}


def test_whole_fiber_mode_keeps_outside_links_when_both_fibers_survive():
    net = FiberNetwork(fibers=[fiber([-2, 1, 5]), fiber([-2, 1, 5], 0.5)])
    net.crosslinks = [link(0, 1, 0, 0, [-2, 0.25, 0])]
    out = trim_to_box(net, LOW, HIGH)
    assert out.num_crosslinks == 1
    assert (out.crosslinks[0].param_i, out.crosslinks[0].param_j) == (0, 0)
    for old, new in zip(net.fibers, out.fibers):
        np.testing.assert_array_equal(new.centerline, old.centerline)


@pytest.mark.parametrize(
    "parameter, expected", [(0.25, 0), (0.375, 0.25), (0.5, 0.5), (0.75, 1)]
)
def test_sampled_mode_remaps_vertices_and_segment_interiors(parameter, expected):
    # Nonuniform lengths distinguish sample-index parameters from arc length.
    net = FiberNetwork(fibers=[fiber([-3, 0, 0.1, 3, 7]), fiber([0, 3], 0.5)])
    net.crosslinks = [link(0, 1, parameter, 0.25)]
    out = trim_to_box(net, LOW, HIGH, remove_outside=False)
    assert out.num_crosslinks == 1
    assert out.crosslinks[0].param_i == pytest.approx(expected)
    assert out.crosslinks[0].param_j == 0.25
    np.testing.assert_array_equal(
        out.fibers[0].centerline, net.fibers[0].centerline[1:4]
    )
    np.testing.assert_array_equal(
        out.crosslinks[0].position, net.crosslinks[0].position
    )


@pytest.mark.parametrize(
    "parameter", [0.0, 0.125, 0.875, 1.0, -0.1, 1.1, np.nan, np.inf]
)
def test_sampled_mode_drops_removed_or_invalid_endpoint_parameters(parameter):
    net = FiberNetwork(fibers=[fiber([-3, 0, 1, 3, 7]), fiber([0, 3], 0.5)])
    net.crosslinks = [link(0, 1, parameter), link(1, 0, pj=parameter)]
    out = trim_to_box(net, LOW, HIGH, remove_outside=False)
    assert out.num_fibers == 2
    assert out.num_crosslinks == 0


@pytest.mark.parametrize(
    "parameter, expected",
    [
        (0, 0),
        (0.125, 1 / 6),
        (0.25, 1 / 3),
        (0.5, None),
        (0.75, 2 / 3),
        (0.875, 5 / 6),
        (1, 1),
    ],
)
def test_separated_runs_do_not_move_links_onto_artificial_chords(parameter, expected):
    net = FiberNetwork(fibers=[fiber([0, 1, 9, 2, 3]), fiber([0, 3], 0.5)])
    net.crosslinks = [link(0, 1, parameter)]
    out = trim_to_box(net, LOW, HIGH, remove_outside=False)
    # Keep the old geometry contract: sampled points are not split or clipped.
    np.testing.assert_array_equal(out.fibers[0].centerline[:, 0], [0, 1, 2, 3])
    if expected is None:
        assert out.num_crosslinks == 0
    else:
        assert out.num_crosslinks == 1
        assert out.crosslinks[0].param_i == pytest.approx(expected)


def test_roundoff_at_surviving_vertex_next_to_removed_segment():
    xs = [-1] * 23
    xs[15], xs[20] = 0, 3
    net = FiberNetwork(fibers=[fiber(xs), fiber([0, 3])])
    net.crosslinks = [link(0, 1, 15 / 22)]
    out = trim_to_box(net, LOW, HIGH, remove_outside=False)
    assert out.num_crosslinks == 1
    assert out.crosslinks[0].param_i == 0


@pytest.mark.parametrize("remove_outside", [True, False])
def test_empty_no_crosslinks_and_no_geometric_intersection_insertion(remove_outside):
    assert trim_to_box(FiberNetwork(), LOW, HIGH, remove_outside).num_crosslinks == 0
    # A crossing segment without an inside sample stays omitted, as before.
    net = FiberNetwork(fibers=[fiber([-1, 4]), fiber([1, 4])])
    net.crosslinks = [link(0, 1)]
    out = trim_to_box(net, LOW, HIGH, remove_outside)
    assert out.num_fibers == int(remove_outside)
    assert out.num_crosslinks == 0


def test_signature_defaults_unchanged():
    params = inspect.signature(trim_to_box).parameters
    assert list(params) == ["network", "box_min", "box_max", "remove_outside"]
    assert params["remove_outside"].default is True
