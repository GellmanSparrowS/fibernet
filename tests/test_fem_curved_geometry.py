"""FEM does not silently solve the chord of a curve shown to the user."""

import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def _graph(points, dimension):
    graph = fn.StructureGraph(dimension=dimension)
    left = graph.add_node([0., 0., 0.])
    right = graph.add_node([1., 0., 0.])
    graph.add_edge(left, right, radius=.05, internal_points=points)
    return graph


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("points", [
    np.array([[.5, .5, 0.]]),
    np.array([[.5, 0., 0.]]),
])
def test_public_fem_rejects_unmodeled_internal_points(dimension, points):
    graph = _graph(points, dimension)
    original = graph.to_dict()
    with pytest.raises(ValueError, match="actual intermediate graph nodes"):
        fn.simulate(graph, backend="fem", stretch_ratio=1.01,
                    nonlinear=False, E=1000.)
    assert graph.to_dict() == original


@pytest.mark.parametrize("points", [None, np.empty((0, 3))])
def test_straight_members_and_empty_curves_keep_analytic_response(points):
    graph = _graph(points, 2)
    result = fn.simulate(graph, backend="fem", stretch_ratio=1.01,
                         nonlinear=False, E=1000.)
    expected = 1000. * np.pi * .05 ** 2 * .01
    assert result.max_force == pytest.approx(expected)


def test_direct_graph_conversion_rejects_curves_before_array_projection(monkeypatch):
    import fibernet.sim.accelerated as accelerated

    def unexpected(*args, **kwargs):
        pytest.fail("curved member reached endpoint array conversion")

    monkeypatch.setattr(accelerated, "_graph_to_arrays", unexpected)
    graph = _graph(np.array([[.5, .5, 0.]]), 2)
    with pytest.raises(ValueError, match="internal_points"):
        BeamFrameFEM.graph_to_fem_input(graph)
