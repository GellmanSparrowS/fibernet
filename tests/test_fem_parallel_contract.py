"""Public beam FEM retains parallel-fiber mechanics and honest summaries."""
import numpy as np
import pytest

pytest.importorskip("torch")

from fibernet.core.structure_graph import StructureGraph
from fibernet.easy import simulate
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.ml.beam_frame_fem_sparse import SparseBeamFrameFEM


def _two_node_graph(radii):
    graph = StructureGraph(dimension=2)
    left = graph.add_node([0, 0])
    right = graph.add_node([1, 0])
    for index, radius in enumerate(radii):
        graph.add_edge(left, right, radius=radius,
                       allow_parallel=index > 0)
    return graph, right


def test_fem_preserves_parallel_grip_reaction():
    one, right = _two_node_graph([.05])
    two, _ = _two_node_graph([.05, .05])
    a = simulate(one, backend="fem", strain=1.01, nonlinear=False, E=1000)
    b = simulate(two, backend="fem", strain=1.01, nonlinear=False, E=1000)
    assert a.n_edges == 1 and b.n_edges == 2
    np.testing.assert_allclose(b.metadata["reactions"][right, 0],
                               2 * a.metadata["reactions"][right, 0],
                               rtol=1e-12)
    np.testing.assert_allclose(b.energy, 2 * a.energy, rtol=1e-12)
    assert len(b.edge_forces) == len(b.edge_stretches) == 2
    np.testing.assert_allclose(b.max_force, max(abs(b.edge_forces)))


def test_fem_uses_actual_radius_and_signed_stretch():
    graph, _ = _two_node_graph([.05, .10])
    stretch = simulate(graph, backend="fem", strain=1.01,
                       nonlinear=False, E=1000)
    np.testing.assert_allclose(abs(stretch.edge_forces[1] /
                                   stretch.edge_forces[0]), 4.0, rtol=1e-12)
    np.testing.assert_allclose(stretch.max_force,
                               abs(stretch.edge_forces[1]), rtol=1e-12)
    compression = simulate(graph, backend="fem", strain=.99,
                           nonlinear=False, E=1000)
    np.testing.assert_allclose(compression.max_stretch, .99, rtol=1e-12)


def test_direct_beam_stiffness_defaults_to_independent_edges():
    points = np.array([[0., 0.], [1., 0.]])
    edges = np.array([[0, 0], [1, 1]])
    radii = np.array([.05, .05])
    dense = BeamFrameFEM(E=1000)
    sparse = SparseBeamFrameFEM(E=1000)
    direct, selected = dense.build_stiffness_2d(edges, points, radii)
    legacy, old_selected = dense.build_stiffness_2d(
        edges, points, radii, deduplicate=True)
    assert len(selected) == 2 and len(old_selected) == 1
    np.testing.assert_allclose(direct[3, 3], 2 * legacy[3, 3], rtol=1e-12)
    fast, fast_selected = sparse.build_sparse_stiffness_2d(edges, points, radii)
    assert len(fast_selected) == 2
    np.testing.assert_allclose(fast[3, 3], direct[3, 3], rtol=1e-12)
