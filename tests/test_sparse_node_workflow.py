"""Public workflows must map graph IDs to positional array indices."""
import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.gen import PlanarManufacturingConfig, compile_planar
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver


def _graph(sparse=False):
    graph = StructureGraph(dimension=2)
    if sparse:
        graph.add_node([99., 99.])
    ids = [graph.add_node(p) for p in ([0., 0.], [10., 0.], [10., 10.], [0., 10.])]
    if sparse:
        graph.remove_node(0)
    for a, b in zip(ids, ids[1:] + ids[:1]):
        graph.add_edge(a, b)
    graph.metadata['weld_triplets'] = [[ids[0], ids[1], ids[2]]]
    return graph


def test_reduced_trajectory_after_node_deletion_matches_dense_graph():
    config = ReducedBeamConfig(num_steps=20, n_increments=2,
                               target_stretch=1.01, use_contact=False)
    dense = ReducedBeamSolver(_graph(), config).run()
    sparse = ReducedBeamSolver(_graph(True), config).run()
    np.testing.assert_array_equal(sparse.edges, dense.edges)
    np.testing.assert_array_equal(sparse.frames_xy, dense.frames_xy)
    np.testing.assert_array_equal(sparse.force_curve, dense.force_curve)
    assert sparse.metadata['node_ids'] == [1, 2, 3, 4]
    assert sparse.metadata['edge_ids'] == list(_graph(True).edges)


def test_custom_cell_after_node_deletion_preserves_route():
    a = compile_planar(PlanarManufacturingConfig(
        base_graph=_graph(), grid_x=1, grid_y=1, n_pts_per_side=1))
    b = compile_planar(PlanarManufacturingConfig(
        base_graph=_graph(True), grid_x=1, grid_y=1, n_pts_per_side=1))
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.route_edges, b.route_edges)


def test_disabled_contact_skips_candidate_construction(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('contact search should not run when disabled')
    monkeypatch.setattr(ReducedBeamSolver, '_contact_candidates', unexpected)
    result = ReducedBeamSolver(_graph(), ReducedBeamConfig(
        num_steps=4, n_increments=2, use_contact=False)).run()
    assert not result.contact_counts.any()


def test_invalid_weld_triplet_fails_with_clear_error():
    graph = _graph(True)
    graph.metadata['weld_triplets'] = [[1, 2, 99]]
    with pytest.raises(ValueError, match='weld_triplets'):
        ReducedBeamSolver(graph)
