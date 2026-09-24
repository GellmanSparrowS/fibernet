"""Mechanical graph identity checks for compacted Euler-cycle deletions."""
import networkx as nx
import numpy as np

from benchmarks.percolation_cycle_intervention import CycleInterventionStudy
from fslab.structure import StructureFactory


def test_ring_cycle_compaction_preserves_remaining_fibers():
    graph = StructureFactory(unit='ring', grid_x=2, grid_y=2,
                             n_pts_per_side=1, seed=23).build()
    edges = np.asarray(graph.edge_array(), dtype=int)
    grips = np.union1d(np.flatnonzero(graph.node_positions()[:, 0]
                                      == graph.node_positions()[:, 0].min()),
                       np.flatnonzero(graph.node_positions()[:, 0]
                                      == graph.node_positions()[:, 0].max()))
    study = CycleInterventionStudy()
    valid = []
    for removed in study.cycle_basis_edge_ids(edges, graph.num_nodes):
        okay, _ = study.topologically_valid(edges, graph.num_nodes, grips, removed)
        if okay:
            valid.append(removed)
    assert valid
    removed = valid[0]
    candidate, mapping = study.compact_graph(graph, removed)
    assert sorted(candidate.nodes) == list(range(candidate.num_nodes))
    assert candidate.num_edges == graph.num_edges - len(removed)
    assert candidate.num_nodes < graph.num_nodes
    assert all(int(node) in mapping for node in grips)
    original_length = graph.edge_lengths()
    expected = float(original_length.sum() - original_length[list(removed)].sum())
    np.testing.assert_allclose(candidate.edge_lengths().sum(), expected)
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from((int(a), int(b)) for a, b in candidate.edge_array())
    assert nx.is_eulerian(multigraph)
