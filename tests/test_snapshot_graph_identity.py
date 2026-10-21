"""Snapshot adapters map graph identity, while raw arrays retain row identity."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fibernet.analysis import ContactConfig, SnapshotFeatureExtractor
from fibernet.core.structure_graph import StructureGraph


def _graph(ids=(0, 1, 2), reverse=False):
    nodes = [{'id': n, 'position': p} for n, p in zip(
        ids, ([0, 0, 0], [3, 0, 0], [1, 2, 0]))]
    # Non-sorted edge IDs, a parallel pair, mixed radii and directed endpoints.
    edges = [{'id': e, 'node_i': ids[a], 'node_j': ids[b], 'radius': r}
             for e, a, b, r in ((90, 1, 2, .1), (7, 0, 1, .2),
                                (50, 2, 0, .3), (2, 0, 1, .4))]
    return StructureGraph.from_dict({'nodes': nodes[::-1] if reverse else nodes,
                                     'edges': edges})


def _serialized(features):
    # Preserve dict and histogram order; no rounding or tolerance-based comparison.
    return json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in features.items()},
                      separators=(',', ':'), allow_nan=False)


@pytest.mark.parametrize('contact', [False, True])
@pytest.mark.parametrize('rect', [None, [-1, -1, 4, 3], [0, -.1, 3, .1],
                                 [10, 10, 11, 11]])
@pytest.mark.parametrize('reverse', [False, True])
def test_sparse_graph_matches_all_dense_features(contact, rect, reverse):
    extractor = SnapshotFeatureExtractor(include_contact=contact,
        contact_config=ContactConfig(width=.16, resolution=64, max_pixels=200000))
    dense, sparse = _graph(), _graph((10, 30, 70), reverse)
    before = copy.deepcopy(sparse.to_dict())
    node_order, edge_order = list(sparse.nodes), list(sparse.edges)
    expected = extractor.extract(dense.node_positions()[:, :2], dense.edge_array(), rect)
    actual = extractor.extract_graph(sparse, rect)
    assert _serialized(actual) == _serialized(expected)
    assert _serialized(extractor.extract_graph(dense, rect)) == _serialized(expected)
    assert sparse.to_dict() == before
    assert list(sparse.nodes) == node_order
    assert list(sparse.edges) == edge_order == [90, 7, 50, 2]
    if rect is None:
        assert actual['n_edge'] == 4
        np.testing.assert_array_equal(actual['edge_lengths'],
                                      [np.sqrt(8), 3, np.sqrt(5), 3])
        np.testing.assert_array_equal(actual['degree_hist'], [3, 3, 2])


def test_raw_api_still_rejects_sparse_endpoint_ids():
    graph = _graph((10, 30, 70))
    with pytest.raises(ValueError, match='invalid node indices'):
        SnapshotFeatureExtractor().extract(graph.node_positions()[:, :2], graph.edge_array())


def test_generic_graph_keeps_dense_row_protocol_without_guessing():
    graph = _graph()
    positions, edges = graph.node_positions(), graph.edge_array()
    before_pos, before_edges = positions.copy(), edges.copy()
    def unsupported_export():
        raise AssertionError('Generic to_numpy must not be used')
    legacy = SimpleNamespace(node_positions=lambda: positions,
                             edge_array=lambda: edges,
                             nodes={70: None, 10: None, 30: None},
                             to_numpy=unsupported_export)
    extractor = SnapshotFeatureExtractor()
    assert _serialized(extractor.extract_graph(legacy)) == _serialized(
        extractor.extract(positions[:, :2], edges))
    np.testing.assert_array_equal(positions, before_pos)
    np.testing.assert_array_equal(edges, before_edges)
    legacy.edge_array = lambda: np.array([[10, 30]])
    with pytest.raises(ValueError, match='invalid node indices'):
        extractor.extract_graph(legacy)


@pytest.mark.parametrize('nodes', [[], [{'id': 70, 'position': [1, 2, 3]}],
    [{'id': 70, 'position': [1, 2, 3]}, {'id': 10, 'position': [3, 4, 3]}]])
def test_empty_and_isolated_graphs(nodes):
    graph = StructureGraph.from_dict({'nodes': nodes, 'edges': []})
    extractor = SnapshotFeatureExtractor()
    actual = extractor.extract_graph(graph)
    expected = extractor.extract(graph.node_positions()[:, :2], np.empty((0, 2), int))
    assert _serialized(actual) == _serialized(expected)
    assert actual['n_node'] == len(nodes)
    assert actual['n_edge'] == 0


@pytest.mark.parametrize('kwargs', [{'max_nodes': 2}, {'max_edges': 3}])
def test_graph_budgets_precede_region_filtering(kwargs):
    with pytest.raises(MemoryError, match='budget'):
        SnapshotFeatureExtractor(**kwargs).extract_graph(_graph((10, 30, 70)),
                                                        rect=[10, 10, 11, 11])


def test_graph_planarity_and_rect_validation_unchanged():
    graph = _graph((10, 30, 70))
    extractor = SnapshotFeatureExtractor()
    with pytest.raises(ValueError, match='four finite bounds'):
        extractor.extract_graph(graph, rect=[0, 1, np.inf, 3])
    graph.nodes[30].position[2] = .1
    with pytest.raises(ValueError, match='planar'):
        extractor.extract_graph(graph)


def test_generic_planarity_error_precedes_edge_access():
    def unexpected_edges():
        raise AssertionError('Nonplanar graph must fail before edge access')
    legacy = SimpleNamespace(node_positions=lambda: np.array([[0, 0, 0], [1, 1, .1]]),
                             edge_array=unexpected_edges)
    with pytest.raises(ValueError, match='planar'):
        SnapshotFeatureExtractor().extract_graph(legacy)
