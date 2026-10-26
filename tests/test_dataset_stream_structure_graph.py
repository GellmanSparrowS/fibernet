"""Dataset adapter regressions: stable graph IDs are not position-row indices."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph
from fibernet.ml.physical_dataset import DatasetStream
from fibernet.ml.physical_surrogate import light_features


def stream_for(graph, feature_fn=light_features):
    def forbidden_sample(factory):
        raise AssertionError('generate-mode adapter tests must not calculate labels')

    return DatasetStream(
        dict(seed0=0, mode='generate', acquisition='random'),
        factory_builder=lambda seed, cfg, rng: SimpleNamespace(
            seed=seed, build=lambda: graph),
        feature_fn=feature_fn,
        sample_fn=forbidden_sample,
        spec_fn=lambda factory: {'seed': factory.seed},
    )


def snapshot(graph):
    # Serialization orders IDs and contains live metadata references: freeze it
    # independently and retain insertion order as a separate part of the oracle.
    return (json.dumps(graph.to_dict(), sort_keys=True),
            list(graph.nodes), list(graph.edges))


def two_parallel_edges(remove_isolated=False):
    graph = StructureGraph(2)
    if remove_isolated:
        isolated = graph.add_node([9., 9.], merge=False)
    u = graph.add_node([0., 0.], merge=False)
    v = graph.add_node([1., 1.], merge=False)
    graph.add_edge(u, v, radius=.1)
    graph.add_edge(v, u, radius=.2, allow_parallel=True)
    if remove_isolated:
        graph.remove_node(isolated)
    return graph


@pytest.mark.parametrize('remove_isolated', [False, True])
def test_removing_isolated_node_does_not_reject_same_parallel_geometry(remove_isolated):
    graph = two_parallel_edges(remove_isolated)
    before = snapshot(graph)
    seen = []

    def features(positions, edges):
        seen.append((positions.copy(), edges.copy()))
        return light_features(positions, edges)

    stream = stream_for(graph, features)
    x, y = stream.generate(1)
    expected_positions = np.array([[0., 0., 0.], [1., 1., 0.]])
    expected_edges = np.array([[0, 1], [1, 0]])
    np.testing.assert_array_equal(seen[0][0], expected_positions)
    np.testing.assert_array_equal(seen[0][1], expected_edges)
    np.testing.assert_allclose(x[0], light_features(expected_positions, expected_edges))
    assert x.shape == (1, 14) and y.shape == (1, 3) and np.isnan(y).all()
    assert stream.specs == [{'seed': 0}] and not stream.failures
    assert x[0, 4] == 1.0  # Both parallel physical members count toward degree 2.
    assert x[0, 10] == .002 and x[0, 11] == 1.0
    assert snapshot(graph) == before


def rich_graph(ids, node_order, edge_order):
    positions = [[0., 0., 0.], [1., 1., .5], [2., 0., 0.], [3., 2., 0.]]
    graph = StructureGraph(3, box_size=[4., 4., 2.])
    for index, position in enumerate(positions):
        graph.add_node(position, merge=False, label={'row': index})
    graph.add_edge(0, 1, radius=.1, internal_points=np.array([[.4, .8, .2]]),
                   material=Material(name='first', youngs_modulus=2e9),
                   label={'member': 'forward'})
    graph.add_edge(1, 0, radius=.2, internal_points=np.array([[.7, .2, .3]]),
                   material=Material(name='second', youngs_modulus=3e9),
                   label={'member': 'reverse'}, allow_parallel=True)
    graph.add_edge(1, 2, radius=.3, label={'member': 'tail'})
    data = graph.to_dict()
    for index, node in enumerate(data['nodes']):
        node['id'] = ids[index]
    for index, edge in enumerate(data['edges']):
        edge['id'] = [30, 4, 80][index]
        edge['node_i'] = ids[edge['node_i']]
        edge['node_j'] = ids[edge['node_j']]
    data['next_node_id'] = max(ids) + 1
    data['next_edge_id'] = 81
    data['nodes'] = [data['nodes'][index] for index in node_order]
    data['edges'] = [data['edges'][index] for index in edge_order]
    return StructureGraph.from_dict(data)


@pytest.mark.parametrize('ids,position_order,base_edges', [
    ((0, 1, 2, 3), (0, 1, 2, 3), [[0, 1], [1, 0], [1, 2]]),
    ((40, 7, 90, 15), (1, 3, 0, 2), [[2, 0], [0, 2], [0, 3]]),
])
@pytest.mark.parametrize('node_order', [(0, 1, 2, 3), (3, 1, 0, 2)])
@pytest.mark.parametrize('edge_order', [(0, 1, 2), (2, 1, 0)])
def test_stable_ids_and_permuted_insertions_preserve_physical_members(
        ids, position_order, base_edges, node_order, edge_order):
    graph = rich_graph(ids, node_order, edge_order)
    before = snapshot(graph)
    seen = []

    def features(positions, edges):
        seen.append((positions.copy(), edges.copy()))
        return light_features(positions, edges)

    x, y = stream_for(graph, features).generate(1)
    all_positions = np.array([[0., 0., 0.], [1., 1., .5], [2., 0., 0.], [3., 2., 0.]])
    expected_positions = all_positions[list(position_order)]
    expected_edges = np.array(base_edges)[list(edge_order)]
    np.testing.assert_array_equal(seen[0][0], expected_positions)
    np.testing.assert_array_equal(seen[0][1], expected_edges)
    # The same three endpoint chords have identical descriptors under all IDs
    # and permutations. Curves/materials remain source provenance, not features.
    np.testing.assert_allclose(x[0], light_features(all_positions, np.array([[0, 1], [1, 0], [1, 2]])))
    assert x[0, 4] == .25 and x[0, 5] == .25
    assert x[0, 9] == .004 and x[0, 10] == .003
    assert np.isnan(y).all() and snapshot(graph) == before
    assert list(graph.nodes) == [ids[index] for index in node_order]
    assert list(graph.edges) == [[30, 4, 80][index] for index in edge_order]


def test_callback_receives_detached_projection_not_mutable_graph_storage():
    graph = rich_graph((40, 7, 90, 15), (3, 1, 0, 2), (2, 1, 0))
    before = snapshot(graph)

    def mutating_features(positions, edges):
        result = light_features(positions, edges)
        positions[:] = -123.
        edges[:] = 0
        return result

    x, _ = stream_for(graph, mutating_features).generate(1)
    assert np.isfinite(x).all() and snapshot(graph) == before


@pytest.mark.parametrize('node_count', [0, 1, 2])
def test_empty_edge_callback_shape_is_unchanged(node_count):
    graph = StructureGraph(2)
    for index in range(node_count):
        graph.add_node([float(index), 0.], merge=False)
    before = snapshot(graph)
    seen = []

    def accepts_empty(positions, edges):
        seen.append((positions.shape, edges.shape))
        return np.zeros(14)

    x, y = stream_for(graph, accepts_empty).generate(1)
    assert seen == [((node_count, 3), (0, 2))]
    assert x.shape == (1, 14) and np.isnan(y).all()
    assert snapshot(graph) == before


@pytest.mark.parametrize('node_count', [0, 1, 2])
def test_empty_graphs_remain_rejected_by_light_features(node_count):
    graph = StructureGraph(2)
    for index in range(node_count):
        graph.add_node([float(index), 0.], merge=False)
    stream = stream_for(graph)
    with pytest.raises(ValueError, match='no valid candidate structures'):
        stream.generate(1)
    assert stream.failures == [{'seed': 0, 'error': 'features require finite nodes and integer edges'}]
    assert len(stream.X) == len(stream.Y) == len(stream.specs) == 0


def test_corrupt_missing_endpoint_is_rejected_without_committing_row():
    graph = two_parallel_edges()
    # Deliberately corrupt mutable public edge data; no valid graph is relabeled.
    graph.edges[0].node_j = 99
    before = snapshot(graph)
    stream = stream_for(graph)
    with pytest.raises(ValueError, match='no valid candidate structures'):
        stream.generate(1)
    assert stream.failures == [{'seed': 0, 'error': 'feature edges contain invalid node IDs'}]
    assert len(stream.X) == len(stream.Y) == len(stream.specs) == 0
    assert snapshot(graph) == before


def test_nonfinite_source_positions_are_still_rejected():
    graph = two_parallel_edges()
    graph.nodes[0].position[0] = np.nan
    stream = stream_for(graph)
    with pytest.raises(ValueError, match='no valid candidate structures'):
        stream.generate(1)
    assert stream.failures == [{'seed': 0, 'error': 'features require finite nodes and integer edges'}]
    assert not stream.X and not stream.Y and not stream.specs


def test_legacy_graph_like_callbacks_keep_their_dense_array_contract():
    positions = np.array([[0., 0.], [1., 1.]])
    edges = np.array([[0, 1], [1, 0]])

    def must_not_project():
        raise AssertionError('Only StructureGraph uses the authoritative projection')

    graph = SimpleNamespace(node_positions=lambda: positions, edge_array=lambda: edges,
                            to_numpy=must_not_project)

    def features(actual_positions, actual_edges):
        assert actual_positions is positions and actual_edges is edges
        return light_features(actual_positions, actual_edges)

    x, _ = stream_for(graph, features).generate(1)
    np.testing.assert_allclose(x[0], light_features(positions, edges))


@pytest.mark.parametrize('positions,edges,error', [
    ([[0., 0.], [1., 1.]], np.array([[0, 2]]), 'invalid node indices'),
    ([[0., 0.], [1., 1.]], np.array([[-1, 1]]), 'invalid node indices'),
    ([[0., 0.], [1., 1.]], np.array([[0, 0]]), 'invalid node indices'),
    ([[0., 0.], [1., 1.]], np.array([[0., 1.]]), 'finite nodes and integer edges'),
    ([[0., 0.], [1., 1.]], np.array([], dtype=int), 'finite nodes and integer edges'),
    ([[0., 0.], [1., 1.]], np.empty((0, 2), dtype=int), 'finite nodes and integer edges'),
    ([[0., 0.], [np.inf, 1.]], np.array([[0, 1]]), 'finite nodes and integer edges'),
])
def test_invalid_legacy_arrays_remain_rejected(positions, edges, error):
    graph = SimpleNamespace(node_positions=lambda: np.asarray(positions), edge_array=lambda: edges)
    stream = stream_for(graph)
    with pytest.raises(ValueError, match='no valid candidate structures'):
        stream.generate(1)
    assert len(stream.failures) == 1 and error in stream.failures[0]['error']
    assert len(stream.X) == len(stream.Y) == len(stream.specs) == 0
