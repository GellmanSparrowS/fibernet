"""Sparse graph IDs survive JSON, including subsequent graph editing."""
import copy

import numpy as np
import pytest

from fibernet import StructureGraph


def _sparse_graph():
    graph = StructureGraph(dimension=2, box_size=[3, 3])
    for p in ([99, 99], [0, 0], [88, 88], [2, 0], [77, 77]):
        graph.add_node(p, merge=False)
    for nid in (0, 2, 4):
        graph.remove_node(nid)
    for radius in (.1, .2, .3, .4):
        graph.add_edge(1, 3, radius=radius, allow_parallel=True,
                       internal_points=np.array([[1., .1, 0.]]))
    graph.remove_edge(0)
    graph.remove_edge(3)
    graph.metadata['weld_triplets'] = [[1, 3, 1]]
    graph.nodes[1].metadata = {'merge': 'metadata, not an add_node argument'}
    graph.edges[1].metadata = {'allow_parallel': 'metadata', 'nested': {'a': 1}}
    return graph


def test_sparse_parallel_json_roundtrip_and_next_ids(tmp_path):
    graph = _sparse_graph()
    path = tmp_path / 'graph.json'
    graph.save_json(path)
    restored = StructureGraph.load_json(path)
    assert restored.to_dict() == graph.to_dict()
    assert list(restored.nodes) == [1, 3]
    assert list(restored.edges) == [1, 2]
    assert [node.node_id for node in restored.nodes.values()] == [1, 3]
    assert [edge.edge_id for edge in restored.edges.values()] == [1, 2]
    assert restored.add_node([0, 0]) == 1  # Rebuilt spatial lookup.
    assert restored.add_node([3, 0]) == 5  # Deleted highest ID is not reused.
    assert restored.add_edge(1, 3) == 1  # Rebuilt parallel-edge lookup.
    assert restored.add_edge(1, 3, allow_parallel=True) == 4


def test_unsorted_records_and_legacy_missing_counters():
    data = _sparse_graph().to_dict()
    data['nodes'].reverse()
    data['edges'].reverse()
    data.pop('next_node_id')
    data.pop('next_edge_id')
    restored = StructureGraph.from_dict(data)
    assert sorted(restored.nodes) == [1, 3]
    assert sorted(restored.edges) == [1, 2]
    assert restored.add_node([3, 0]) == 4
    assert restored.add_edge(1, 3, allow_parallel=True) == 3


def test_legacy_idless_records_and_independent_metadata():
    data = {'nodes': [{'position': [0, 0]}, {'position': [1, 0]}],
            'edges': [{'node_i': 0, 'node_j': 1}, {'node_i': 0, 'node_j': 1}],
            'metadata': {'nested': {'value': 1}}}
    restored = StructureGraph.from_dict(data)
    assert list(restored.nodes) == [0, 1]
    assert list(restored.edges) == [0, 1]
    restored.metadata['nested']['value'] = 2
    assert data['metadata']['nested']['value'] == 1


@pytest.mark.parametrize('kind', ['duplicate_node', 'duplicate_edge', 'bad_endpoint',
                                 'self_loop', 'boolean_id', 'fractional_id',
                                 'negative_id', 'bad_node_counter', 'bad_edge_counter'])
def test_invalid_serialized_ids_are_rejected(kind):
    data = copy.deepcopy(_sparse_graph().to_dict())
    if kind == 'duplicate_node':
        data['nodes'][1]['id'] = data['nodes'][0]['id']
    elif kind == 'duplicate_edge':
        data['edges'][1]['id'] = data['edges'][0]['id']
    elif kind == 'bad_endpoint':
        data['edges'][0]['node_j'] = 999
    elif kind == 'self_loop':
        data['edges'][0]['node_j'] = data['edges'][0]['node_i']
    elif kind in ('boolean_id', 'fractional_id', 'negative_id'):
        data['nodes'][0]['id'] = {'boolean_id': True, 'fractional_id': 1.5,
                                 'negative_id': -1}[kind]
    elif kind == 'bad_node_counter':
        data['next_node_id'] = 3
    else:
        data['next_edge_id'] = 2
    with pytest.raises(ValueError):
        StructureGraph.from_dict(data)
