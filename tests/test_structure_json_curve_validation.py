"""Native curve deserialization rejects malformed coordinates before admission."""
import json
import pickle

import numpy as np
import pytest

from fibernet import api as fn
from fibernet.core.structure_graph import StructureGraph


def record(points):
    return {'version': '3.1', 'dimension': 3,
            'nodes': [{'id': 10, 'position': [0, 0, 0]},
                      {'id': 30, 'position': [2, 0, 0]}],
            'edges': [{'id': 101, 'node_i': 10, 'node_j': 30,
                       'internal_points': points},
                      {'id': 205, 'node_i': 30, 'node_j': 10,
                       'internal_points': [[1, 0, -2]]}],
            'next_node_id': 80, 'next_edge_id': 300}


@pytest.mark.parametrize('points', [
    [['bad', 0, 0]], [['1', '0', '0']], [[True, 0, 0]],
    [[1j, 0, 0]], [[None, 0, 0]], [[float('nan'), 0, 0]],
    [[float('inf'), 0, 0]], [[-float('inf'), 0, 0]],
    np.array([[1, 0, 0]], dtype=object), np.ones((1, 3), dtype=bool),
    np.ma.array([[1., 0., 0.]], mask=False),
    [np.ma.array([1., 0., 0.], mask=[True, False, False])],
    [[1, 2]], [[1, 2, 3], [4]],
])
def test_from_dict_rejects_invalid_curve_without_input_mutation(points):
    data = record(points)
    before = pickle.dumps(data)
    with pytest.raises(ValueError, match='internal_points.*edge 101'):
        StructureGraph.from_dict(data)
    assert pickle.dumps(data) == before


@pytest.mark.parametrize('points', [None, [], [[1, 0, 1]], [[1.5, 0., 1.]],
                                    np.empty((0, 3)), np.array([[1, 0, 1]], dtype=np.int64),
                                    np.array([[1.5, 0., 1.]], dtype=np.float32)])
def test_valid_sparse_parallel_roundtrip_detaches_input(points, tmp_path):
    data = record(points)
    before = pickle.dumps(data)
    graph = StructureGraph.from_dict(data)
    assert pickle.dumps(data) == before
    assert sorted(graph.nodes) == [10, 30]
    assert sorted(graph.edges) == [101, 205]
    assert graph._next_node_id == 80 and graph._next_edge_id == 300
    stored = graph.edges[101].internal_points
    if points is None:
        assert stored is None
    else:
        expected = np.array(points)
        if expected.shape == (0,):
            assert stored.shape == (0, 3)
            assert stored.dtype == np.dtype(float)
        else:
            assert stored.shape == expected.shape
            assert stored.dtype == expected.dtype
            np.testing.assert_array_equal(stored, expected)
    path = tmp_path / 'graph.json'
    graph.save_json(path)
    for restored in [StructureGraph.load_json(path), fn.load(path)]:
        assert restored.to_dict() == graph.to_dict()
        assert sorted(restored.edges) == [101, 205]
        if points is None:
            assert restored.edges[101].internal_points is None
        elif stored.size == 0:
            assert restored.edges[101].internal_points.shape == (0, 3)
    if points is not None:
        stored = graph.edges[101].internal_points
        assert stored.shape[1] == 3
        if stored.size:
            stored[0, 0] = 99
        assert pickle.dumps(data) == before


@pytest.mark.parametrize('reader', [StructureGraph.load_json, fn.load])
@pytest.mark.parametrize('bad_json', ['[["bad",0,0]]', '[[true,0,0]]', '[[1e309,0,0]]'])
def test_json_readers_reject_strings_bools_and_parser_overflow(reader, bad_json, tmp_path):
    data = record('CURVE_PLACEHOLDER')
    payload = json.dumps(data).replace('"CURVE_PLACEHOLDER"', bad_json)
    path = tmp_path / 'invalid.json'
    path.write_text(payload)
    with pytest.raises(ValueError, match='internal_points.*edge 101'):
        reader(path)
    assert path.read_text() == payload


def test_finite_mixed_numeric_coordinates_use_ordinary_numpy_promotion():
    points = [[2**53 + 1, 0., 0.]]
    data = record(points)
    before = pickle.dumps(data)
    graph = StructureGraph.from_dict(data)
    np.testing.assert_array_equal(graph.edges[101].internal_points, np.array(points))
    assert pickle.dumps(data) == before
