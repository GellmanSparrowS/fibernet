"""Native curve serialization validates coordinates before admission or writes."""
import json
import pickle
from unittest.mock import patch

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


@pytest.mark.parametrize('points', [
    pytest.param(np.array([[np.nan, 0., 0.]]), id='nan'),
    pytest.param(np.array([[np.inf, 0., 0.]]), id='positive-inf'),
    pytest.param(np.array([[-np.inf, 0., 0.]]), id='negative-inf'),
    pytest.param(np.array([1., 2., 3.]), id='one-dimensional'),
    pytest.param(np.ones((1, 2)), id='two-columns'),
    pytest.param(np.ones((1, 4)), id='four-columns'),
    pytest.param(np.empty((0, 2)), id='empty-wrong-columns'),
    pytest.param(np.empty((1, 0)), id='empty-row'),
    pytest.param(np.ones((1, 1, 3)), id='three-dimensional'),
    pytest.param(np.array(1.), id='scalar-array'),
    pytest.param([[1, 2, 3], [4]], id='ragged'),
    pytest.param([[True, 0, 0]], id='mixed-bool'),
    pytest.param(np.ones((1, 3), dtype=bool), id='bool-array'),
    pytest.param([['1', '0', '0']], id='numeric-strings'),
    pytest.param(np.array([['1', '0', '0']]), id='string-array'),
    pytest.param(np.array([[1, 0, 0]], dtype=object), id='object-array'),
    pytest.param(np.array([[1j, 0, 0]]), id='complex-array'),
    pytest.param([[None, 0, 0]], id='null-coordinate'),
    pytest.param(np.ma.array([[1., 0., 0.]], mask=False), id='unmasked-array'),
    pytest.param(np.ma.array([[1., 0., 0.]], mask=[[True, False, False]]),
                 id='masked-array'),
    pytest.param([np.ma.array([1., 0., 0.], mask=False)], id='nested-masked'),
])
def test_writer_rejects_raw_curve_before_open_and_preserves_destinations(points, tmp_path):
    graph = StructureGraph.from_dict(record([[1., 0., 1.]]))
    existing = tmp_path / 'existing.json'
    graph.save_json(existing)
    previous_bytes = existing.read_bytes()
    previous_record = graph.to_dict()
    new = tmp_path / 'new.json'
    graph.edges[101].internal_points = points
    before = pickle.dumps(points)

    with pytest.raises(ValueError, match='internal_points.*edge 101'):
        graph.to_dict()
    with patch('fibernet.core.structure_graph.open', create=True) as open_destination:
        for destination in (existing, new):
            with pytest.raises(ValueError, match='internal_points.*edge 101'):
                graph.save_json(destination)
        open_destination.assert_not_called()

    assert existing.read_bytes() == previous_bytes
    assert StructureGraph.load_json(existing).to_dict() == previous_record
    assert not new.exists()
    assert graph.edges[101].internal_points is points
    assert pickle.dumps(points) == before


@pytest.mark.parametrize('points', [
    None, np.empty((0, 3)), np.array([[1, 0, 1]], dtype=np.int64),
    np.array([[1.5, 0., 1.]], dtype=np.float32),
    np.array([[1.5, 0., 1.]], dtype=np.float64),
    [[1, 0, 1]], ((1., 0., 1.),), [[2**53 + 1, 0., 0.]],
])
def test_writer_valid_raw_curves_keep_format_roundtrip_and_input(points, tmp_path):
    graph = StructureGraph.from_dict(record(None))
    graph.edges[101].internal_points = points
    before = pickle.dumps(points)
    expected = record(None if points is None else np.array(points).tolist())
    expected = StructureGraph.from_dict(expected).to_dict()
    data = graph.to_dict()
    assert data == expected

    destination = tmp_path / 'graph.json'
    graph.save_json(destination)
    reference = tmp_path / 'reference.json'
    with open(reference, 'w') as stream:
        json.dump(expected, stream, indent=2)
    assert destination.read_bytes() == reference.read_bytes()
    assert not destination.read_bytes().endswith(b'\n')
    for restored in (StructureGraph.load_json(destination), fn.load(destination)):
        assert restored.to_dict() == expected
        if points is None:
            assert restored.edges[101].internal_points is None
        elif np.array(points).size == 0:
            assert restored.edges[101].internal_points.shape == (0, 3)
    assert graph.edges[101].internal_points is points
    assert pickle.dumps(points) == before
