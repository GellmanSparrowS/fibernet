"""Stable-ID field interchange, including actual VTU reloads and legacy bytes."""
from collections.abc import Mapping
import copy
import json

import numpy as np
import pytest

import fibernet as fn
from fibernet.io.meshio_adapter import to_meshio, write_vtu_bundle


def fixture():
    graph = fn.StructureGraph(dimension=3)
    for position in ([0, 0, 0], [1, 2, 3], [9, 8, 7]):
        graph.add_node(position)
    graph.add_edge(0, 1, radius=.1)
    graph.add_edge(1, 0, radius=.2, allow_parallel=True)
    data = graph.to_dict()
    ids = {0: 7, 1: 29, 2: 103}
    for node in data['nodes']:
        node['id'] = ids[node['id']]
    for edge in data['edges']:
        edge['id'] = 11 + 13 * edge['id']
        edge['node_i'], edge['node_j'] = ids[edge['node_i']], ids[edge['node_j']]
    data['nodes'].reverse(); data['edges'].reverse()
    data.pop('next_node_id'); data.pop('next_edge_id')
    return fn.StructureGraph.from_dict(data)


def fields(graph):
    point = {'temperature': {i: float(i) for i in reversed(sorted(graph.nodes))},
             'displacement': {i: np.array([i, i + 1, i + 2.]) for i in graph.nodes}}
    cell = {'current': {i: float(i) for i in reversed(graph.edges)},
            'direction': {i: np.array([i, -i, 0.]) for i in reversed(graph.edges)}}
    metadata = {'point': {name: {'units': 'K' if name == 'temperature' else 'mm',
                                'meaning': 'synthetic point field'} for name in point},
                'cell': {name: {'units': 'A' if name == 'current' else '1',
                               'meaning': 'synthetic cell field'} for name in cell}}
    return point, cell, metadata


def test_id_fields_reload_legacy_bytes_and_immutability(tmp_path):
    meshio = pytest.importorskip('meshio')
    graph = fixture()
    point, cell, metadata = fields(graph)
    before = copy.deepcopy((graph.to_dict(), point, cell, metadata))
    kwargs = dict(length_unit='mm', field_metadata=metadata,
                  provenance={'source': 'ID fixture'})
    array_point = {n: np.array([v[i] for i in sorted(graph.nodes)]) for n, v in point.items()}
    array_cell = {n: np.array([v[i] for i in graph.edges]) for n, v in cell.items()}
    old = write_vtu_bundle(tmp_path / 'arrays', graph, point_data=array_point,
                           cell_data=array_cell, **kwargs)
    new = write_vtu_bundle(tmp_path / 'ids', graph, point_data_by_id=point,
                           cell_data_by_id=cell, **kwargs)
    for filename in ('network.vtu', 'manifest.json'):
        assert (old / filename).read_bytes() == (new / filename).read_bytes()
    loaded = meshio.read(new / 'network.vtu')
    np.testing.assert_array_equal(loaded.points, [graph.nodes[i].position for i in sorted(graph.nodes)])
    np.testing.assert_array_equal(loaded.cells_dict['line'], [[1, 0], [0, 1]])
    assert loaded.point_data['node_id'].tolist() == [7, 29, 103]
    assert loaded.cell_data['edge_id'][0].tolist() == [24, 11]
    for source, ids, loaded_fields in ((point, loaded.point_data['node_id'], loaded.point_data),
                                     (cell, loaded.cell_data['edge_id'][0],
                                      {k: v[0] for k, v in loaded.cell_data.items()})):
        for name, values in source.items():
            for identifier, value in zip(ids, loaded_fields[name]):
                np.testing.assert_array_equal(value, values[identifier])
    assert graph.to_dict() == before[0]
    assert metadata == before[3]
    for current, original in ((point, before[1]), (cell, before[2])):
        assert list(current) == list(original)
        for name in current:
            assert list(current[name]) == list(original[name])
            for key in current[name]:
                np.testing.assert_array_equal(current[name][key], original[name][key])
    for name in array_point:
        np.testing.assert_array_equal(array_point[name], loaded.point_data[name])
    for name in array_cell:
        np.testing.assert_array_equal(array_cell[name], loaded.cell_data[name][0])
    mesh, manifest = to_meshio(graph, point_data_by_id=point, cell_data_by_id=cell, **kwargs)
    mesh.point_data['displacement'][0, 0] = -999
    mesh.cell_data['direction'][0][0, 0] = -999
    manifest['field_metadata']['point']['temperature']['units'] = 'changed'
    np.testing.assert_array_equal(point['displacement'][7], [7, 8, 9])
    np.testing.assert_array_equal(cell['direction'][24], [24, -24, 0])
    assert metadata == before[3]


@pytest.mark.parametrize('location', ['point', 'cell'])
@pytest.mark.parametrize('key', [True, np.bool_(False), '7', 7.0, 7.5, -1, 2**63, None])
def test_invalid_ids(location, key):
    graph = fixture()
    with pytest.raises(ValueError, match='nonnegative int64'):
        to_meshio(graph, length_unit='m', **{location + '_data_by_id': {'x': {key: 1}}})


@pytest.mark.parametrize('location', ['point', 'cell'])
@pytest.mark.parametrize('change', ['missing', 'foreign', 'empty', 'none', 'nan', 'infinite',
                                    'vector2', 'mixed', 'string', 'bool', 'reserved',
                                    'no_metadata', 'empty_units', 'empty_meaning',
                                    'bad_outer', 'bad_inner'])
def test_rejected_fields_do_not_publish(tmp_path, location, change):
    graph = fixture()
    ids = sorted(graph.nodes) if location == 'point' else list(graph.edges)
    field = {i: 1. for i in ids}
    name = 'radius' if change == 'reserved' else 'x'
    values = {name: field}
    spec = {'units': '1', 'meaning': 'synthetic'}
    metadata = {location: {name: spec}}
    if change == 'missing': field.pop(ids[0])
    if change == 'foreign': field[999] = 1.
    if change == 'empty': field.clear()
    if change in ('none', 'nan', 'infinite', 'vector2', 'mixed', 'string', 'bool'):
        field[ids[0]] = {'none': None, 'nan': np.nan, 'infinite': np.inf,
                          'vector2': [1, 2], 'mixed': [1, 2, 3], 'string': 'bad', 'bool': True}[change]
        if change == 'bool': field.update({i: True for i in ids})
    if change == 'no_metadata': metadata = {}
    if change == 'empty_units': spec['units'] = ''
    if change == 'empty_meaning': spec['meaning'] = ''
    if change == 'bad_outer': values = []
    if change == 'bad_inner': values = {name: [(i, 1) for i in ids]}
    with pytest.raises(ValueError):
        write_vtu_bundle(tmp_path / 'invalid', graph, length_unit='m',
                         field_metadata=metadata, **{location + '_data_by_id': values})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('location', ['point', 'cell'])
@pytest.mark.parametrize('array,idmap', [({}, {}), ({'a': []}, {}), ({}, {'b': {}}), ([], {})])
def test_exclusive_even_empty(location, array, idmap):
    with pytest.raises(ValueError, match='mutually exclusive'):
        to_meshio(fixture(), length_unit='m', **{location + '_data': array,
                                               location + '_data_by_id': idmap})


def test_independent_locations_and_numpy_integer_keys():
    pytest.importorskip('meshio')
    graph = fixture()
    point, cell, metadata = fields(graph)
    point = {n: {np.int64(k): v for k, v in values.items()} for n, values in point.items()}
    mesh, _ = to_meshio(graph, length_unit='mm', point_data_by_id=point,
                        cell_data={n: [v[i] for i in graph.edges] for n, v in cell.items()},
                        field_metadata=metadata)
    assert mesh.point_data['temperature'].tolist() == [7, 29, 103]
    to_meshio(graph, length_unit='mm', point_data={}, cell_data_by_id={})


class RepeatedItems(Mapping):
    """An observable duplicate sequence, unlike already-overwritten dict keys."""
    def __getitem__(self, key): return 1.
    def __iter__(self): return iter([7, 7, 29, 103])
    def __len__(self): return 4
    def items(self): return [(7, 1.), (np.int64(7), 2.), (29, 3.), (103, 4.)]


def test_observable_duplicate_ids_rejected():
    with pytest.raises(ValueError, match='duplicate IDs'):
        to_meshio(fixture(), length_unit='m', point_data_by_id={'x': RepeatedItems()})


def test_generated_transport_id_export_journey(tmp_path):
    meshio = pytest.importorskip('meshio')
    graph = fn.pattern_2d(unit='square', box=(1, 1), grid=(1, 1), n_pts_per_side=0)
    data = graph.to_dict()
    mapping = {i: 9 + 13 * i for i in graph.nodes}
    for node in data['nodes']: node['id'] = mapping[node['id']]
    for edge in data['edges']:
        edge['id'] += 17
        edge['node_i'], edge['node_j'] = mapping[edge['node_i']], mapping[edge['node_j']]
    data['nodes'].reverse(); data['edges'].reverse()
    data.pop('next_node_id'); data.pop('next_edge_id')
    graph = fn.StructureGraph.from_dict(data)
    graph.save_json(tmp_path / 'graph.json')
    graph = fn.StructureGraph.load_json(tmp_path / 'graph.json')
    result = fn.solve_linear_transport(
        graph, {i: 1. for i in graph.edges},
        [(i, 1. - node.position[0]) for i, node in graph.nodes.items()], length_unit='mm')
    point = dict(reversed(list(zip(result.node_ids, result.voltage_V))))
    cell = dict(reversed(list(zip(result.edge_ids, result.current_A))))
    target = write_vtu_bundle(tmp_path / 'electrical', graph, length_unit='mm',
        point_data_by_id={'potential': point}, cell_data_by_id={'current': cell},
        field_metadata={'point': {'potential': {'units': 'V', 'meaning': 'electric potential'}},
                        'cell': {'current': {'units': 'A', 'meaning': 'signed node_i to node_j'}}},
        provenance={'electrical_result': result.to_dict()})
    mesh = meshio.read(target / 'network.vtu')
    for identifier, value in zip(mesh.point_data['node_id'], mesh.point_data['potential']):
        assert value == point[identifier]
    for identifier, value in zip(mesh.cell_data['edge_id'][0], mesh.cell_data['current'][0]):
        assert value == cell[identifier]
    np.testing.assert_array_equal(mesh.points, [graph.nodes[i].position for i in sorted(graph.nodes)])
    rows = {i: row for row, i in enumerate(sorted(graph.nodes))}
    np.testing.assert_array_equal(mesh.cells_dict['line'], [[rows[e.node_i], rows[e.node_j]] for e in graph.edges.values()])
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['length_unit'] == 'mm'
    assert manifest['field_metadata']['cell']['current']['units'] == 'A'
    assert manifest['provenance']['electrical_result'] == result.to_dict()
