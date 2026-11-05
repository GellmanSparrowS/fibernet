"""Masked fields must not silently become available numeric export data."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fibernet import Material, StructureGraph
from fibernet.io import meshio_adapter


@pytest.fixture
def adapter():
    return meshio_adapter


@pytest.fixture
def graph():
    graph = StructureGraph(dimension=3)
    a = graph.add_node([1., 0., 0.])
    b = graph.add_node([0., 0., 0.])
    graph.add_edge(a, b, radius=.2, material=Material(name='second'))
    graph.add_edge(b, a, radius=.1, material=Material(name='first'), allow_parallel=True)
    data = graph.to_dict()
    node_ids = {a: 29, b: 7}
    for node in data['nodes']:
        node['id'] = node_ids[node['id']]
    for edge, edge_id in zip(data['edges'], [24, 11]):
        edge['id'] = edge_id
        edge['node_i'] = node_ids[edge['node_i']]
        edge['node_j'] = node_ids[edge['node_j']]
    data.pop('next_node_id')
    data.pop('next_edge_id')
    return StructureGraph.from_dict(data)


@pytest.fixture
def fake_meshio(monkeypatch):
    calls = []
    class Mesh:
        def __init__(self, points, cells, *, point_data, cell_data):
            calls.append('Mesh')
            self.points, self.cells = points, cells
            self.point_data, self.cell_data = point_data, cell_data
        def write(self, *args, **kwargs):
            calls.append('write')
            raise AssertionError('unexpected writer invocation')
    monkeypatch.setitem(sys.modules, 'meshio', SimpleNamespace(__version__='test-double', Mesh=Mesh))
    return calls


def metadata(location):
    return {location: {'measurement': {'units': 'Pa', 'meaning': 'synthetic measurement'}}}


def masked_field(vector, mask_kind):
    data = np.array([[12., 13., 14.], [987654., 98., 99.]]) if vector else np.array([12., 987654.])
    mask = np.zeros(data.shape, dtype=bool)
    if mask_kind == 'partial': mask.flat[-1] = True
    if mask_kind == 'all': mask[...] = True
    return np.ma.array(data, mask=mask)


@pytest.mark.parametrize('location', ['point', 'cell'])
@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('mask_kind', ['partial', 'all', 'none'])
@pytest.mark.parametrize('container', ['array', 'list', 'tuple', 'by_id'])
def test_masks_rejected_without_mesh_or_files(adapter, graph, fake_meshio, tmp_path,
                                             location, vector, mask_kind, container):
    value = masked_field(vector, mask_kind)
    before_data, before_mask = value.data.copy(), np.ma.getmaskarray(value).copy()
    if container == 'array':
        field = value
    else:
        # Wrap every row as a MaskedArray, including 0-D scalars with mask=False.
        rows = [np.ma.array(value.data[i], mask=before_mask[i]) for i in range(2)]
        if container == 'by_id':
            ids = [7, 29] if location == 'point' else [24, 11]
            field = dict(reversed(list(zip(ids, rows))))
        else:
            field = rows if container == 'list' else tuple(rows)
    suffix = '_data_by_id' if container == 'by_id' else '_data'
    kwargs = {location + suffix: {'measurement': field}, 'field_metadata': metadata(location)}
    positions = {i: n.position.copy() for i, n in graph.nodes.items()}
    with pytest.raises(ValueError, match='masked arrays/scalars are unsupported'):
        adapter.write_vtu_bundle(tmp_path / 'bundle', graph, length_unit='mm', **kwargs)
    assert fake_meshio == []
    assert list(tmp_path.iterdir()) == []
    np.testing.assert_array_equal(value.data, before_data)
    np.testing.assert_array_equal(np.ma.getmaskarray(value), before_mask)
    for i, node in graph.nodes.items():
        np.testing.assert_array_equal(node.position, positions[i])
    assert list(graph.edges) == [24, 11]


@pytest.mark.parametrize('location', ['point', 'cell'])
@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('container', ['array', 'list', 'tuple', 'by_id'])
def test_plain_fields_keep_values_ids_units_and_detachment(adapter, graph, fake_meshio,
                                                          location, vector, container):
    value = masked_field(vector, 'none').data.copy()
    if container == 'array': field = value
    elif container == 'list': field = value.tolist()
    elif container == 'tuple': field = tuple(tuple(r) for r in value) if vector else tuple(value)
    else:
        ids = [7, 29] if location == 'point' else [24, 11]
        field = dict(reversed(list(zip(ids, value))))
    kwargs = {location + ('_data_by_id' if container == 'by_id' else '_data'): {'measurement': field}}
    mesh, manifest = adapter.to_meshio(graph, length_unit='mm', field_metadata=metadata(location), **kwargs)
    actual = mesh.point_data['measurement'] if location == 'point' else mesh.cell_data['measurement'][0]
    np.testing.assert_array_equal(actual, value)
    np.testing.assert_array_equal(mesh.point_data['node_id'], [7, 29])
    np.testing.assert_array_equal(mesh.cell_data['edge_id'][0], [24, 11])
    np.testing.assert_array_equal(mesh.cells[0][1], [[1, 0], [0, 1]])
    assert manifest['length_unit'] == 'mm'
    assert manifest['coordinate_conversion'] == 'none'
    assert [m['name'] for m in manifest['materials']] == ['second', 'first']
    assert manifest['field_metadata'] == metadata(location)
    actual.flat[0] = -1
    assert value.flat[0] == 12.
    assert fake_meshio == ['Mesh']


def test_nested_masked_scalar_rejected(adapter):
    with pytest.raises(ValueError, match='masked arrays/scalars are unsupported'):
        adapter._fields({'measurement': [(1., np.ma.masked, 3.), (4., 5., 6.)]},
                        2, metadata('point'), 'point')


@pytest.mark.parametrize('bad', [np.array([1., np.nan]), np.ones((2, 2)), np.array([True, False])])
def test_existing_numeric_checks_preserved(adapter, bad):
    with pytest.raises(ValueError):
        adapter._fields({'measurement': bad}, 2, metadata('cell'), 'cell')


@pytest.mark.parametrize('ids', [{24: 1.}, {24: 1., 99: 2.}])
def test_existing_id_coverage_preserved(adapter, graph, fake_meshio, ids):
    with pytest.raises(ValueError, match='exact ID coverage'):
        adapter.to_meshio(graph, length_unit='mm', cell_data_by_id={'measurement': ids},
                          field_metadata=metadata('cell'))
    assert fake_meshio == []
