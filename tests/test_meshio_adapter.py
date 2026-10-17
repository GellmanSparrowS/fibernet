"""Synthetic software checks for optional straight-line interchange."""
import builtins
import hashlib
import json

import numpy as np
import pytest

from fibernet import Material, StructureGraph
from fibernet.io.meshio_adapter import to_meshio, write_vtu_bundle


def graph():
    g = StructureGraph(dimension=3)
    a = g.add_node([0, 0, 0])
    ghost = g.add_node([9, 9, 9])
    b = g.add_node([1, 2, 3])
    g.remove_node(ghost)
    g.add_node([4, 0, 2])  # isolated node
    g.add_edge(a, b, radius=.1, material=Material(name='first', youngs_modulus=2e9))
    g.add_edge(a, b, radius=.2, material=Material(name='second', youngs_modulus=3e9), allow_parallel=True)
    # Force nontrivial edge insertion order while keeping stable IDs intact.
    edge = g.edges.pop(0)
    g.edges[0] = edge
    return g


def fields():
    return dict(point_data={'temperature': np.array([2., 4., 8.]),
                            'displacement': np.arange(9.).reshape(3, 3)},
                cell_data={'force': np.array([11., 13.])},
                field_metadata={'point': {'temperature': {'units': 'K', 'meaning': 'synthetic scalar'},
                                           'displacement': {'units': 'mm', 'meaning': 'synthetic vector'}},
                                'cell': {'force': {'units': 'N', 'meaning': 'synthetic axial force'}}})


def test_actual_vtu_roundtrip(tmp_path):
    meshio = pytest.importorskip('meshio')
    g = graph()
    before = json.dumps(g.to_dict(), sort_keys=True)
    target = write_vtu_bundle(tmp_path / 'bundle', g, length_unit='mm',
                              provenance={'backend': 'synthetic fixture'}, **fields())
    mesh = meshio.read(target / 'network.vtu')
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['node_ids'] == [0, 2, 3]
    assert manifest['edge_ids'] == [1, 0]
    assert manifest['length_unit'] == 'mm'
    assert manifest['coordinate_conversion'] == 'none'
    assert manifest['geometry_dimension'] == 3
    assert [m['name'] for m in manifest['materials']] == ['second', 'first']
    assert manifest['materials'][0]['youngs_modulus'] == 3e9
    assert manifest['mesh_sha256'] == hashlib.sha256((target / 'network.vtu').read_bytes()).hexdigest()
    np.testing.assert_allclose(mesh.points, g.node_positions(), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(mesh.cells_dict['line'], [[0, 1], [0, 1]])
    np.testing.assert_array_equal(mesh.point_data['node_id'], [0, 2, 3])
    np.testing.assert_array_equal(mesh.cell_data_dict['edge_id']['line'], [1, 0])
    np.testing.assert_allclose(mesh.cell_data_dict['radius']['line'], [.2, .1])
    np.testing.assert_array_equal(mesh.cell_data_dict['material_index']['line'], [0, 1])
    for name, values in fields()['point_data'].items():
        np.testing.assert_array_equal(mesh.point_data[name], values)
    np.testing.assert_array_equal(mesh.cell_data_dict['force']['line'], [11., 13.])
    assert json.dumps(g.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize('bad', ['nan', 'radius', 'endpoint', 'curve', 'field', 'units', 'metadata'])
def test_rejects_before_publishing(tmp_path, bad):
    g = graph()
    kwargs = dict(length_unit='mm')
    if bad == 'nan': g.nodes[0].position[0] = np.nan
    if bad == 'radius': g.edges[0].radius = 0
    if bad == 'endpoint': g.edges[0].node_j = 99
    if bad == 'curve': g.edges[0].internal_points = np.array([[.5, .1, 1.]])
    if bad == 'field': kwargs.update(point_data={'bad': [1, 2]})
    if bad == 'units': kwargs['length_unit'] = 'unknown'
    if bad == 'metadata': g.metadata['invalid'] = np.inf
    with pytest.raises(ValueError):
        write_vtu_bundle(tmp_path / 'bundle', g, **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_existing_destination_unchanged(tmp_path):
    target = tmp_path / 'bundle'
    target.mkdir()
    (target / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        write_vtu_bundle(target, graph(), length_unit='m')
    assert (target / 'sentinel').read_text() == 'keep'


def test_missing_dependency_actionable(monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'meshio': raise ImportError('deliberately blocked')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(ImportError, match='Optional line export requires meshio'):
        to_meshio(graph(), length_unit='m')


def test_write_failure_cleans_staging(tmp_path, monkeypatch):
    meshio = pytest.importorskip('meshio')
    def fail(*args, **kwargs):
        raise OSError('simulated writer failure')
    monkeypatch.setattr(meshio.Mesh, 'write', fail)
    with pytest.raises(OSError, match='simulated'):
        write_vtu_bundle(tmp_path / 'bundle', graph(), length_unit='m')
    assert list(tmp_path.iterdir()) == []


def test_manifest_and_mesh_are_detached():
    pytest.importorskip('meshio')
    g = graph()
    g.edges[0].material.extra['nested'] = {'a': [1, 2]}
    mesh, manifest = to_meshio(g, length_unit='m')
    mesh.points[0, 0] = 9
    manifest['materials'][1]['extra']['nested']['a'][0] = 99
    assert g.nodes[0].position[0] == 0
    assert g.edges[0].material.extra['nested']['a'][0] == 1


@pytest.mark.parametrize('spec', [{'point': []}, {'cell': []}, []])
def test_malformed_metadata_rejected(spec):
    with pytest.raises(ValueError):
        to_meshio(graph(), length_unit='m', point_data={'a': [1, 2, 3]},
                  field_metadata=spec)


def test_numeric_metadata_keys_rejected():
    g = graph()
    g.metadata['ambiguous'] = {1: 'a', '1': 'b'}
    with pytest.raises(ValueError, match='keys must be strings'):
        to_meshio(g, length_unit='m')


def test_two_dimensional_graph():
    pytest.importorskip('meshio')
    g = StructureGraph(dimension=2)
    a = g.add_node([0, 0])
    b = g.add_node([1, 2])
    g.add_edge(a, b)
    mesh, manifest = to_meshio(g, length_unit='um')
    assert manifest['geometry_dimension'] == 2
    assert mesh.points.shape == (2, 3)
    np.testing.assert_array_equal(mesh.points[:, 2], [0, 0])
