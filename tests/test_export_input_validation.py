"""Validate public export boundaries before replacing existing files."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fibernet.gen.manufacturing import ManufacturingNetwork, PlanarManufacturingConfig, compile_planar
from fibernet.gen.solid_export import FiberSolid, PrintSettings, export_route, export_solid


@pytest.fixture
def network():
    return compile_planar(PlanarManufacturingConfig(grid_x=1, grid_y=1, n_pts_per_side=0))


@pytest.mark.parametrize('kind', ['reference', 'fractional', 'negative', 'out_of_range'])
def test_network_rejects_invalid_input(network, kind):
    reference = network.reference.copy()
    edges = network.edges.copy()
    if kind == 'reference':
        reference[0, 0] = np.nan
    elif kind == 'fractional':
        edges = edges.astype(float) + .1
    else:
        edges[0, 0] = -1 if kind == 'negative' else len(reference)
    with pytest.raises(ValueError):
        ManufacturingNetwork(reference, network.positions, edges)


@pytest.mark.parametrize('kind', ['nan', 'two_columns', 'duplicate_edge', 'wrong_order', 'negative_node'])
def test_route_rejects_invalid_without_replacing_file(tmp_path, network, kind):
    centers = network.positions.copy()
    if kind == 'nan':
        centers[0, 0] = np.nan
    elif kind == 'two_columns':
        centers = centers[:, :2]
    elif kind == 'duplicate_edge':
        network.route_edges[:] = 0
    elif kind == 'wrong_order':
        network.route_edges = np.roll(network.route_edges, 1)
    else:
        network.route_nodes[0] = -1
    path = tmp_path / 'route.csv'
    path.write_text('previous valid file')
    with pytest.raises(ValueError):
        export_route(path, network, centers)
    assert path.read_text() == 'previous valid file'
    assert not path.with_name(path.name + '.tmp').exists()


def test_parallel_route_csv_keeps_edge_identity(tmp_path, network):
    path = tmp_path / 'route.csv'
    export_route(path, network, network.positions)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == len(network.edges) + 1
    assert sorted(int(r['incoming_edge_id']) for r in rows[1:]) == list(range(len(network.edges)))
    assert len(np.unique(np.sort(network.edges, axis=1), axis=0)) < len(network.edges)
    for i, row in enumerate(rows):
        assert int(row['node_id']) == network.route_nodes[i]
        assert np.array_equal([float(row[k]) for k in ('x_mm', 'y_mm', 'z_mm')], network.positions[network.route_nodes[i]])


@pytest.mark.parametrize('kind', ['negative', 'fractional', 'out_of_range', 'duplicate_vertex'])
def test_solid_rejects_invalid_faces(tmp_path, kind):
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    if kind == 'fractional':
        faces = faces.astype(float) + .1
    elif kind == 'duplicate_vertex':
        faces[0, 0] = faces[0, 1]
    else:
        faces[faces == 3] = -1 if kind == 'negative' else 4
    path = tmp_path / 'solid.3mf'
    path.write_text('previous valid file')
    with pytest.raises(ValueError):
        export_solid(path, FiberSolid(vertices, faces, vertices, .01, PrintSettings()))
    assert path.read_text() == 'previous valid file'


def test_json_config_grid_has_numeric_csv_columns(tmp_path, monkeypatch):
    import fibernet.easy as easy
    configs = json.loads('[{"unit":"square","grid":[2,3]}]')
    monkeypatch.setattr(easy, 'simulate', lambda *a, **k: SimpleNamespace(max_force=1., max_displacement=.1, energy=.01))
    output = tmp_path / 'batch.csv'
    easy.batch_simulate(configs, str(output))
    row = next(csv.DictReader(output.open()))
    assert int(row['grid_x']) == 2
    assert int(row['grid_y']) == 3


def test_tetrahedron_formats_without_optional_mesher(tmp_path):
    import struct
    import zipfile
    import xml.etree.ElementTree as ET
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    solid = FiberSolid(vertices, faces, vertices, .01, PrintSettings())
    assert solid.validate() == pytest.approx(1 / 6)
    export_solid(tmp_path / 'tetra.stl', solid)
    raw = (tmp_path / 'tetra.stl').read_bytes()
    assert len(raw) == 284
    assert struct.unpack('<I', raw[80:84])[0] == 4
    export_solid(tmp_path / 'tetra.3mf', solid)
    with zipfile.ZipFile(tmp_path / 'tetra.3mf') as archive:
        root = ET.fromstring(archive.read('3D/3dmodel.model'))
    assert root.attrib['unit'] == 'millimeter'
    assert np.array_equal([[float(v.attrib[k]) for k in ('x', 'y', 'z')]
                           for v in root.findall('.//{*}vertex')], vertices)
    assert np.array_equal([[int(f.attrib[k]) for k in ('v1', 'v2', 'v3')]
                           for f in root.findall('.//{*}triangle')], solid.faces)


@pytest.mark.parametrize('field', ['reference', 'positions'])
@pytest.mark.parametrize('shape', [(3, 2), (6,), (1, 2, 3), (0, 3)])
def test_network_rejects_coordinate_regrouping(field, shape):
    coordinates = np.arange(6., dtype=float).reshape(2, 3)
    inputs = dict(reference=coordinates, positions=coordinates, edges=[[0, 1], [1, 0]])
    inputs[field] = np.empty(shape) if shape == (0, 3) else coordinates.reshape(shape)
    with pytest.raises(ValueError, match=r'coordinates require nonempty \(N, 3\)'):
        ManufacturingNetwork(**inputs)


@pytest.mark.parametrize('edges', [[0, 1, 1, 0], [[[0, 1], [1, 0]]], np.empty((0, 2), dtype=int)])
def test_network_rejects_edge_regrouping(edges):
    coordinates = [[0., 0., 0.], [1., 0., 0.]]
    with pytest.raises(ValueError, match=r'edges require a nonempty \(M, 2\)'):
        ManufacturingNetwork(coordinates, coordinates, edges)


@pytest.mark.parametrize('nested_lists', [False, True])
def test_network_preserves_explicit_coordinate_and_edge_shapes(nested_lists):
    coordinates = np.array([[0., 0., 0.], [1., 0., 0.]])
    edges = np.array([[0, 1], [1, 0]])
    positions_input = coordinates.tolist() if nested_lists else coordinates
    edges_input = edges.tolist() if nested_lists else edges
    network = ManufacturingNetwork(positions_input, positions_input, edges_input)
    assert np.array_equal(network.positions, coordinates)
    assert np.array_equal(network.reference, coordinates)
    assert np.array_equal(network.edges, edges)
    assert network.route_nodes[0] == network.route_nodes[-1]
    assert sorted(network.route_edges) == [0, 1]
    assert np.array_equal(coordinates, [[0., 0., 0.], [1., 0., 0.]])
    assert np.array_equal(edges, [[0, 1], [1, 0]])
