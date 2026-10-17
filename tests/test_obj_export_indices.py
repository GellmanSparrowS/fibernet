"""Index and compatibility checks for the legacy FiberNetwork OBJ writer."""

import inspect
import warnings

import numpy as np
import pytest

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.io.mesh_export import _generate_cylinder_mesh, export_obj


SEGMENTS = (
    ((0, 0, 0), (2, 0, 0)),
    ((5, 1, 0), (5, 4, 0)),
    ((9, 0, 1), (10, 2, 4)),
    ((13, 2, 5), (13, 2, 1)),
)


def _network(count=1, reverse=False):
    fibers = []
    for index, endpoints in enumerate(SEGMENTS[:count]):
        points = endpoints[::-1] if reverse else endpoints
        fibers.append(Fiber(np.asarray(points, dtype=float), radius=0.125,
                            fiber_id=100 + index))
    return FiberNetwork(fibers=fibers, dimension=3)


def _parse_obj(path):
    """Read the writer's plain triangular subset, retaining one-based indices."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] == '#':
            continue
        assert len(fields) == 4
        if fields[0] == 'v':
            vertices.append(tuple(float(value) for value in fields[1:]))
        else:
            assert fields[0] == 'f'
            faces.append(tuple(int(value) for value in fields[1:]))
    return np.asarray(vertices), np.asarray(faces)


def _expected_obj_faces(n_sides, count):
    """Independent one-based ring/cap connectivity, without the mesh helper."""
    result = []
    for segment in range(count):
        base = segment * (2 * n_sides + 2)
        bottom = [base + j + 1 for j in range(n_sides)]
        top = [base + n_sides + j + 1 for j in range(n_sides)]
        for j in range(n_sides):
            nxt = (j + 1) % n_sides
            result.extend(((bottom[j], bottom[nxt], top[j]),
                           (top[j], bottom[nxt], top[nxt])))
        for j in range(n_sides):
            result.append((base + 2 * n_sides + 1,
                           bottom[(j + 1) % n_sides], bottom[j]))
        for j in range(n_sides):
            result.append((base + 2 * n_sides + 2,
                           top[j], top[(j + 1) % n_sides]))
    return np.asarray(result)


@pytest.mark.parametrize('n_sides', [3, 4, 8, 16])
@pytest.mark.parametrize('count', [1, 2, 4])
@pytest.mark.parametrize('reverse', [False, True])
def test_obj_face_indices_match_each_segment(tmp_path, n_sides, count, reverse):
    path = tmp_path / 'network.obj'
    assert export_obj(_network(count, reverse), str(path), n_sides) is None
    vertices, faces = _parse_obj(path)
    per_segment = 2 * n_sides + 2
    assert vertices.shape == (count * per_segment, 3)
    assert faces.shape == (count * 4 * n_sides, 3)
    assert faces.min() == 1
    assert faces.max() == len(vertices)
    np.testing.assert_array_equal(faces, _expected_obj_faces(n_sides, count))
    for segment in range(count):
        block = faces[segment * 4 * n_sides:(segment + 1) * 4 * n_sides]
        assert block.min() == segment * per_segment + 1
        assert block.max() == (segment + 1) * per_segment
        assert set(block.ravel()) == set(range(segment * per_segment + 1,
                                               (segment + 1) * per_segment + 1))


def test_obj_default_matches_explicit_eight_sides(tmp_path):
    default, explicit = tmp_path / 'default.obj', tmp_path / 'explicit.obj'
    network = _network(4)
    export_obj(network, str(default))
    export_obj(network, str(explicit), n_sides=8)
    assert default.read_bytes() == explicit.read_bytes()
    parameters = inspect.signature(export_obj).parameters
    assert list(parameters) == ['network', 'filename', 'n_sides']
    assert parameters['n_sides'].default == 8


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('collapsed', [False, True])
def test_obj_empty_preserves_warning_and_file_behavior(tmp_path, existing, collapsed):
    path = tmp_path / 'empty.obj'
    if existing:
        path.write_bytes(b'previous output\n')
    network = FiberNetwork()
    if collapsed:
        network.fibers.append(Fiber(np.zeros((2, 3))))
    with pytest.warns(UserWarning, match='^No fibers to export$'):
        assert export_obj(network, str(path)) is None
    if existing:
        assert path.read_bytes() == b'previous output\n'
    else:
        assert not path.exists()


def test_obj_collapsed_segment_does_not_shift_later_faces(tmp_path):
    path = tmp_path / 'mixed.obj'
    network = _network(2)
    network.fibers.insert(1, Fiber(np.zeros((2, 3))))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        export_obj(network, str(path))
    assert not records
    vertices, faces = _parse_obj(path)
    assert vertices.shape == (36, 3)
    np.testing.assert_array_equal(faces, _expected_obj_faces(8, 2))


@pytest.mark.parametrize('n_sides', [3, 8, 16])
@pytest.mark.parametrize('caps', [False, True])
def test_cylinder_helper_retains_zero_based_vertex_counts(n_sides, caps):
    vertices, faces = _generate_cylinder_mesh(
        np.array([0., 0, 0]), np.array([1., 2, 3]), 0.125,
        n_sides=n_sides, n_caps=caps)
    count = 2 * n_sides + (2 if caps else 0)
    assert vertices.shape == (count, 3)
    assert faces.shape == ((4 if caps else 2) * n_sides, 3)
    assert faces.min() == 0
    assert faces.max() == count - 1


def test_obj_preserves_endpoint_chord_and_plain_triangle_contract(tmp_path):
    straight, curved = tmp_path / 'straight.obj', tmp_path / 'curved.obj'
    network = _network()
    export_obj(network, str(straight))
    fiber = network.fibers[0]
    fiber.centerline = np.vstack((fiber.centerline[0], [0., 10., 10.],
                                 fiber.centerline[-1]))
    fiber.fiber_id = 999
    export_obj(network, str(curved))
    assert straight.read_bytes() == curved.read_bytes()
    assert all(line.startswith(('#', 'v ', 'f ')) or not line
               for line in curved.read_text().splitlines())


@pytest.mark.parametrize('count', [1, 4])
@pytest.mark.parametrize('reverse', [False, True])
def test_obj_meshio_roundtrip_when_available(tmp_path, count, reverse):
    meshio = pytest.importorskip('meshio', reason='optional independent OBJ parser')
    path = tmp_path / 'parsed.obj'
    export_obj(_network(count, reverse), str(path))
    vertices, faces = _parse_obj(path)
    mesh = meshio.read(path)
    np.testing.assert_array_equal(mesh.points, vertices)
    parsed_faces = mesh.cells_dict['triangle']
    assert parsed_faces.min() == 0
    assert parsed_faces.max() == len(mesh.points) - 1
    np.testing.assert_array_equal(parsed_faces + 1, faces)
    # Force coordinate lookup too: a permissive parser may accept bad references.
    assert mesh.points[parsed_faces].shape == (count * 32, 3, 3)
