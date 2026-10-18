"""Mesh64 adapter contract, using pure provider doubles and leaf-module loading.

No test imports the optional native provider, constructs native geometry, or
calls FiberSolid.validate. These tests do not qualify a whole-mesh benchmark.
"""
import ast
import builtins
import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'fibernet' / 'gen' / 'tube_solid.py'
POINTS = np.array([[0., 0., 0.], [1., 0., 0.]])
EDGES = np.array([[0, 1]])
VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)


def load_leaf():
    spec = importlib.util.spec_from_file_location('isolated_tube_solid', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tube(monkeypatch):
    # Fail closed even when manifold3d is installed in the test environment.
    monkeypatch.setitem(sys.modules, 'manifold3d', None)
    return load_leaf()


class Provider:
    """Only Python call recording and supplied NumPy arrays; no mesh engine."""
    def __init__(self, vertices=None, faces=None, status='NoError'):
        self.vertices = VERTICES.copy() if vertices is None else vertices
        self.faces = FACES.copy() if faces is None else faces
        self.status = status
        self.calls = []
        self.mesh_refs = []
        self.sphere_triangles = 8
        self.union_triangles = 8
        owner = self

        class MeshView:
            pass

        class Manifold:
            def __init__(self, *args, **kwargs):
                raise AssertionError('native-style reconstruction is forbidden')

            @staticmethod
            def _shape(kind):
                shape = object.__new__(Manifold)
                shape.kind = kind
                return shape

            @staticmethod
            def sphere(radius, circular_segments):
                owner.calls.append(('sphere', radius, circular_segments))
                return Manifold._shape('sphere')

            @staticmethod
            def cylinder(length, radius, circular_segments):
                owner.calls.append(('cylinder', float(length), radius, circular_segments))
                return Manifold._shape('cylinder')

            @staticmethod
            def batch_boolean(pieces, operation):
                owner.calls.append(('batch_boolean', tuple(p.kind for p in pieces), operation))
                return Manifold._shape('union')

            def transform(self, transform):
                owner.calls.append(('transform', transform.copy()))
                return self

            def translate(self, point):
                owner.calls.append(('translate', point.copy()))
                return Manifold._shape('joint')

            def num_tri(self):
                owner.calls.append(('num_tri', self.kind))
                return owner.sphere_triangles if self.kind == 'sphere' else owner.union_triangles

            def as_original(self):
                owner.calls.append(('as_original',))
                return self

            def simplify(self, tolerance):
                owner.calls.append(('simplify', tolerance))
                return self

            def to_mesh64(self):
                owner.calls.append(('to_mesh64',))
                mesh = MeshView()
                mesh.vert_properties = owner.vertices
                mesh.tri_verts = owner.faces
                owner.mesh_refs.append(weakref.ref(mesh))
                return mesh

            def to_mesh(self):
                raise AssertionError('float32 extraction is forbidden')

            def status(self):
                owner.calls.append(('status',))
                return owner.status

        def reject_mesh(*args, **kwargs):
            raise AssertionError('Mesh reconstruction is forbidden')

        self.module = ModuleType('manifold3d')
        self.module.Manifold = Manifold
        self.module.Mesh = reject_mesh
        self.module.Mesh64 = reject_mesh
        self.module.OpType = SimpleNamespace(Add='Add')
        self.module.Error = SimpleNamespace(NoError='NoError')

    def install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'manifold3d', self.module)
        return self

    def names(self):
        return [call[0] for call in self.calls]


def test_leaf_import_needs_no_optional_provider(monkeypatch):
    imported = []
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        imported.append(name)
        if name.split('.')[0] in {'manifold3d', 'torch', 'scipy', 'taichi', 'skimage'}:
            raise AssertionError('unexpected optional import: ' + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', checked)
    module = load_leaf()
    assert callable(module.tube_mesh)
    assert 'manifold3d' not in imported


def test_missing_package_keeps_import_error(tube):
    with pytest.raises(ModuleNotFoundError, match='manifold3d'):
        tube.tube_mesh(POINTS, EDGES, 1.)


@pytest.mark.parametrize('capability', ['missing', None, 17])
def test_missing_mesh64_capability_refuses_before_primitive(tube, monkeypatch, capability):
    provider = Provider().install(monkeypatch)
    if capability == 'missing':
        delattr(provider.module.Manifold, 'to_mesh64')
    else:
        provider.module.Manifold.to_mesh64 = capability
    with pytest.raises(RuntimeError, match='(?i)(binary64|to_mesh64|mesh64)'):
        tube.tube_mesh(POINTS, EDGES, 1.)
    assert provider.calls == []


def test_source_contains_one_mesh64_and_no_surface_roundtrip():
    tree = ast.parse(SOURCE.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    assert attributes.count('to_mesh64') == 1
    assert attributes.count('as_original') == 1
    assert attributes.count('simplify') == 1
    assert 'to_mesh' not in attributes
    assert 'Mesh' not in attributes and 'Mesh64' not in attributes
    assert not any(isinstance(node, ast.Attribute) and node.attr == 'float32'
                   for node in ast.walk(tree))
    round_calls = [node for node in calls if isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'round']
    assert all(not (len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                       and node.args[1].value == 4) for node in round_calls)


def test_single_extraction_order_and_unchanged_progress(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    progress = []
    round_digits = []
    original_round = np.round

    def recorded_round(values, decimals=0, *args, **kwargs):
        round_digits.append(decimals)
        return original_round(values, decimals, *args, **kwargs)

    monkeypatch.setattr(tube.np, 'round', recorded_round)
    vertices, faces = tube.tube_mesh(POINTS, EDGES, .25, progress=progress.append, sides=12)
    assert provider.names() == ['sphere', 'num_tri', 'cylinder', 'transform', 'translate',
                                'translate', 'batch_boolean', 'num_tri', 'as_original',
                                'simplify', 'to_mesh64', 'status']
    assert provider.calls[0] == ('sphere', .25, 12)
    assert provider.calls[2] == ('cylinder', 1., .25, 12)
    assert provider.calls[6] == ('batch_boolean', ('cylinder', 'joint', 'joint'), 'Add')
    assert provider.calls[9] == ('simplify', .00025)
    assert round_digits == [8, 5, 5]
    assert progress == [0, 85]
    np.testing.assert_array_equal(vertices, VERTICES)
    np.testing.assert_array_equal(faces, FACES)


def test_graph_round8_dedup_and_degree_two_merge_remain(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    points = np.array([[2., 0, 0], [0., 0, 0], [1., 0, 0], [0.000000001, 0, 0]])
    edges = np.array([[0, 2], [2, 1], [3, 1], [2, 0]])
    before = points.copy(), edges.copy()
    tube.tube_mesh(points, edges, .25)
    cylinders = [call for call in provider.calls if call[0] == 'cylinder']
    assert cylinders == [('cylinder', 2., .25, 16)]
    translations = [call[1] for call in provider.calls if call[0] == 'translate']
    np.testing.assert_array_equal(translations, [[0., 0, 0], [2., 0, 0]])
    np.testing.assert_array_equal(points, before[0])
    np.testing.assert_array_equal(edges, before[1])


def test_spatial_order_and_deterministic_phase_remain(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    points = np.array([[4., 0, 0], [5., 0, 0], [0., 0, 0], [1., 0, 0]])
    tube.tube_mesh(points, np.array([[1, 0], [3, 2]]), .25)
    transforms = [call[1] for call in provider.calls if call[0] == 'transform']
    for i, transform in enumerate(transforms):
        phase = (i * .61803398875 % 1.) * 2 * np.pi / 16
        expected = np.column_stack([[0., np.sin(phase), -np.cos(phase)],
                                    [0., np.cos(phase), np.sin(phase)],
                                    [1., 0, 0], [4. * i, 0, 0]])
        np.testing.assert_allclose(transform, expected, rtol=0, atol=1e-15)


def test_bounded_batches_union_tree_and_progress_remain(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    points = np.column_stack([np.arange(100, dtype=float), np.zeros((100, 2))])
    edges = np.arange(100).reshape(-1, 2)
    progress = []
    tube.tube_mesh(points, edges, .25, progress=progress.append)
    batches = [call for call in provider.calls if call[0] == 'batch_boolean']
    assert [len(call[1]) for call in batches] == [96, 54, 2]
    assert batches[-1][1] == ('union', 'union')
    assert progress == [0, 17, 34, 51, 68, 85, 95]
    assert provider.names().count('as_original') == 1
    assert provider.names().count('to_mesh64') == 1


@pytest.mark.parametrize('dtype', ['<f8', '>f8'])
def test_readonly_strided_properties_are_owned_writable_and_survive_gc(tube, monkeypatch, dtype):
    storage = np.zeros((8, 10), dtype=dtype)
    properties = storage[::2, ::2]
    properties[:, :3] = VERTICES
    properties[:, 3:] = np.nan  # Non-position properties are not public output.
    face_storage = np.zeros((4, 6), dtype=np.uint64)
    face_view = face_storage[:, ::2]
    face_view[:] = FACES
    properties.setflags(write=False)
    face_view.setflags(write=False)
    provider = Provider(properties, face_view).install(monkeypatch)
    vertices, faces = tube.tube_mesh(POINTS, EDGES, .25)
    gc.collect()
    assert all(reference() is None for reference in provider.mesh_refs)
    for output, dtype in [(vertices, np.dtype('float64')), (faces, np.dtype('int32'))]:
        assert output.dtype == dtype
        assert output.flags.owndata and output.flags.writeable and output.flags.c_contiguous
    assert vertices.shape == (4, 3)
    assert not np.shares_memory(vertices, storage)
    assert not np.shares_memory(faces, face_storage)
    vertices[:, 2] *= 25.
    vertices += [2., 3., 4.]
    faces[0] = [3, 2, 1]
    np.testing.assert_array_equal(properties[:, :3], VERTICES)
    np.testing.assert_array_equal(face_view, FACES)
    storage[:] = 123.
    face_storage[:] = 3
    np.testing.assert_array_equal(vertices, VERTICES * [1., 1., 25.] + [2., 3., 4.])
    np.testing.assert_array_equal(faces[1:], FACES[1:])


@pytest.mark.parametrize('vertices', [
    np.zeros(12), np.zeros((2, 2, 3)), np.zeros((4, 2)),
    VERTICES.astype(np.float32), VERTICES.astype(np.float16), VERTICES.astype(np.int64),
    VERTICES.astype(bool), VERTICES.astype(complex), VERTICES.astype(object),
    VERTICES.astype('U4'),
], ids=['rank1', 'rank3', 'two_columns', 'float32', 'float16', 'integer',
        'bool', 'complex', 'object', 'string'])
def test_malformed_or_nonbinary64_positions_refuse(tube, monkeypatch, vertices):
    Provider(vertices=vertices).install(monkeypatch)
    with pytest.raises(ValueError):
        tube.tube_mesh(POINTS, EDGES, .25)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_positions_refuse(tube, monkeypatch, value):
    vertices = VERTICES.copy()
    vertices[2, 1] = value
    Provider(vertices=vertices).install(monkeypatch)
    with pytest.raises(ValueError, match='finite'):
        tube.tube_mesh(POINTS, EDGES, .25)


@pytest.mark.parametrize('faces', [
    np.array([0, 1, 2]), np.zeros((1, 1, 3), dtype=int),
    np.zeros((2, 2), dtype=int), np.zeros((2, 4), dtype=int),
    np.array([[0., 1., 2.]]), np.array([[0., 1., 1.5]]),
    np.array([[False, True, False]]), np.array([[0, 1, 2]], dtype=object),
    np.array([[0, 1, 2]], dtype=complex), np.array([[0, 1, 2]], dtype='U1'),
    np.array([[-1, 1, 2]]), np.array([[0, 1, 4]]),
    np.array([[0, 1, 2**31]], dtype=np.uint64),
    np.array([[0, 1, 2**64 - 1]], dtype=np.uint64),
], ids=['rank1', 'rank3', 'two_columns', 'four_columns', 'integral_float',
        'fractional_float', 'bool', 'object', 'complex', 'string', 'negative',
        'index_at_N', 'uint64_int32_overflow', 'uint64_max'])
def test_bad_face_arrays_refuse_before_indexing_or_narrowing(tube, monkeypatch, faces):
    Provider(faces=faces).install(monkeypatch)
    with pytest.raises(ValueError):
        tube.tube_mesh(POINTS, EDGES, .25)


@pytest.mark.parametrize('dtype', ['i1', 'u1', '<i2', '>i2', '<u2', '>u2',
                                  '<i4', '>i4', '<u4', '>u4', '<i8', '>i8', '<u8', '>u8'])
def test_signed_unsigned_integer_faces_are_copied(tube, monkeypatch, dtype):
    incoming = FACES.astype(dtype)
    Provider(faces=incoming).install(monkeypatch)
    _, faces = tube.tube_mesh(POINTS, EDGES, .25)
    assert faces.dtype == np.int32
    assert faces.flags.owndata and faces.flags.c_contiguous
    assert not np.shares_memory(faces, incoming)
    np.testing.assert_array_equal(faces, FACES)


@pytest.mark.parametrize('dtype,value', [
    ('<i2', -1), ('>i2', -1), ('<i8', -(2**63)), ('>i8', -(2**63)),
    ('<u2', 4), ('>u2', 4), ('<u2', 2**16-1), ('>u2', 2**16-1),
    ('<i4', 2**31-1), ('>i4', 2**31-1),
    ('<u4', 2**31), ('>u4', 2**31),
    ('<u8', 2**64-1), ('>u8', 2**64-1),
])
def test_index_extrema_refuse_exactly_before_narrowing(tube, monkeypatch, dtype, value):
    storage = np.zeros((2, 6), dtype=dtype)
    incoming = storage[:, ::2]
    incoming[:] = [[0, 1, 2], [0, 1, value]]
    incoming.setflags(write=False)
    before = storage.tobytes()
    Provider(faces=incoming).install(monkeypatch)
    with pytest.raises(ValueError, match='^mesh face index is outside the vertex table or int32 range$'):
        tube.tube_mesh(POINTS, EDGES, .25)
    assert storage.tobytes() == before


def test_empty_and_degenerate_faces_are_not_repaired_or_validated(tube, monkeypatch):
    Provider(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int)).install(monkeypatch)
    vertices, faces = tube.tube_mesh(POINTS, EDGES, .25)
    assert vertices.shape == faces.shape == (0, 3)
    supplied = np.array([[0, 0, 1], [0, 1, 2], [0, 1, 2]])
    Provider(faces=supplied).install(monkeypatch)
    _, faces = tube.tube_mesh(POINTS, EDGES, .25)
    np.testing.assert_array_equal(faces, supplied)


def test_minimal_positive_triangle_is_not_quantized_to_zero(tube, monkeypatch):
    vertices = np.array([[0., 0., 0.], [1., 1., 0.], [2., 2.00003, 0.]])
    faces = np.array([[0, 1, 2]])
    assert np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])[2] > 0
    rounded = np.round(vertices, 4)
    assert np.cross(rounded[1] - rounded[0], rounded[2] - rounded[0])[2] == 0
    Provider(vertices, faces).install(monkeypatch)
    actual, actual_faces = tube.tube_mesh(POINTS, EDGES, .25)
    np.testing.assert_array_equal(actual, vertices)
    np.testing.assert_array_equal(actual_faces, faces)
    assert np.cross(actual[1] - actual[0], actual[2] - actual[0])[2] > 0


def test_failed_native_status_remains_failure(tube, monkeypatch):
    provider = Provider(status='BadGeometry').install(monkeypatch)
    with pytest.raises(ValueError, match='cylinder union failed: BadGeometry'):
        tube.tube_mesh(POINTS, EDGES, .25)
    assert provider.names()[-3:] == ['to_mesh64', 'status', 'status']


def test_cancellation_during_first_add(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    progress = []
    with pytest.raises(InterruptedError, match='solid export cancelled'):
        tube.tube_mesh(POINTS, EDGES, .25, progress=progress.append, stop_cb=lambda: True)
    assert 'batch_boolean' not in provider.names()
    assert 'to_mesh64' not in provider.names()
    assert progress == []


def test_cancellation_after_mesh64_precedes_status(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    counts = iter([False, False, False, True])
    with pytest.raises(InterruptedError, match='solid export cancelled'):
        tube.tube_mesh(POINTS, EDGES, .25, stop_cb=lambda: next(counts))
    assert provider.names()[-1] == 'to_mesh64'
    assert 'status' not in provider.names()


def test_touching_displacement_changes_only_coincident_vertices(tube, monkeypatch):
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0],
                         [0., 0, 0], [-1., 0, 0], [0., 0, 1.]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    Provider(vertices, faces).install(monkeypatch)
    result, output_faces = tube.tube_mesh(POINTS, EDGES, 1.)
    expected = vertices.copy()
    expected[0, 2] = -.001
    expected[3, 1] = -.001
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(output_faces, faces)
    np.testing.assert_array_equal(vertices[[0, 3]], np.zeros((2, 3)))


def test_ambiguous_zero_thickness_contact_remains_failure(tube, monkeypatch):
    vertices = np.vstack([VERTICES, VERTICES[0]])
    Provider(vertices, FACES).install(monkeypatch)
    with pytest.raises(ValueError, match='ambiguous zero-thickness solid contact'):
        tube.tube_mesh(POINTS, EDGES, 1.)


def test_unresolved_contact_retains_three_attempts_and_displacements(tube, monkeypatch):
    vertices = np.tile(np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0]]), (2, 1))
    Provider(vertices, np.array([[0, 1, 2], [3, 4, 5]])).install(monkeypatch)
    original_round = np.round
    classification_inputs = []

    def recorded_round(values, decimals=0, *args, **kwargs):
        if decimals == 5:
            classification_inputs.append(values.copy())
        return original_round(values, decimals, *args, **kwargs)

    monkeypatch.setattr(tube.np, 'round', recorded_round)
    with pytest.raises(ValueError, match='unresolved coincident solid vertices'):
        tube.tube_mesh(POINTS, EDGES, 1.)
    assert len(classification_inputs) == 4
    for coordinates, displacement in zip(classification_inputs, [0., -.001, -.003, -.006]):
        np.testing.assert_allclose(coordinates[:, 2], displacement, rtol=0, atol=1e-18)


def test_new_nonfinite_positions_after_touching_are_refused(tube, monkeypatch):
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0],
                         [0., 0, 0], [-1., 0, 0], [0., 0, 1.]])
    Provider(vertices, np.array([[0, 1, 2], [3, 4, 5]])).install(monkeypatch)
    original_cross = np.cross

    def overflow_face_normals(a, b, *args, **kwargs):
        if a.ndim == 2:
            return np.full(a.shape, np.inf)
        return original_cross(a, b, *args, **kwargs)

    monkeypatch.setattr(tube.np, 'cross', overflow_face_normals)
    with np.errstate(invalid='ignore', over='ignore'):
        with pytest.raises(ValueError, match='finite'):
            tube.tube_mesh(POINTS, EDGES, 1.)


def test_primitive_geometry_cap_refuses_before_cylinder(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    provider.sphere_triangles = 6_000_000
    with pytest.raises(MemoryError, match='tube mesh exceeds available geometry budget'):
        tube.tube_mesh(POINTS, EDGES, .25)
    assert provider.names() == ['sphere', 'num_tri']


def test_intermediate_union_cap_remains_enforced(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    provider.union_triangles = 8_000_001
    points = np.column_stack([np.arange(100, dtype=float), np.zeros((100, 2))])
    with pytest.raises(MemoryError, match='solid union exceeds geometry budget'):
        tube.tube_mesh(points, np.arange(100).reshape(-1, 2), .25)
    assert provider.names().count('batch_boolean') == 3
    assert 'as_original' not in provider.names()
    assert 'to_mesh64' not in provider.names()


def test_cancellation_before_union_tree_merge(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    points = np.column_stack([np.arange(100, dtype=float), np.zeros((100, 2))])
    checks = iter([False] * 150 + [True])
    with pytest.raises(InterruptedError, match='solid export cancelled'):
        tube.tube_mesh(points, np.arange(100).reshape(-1, 2), .25, stop_cb=lambda: next(checks))
    assert provider.names().count('batch_boolean') == 2
    assert 'as_original' not in provider.names()
    assert 'to_mesh64' not in provider.names()


def test_initial_simplification_floor_is_unchanged(tube, monkeypatch):
    provider = Provider().install(monkeypatch)
    tube.tube_mesh(POINTS, EDGES, 1e-4)
    assert [call for call in provider.calls if call[0] == 'simplify'] == [('simplify', 1e-5)]


def test_extended_float_positions_follow_binary64_representation(tube, monkeypatch):
    vertices = VERTICES.astype(np.longdouble)
    Provider(vertices=vertices).install(monkeypatch)
    if vertices.dtype.itemsize == 8:
        actual, _ = tube.tube_mesh(POINTS, EDGES, .25)
        np.testing.assert_array_equal(actual, VERTICES)
    else:
        with pytest.raises(ValueError):
            tube.tube_mesh(POINTS, EDGES, .25)
