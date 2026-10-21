"""Cleanup contract tests using Python provider doubles and leaf-module loading.

Run with unittest. These tests never import the native provider or construct
native geometry, and do not qualify whole-mesh behavior.
"""
import ast
import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'fibernet' / 'gen' / 'tube_solid.py'
POINTS = np.array([[0., 0., 0.], [1., 0., 0.]])
EDGES = np.array([[0, 1]])
VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)


def load_leaf():
    spec = importlib.util.spec_from_file_location('fresh_cleanup_tube_leaf', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProviderDouble:
    """Record a single native-style cleanup without importing any native code."""
    def __init__(self, vertices=None, faces=None, tangents=None):
        self.vertices = VERTICES.copy() if vertices is None else vertices
        self.faces = FACES.copy() if faces is None else faces
        self.tangents = np.empty((0, 4)) if tangents is None else tangents
        self.calls = []
        self.raw_mesh = None
        self.cleaned_mesh = None
        self.constructor_arg = None
        self.constructor_exception = None
        self.union_status = 'NoError'
        self.cleanup_status = 'NoError'
        self.sphere_triangles = 8
        self.union_triangles = 8
        self.cleanup_triangles = 4
        self.metadata = {
            'merge_from_vert': np.array([2], dtype=np.uint64),
            'merge_to_vert': np.array([3], dtype=np.uint64),
            'run_index': np.array([0, 12], dtype=np.uint64),
            'run_original_id': np.array([177], dtype=np.uint32),
            'run_transform': np.arange(12, dtype=np.float64).reshape(1, 4, 3),
            'face_id': np.array([4, 3, 2, 1], dtype=np.uint64),
            'tolerance': 1.123456789012345e-9,
        }
        owner = self

        class Manifold:
            def __init__(self, native_mesh):
                owner.calls.append(('cleanup', id(native_mesh)))
                owner.constructor_arg = native_mesh
                if native_mesh is not owner.raw_mesh:
                    raise AssertionError('cleanup must receive exact extracted native object')
                for key, value in owner.metadata.items():
                    if getattr(native_mesh, key) is not value:
                        raise AssertionError('native metadata identity changed: ' + key)
                if owner.constructor_exception is not None:
                    raise owner.constructor_exception
                self.kind = 'cleanup'

            @staticmethod
            def shape(kind):
                shape = object.__new__(Manifold)
                shape.kind = kind
                return shape

            @staticmethod
            def sphere(radius, circular_segments):
                owner.calls.append(('sphere', radius, circular_segments))
                return Manifold.shape('sphere')

            @staticmethod
            def cylinder(length, radius, circular_segments):
                owner.calls.append(('cylinder', float(length), radius, circular_segments))
                return Manifold.shape('cylinder')

            @staticmethod
            def batch_boolean(pieces, operation):
                owner.calls.append(('batch_boolean', tuple(x.kind for x in pieces), operation))
                return Manifold.shape('union')

            def transform(self, transform):
                owner.calls.append(('transform', transform.copy()))
                return self

            def translate(self, point):
                owner.calls.append(('translate', point.copy()))
                return Manifold.shape('joint')

            def num_tri(self):
                owner.calls.append(('num_tri', self.kind))
                return {'sphere': owner.sphere_triangles,
                        'union': owner.union_triangles,
                        'cleanup': owner.cleanup_triangles}[self.kind]

            def as_original(self):
                owner.calls.append(('as_original',))
                return self

            def simplify(self, tolerance):
                owner.calls.append(('simplify', tolerance))
                return self

            def to_mesh64(self):
                owner.calls.append(('to_mesh64', self.kind))
                mesh = SimpleNamespace()
                if self.kind == 'cleanup':
                    mesh.vert_properties = owner.vertices
                    mesh.tri_verts = owner.faces
                    owner.cleaned_mesh = mesh
                else:
                    # Deliberately distinct from output, proving final extraction is used.
                    mesh.vert_properties = VERTICES + 100.
                    mesh.tri_verts = FACES[::-1].copy()
                    mesh.halfedge_tangent = owner.tangents
                    for key, value in owner.metadata.items():
                        setattr(mesh, key, value)
                    owner.raw_mesh = mesh
                return mesh

            def to_mesh(self):
                raise AssertionError('float32 extraction forbidden')

            def status(self):
                owner.calls.append(('status', self.kind))
                return owner.cleanup_status if self.kind == 'cleanup' else owner.union_status

        def reject_mesh_constructor(*args, **kwargs):
            raise AssertionError('Python-side Mesh/Mesh64 reconstruction forbidden')

        self.module = ModuleType('manifold3d')
        self.module.Manifold = Manifold
        self.module.Mesh = reject_mesh_constructor
        self.module.Mesh64 = reject_mesh_constructor
        self.module.OpType = SimpleNamespace(Add='Add')
        self.module.Error = SimpleNamespace(NoError='NoError')

    def names(self):
        return [call[0] for call in self.calls]


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.module_guard = patch.dict(sys.modules, {'manifold3d': None})
        self.module_guard.start()
        self.addCleanup(self.module_guard.stop)
        self.tube = load_leaf()

    def run_mesh(self, provider=None, **kwargs):
        provider = ProviderDouble() if provider is None else provider
        with patch.dict(sys.modules, {'manifold3d': provider.module}):
            return self.tube.tube_mesh(POINTS, EDGES, .25, **kwargs)

    def test_leaf_import_avoids_optional_providers(self):
        original = builtins.__import__
        def checked(name, *args, **kwargs):
            if name.split('.')[0] in {'manifold3d', 'torch', 'scipy', 'taichi', 'skimage'}:
                raise AssertionError('unexpected optional import: ' + name)
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', checked):
            self.assertTrue(callable(load_leaf().tube_mesh))

    def test_missing_package_retains_import_failure(self):
        with self.assertRaisesRegex(ModuleNotFoundError, 'manifold3d'):
            self.tube.tube_mesh(POINTS, EDGES, .25)

    def test_source_has_two_extractions_one_cleanup_no_float32_or_round4(self):
        tree = ast.parse(SOURCE.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        attrs = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
        self.assertEqual(attrs.count('to_mesh64'), 2)
        self.assertEqual(attrs.count('Manifold'), 1)
        self.assertEqual(attrs.count('as_original'), 1)
        self.assertEqual(attrs.count('simplify'), 1)
        for forbidden in ['to_mesh', 'Mesh', 'Mesh64']:
            self.assertNotIn(forbidden, attrs)
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'float32'
                             for n in ast.walk(tree)))
        for n in calls:
            if isinstance(n.func, ast.Attribute) and n.func.attr == 'round':
                for arg in n.args[1:] + [k.value for k in n.keywords if k.arg == 'decimals']:
                    self.assertFalse(isinstance(arg, ast.Constant) and arg.value == 4)
        bound_assignments = {n.targets[0].id: n.value for n in ast.walk(tree)
                             if isinstance(n, ast.Assign) and len(n.targets) == 1
                             and isinstance(n.targets[0], ast.Name)}
        for name, reducer in [('first_index', 'min'), ('last_index', 'max')]:
            call = bound_assignments[name]
            self.assertIsInstance(call, ast.Call)
            self.assertEqual(call.func.id, 'int')
            self.assertEqual(call.args[0].func.attr, reducer)

    def test_exact_native_identity_metadata_and_two_extraction_order(self):
        provider = ProviderDouble()
        before = {k: v.copy() for k, v in provider.metadata.items() if isinstance(v, np.ndarray)}
        progress = []
        round_digits = []
        original_round = np.round
        def recorded_round(values, decimals=0, *args, **kwargs):
            round_digits.append(decimals)
            return original_round(values, decimals, *args, **kwargs)
        with patch.object(self.tube.np, 'round', recorded_round):
            vertices, faces = self.run_mesh(provider, progress=progress.append, sides=12)
        self.assertIs(provider.constructor_arg, provider.raw_mesh)
        self.assertIsNot(provider.cleaned_mesh, provider.raw_mesh)
        self.assertEqual(provider.names(), ['sphere', 'num_tri', 'cylinder', 'transform',
                         'translate', 'translate', 'batch_boolean', 'num_tri', 'as_original',
                         'simplify', 'to_mesh64', 'status', 'cleanup', 'status', 'num_tri', 'to_mesh64'])
        self.assertEqual([c for c in provider.calls if c[0] == 'to_mesh64'],
                         [('to_mesh64', 'union'), ('to_mesh64', 'cleanup')])
        self.assertEqual(provider.calls[0], ('sphere', .25, 12))
        self.assertEqual(provider.calls[2], ('cylinder', 1., .25, 12))
        self.assertEqual(provider.calls[9], ('simplify', .00025))
        self.assertEqual(progress, [0, 85])
        self.assertEqual(round_digits, [8, 5, 5])
        for name, snapshot in before.items():
            np.testing.assert_array_equal(getattr(provider.constructor_arg, name), snapshot)
        np.testing.assert_array_equal(vertices, VERTICES)
        np.testing.assert_array_equal(faces, FACES)

    def test_tangents_refuse_before_constructor(self):
        provider = ProviderDouble(tangents=np.zeros((1, 4)))
        with self.assertRaisesRegex(ValueError, 'empty halfedge tangents'):
            self.run_mesh(provider)
        self.assertNotIn('cleanup', provider.names())
        self.assertEqual(provider.names().count('to_mesh64'), 1)

    def test_empty_python_tangent_sequence_allowed(self):
        provider = ProviderDouble(tangents=[])
        self.run_mesh(provider)
        self.assertEqual(provider.names().count('cleanup'), 1)

    def test_union_status_refuses_before_constructor(self):
        provider = ProviderDouble()
        provider.union_status = 'BrokenUnion'
        with self.assertRaisesRegex(ValueError, 'cylinder union failed: BrokenUnion'):
            self.run_mesh(provider)
        self.assertNotIn('cleanup', provider.names())
        self.assertEqual(provider.names()[-3:], ['to_mesh64', 'status', 'status'])

    def test_cleanup_status_refuses_before_final_extraction(self):
        provider = ProviderDouble()
        provider.cleanup_status = 'BrokenCleanup'
        with self.assertRaisesRegex(ValueError, 'full-precision mesh cleanup failed: BrokenCleanup'):
            self.run_mesh(provider)
        self.assertEqual(provider.names().count('to_mesh64'), 1)
        self.assertEqual(provider.calls[-2:], [('status', 'cleanup'), ('status', 'cleanup')])

    def test_empty_cleanup_refuses_before_final_extraction(self):
        provider = ProviderDouble()
        provider.cleanup_triangles = 0
        with self.assertRaisesRegex(ValueError, 'cleanup returned empty geometry'):
            self.run_mesh(provider)
        self.assertEqual(provider.names().count('to_mesh64'), 1)
        self.assertEqual(provider.calls[-1], ('num_tri', 'cleanup'))

    def test_constructor_exception_propagates_unchanged(self):
        provider = ProviderDouble()
        error = RuntimeError('constructor marker')
        provider.constructor_exception = error
        with self.assertRaises(RuntimeError) as caught:
            self.run_mesh(provider)
        self.assertIs(caught.exception, error)
        self.assertEqual(provider.names()[-1], 'cleanup')
        self.assertEqual(provider.names().count('to_mesh64'), 1)

    def test_cancellation_during_first_add(self):
        provider = ProviderDouble()
        progress = []
        with self.assertRaisesRegex(InterruptedError, 'solid export cancelled'):
            self.run_mesh(provider, progress=progress.append, stop_cb=lambda: True)
        self.assertNotIn('batch_boolean', provider.names())
        self.assertNotIn('to_mesh64', provider.names())
        self.assertEqual(progress, [])

    def test_empty_adapter_arrays_do_not_attempt_minmax(self):
        provider = ProviderDouble(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.uint64))
        # Deliberately inconsistent double: native triangle count stays nonzero.
        vertices, faces = self.run_mesh(provider)
        self.assertEqual(vertices.shape, (0, 3))
        self.assertEqual(faces.shape, (0, 3))

    def test_positive_small_triangle_preserves_full_precision(self):
        vertices = np.array([[0., 0., 0.], [1., 1., 0.], [2., 2.00003, 0.]])
        faces = np.array([[0, 1, 2]])
        actual, actual_faces = self.run_mesh(ProviderDouble(vertices, faces))
        np.testing.assert_array_equal(actual, vertices)
        np.testing.assert_array_equal(actual_faces, faces)
        self.assertGreater(np.cross(actual[1] - actual[0], actual[2] - actual[0])[2], 0)
        rounded = np.round(vertices, 4)
        self.assertEqual(np.cross(rounded[1] - rounded[0], rounded[2] - rounded[0])[2], 0)

    def test_degenerate_and_duplicate_faces_are_not_python_repaired(self):
        faces = np.array([[0, 0, 1], [0, 1, 2], [0, 1, 2]])
        _, actual = self.run_mesh(ProviderDouble(faces=faces))
        np.testing.assert_array_equal(actual, faces)

    def test_graph_dedup_and_degree_two_merge_preserved(self):
        provider = ProviderDouble()
        points = np.array([[2., 0, 0], [0., 0, 0], [1., 0, 0], [1e-9, 0, 0]])
        edges = np.array([[0, 2], [2, 1], [3, 1], [2, 0]])
        before_points, before_edges = points.copy(), edges.copy()
        with patch.dict(sys.modules, {'manifold3d': provider.module}):
            self.tube.tube_mesh(points, edges, .25)
        self.assertEqual([c for c in provider.calls if c[0] == 'cylinder'], [('cylinder', 2., .25, 16)])
        np.testing.assert_array_equal([c[1] for c in provider.calls if c[0] == 'translate'], [[0, 0, 0], [2, 0, 0]])
        np.testing.assert_array_equal(points, before_points)
        np.testing.assert_array_equal(edges, before_edges)

    def test_spatial_order_and_deterministic_phase_preserved(self):
        provider = ProviderDouble()
        points = np.array([[4., 0, 0], [5., 0, 0], [0., 0, 0], [1., 0, 0]])
        with patch.dict(sys.modules, {'manifold3d': provider.module}):
            self.tube.tube_mesh(points, np.array([[1, 0], [3, 2]]), .25)
        for i, transform in enumerate(c[1] for c in provider.calls if c[0] == 'transform'):
            phase = (i * .61803398875 % 1.) * 2 * np.pi / 16
            expected = np.column_stack([[0., np.sin(phase), -np.cos(phase)],
                                        [0., np.cos(phase), np.sin(phase)],
                                        [1., 0, 0], [4. * i, 0, 0]])
            np.testing.assert_allclose(transform, expected, rtol=0, atol=1e-15)

    def test_bounded_batches_progress_preserved(self):
        provider = ProviderDouble()
        points = np.column_stack([np.arange(100, dtype=float), np.zeros((100, 2))])
        progress = []
        with patch.dict(sys.modules, {'manifold3d': provider.module}):
            self.tube.tube_mesh(points, np.arange(100).reshape(-1, 2), .25, progress=progress.append)
        batches = [c for c in provider.calls if c[0] == 'batch_boolean']
        self.assertEqual([len(c[1]) for c in batches], [96, 54, 2])
        self.assertEqual(batches[-1][1], ('union', 'union'))
        self.assertEqual(progress, [0, 17, 34, 51, 68, 85, 95])
        self.assertEqual(provider.names().count('cleanup'), 1)
        self.assertEqual(provider.names().count('to_mesh64'), 2)

    def test_primitive_cap_preserved(self):
        provider = ProviderDouble()
        provider.sphere_triangles = 6_000_000
        with self.assertRaisesRegex(MemoryError, 'tube mesh exceeds available geometry budget'):
            self.run_mesh(provider)
        self.assertEqual(provider.names(), ['sphere', 'num_tri'])

    def test_intermediate_union_cap_preserved(self):
        provider = ProviderDouble()
        provider.union_triangles = 8_000_001
        points = np.column_stack([np.arange(100, dtype=float), np.zeros((100, 2))])
        with patch.dict(sys.modules, {'manifold3d': provider.module}):
            with self.assertRaisesRegex(MemoryError, 'solid union exceeds geometry budget'):
                self.tube.tube_mesh(points, np.arange(100).reshape(-1, 2), .25)
        self.assertEqual(provider.names().count('batch_boolean'), 3)
        self.assertNotIn('as_original', provider.names())

    def test_cancellation_before_union_tree_merge_preserved(self):
        provider = ProviderDouble()
        points = np.column_stack([np.arange(100, dtype=float), np.zeros((100, 2))])
        checks = iter([False] * 150 + [True])
        with patch.dict(sys.modules, {'manifold3d': provider.module}):
            with self.assertRaisesRegex(InterruptedError, 'solid export cancelled'):
                self.tube.tube_mesh(points, np.arange(100).reshape(-1, 2), .25, stop_cb=lambda: next(checks))
        self.assertEqual(provider.names().count('batch_boolean'), 2)
        self.assertNotIn('as_original', provider.names())

    def test_simplification_floor_preserved(self):
        provider = ProviderDouble()
        with patch.dict(sys.modules, {'manifold3d': provider.module}):
            self.tube.tube_mesh(POINTS, EDGES, 1e-4)
        self.assertEqual([c for c in provider.calls if c[0] == 'simplify'], [('simplify', 1e-5)])

    def test_contact_displacement_changes_only_coincident_vertices(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0], [0., 0, 0], [-1., 0, 0], [0., 0, 1.]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        actual, actual_faces = self.run_mesh(ProviderDouble(vertices, faces))
        expected = vertices.copy()
        expected[0, 2] = -.00025
        expected[3, 1] = -.00025
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_faces, faces)
        np.testing.assert_array_equal(vertices[[0, 3]], np.zeros((2, 3)))

    def test_ambiguous_contact_refuses(self):
        vertices = np.vstack([VERTICES, VERTICES[0]])
        with self.assertRaisesRegex(ValueError, 'ambiguous zero-thickness solid contact'):
            self.run_mesh(ProviderDouble(vertices, FACES))

    def test_unresolved_contact_retains_three_displacements(self):
        vertices = np.tile(np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0]]), (2, 1))
        provider = ProviderDouble(vertices, np.array([[0, 1, 2], [3, 4, 5]]))
        original_round = np.round
        classification_inputs = []
        def recorded_round(values, decimals=0, *args, **kwargs):
            if decimals == 5:
                classification_inputs.append(values.copy())
            return original_round(values, decimals, *args, **kwargs)
        with patch.object(self.tube.np, 'round', recorded_round):
            with self.assertRaisesRegex(ValueError, 'unresolved coincident solid vertices'):
                self.run_mesh(provider)
        self.assertEqual(len(classification_inputs), 4)
        for coordinates, displacement in zip(classification_inputs, [0., -.00025, -.00075, -.0015]):
            np.testing.assert_allclose(coordinates[:, 2], displacement, rtol=0, atol=1e-18)

    def test_contact_nonfinite_result_refuses(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0], [0., 0, 0], [-1., 0, 0], [0., 0, 1.]])
        provider = ProviderDouble(vertices, np.array([[0, 1, 2], [3, 4, 5]]))
        original_cross = np.cross
        def overflow(a, b, *args, **kwargs):
            return np.full(a.shape, np.inf) if a.ndim == 2 else original_cross(a, b, *args, **kwargs)
        with patch.object(self.tube.np, 'cross', overflow), np.errstate(invalid='ignore', over='ignore'):
            with self.assertRaisesRegex(ValueError, 'finite'):
                self.run_mesh(provider)

    def test_extended_float_matches_binary64_itemsize_contract(self):
        vertices = VERTICES.astype(np.longdouble)
        provider = ProviderDouble(vertices=vertices)
        if vertices.dtype.itemsize == 8:
            actual, _ = self.run_mesh(provider)
            np.testing.assert_array_equal(actual, VERTICES)
        else:
            with self.assertRaises(ValueError):
                self.run_mesh(provider)


def add_test(name, body):
    body.__name__ = name
    setattr(CleanupTests, name, body)


for label, capability in [('absent', 'absent'), ('none', None), ('integer', 17)]:
    def check(self, capability=capability):
        provider = ProviderDouble()
        if capability == 'absent':
            delattr(provider.module.Manifold, 'to_mesh64')
        else:
            provider.module.Manifold.to_mesh64 = capability
        with self.assertRaisesRegex(RuntimeError, '(binary64|to_mesh64|mesh64)'):
            self.run_mesh(provider)
        self.assertEqual(provider.calls, [])
    add_test('test_missing_mesh64_capability_' + label, check)

for label, ordinal, expected in [('first_extract', 4, ('to_mesh64', 'union')),
                                 ('after_constructor', 5, None),
                                 ('second_extract', 6, ('to_mesh64', 'cleanup'))]:
    def check(self, ordinal=ordinal, expected=expected):
        provider = ProviderDouble()
        checks = iter([False] * (ordinal - 1) + [True])
        with self.assertRaisesRegex(InterruptedError, 'solid export cancelled'):
            self.run_mesh(provider, stop_cb=lambda: next(checks))
        if expected is None:
            self.assertEqual(provider.names()[-1], 'cleanup')
        else:
            self.assertEqual(provider.calls[-1], expected)
        if ordinal == 4:
            self.assertNotIn('status', provider.names())
            self.assertNotIn('cleanup', provider.names())
        elif ordinal == 5:
            self.assertNotIn(('status', 'cleanup'), provider.calls)
            self.assertEqual(provider.names().count('to_mesh64'), 1)
        else:
            self.assertEqual(provider.names().count('to_mesh64'), 2)
    add_test('test_cancellation_' + label, check)

for label, dtype in [('little', '<f8'), ('big', '>f8')]:
    def check(self, dtype=dtype):
        storage = np.zeros((8, 10), dtype=dtype)
        properties = storage[::2, ::2]
        properties[:, :3] = VERTICES
        properties[:, 3:] = np.nan
        face_storage = np.zeros((4, 6), dtype='>u8')
        incoming = face_storage[:, ::2]
        incoming[:] = FACES
        properties.setflags(write=False)
        incoming.setflags(write=False)
        vertices, faces = self.run_mesh(ProviderDouble(properties, incoming))
        for output, expected_dtype in [(vertices, np.dtype('float64')), (faces, np.dtype('int32'))]:
            self.assertEqual(output.dtype, expected_dtype)
            self.assertTrue(output.flags.owndata and output.flags.writeable and output.flags.c_contiguous)
        self.assertFalse(np.shares_memory(vertices, storage))
        self.assertFalse(np.shares_memory(faces, face_storage))
        vertices[:, 2] *= 25.
        vertices += [2., 3., 4.]
        faces[0] = [3, 2, 1]
        np.testing.assert_array_equal(properties[:, :3], VERTICES)
        np.testing.assert_array_equal(incoming, FACES)
        storage[:] = 123.
        face_storage[:] = 3
        np.testing.assert_array_equal(vertices, VERTICES * [1., 1., 25.] + [2., 3., 4.])
        np.testing.assert_array_equal(faces[1:], FACES[1:])
    add_test('test_owned_readonly_strided_outputs_' + label, check)

bad_positions = [('rank1', np.zeros(12)), ('rank3', np.zeros((2, 2, 3))),
                 ('two_columns', np.zeros((4, 2)))]
for label, dtype in [('float32', np.float32), ('float16', np.float16), ('integer', np.int64),
                     ('bool', bool), ('complex', complex), ('object', object), ('string', 'U4')]:
    bad_positions.append((label, VERTICES.astype(dtype)))
for label, positions in bad_positions:
    def check(self, positions=positions):
        with self.assertRaisesRegex(ValueError, 'binary64'):
            self.run_mesh(ProviderDouble(vertices=positions))
    add_test('test_malformed_positions_' + label, check)

for label, value in [('nan', np.nan), ('posinf', np.inf), ('neginf', -np.inf)]:
    def check(self, value=value):
        positions = VERTICES.copy()
        positions[2, 1] = value
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.run_mesh(ProviderDouble(vertices=positions))
    add_test('test_nonfinite_positions_' + label, check)

bad_faces = [('rank1', np.array([0, 1, 2])), ('rank3', np.zeros((1, 1, 3), dtype=int)),
             ('two_columns', np.zeros((2, 2), dtype=int)), ('four_columns', np.zeros((2, 4), dtype=int)),
             ('integral_float', np.array([[0., 1., 2.]])), ('fractional_float', np.array([[0., 1., 1.5]])),
             ('bool', np.array([[False, True, False]])), ('object', np.array([[0, 1, 2]], dtype=object)),
             ('complex', np.array([[0, 1, 2]], dtype=complex)), ('string', np.array([[0, 1, 2]], dtype='U1'))]
for label, faces in bad_faces:
    def check(self, faces=faces):
        with self.assertRaisesRegex(ValueError, 'integer array'):
            self.run_mesh(ProviderDouble(faces=faces))
    add_test('test_malformed_faces_' + label, check)

for dtype in ['i1', 'u1', '<i2', '>i2', '<u2', '>u2', '<i4', '>i4', '<u4', '>u4', '<i8', '>i8', '<u8', '>u8']:
    label = dtype.replace('<', 'little_').replace('>', 'big_')
    def check(self, dtype=dtype):
        incoming = FACES.astype(dtype)
        _, faces = self.run_mesh(ProviderDouble(faces=incoming))
        self.assertEqual(faces.dtype, np.dtype('int32'))
        self.assertTrue(faces.flags.owndata and faces.flags.c_contiguous and faces.flags.writeable)
        self.assertFalse(np.shares_memory(faces, incoming))
        np.testing.assert_array_equal(faces, FACES)
    add_test('test_valid_integer_faces_' + label, check)
    def check_bad_bound(self, dtype=dtype):
        incoming = np.array([[0, 1, 4]], dtype=dtype)
        with self.assertRaisesRegex(ValueError, 'outside the vertex table or int32 range'):
            self.run_mesh(ProviderDouble(faces=incoming))
        np.testing.assert_array_equal(incoming, [[0, 1, 4]])
    add_test('test_index_at_vertex_count_' + label, check_bad_bound)
    if np.dtype(dtype).kind == 'i':
        def check_negative(self, dtype=dtype):
            with self.assertRaisesRegex(ValueError, 'outside the vertex table or int32 range'):
                self.run_mesh(ProviderDouble(faces=np.array([[-1, 1, 2]], dtype=dtype)))
        add_test('test_negative_index_' + label, check_negative)

for dtype in ['<u4', '>u4', '<i8', '>i8', '<u8', '>u8']:
    label = dtype.replace('<', 'little_').replace('>', 'big_')
    def check(self, dtype=dtype):
        with self.assertRaisesRegex(ValueError, 'outside the vertex table or int32 range'):
            self.run_mesh(ProviderDouble(faces=np.array([[0, 1, 2**31]], dtype=dtype)))
    add_test('test_int32_overflow_' + label, check)

for dtype in ['<u8', '>u8']:
    label = dtype.replace('<', 'little_').replace('>', 'big_')
    def check(self, dtype=dtype):
        with self.assertRaisesRegex(ValueError, 'outside the vertex table or int32 range'):
            self.run_mesh(ProviderDouble(faces=np.array([[0, 1, 2**64 - 1]], dtype=dtype)))
    add_test('test_uint64_maximum_' + label, check)

if __name__ == '__main__':
    unittest.main(verbosity=2)
