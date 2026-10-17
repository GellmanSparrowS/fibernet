"""Exact stored-coordinate degeneracy, without an optional native mesher."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from fibernet.gen import solid_export as module


FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
REAL_DTYPES = ['i1', 'u1', '<i2', '>i2', '<u2', '>u2', '<i4', '>i4',
               '<u4', '>u4', '<i8', '>i8', '<u8', '>u8', '<f2', '>f2',
               '<f4', '>f4', '<f8', '>f8']


def solid(vertices, faces=FACES):
    return module.FiberSolid(vertices, faces.copy(), vertices, .01, module.PrintSettings())


def regression():
    # A closed subdivided tetrahedron with one exactly collinear triangle.
    a = [.25679999589920044, 50.25, 100.4999008178711]
    b = [.25679999589920044, 50.25709915161133, 100.4999008178711]
    c = [.25679999589920044, 75.247802734375, 100.4999008178711]
    vertices = np.array([a, c, [a[0] + 1, a[1], a[2]],
                         [a[0], a[1], a[2] - 1], b])
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 0, 4],
                      [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


@pytest.mark.parametrize('dtype', REAL_DTYPES)
def test_valid_tetrahedron_preserves_dtype_and_orientation(dtype):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype)
    instance = solid(vertices)
    original_faces = instance.faces.copy()
    triangles = vertices[original_faces]
    expected = float(np.einsum('ij,ij->i', triangles[:, 0],
                              np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6)
    assert instance.validate() == expected
    assert instance.vertices is vertices and instance.vertices.dtype == vertices.dtype
    assert np.array_equal(instance.faces, original_faces)


@pytest.mark.parametrize('dtype', REAL_DTYPES)
@pytest.mark.parametrize('strided', [False, True])
def test_exact_collinear_and_positive_faces(dtype, strided):
    vertices = np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4], [4, 4, 5]], dtype=dtype)
    if strided:
        storage = np.zeros((8, 6), dtype=dtype)
        storage[::2, ::2] = vertices
        vertices = storage[::2, ::2]
        assert not vertices.flags.c_contiguous
    assert module._first_zero_area_face(vertices, np.array([[0, 1, 3]])) is None
    assert module._first_zero_area_face(vertices, np.array([[0, 1, 3], [0, 1, 2]])) == 1


@pytest.mark.parametrize('dtype', ['<f2', '>f2', '<f4', '>f4', '<f8', '>f8'])
def test_stored_subnormals_and_signed_zero(dtype):
    dtype = np.dtype(dtype)
    endian = 'little' if dtype.byteorder in ('<', '=') and sys.byteorder == 'little' else 'big'
    if dtype.byteorder == '<':
        endian = 'little'
    if dtype.byteorder == '>':
        endian = 'big'
    width = dtype.itemsize
    negative_zero = 1 << (8 * width - 1)
    # Build bits directly: no arithmetic can flush the smallest subnormal.
    packed = [negative_zero, 0, 0, 1, 0, 0, 0, 1, 0]
    vertices = np.frombuffer(b''.join(x.to_bytes(width, endian) for x in packed), dtype=dtype).reshape(3, 3)
    assert module._first_zero_area_face(vertices, np.array([[0, 1, 2]])) is None
    zeros = np.frombuffer(b''.join(x.to_bytes(width, endian) for x in [negative_zero] * 9), dtype=dtype).reshape(3, 3)
    assert module._first_zero_area_face(zeros, np.array([[0, 1, 2]])) == 0


@pytest.mark.parametrize('dtype', ['i8', 'u8', '>i8', '>u8'])
def test_integer_offsets_above_float_exact_range(dtype):
    base = 2**63 - 4 if np.dtype(dtype).kind == 'i' else 2**64 - 4
    vertices = np.array([[base, base, base], [base + 1, base, base],
                         [base, base + 1, base]], dtype=dtype)
    assert module._first_zero_area_face(vertices, np.array([[0, 1, 2]])) is None
    vertices[2] = [base + 2, base, base]
    assert module._first_zero_area_face(vertices, np.array([[0, 1, 2]])) == 0


@pytest.mark.parametrize('dtype', ['<i8', '>i8'])
def test_signed_integer_subtraction_does_not_overflow(dtype):
    vertices = np.array([[-2**63, 0, 0], [2**63 - 1, 0, 0], [0, 1, 0]], dtype=dtype)
    assert module._first_zero_area_face(vertices, np.array([[0, 1, 2]])) is None


def test_large_tiny_mixed_binary64_and_cancellation():
    maximum = np.finfo(np.float64).max
    tiny = np.frombuffer((1).to_bytes(8, sys.byteorder), dtype=np.float64)[0]
    vertices = np.array([[maximum, 0, 0], [-maximum, 0, 0], [0, tiny, 0]])
    assert module._first_zero_area_face(vertices, np.array([[0, 1, 2]])) is None
    # The determinant is one, while rounded multiplication can cancel it.
    n = 2**52
    vertices = np.array([[0, 0, 0], [n, n - 1, 0], [n + 1, n, 0]], dtype=np.float64)
    assert module._first_zero_area_face(vertices, np.array([[0, 1, 2]])) is None


@pytest.mark.parametrize('dtype', ['?', 'complex64', 'complex128', 'O', 'U2', 'datetime64[D]'])
def test_unsupported_coordinate_kinds_are_explicit(dtype):
    vertices = np.zeros((4, 3), dtype=dtype)
    instance = solid(vertices)
    with pytest.raises(ValueError, match='unsupported dtype'):
        instance.validate()
    assert instance.vertices is vertices


def test_longdouble_is_selected_by_representation():
    dtype = np.dtype(np.longdouble)
    info = np.finfo(dtype)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=dtype)
    binary64_alias = (dtype.itemsize, info.bits, info.iexp, info.nmant) == (8, 64, 11, 52)
    if binary64_alias:
        assert module._first_zero_area_face(vertices, np.array([[0, 1, 2]])) is None
    else:
        with pytest.raises(ValueError, match='unsupported dtype'):
            module._first_zero_area_face(vertices, np.array([[0, 1, 2]]))


@pytest.mark.parametrize('kind', ['nan', 'inf', '-inf'])
def test_nonfinite_values_are_refused(kind):
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vertices[0, 0] = float(kind)
    with pytest.raises(ValueError, match='finite'):
        solid(vertices).validate()
    with pytest.raises(ValueError, match='finite'):
        module._first_zero_area_face(vertices, FACES)


def test_regression_refuses_before_assigning_wrapper_arrays():
    vertices, faces = regression()
    inputs = vertices.tolist(), faces.tolist()
    instance = module.FiberSolid(inputs[0], inputs[1], vertices, .01, module.PrintSettings())
    with pytest.raises(ValueError, match='solid face 2 has exactly zero area'):
        instance.validate()
    assert instance.vertices is inputs[0] and instance.faces is inputs[1]
    assert np.array_equal(instance.vertices, vertices) and np.array_equal(instance.faces, faces)


@pytest.mark.parametrize('suffix', ['stl', '3mf'])
def test_zero_area_refusal_precedes_export_staging(tmp_path, monkeypatch, suffix):
    vertices, faces = regression()
    instance = solid(vertices, faces)
    path = tmp_path / ('existing.' + suffix)
    path.write_bytes(b'existing destination')
    def unexpected_stage(*args):
        raise AssertionError('invalid geometry reached staging')
    monkeypatch.setattr(module, '_owned_export_file', unexpected_stage)
    with pytest.raises(ValueError, match='solid face 2 has exactly zero area'):
        module.export_solid(path, instance)
    assert path.read_bytes() == b'existing destination'
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    assert np.array_equal(instance.faces, faces)


def test_duplicate_coordinates_with_distinct_indices_and_all_zero():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match='solid face 0 has exactly zero area'):
        solid(vertices).validate()
    assert module._first_zero_area_face(np.zeros((3, 3)), np.array([[0, 1, 2]])) == 0


def test_positive_area_does_not_override_existing_volume_floor():
    vertices = np.array([[0., 0, 0], [1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]])
    assert module._first_zero_area_face(vertices, FACES) is None
    with pytest.raises(ValueError, match='solid has no volume'):
        solid(vertices).validate()


def test_negative_orientation_retains_existing_correction():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    instance = solid(vertices, FACES[:, [0, 2, 1]])
    assert instance.validate() == pytest.approx(1 / 6)
    assert np.array_equal(instance.faces, FACES)


def test_independent_fraction_reference():
    rng = np.random.default_rng(1729)
    for dtype in ['f2', 'f4', 'f8', '>f8', 'i8', 'u8']:
        for trial in range(40):
            vertices = rng.integers(0, 64, size=(3, 3)).astype(dtype)
            if trial % 2 == 0:
                vertices[2] = vertices[0] + 2 * (vertices[1] - vertices[0]) if np.dtype(dtype).kind != 'u' else vertices[0]
            values = [[Fraction(int(v)) if np.dtype(dtype).kind in 'iu'
                       else Fraction(*v.as_integer_ratio()) for v in row] for row in vertices]
            u = [values[1][i] - values[0][i] for i in range(3)]
            v = [values[2][i] - values[0][i] for i in range(3)]
            zero = all(u[a] * v[b] == u[b] * v[a] for a, b in [(0, 1), (1, 2), (2, 0)])
            assert (module._first_zero_area_face(vertices, np.array([[0, 1, 2]])) == 0) == zero


def test_leaf_import_needs_no_native_mesher_or_ml_provider():
    source = str(Path(module.__file__).resolve())
    code = '''import importlib.util, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('manifold3d', 'torch', 'torch_geometric', 'taichi', 'scipy', 'skimage'):
   raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Block())
spec=importlib.util.spec_from_file_location('isolated_solid', sys.argv[1])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
assert module._first_zero_area_face(module.np.array([[0,0,0],[1,0,0],[0,1,0]]),module.np.array([[0,1,2]])) is None
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, source], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
