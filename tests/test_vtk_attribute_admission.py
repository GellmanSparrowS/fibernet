"""Reject malformed legacy VTK attributes before opening the destination."""
from collections import UserDict
import warnings

import numpy as np
import pytest

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.io.vtk import to_vtk


def _network():
    network = FiberNetwork()
    network.add_fiber(Fiber(np.array([[0., 0., 0.], [1., 0., 0.],
                                     [2., 1., 0.], [3., 1., 0.]])))
    return network


@pytest.mark.parametrize('location,shape', [
    ('point_data', (4,)), ('point_data', (4, 3)), ('cell_data', (3,)),
])
@pytest.mark.parametrize('bad', [
    'text', 'complex', 'object', 'nan', 'positive_infinity',
    'negative_infinity', 'positive_overflow', 'negative_overflow',
])
@pytest.mark.parametrize('existing', [False, True])
def test_invalid_values_do_not_touch_destination(tmp_path, location, shape, bad, existing):
    values = np.ones(shape)
    if bad == 'text':
        values = values.astype(str)
    elif bad == 'complex':
        values = values.astype(complex)
    elif bad == 'object':
        values = values.astype(object)
    else:
        values.flat[-1] = {
            'nan': np.nan, 'positive_infinity': np.inf,
            'negative_infinity': -np.inf,
            'positive_overflow': float(np.finfo(np.float32).max) * 2,
            'negative_overflow': -float(np.finfo(np.float32).max) * 2,
        }[bad]
    path = tmp_path / 'network.vtk'
    if existing:
        path.write_bytes(b'previous valid VTK')
    with pytest.raises(ValueError, match='finite real numeric|float32'):
        to_vtk(_network(), path, **{location: {'value': values}})
    assert path.read_bytes() == b'previous valid VTK' if existing else not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


@pytest.mark.parametrize('location,count', [('point_data', 4), ('cell_data', 3)])
@pytest.mark.parametrize('name', ['', 'two words', 'tab\tname', 'new\nline',
                                  'null\x00name', 'delete\x7f', '温度', 4, None, 'x' * 256,
                                  'a%20b', 'a%00b', 'name%ff'])
def test_invalid_names_do_not_open_destination(tmp_path, location, count, name):
    path = tmp_path / 'network.vtk'
    path.write_bytes(b'previous valid VTK')
    with pytest.raises(ValueError, match='ASCII token'):
        to_vtk(_network(), path, **{location: {name: np.ones(count)}})
    assert path.read_bytes() == b'previous valid VTK'


@pytest.mark.parametrize('location', ['point_data', 'cell_data'])
@pytest.mark.parametrize('bad', [[], (), '', np.array([]), [('name', [1, 2, 3])]])
def test_nonmapping_attributes_are_not_silently_ignored(tmp_path, location, bad):
    path = tmp_path / 'network.vtk'
    with pytest.raises(ValueError, match='mapping'):
        to_vtk(_network(), path, **{location: bad})
    assert not path.exists()


def test_valid_numeric_and_boolean_fields_keep_legacy_format(tmp_path):
    path = tmp_path / 'network.vtk'
    floats = np.array([-np.finfo(np.float32).max, 0., .5, np.finfo(np.float32).max])
    vectors = np.arange(12, dtype=np.int64).reshape(4, 3)
    flags = np.array([True, False, True])
    original = [array.copy() for array in (floats, vectors, flags)]
    returned = to_vtk(_network(), path,
                      point_data=UserDict({'field.with-dots': floats, 'vectors': vectors}),
                      cell_data=UserDict({'flags': flags}))
    assert returned == path
    text = path.read_text()
    assert 'SCALARS field.with-dots float 1\nLOOKUP_TABLE default\n' in text
    assert '-3.402823e+38\n0.000000e+00\n5.000000e-01\n3.402823e+38\n' in text
    assert 'VECTORS vectors float\n' in text
    assert '9.000000e+00 1.000000e+01 1.100000e+01\n' in text
    assert 'SCALARS flags float 1\nLOOKUP_TABLE default\n1.000000e+00\n0.000000e+00\n1.000000e+00\n' in text
    for array, before in zip((floats, vectors, flags), original):
        np.testing.assert_array_equal(array, before)


def test_all_false_masks_remain_accepted(tmp_path):
    values = np.ma.array([1., 2., 3., 4.], mask=False)
    to_vtk(_network(), tmp_path / 'network.vtk', point_data={'s': values})
    assert '4.000000e+00\n' in (tmp_path / 'network.vtk').read_text()


def test_255_character_name_is_supported(tmp_path):
    name = 'x' * 255
    path = tmp_path / 'network.vtk'
    to_vtk(_network(), path, point_data={name: np.arange(4)})
    assert f'SCALARS {name} float 1' in path.read_text()


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.longdouble,
                                  np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64, np.bool_])
def test_valid_numpy_dtypes_do_not_warn(tmp_path, dtype):
    point = np.arange(4).astype(dtype)
    cell = np.arange(3).astype(dtype)
    if np.dtype(dtype).kind in 'iu':
        point[-1] = np.iinfo(dtype).max
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        to_vtk(_network(), tmp_path / 'network.vtk',
               point_data={'s': point, 'v': np.repeat(point[:, None], 3, axis=1)},
               cell_data={'s': cell})
