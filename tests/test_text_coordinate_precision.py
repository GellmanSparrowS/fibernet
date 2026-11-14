"""Geometry survives text export independently of coordinate scale."""
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.io.vtk import to_vtk, to_vtk_xml
from fibernet.io.xyz import to_xyz, from_xyz


@pytest.mark.parametrize('kind', ['vtk', 'vtp', 'xyz'])
@pytest.mark.parametrize('points', [
    [[0., 0., 0.], [1e-7, 2e-7, 3e-7]],
    [[0., 0., 0.], [1.123456789012345, 2.123456789012345, 3.123456789012345]],
    [[1e10, -1e10, 1e10], [1e10 + .125, -1e10 + .25, 1e10 + .5]],
    [[-0., 1e-200, -1e-200], [1e-200, -0., 0.]],
    [[0., 0., 0.], [1e100, -1e100, 1e100]],
])
def test_float64_coordinates_roundtrip(tmp_path, kind, points):
    source = np.asarray(points, dtype=np.float64)
    net = FiberNetwork(fibers=[Fiber(source)])
    path = tmp_path / ('network.' + kind)
    if kind == 'vtk':
        to_vtk(net, path)
        lines = path.read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith('POINTS '))
        assert lines[start] == 'POINTS 2 double'
        actual = np.loadtxt(lines[start + 1:start + 3], dtype=np.float64)
        assert 'LINES 1 3\n2 0 1\n' in path.read_text()
    elif kind == 'vtp':
        to_vtk_xml(net, path)
        root = ET.parse(path)
        array = root.find('.//Points/DataArray')
        assert array.attrib['type'] == 'Float64'
        actual = np.fromstring(array.text, sep=' ').reshape(-1, 3)
        assert root.find('.//Piece').attrib['NumberOfLines'] == '1'
    else:
        to_xyz(net, path, bead_spacing=1e101)
        actual = from_xyz(path, strict=True)
    np.testing.assert_array_equal(actual, source)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(source))
