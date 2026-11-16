"""FEA node coordinates use the same float64 geometry as preflight."""

import numpy as np
import pytest

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.io import fea_export as fea


EXPORTERS = (fea.export_to_abaqus, fea.export_to_ansys, fea.export_to_gmsh)


def node_coordinates(path, exporter):
    text = path.read_text()
    if exporter is fea.export_to_abaqus:
        lines = text.split('*NODE\n')[1].split('*ELEMENT')[0].strip().splitlines()
        return np.array([[float(x) for x in line.split(',')[1:]] for line in lines])
    if exporter is fea.export_to_ansys:
        return np.array([[float(x) for x in line.split(',')[2:]]
                         for line in text.splitlines() if line.startswith('N, ')])
    lines = text.split('$Nodes\n')[1].split('$EndNodes')[0].strip().splitlines()[1:]
    return np.array([[float(x) for x in line.split()[1:]] for line in lines])


@pytest.mark.parametrize('exporter', EXPORTERS)
@pytest.mark.parametrize('coordinates', [
    np.array([[-2**62, 0, 0], [2**62, 0, 0]], dtype=np.int64),
    np.array([[-2e38, 0, 0], [2e38, 0, 0]], dtype=np.float32),
    [[-2., 0., 0.], [2., 0., 0.]],
])
def test_mutable_centerline_uses_validated_float64(tmp_path, exporter, coordinates):
    fiber = Fiber([[-1., 0., 0.], [1., 0., 0.]])
    fiber.centerline = coordinates
    original = np.array(coordinates, copy=True)
    net = FiberNetwork(fibers=[fiber])
    path = tmp_path / 'network.txt'
    exporter(net, path, segments_per_fiber=2)
    expected = np.array([original[0], [0., 0., 0.], original[-1]], dtype=np.float64)
    np.testing.assert_array_equal(node_coordinates(path, exporter), expected)
    assert fiber.centerline is coordinates
    np.testing.assert_array_equal(np.asarray(coordinates), original)


@pytest.mark.parametrize('exporter', EXPORTERS)
def test_float32_geometry_matches_explicit_float64_export(tmp_path, exporter):
    points = np.array([[0.125, 0.25, 0.5], [1.75, 2.5, 3.25]], dtype=np.float32)
    native = FiberNetwork(fibers=[Fiber(points)])
    mutable = FiberNetwork(fibers=[Fiber(points)])
    mutable.fibers[0].centerline = points
    first, second = tmp_path / 'native', tmp_path / 'mutable'
    exporter(native, first, segments_per_fiber=7)
    exporter(mutable, second, segments_per_fiber=7)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize('exporter', EXPORTERS)
def test_many_fiber_export_matches_normalized_network(tmp_path, exporter):
    """Mixed mutable coordinate containers preserve every node and element."""
    normalized, reassigned = [], []
    for i in range(36):
        start = np.array([i % 6, i // 6, i % 3], dtype=np.float32)
        end = start + np.array([0.75, 0.5, 0.25], dtype=np.float32)
        points = np.array([start, end])
        normalized.append(Fiber(points))
        fiber = Fiber(points)
        fiber.centerline = points.tolist() if i % 2 else points
        reassigned.append(fiber)
    first, second = tmp_path / 'normalized', tmp_path / 'reassigned'
    exporter(FiberNetwork(fibers=normalized), first, segments_per_fiber=9)
    exporter(FiberNetwork(fibers=reassigned), second, segments_per_fiber=9)
    assert first.read_bytes() == second.read_bytes()
    assert node_coordinates(second, exporter).shape == (360, 3)


@pytest.mark.parametrize('exporter', EXPORTERS)
@pytest.mark.parametrize('coordinates', [
    np.array([[3j, 0, 0], [1 + 4j, 0, 0]]),
    [[0., 0., 0.], [1., True, 0.]],
    np.array([[0., 0., 0.], [1., 0., 0.]], dtype=str),
    np.ma.array([[0., 0., 0.], [1., 0., 0.]], mask=False),
    [[0., 0., 0.], np.ma.array([1., 0., 0.], mask=[False, True, False])],
    np.array([[0., 0., 0.], [1., 0., 4j]], dtype=object),
    np.array([[0., 0., 0.], [1., False, 0.]], dtype=object),
])
def test_lossy_coordinate_conversion_rejected_before_open(tmp_path, exporter, coordinates):
    fiber = Fiber([[0., 0., 0.], [1., 0., 0.]])
    fiber.centerline = coordinates
    net = FiberNetwork(fibers=[fiber])
    for existing in (False, True):
        path = tmp_path / str(existing)
        if existing:
            path.write_bytes(b'preserve original file')
        with pytest.raises(ValueError, match='unmasked real numeric'):
            exporter(net, path)
        if existing:
            assert path.read_bytes() == b'preserve original file'
        else:
            assert not path.exists()
