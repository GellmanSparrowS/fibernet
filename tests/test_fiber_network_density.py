"""Analytical strip area and volume fractions for the public FiberNetwork."""

import copy
import math

import numpy as np
import pytest

from fibernet import Fiber, FiberNetwork
from fibernet.core.fiber import CrossSection


@pytest.mark.parametrize(('dimension', 'box_size', 'box_area'), [
    (2, [10., 10., 4.], 100.),
    (3, [10., 10., 0.], 100.),
    (2, None, 49.),
    (3, None, 49.),
])
def test_planar_density_adds_each_fiber_strip_and_is_partition_invariant(
        dimension, box_size, box_area):
    # The thin fiber has length 1 and the thick fiber has legs of length 4
    # and 6. Splitting the thick fiber at the bend retains both line segments.
    thin = Fiber([[1., 1., 0.], [2., 1., 0.]], radius=.1)
    thick = Fiber([[4., 2., 0.], [8., 2., 0.], [8., 8., 0.]], radius=1.)
    box = None if box_size is None else np.array(box_size)
    source = FiberNetwork(fibers=[thin, thick], dimension=dimension, box_size=box)
    split = FiberNetwork(fibers=[
        thin,
        Fiber([[4., 2., 0.], [8., 2., 0.]], radius=1.),
        Fiber([[8., 2., 0.], [8., 8., 0.]], radius=1.),
    ], dimension=dimension, box_size=box)
    expected = (2 * .1 * 1 + 2 * 1. * (4 + 6)) / box_area
    for network in (source, split):
        strip_area = math.fsum(2 * fiber.radius * math.dist(a, b)
                               for fiber in network.fibers
                               for a, b in zip(fiber.centerline[:-1], fiber.centerline[1:]))
        assert strip_area / box_area == pytest.approx(expected)

    source_before = copy.deepcopy(source.to_dict())
    split_before = copy.deepcopy(split.to_dict())
    assert source.density() == pytest.approx(expected)
    assert split.density() == pytest.approx(expected)
    assert source.to_dict() == source_before
    assert split.to_dict() == split_before


@pytest.mark.parametrize(('lengths', 'radii', 'strip_area'), [
    ((1., 10.), (.4, .4), 8.8),
    ((4., 4.), (.1, 1.), 8.8),
])
def test_planar_density_for_equal_radii_or_equal_lengths(lengths, radii, strip_area):
    fibers = [Fiber([[0., y, 0.], [length, y, 0.]], radius=radius)
              for y, (length, radius) in enumerate(zip(lengths, radii))]
    network = FiberNetwork(fibers=fibers, dimension=2, box_size=np.array([12., 3.]))
    assert network.density() == pytest.approx(strip_area / 36.)


def test_3d_density_retains_actual_cross_section_volumes():
    fibers = [
        Fiber([[0., 0., 0.], [4., 0., 0.]], radius=.25),
        Fiber([[5., 6., 1.], [5., 6., 4.]], radius=.6,
              cross_section=CrossSection.RECTANGULAR,
              cross_section_params={'width': .7, 'height': .2}),
    ]
    network = FiberNetwork(fibers=fibers, dimension=3, box_size=np.array([10., 10., 4.]))
    expected = (4 * math.pi * .25**2 + 3 * .7 * .2) / 400.
    assert network.density() == pytest.approx(expected)


def test_planar_density_of_empty_network_or_degenerate_xy_box_is_zero():
    empty = FiberNetwork(dimension=2, box_size=np.array([10., 10., 0.]))
    line = FiberNetwork(fibers=[Fiber([[0., 0., 0.], [1., 0., 0.]])], dimension=2)
    assert empty.density() == line.density() == 0.


def test_planar_density_accumulates_numpy_half_radii_in_native_precision():
    count = 3000
    fibers = [Fiber([[i, 0., 0.], [i + 1, 0., 0.]],
                    radius=np.float16(1.), fiber_id=i)
              for i in range(count)]
    network = FiberNetwork(fibers=fibers, dimension=2, box_size=np.array([count, 10.]))
    assert network.density() == pytest.approx(.2)


def test_planar_density_avoids_overflow_when_a_wide_strip_is_short():
    fiber = Fiber([[0., 0., 0.], [1e-100, 0., 0.]], radius=1e308,
                  cross_section=CrossSection.RECTANGULAR,
                  cross_section_params={'width': .1, 'height': .2})
    network = FiberNetwork(fibers=[fiber], dimension=2, box_size=np.array([1., 1.]))
    assert network.density() == pytest.approx(2e208)


def test_planar_density_retains_a_representable_subnormal_strip():
    radius = math.nextafter(0., 1.)
    if not 0. < radius or radius + radius == 0.:
        pytest.skip('subnormal float arithmetic is unavailable')
    fiber = Fiber([[0., 0., 0.], [.4, 0., 0.]], radius=radius)
    network = FiberNetwork(fibers=[fiber], dimension=2, box_size=np.array([1., 1.]))
    assert network.density() == radius
