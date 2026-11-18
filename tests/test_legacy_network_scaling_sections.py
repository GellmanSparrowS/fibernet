"""Keep explicit cross-section dimensions consistent with legacy geometric scaling."""

from copy import deepcopy
import math

import numpy as np
import pytest

from fibernet import Crosslink, Fiber, FiberNetwork, Material, api
from fibernet.core.fiber import CrossSection
from fibernet.core.transform import scale


@pytest.mark.parametrize('factor', [.5, 2.])
def test_public_scale_preserves_mixed_section_area_and_volume_ratios(factor):
    material = Material(name='fixture', extra={'label': ['retain']})
    profiles = [
        ([[0., 0., 0.], [.4, .3, 0.], [1., 0., 0.]], .25, CrossSection.RECTANGULAR,
         {'width': .3, 'height': .2, 'annotation': {'values': [4, 5]}}, 31),
        ([[1., 0., 0.], [1., 1., 0.]], .25, CrossSection.ELLIPTICAL,
         {'semi_minor': .1}, 9),
        ([[1., 1., 0.], [1., 1., 2.]], .2, CrossSection.HOLLOW_CIRCULAR,
         {'inner_radius': .15}, 17),
        ([[1., 1., 2.], [0., 1., 2.]], .07, CrossSection.CIRCULAR,
         {'manufacturing_pass': 2}, 47),
    ]
    fibers = [Fiber(points, radius=radius, material=material, cross_section=shape,
                    cross_section_params=params, fiber_id=label)
              for points, radius, shape, params, label in profiles]
    links = [Crosslink(i, i + 1, 1., 0., fiber.end_point.copy(), crosslink_type='bonded',
                       strength=3. + i, stiffness=7. + i)
             for i, fiber in enumerate(fibers[:-1])]
    source = FiberNetwork(fibers=fibers, crosslinks=links, box_size=np.array([2., 3., 4.]),
                          metadata={'name': 'mixed'})
    before = deepcopy(source.to_dict())
    scaled = api.scale(source, factor=factor)

    sections = [
        .3 * .2, math.pi * .25 * .1, math.pi * (.2**2 - .15**2), math.pi * .07**2,
    ]
    lengths = [math.fsum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))
               for points, *_ in profiles]
    expected_volume = math.fsum(area * length for area, length in zip(sections, lengths))
    origin = np.array([.5, .5, 1.])
    np.testing.assert_allclose([fiber.cross_section_area for fiber in scaled.fibers],
                               np.asarray(sections) * factor**2, rtol=1e-14)
    assert scaled.fibers[2].cross_section_area > 0.
    assert scaled.total_volume == pytest.approx(expected_volume * factor**3)
    assert scaled.density() == pytest.approx(expected_volume / 24.)
    assert scaled.fibers[0].second_moment_area == pytest.approx(
        (.3 * .2**3 / 12 * factor**4, .2 * .3**3 / 12 * factor**4))
    assert scaled.fibers[0].cross_section_params == {
        'width': .3 * factor, 'height': .2 * factor, 'annotation': {'values': [4, 5]},
    }
    assert scaled.fibers[1].cross_section_params == {'semi_minor': .1 * factor}
    assert scaled.fibers[2].cross_section_params == {'inner_radius': .15 * factor}
    assert scaled.fibers[3].cross_section_params == {'manufacturing_pass': 2}
    for reference, result in zip(fibers, scaled.fibers):
        np.testing.assert_allclose(result.centerline, origin + factor * (reference.centerline - origin))
        assert result.radius == reference.radius * factor
        assert result.fiber_id == reference.fiber_id
        assert result.material.to_dict() == material.to_dict()
    for reference, result in zip(links, scaled.crosslinks):
        np.testing.assert_allclose(result.position, origin + factor * (reference.position - origin))
        assert (result.fiber_i, result.fiber_j, result.param_i, result.param_j,
                result.crosslink_type, result.strength, result.stiffness) == (
                    reference.fiber_i, reference.fiber_j, reference.param_i, reference.param_j,
                    reference.crosslink_type, reference.strength, reference.stiffness)
    np.testing.assert_array_equal(scaled.box_size, source.box_size * factor)
    assert scaled.metadata == source.metadata
    scaled.fibers[0].cross_section_params['annotation']['values'].append(6)
    assert source.to_dict() == before


def test_vector_inplace_scale_handles_shared_dimension_mappings_once_per_fiber():
    params = {'width': .3, 'height': .2, 'semi_minor': .08, 'tag': ['shared']}
    before_params = deepcopy(params)
    fibers = [Fiber([[0., 0., 0.], [1., 1., 1.]], radius=.25, cross_section=shape,
                    cross_section_params=params) for shape in
              (CrossSection.RECTANGULAR, CrossSection.RECTANGULAR,
               CrossSection.ELLIPTICAL, CrossSection.CUSTOM)]
    source = FiberNetwork(fibers=fibers, box_size=np.array([2., 3., 4.]))
    factor = np.array([2., 4., 6.])
    result = scale(source, factor=factor, origin=np.zeros(3), inplace=True)
    assert result is source
    for current, original in zip(result.fibers, fibers):
        assert current is original
        assert current.radius == 1.
        np.testing.assert_array_equal(current.centerline[-1], factor)
    rectangle = dict(params, width=1.2, height=.8)
    assert result.fibers[0].cross_section_params == result.fibers[1].cross_section_params == rectangle
    assert result.fibers[2].cross_section_params == dict(params, semi_minor=.32)
    assert result.fibers[3].cross_section_params == params == before_params
    np.testing.assert_array_equal(result.box_size, [4., 12., 24.])


@pytest.mark.parametrize('shape', [
    CrossSection.CIRCULAR, CrossSection.RECTANGULAR,
    CrossSection.ELLIPTICAL, CrossSection.HOLLOW_CIRCULAR,
])
def test_uniform_scale_keeps_implicit_section_dimensions_implicit(shape):
    source = FiberNetwork(fibers=[
        Fiber([[0., 0., 0.], [1., 1., 1.]], radius=.25, cross_section=shape),
    ])
    result = api.scale(source, factor=2.)
    assert result.fibers[0].cross_section_params == {}
    assert result.fibers[0].cross_section_area == pytest.approx(
        4 * source.fibers[0].cross_section_area)


@pytest.mark.parametrize('factor', [
    np.float16(2.),
    np.array(2., dtype=np.float16),
    np.array([2., 2., 2.], dtype=np.float16),
], ids=['scalar', 'zero_dimensional', 'vector'])
def test_half_precision_scale_factor_keeps_large_sections_representable(factor):
    fibers = [
        Fiber([[0., 0., 0.], [2., 0., 0.]], radius=1., cross_section=CrossSection.RECTANGULAR,
              cross_section_params={'width': 256., 'height': 256.}),
        Fiber([[0., 1., 0.], [2., 1., 0.]], radius=40000., cross_section=CrossSection.ELLIPTICAL,
              cross_section_params={'semi_minor': 16.}),
        Fiber([[0., 2., 0.], [2., 2., 0.]], radius=256., cross_section=CrossSection.HOLLOW_CIRCULAR,
              cross_section_params={'inner_radius': 192.}),
    ]
    result = scale(FiberNetwork(fibers=fibers), factor=factor, origin=np.zeros(3))
    areas = [262144., math.pi * 80000. * 32., math.pi * (512.**2 - 384.**2)]
    np.testing.assert_allclose([fiber.cross_section_area for fiber in result.fibers],
                               areas, rtol=1e-14)
    assert result.total_volume == pytest.approx(4 * math.fsum(areas))


def test_half_precision_identity_scale_preserves_a_thin_wall():
    fiber = Fiber([[0., 0., 0.], [1., 0., 0.]], radius=.2, cross_section=CrossSection.HOLLOW_CIRCULAR,
                  cross_section_params={'inner_radius': .19999})
    result = api.scale(FiberNetwork(fibers=[fiber]), factor=np.float16(1.)).fibers[0]
    assert result.radius == .2 and result.cross_section_params['inner_radius'] == .19999
    assert result.cross_section_area == pytest.approx(math.pi * (.2**2 - .19999**2))
    assert result.cross_section_area > 0.


def test_section_scaling_preserves_extended_precision():
    extended = np.nextafter(np.longdouble(1.), np.longdouble(2.))
    fiber = Fiber([[0., 0., 0.], [1., 0., 0.]], radius=extended, cross_section=CrossSection.ELLIPTICAL,
                  cross_section_params={'semi_minor': extended})
    result = api.scale(FiberNetwork(fibers=[fiber]), factor=np.longdouble(2.)).fibers[0]
    assert isinstance(result.radius, np.longdouble)
    assert isinstance(result.cross_section_params['semi_minor'], np.longdouble)
    assert result.radius == result.cross_section_params['semi_minor'] == extended * np.longdouble(2.)
