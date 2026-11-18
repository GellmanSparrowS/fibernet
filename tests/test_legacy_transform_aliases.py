"""Legacy affine transforms apply one operation to every physical entity."""

from copy import deepcopy

import numpy as np
import pytest

from fibernet import Crosslink, Fiber, FiberNetwork, Material
from fibernet.core.fiber import CrossSection
from fibernet.core.transform import mirror, rotate, scale, translate


def _network(with_aliases):
    points = np.array([[0., 1., 2.], [1., 2., 3.], [2., 4., 3.],
                       [3., 5., 8.], [4., 6., 9.]])
    common = points[:3]
    positions = [common, common, points[1:4], common.copy(), common]
    material = Material(name='fixture', extra={'label': ['keep']})
    fibers = [Fiber(line if with_aliases else line.copy(), radius=.3, material=material,
                    fiber_id=7, cross_section=CrossSection.RECTANGULAR,
                    cross_section_params={'width': .2, 'height': .4, 'label': ['section']})
              for line in positions]
    if with_aliases:
        fibers.append(fibers[0])
    control = points[1]
    link_points = [control, control, points[2], control.copy()]
    links = [Crosslink(i, (i + 1) % len(fibers), .25, .75,
                       point if with_aliases else point.copy(), crosslink_type='bonded',
                       strength=1. + i, stiffness=2. + i)
             for i, point in enumerate(link_points)]
    if with_aliases:
        links.append(links[0])
    return FiberNetwork(fibers=fibers, crosslinks=links, periodic=True,
                        box_size=np.array([10., 20., 30.]),
                        metadata={'name': 'fixture'}), points, control


@pytest.mark.parametrize('operation', ['mirror', 'rotate', 'scale', 'translate'])
@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('with_aliases', [False, True])
def test_legacy_affine_transforms_use_frozen_geometry_once(operation, inplace, with_aliases):
    source, storage, control = _network(with_aliases)
    if with_aliases:
        assert source.fibers[0].centerline is source.fibers[1].centerline
        assert np.shares_memory(source.fibers[0].centerline, source.fibers[2].centerline)
        assert source.crosslinks[0].position is source.crosslinks[1].position is control
        assert np.shares_memory(source.fibers[0].centerline, source.crosslinks[0].position)

    fibers, links = source.fibers[:], source.crosslinks[:]
    before = deepcopy(source.to_dict())
    storage_before = storage.copy()
    fiber_points = [fiber.centerline.copy() for fiber in fibers]
    link_points = [link.position.copy() for link in links]
    all_points = np.vstack(fiber_points)
    origin = control.copy() if with_aliases else (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    origin_argument = control if with_aliases else None
    shift = control.copy()
    factors = np.array([2., .5, 1.5])
    radius_factor = 4. / 3 if operation == 'scale' else 1.

    def oracle(points):
        relative = points - origin
        if operation == 'mirror':
            return origin + relative * [-1., 1., 1.]
        if operation == 'rotate':
            return origin + np.stack((-relative[..., 1], relative[..., 0], relative[..., 2]), axis=-1)
        if operation == 'scale':
            return origin + relative * factors
        return points + shift

    if operation == 'mirror':
        result = mirror(source, axis='x', origin=origin_argument, inplace=inplace)
    elif operation == 'rotate':
        result = rotate(source, np.pi / 2, np.array([0., 0., 1.]), origin=origin_argument, inplace=inplace)
    elif operation == 'scale':
        result = scale(source, factors, origin=origin_argument, inplace=inplace)
    else:
        result = translate(source, control, inplace=inplace)

    assert (result is source) == inplace
    assert len(result.fibers) == len(fibers) and len(result.crosslinks) == len(links)
    assert (result.fibers[0] is result.fibers[-1]) == with_aliases
    assert (result.crosslinks[0] is result.crosslinks[-1]) == with_aliases
    assert len({id(fiber) for fiber in result.fibers}) == 5
    assert len({id(link) for link in result.crosslinks}) == 4
    for fiber, original, points in zip(result.fibers, fibers, fiber_points):
        assert (fiber is original) == inplace
        assert (fiber.material is original.material) == inplace
        assert fiber.material.extra == {'label': ['keep']}
        assert fiber.fiber_id == 7
        assert fiber.radius == pytest.approx(.3 * radius_factor)
        assert fiber.cross_section_params == {
            'width': .2 * radius_factor, 'height': .4 * radius_factor, 'label': ['section'],
        }
        np.testing.assert_allclose(fiber.centerline, oracle(points), atol=1e-14, rtol=0.)
    for link, original, points in zip(result.crosslinks, links, link_points):
        assert (link is original) == inplace
        assert (link.fiber_i, link.fiber_j, link.param_i, link.param_j, link.crosslink_type,
                link.strength, link.stiffness) == (
                    original.fiber_i, original.fiber_j, original.param_i, original.param_j,
                    original.crosslink_type, original.strength, original.stiffness)
        np.testing.assert_allclose(link.position, oracle(points), atol=1e-14, rtol=0.)
    expected_box = np.array([10., 20., 30.])
    if operation == 'scale':
        expected_box *= factors
    np.testing.assert_array_equal(result.box_size, expected_box)
    assert result.metadata == {'name': 'fixture'} and result.periodic
    np.testing.assert_array_equal(storage, storage_before)
    np.testing.assert_array_equal(control, shift)
    if not inplace:
        assert source.to_dict() == before


def test_mirror_and_translate_retain_legacy_crosslink_container_and_dtype():
    list_link = Crosslink(0, 1, 0., 0., [1., 2., 3.])
    integer_link = Crosslink(0, 1, 0., 0., np.array([1, 2, 3], dtype=np.int32))
    network = FiberNetwork(crosslinks=[list_link, integer_link])
    mirrored = mirror(network, axis=0, origin=np.array([.25, 0., 0.]))
    assert isinstance(mirrored.crosslinks[0].position, list)
    assert mirrored.crosslinks[0].position == [-.5, 2., 3.]
    assert mirrored.crosslinks[1].position.dtype == np.int32
    np.testing.assert_array_equal(mirrored.crosslinks[1].position, [0, 2, 3])

    translated = translate(network, np.array([1, 2, 3]))
    np.testing.assert_array_equal(translated.crosslinks[0].position, [2., 4., 6.])
    assert translated.crosslinks[1].position.dtype == np.int32
    np.testing.assert_array_equal(translated.crosslinks[1].position, [2, 4, 6])
    with pytest.raises(TypeError, match='cast'):
        translate(network, np.array([.1, .2, .3]))


@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('dtype', [np.float16, np.float64])
def test_scale_preserves_shared_radius_views_and_crosslink_coordinates(inplace, dtype):
    storage = np.array([[2., 3., 4.], [5., 6., 7.]], dtype=dtype)
    control = storage[0]
    radius = storage[0, :1].reshape(())
    other_view = storage[0, :1].reshape(())
    assert other_view is not radius and np.shares_memory(other_view, radius)
    fibers = [
        Fiber(storage, radius=radius),
        Fiber([[1., 0., 0.], [2., 1., 1.]], radius=radius),
        Fiber([[0., 1., 0.], [2., 2., 0.]], radius=other_view),
    ]
    link = Crosslink(0, 1, 0., 0., control)
    network = FiberNetwork(fibers=fibers, crosslinks=[link])
    before = storage.copy()
    result = scale(network, factor=control, origin=np.zeros(3), inplace=inplace)

    assert (result is network) == inplace
    for original, scaled in zip(fibers, result.fibers):
        assert (original is scaled) == inplace
        assert scaled.radius.shape == () and scaled.radius.dtype == dtype
        assert scaled.radius.item() == 6.
    np.testing.assert_array_equal(result.crosslinks[0].position, [4., 9., 16.])
    np.testing.assert_array_equal(result.fibers[0].centerline, [[4., 9., 16.], [10., 18., 28.]])
    np.testing.assert_array_equal(storage, before)
    assert radius.item() == other_view.item() == 2.


def test_scale_retains_legacy_integer_radius_casting_refusal():
    fiber = Fiber([[0., 0., 0.], [1., 0., 0.]], radius=np.array(2, dtype=np.int16))
    network = FiberNetwork(fibers=[fiber])
    with pytest.raises(TypeError, match='cast'):
        scale(network, factor=2.)
    assert fiber.radius.dtype == np.int16 and fiber.radius.item() == 2


@pytest.mark.parametrize('operation', ['mirror', 'translate'])
def test_inplace_transform_supports_crosslink_memoryview_positions(operation):
    storage = np.array([1., 2., 3.])
    position = memoryview(storage)
    links = [Crosslink(0, 1, 0., 0., position), Crosslink(1, 0, 0., 0., position)]
    network = FiberNetwork(crosslinks=links)
    if operation == 'mirror':
        result = mirror(network, axis=0, origin=np.zeros(3), inplace=True)
        expected, position_type = [-1., 2., 3.], memoryview
    else:
        result = translate(network, np.ones(3), inplace=True)
        expected, position_type = [2., 3., 4.], np.ndarray

    assert result is network and result.crosslinks is links
    for link in links:
        assert isinstance(link.position, position_type)
        assert np.asarray(link.position).dtype == storage.dtype
        np.testing.assert_array_equal(link.position, expected)
        assert not np.shares_memory(link.position, storage)
    np.testing.assert_array_equal(storage, [1., 2., 3.])
