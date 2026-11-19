"""Legacy network diagnostics use finite geometry and actual fiber references."""

from enum import IntEnum

import numpy as np
import pytest

from fibernet import Crosslink, Fiber, FiberNetwork
from fibernet.gen import ZigZagGenerator


def network_with_fibers(count=3):
    return FiberNetwork(
        fibers=[Fiber([[0., i, 0.], [i + 1., i, 0.]],
                      radius=(i + 1) * .125, fiber_id=70 - i)
                for i in range(count)],
        dimension=2,
    )


def contact(first, second):
    return Crosslink(first, second, .4, .6, np.array([10., 20., 0.]),
                     crosslink_type='contact')


def test_empty_and_ordinary_reports_keep_existing_fields_and_warnings():
    assert FiberNetwork().validate() == {
        'valid': False, 'errors': ['Network has no fibers'], 'warnings': [],
        'stats': {'num_fibers': 0, 'num_crosslinks': 0, 'dimension': 3},
    }
    network = network_with_fibers()
    stats = {'num_fibers': 3, 'num_crosslinks': 0, 'dimension': 2,
             'mean_length': 2., 'total_length': 6., 'mean_radius': .25}
    assert network.validate() == {
        'valid': True, 'errors': [], 'stats': stats,
        'warnings': ['No crosslinks - fibers may not be mechanically connected'],
    }
    network.crosslinks.append(contact(np.int64(0), np.uint32(1)))
    assert network.validate() == {
        'valid': True, 'errors': [],
        'stats': {**stats, 'num_crosslinks': 1, 'num_components': 2},
        'warnings': ['Network has 2 disconnected components'],
    }
    network.fibers[0].radius = 0.
    assert network.validate()['warnings'] == [
        'Fiber 0 has non-positive radius: 0.0',
        'Network has 2 disconnected components',
    ]
    assert network.validate()['valid']


@pytest.mark.parametrize('coordinate', [np.inf, -np.inf])
def test_infinite_coordinates_give_diagnostics_without_length_evaluation(coordinate):
    network = network_with_fibers(2)
    network.fibers[0].centerline[:, 0] = coordinate
    with np.errstate(all='raise'):
        report = network.validate()
    assert not report['valid']
    assert report['errors'] == ['1 fiber(s) have infinite values in centerline']
    assert np.isnan(report['stats']['mean_length'])
    assert np.isnan(report['stats']['total_length'])


def test_nan_and_zero_length_diagnostics_remain_distinct():
    network = FiberNetwork([
        Fiber([[0., 0., 0.], [np.nan, 0., 0.]]),
        Fiber([[0., 0., 0.], [5e-13, 0., 0.]]),
        Fiber([[0., 1., 0.], [1e-12, 1., 0.]]),
    ])
    with np.errstate(all='raise'):
        report = network.validate()
    assert report['errors'] == [
        '1 fiber(s) have NaN values in centerline',
        '1 fiber(s) have zero length',
    ]
    assert not report['valid']
    assert np.isnan(report['stats']['total_length'])


@pytest.mark.parametrize('coordinate, diagnostic', [
    (np.nan, 'NaN'), (np.inf, 'infinite'), (-np.inf, 'infinite'),
])
def test_nonfinite_single_sample_retains_the_zero_length_diagnostic(coordinate, diagnostic):
    network = FiberNetwork([Fiber([[coordinate, 0., 0.]])])
    with np.errstate(all='raise'):
        report = network.validate()
    assert not report['valid']
    assert report['errors'] == [
        f'1 fiber(s) have {diagnostic} values in centerline',
        '1 fiber(s) have zero length',
    ]
    assert np.isnan(report['stats']['mean_length'])
    assert np.isnan(report['stats']['total_length'])


@pytest.mark.parametrize('reference', [
    .5, 1., 'bad', None, True, np.bool_(False), np.nan,
    np.array([1]), np.array(1), np.timedelta64(1, 'ns'),
    -1, 2, np.uint64(2**64 - 1),
])
def test_bad_fiber_reference_returns_diagnostic(reference):
    network = network_with_fibers(2)
    network.crosslinks.append(contact(0, reference))
    report = network.validate()
    assert not report['valid']
    assert report['errors'] == ['1 crosslink(s) reference invalid fiber IDs']
    assert report['stats']['num_crosslinks'] == 1
    assert report['stats']['num_components'] == 2


@pytest.mark.parametrize('ghost', [.5, 12])
def test_invalid_shared_reference_cannot_join_other_components(ghost):
    network = network_with_fibers()
    active = contact(0, 1)
    invalid = contact(1, ghost)
    network.crosslinks.extend([active, invalid, contact(ghost, 2), invalid, contact(-1, 100)])
    before = network.to_dict()
    report = network.validate()
    assert not report['valid']
    assert report['errors'] == ['4 crosslink(s) reference invalid fiber IDs']
    assert report['stats']['num_crosslinks'] == 5
    assert report['stats']['num_components'] == 2
    assert network.to_dict() == before
    assert network.crosslinks[1] is network.crosslinks[3] is invalid


def test_integer_index_protocol_keeps_valid_integer_subclasses():
    class FiberRow(IntEnum):
        SECOND = 1

        def __int__(self):
            return 0

    class NumpyRow(np.int64):
        def __int__(self):
            return 0

    for index in (FiberRow.SECOND, NumpyRow(1)):
        network = network_with_fibers()
        network.crosslinks.extend([contact(index, 2), contact(1, 2)])
        report = network.validate()
        assert report['valid']
        assert report['stats']['num_components'] == 2


def test_generated_zigzag_valid_report_matches_independent_geometry_counts():
    network = ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network()
    report = network.validate()
    assert report['valid'] and report['errors'] == report['warnings'] == []
    stats = report['stats']
    assert (stats['num_fibers'], stats['num_crosslinks'], stats['num_components']) == (45, 90, 1)
    lengths = [np.linalg.norm(fiber.end_point - fiber.start_point) for fiber in network.fibers]
    np.testing.assert_allclose([stats['mean_length'], stats['total_length']],
                               [np.mean(lengths), np.sum(lengths)])
