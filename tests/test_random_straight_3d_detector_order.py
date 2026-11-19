"""Contact ordering and geometric boundaries of the 3D proximity detector."""
import numpy as np
import pytest

from fibernet.core.fiber import Fiber
from fibernet.gen import disordered


def _fiber(points):
    return Fiber(centerline=np.array(points, dtype=np.float64), radius=.25)


@pytest.mark.parametrize('distance, expected', [
    (np.nextafter(1., 0.), 1),
    (1., 0),
    (np.nextafter(1., np.inf), 0),
])
@pytest.mark.parametrize('adjacent', [False, True])
def test_strict_threshold_at_float_boundary(distance, expected, adjacent):
    a = _fiber([[0, 4.75, 0], [0, 5., 0]])
    low, high = (4.875, 5.375) if adjacent else (4.75, 5.)
    b = _fiber([[distance, low, 0], [distance, high, 0]])
    links = disordered._detect_intersections_3d([a, b], threshold_factor=2.)
    assert len(links) == expected
    if links:
        assert links[0][:2] == (0, 1)
        assert links[0][4][0] == distance / 2


def test_same_cell_retains_all_records_in_segment_pair_order():
    a = _fiber([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    b = _fiber([[.25, .25, 0], [.75, .25, 0], [1.75, .25, 0]])
    links = disordered._detect_intersections_3d([a, b])
    assert [record[:4] for record in links] == [
        (0, 1, .125, 0.),
        (0, 1, .375, .5),
        (0, 1, .5, .5),
        (0, 1, .5, .625),
    ]
    assert all(type(i) is int and type(j) is int for i, j, *_ in links)
    np.testing.assert_array_equal([record[4] for record in links], [
        [.25, .125, 0.], [.75, .125, 0.], [.875, .125, 0.], [1., .125, 0.],
    ])


def test_adjacent_cell_chooses_closest_of_multiple_segment_pairs():
    a = _fiber([[4.75, 0, 0], [4.75, .25, 0], [4.75, .5, 0]])
    b = _fiber([[5.25, 0, 0], [5.25, .25, 0], [5.125, .5, 0]])
    record, = disordered._detect_intersections_3d([a, b])
    assert record[:4] == (0, 1, 1., 1.)
    np.testing.assert_array_equal(record[4], [4.9375, .5, 0.])



@pytest.mark.parametrize('endpoints, b_param', [((.5, 1.5), 0.), ((1.5, .5), 1.)])
def test_collinear_nonzero_overlap(endpoints, b_param):
    a = _fiber([[0, 0, 0], [2, 0, 0]])
    b = _fiber([[x, 0, 0] for x in endpoints])
    record, = disordered._detect_intersections_3d([a, b])
    assert record[:4] == (0, 1, .25, b_param)
    np.testing.assert_array_equal(record[4], [.5, 0., 0.])


def test_repeated_samples_keep_point_pair_result():
    a = _fiber([[0, 0, 0], [0, 0, 0]])
    b = _fiber([[.25, 0, 0], [.25, 0, 0]])
    record, = disordered._detect_intersections_3d([a, b])
    assert record[:4] == (0, 1, 0., 0.)
    np.testing.assert_array_equal(record[4], [.125, 0., 0.])


def test_default_repair_retains_order_and_rng_consumption():
    options = dict(num_fibers=72, box_size=40., seed=72)
    prior_state = np.random.get_state()
    try:
        np.random.seed(8)
        raw = disordered.random_straight_3d(**options, ensure_connected=False)
        expected = disordered._ensure_connected(raw, max_gap_factor=2.)
        expected_rng = np.random.bytes(32)

        np.random.seed(8)
        actual = disordered.random_straight_3d(**options)
        assert actual.num_fibers > options['num_fibers']
        assert actual.to_dict() == expected.to_dict()
        assert np.random.bytes(32) == expected_rng
    finally:
        np.random.set_state(prior_state)
