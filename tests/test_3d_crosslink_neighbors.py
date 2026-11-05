"""Recover nearby segment pairs missed at midpoint-cell boundaries."""
import unittest
from unittest.mock import patch
import numpy as np
from fibernet.core.fiber import Fiber
from fibernet.gen.disordered import _detect_intersections_3d, random_straight_3d


def fiber(points, radius=0.1):
    return Fiber(centerline=np.array(points, dtype=float), radius=radius)


class Test3DCrosslinkNeighbors(unittest.TestCase):
    def test_adjacent_cells_near_and_far(self):
        a = fiber([[4.9, 0, 0], [4.9, 1, 0]])
        near = fiber([[5.1, 0, 0], [5.1, 1, 0]])
        result = _detect_intersections_3d([a, near], 2)
        self.assertEqual(len(result), 1)
        i, j, pi, pj, point = result[0]
        self.assertEqual((i, j), (0, 1))
        self.assertTrue(0 <= pi <= 1 and 0 <= pj <= 1)
        np.testing.assert_allclose(point, [5.0, 0.0, 0.0], atol=1e-12)
        far = fiber([[6, 0, 0], [6, 1, 0]])
        self.assertEqual(_detect_intersections_3d([a, far], 2), [])

    def test_later_same_cell_contact_after_early_miss(self):
        a = fiber([[0, 0, 0], [1, 0, 0], [6, 0, 0], [7, 0, 0]])
        b = fiber([[0, 3, 0], [1, 3, 0], [6, 2, 0], [7, 0.1, 0]])
        result = _detect_intersections_3d([a, b], 2)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0][4], [7, 0.05, 0], atol=1e-12)
        np.testing.assert_allclose(result[0][2:4], [1, 1], atol=1e-12)

    def test_existing_same_cell_result_is_not_duplicated(self):
        a = fiber([[0, 0, 0], [1, 0, 0]])
        b = fiber([[0, 0.2, 0], [1, 0.2, 0]])
        result = _detect_intersections_3d([a, b], 2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][:4], (0, 1, 0, 0))
        np.testing.assert_allclose(result[0][4], [0, 0.1, 0], atol=1e-12)

    def test_generator_returns_recovered_crosslink(self):
        class FixedRandom:
            def __init__(self):
                self.centers = iter(([4.9, 0.5, 0], [5.1, 0.5, 0]))
            def uniform(self, low, high, size=None):
                if size == 3:
                    return np.array(next(self.centers))
                return np.pi / 2 if low == 0 else 0.0
        with patch('fibernet.gen.disordered.np.random.RandomState',
                   return_value=FixedRandom()):
            net = random_straight_3d(num_fibers=2, fiber_length=1, box_size=10,
                                     radius=0.1, threshold_factor=2,
                                     ensure_connected=False)
        self.assertEqual(net.num_fibers, 2)
        self.assertEqual(net.num_crosslinks, 1)
        self.assertEqual((net.crosslinks[0].fiber_i, net.crosslinks[0].fiber_j), (0, 1))
        self.assertAlmostEqual(net.crosslinks[0].position[0], 5.0)


if __name__ == '__main__':
    unittest.main()
