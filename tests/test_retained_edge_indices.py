"""Retained-edge indexing and orientation-correct endpoint parameters."""
import unittest

import numpy as np

from fibernet.core.material import Material
from fibernet.gen.unified import _edges_to_network


class RetainedEdgeIndexTests(unittest.TestCase):
    def make_network(self, nodes, edges):
        material = Material()
        net = _edges_to_network(nodes, edges, radius=.23, material=material, dimension=2)
        self.assertEqual(net.dimension, 2)
        for fiber in net.fibers:
            self.assertEqual(fiber.radius, .23)
            self.assertIs(fiber.material, material)
        return net

    def assert_geometry(self, net, expected):
        self.assertEqual(len(net.fibers), len(expected))
        for fiber, centerline in zip(net.fibers, expected):
            np.testing.assert_array_equal(fiber.centerline, centerline)

    def assert_crosslinks(self, net, expected):
        self.assertEqual(len(net.crosslinks), len(expected))
        for link, (i, j, position, pi, pj) in zip(net.crosslinks, expected):
            self.assertEqual((link.fiber_i, link.fiber_j), (i, j))
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, j)
            self.assertLess(j, len(net.fibers))
            np.testing.assert_array_equal(link.position, position)
            self.assertEqual(link.param_i, pi)
            self.assertEqual(link.param_j, pj)

    def test_all_dropped(self):
        net = self.make_network({0: (0, 0), 1: (0, 0), 2: (5e-11, 0)},
                                [(0, 1), (0, 2), (0, 0)])
        self.assert_geometry(net, [])
        self.assert_crosslinks(net, [])

    def test_mixed_dropped_and_retained(self):
        net = self.make_network(
            {0: (0, 0), 1: (2, 0), 2: (0, 3), 3: (2, 4), 4: (0, 0)},
            [(4, 4), (0, 1), (0, 4), (0, 2), (1, 3)])
        self.assert_geometry(net, [[[0, 0, 0], [2, 0, 0]],
                                  [[0, 0, 0], [0, 3, 0]],
                                  [[2, 0, 0], [2, 4, 0]]])
        self.assert_crosslinks(net, [(0, 1, [0, 0, 0], 0., 0.), (0, 2, [2, 0, 0], 1., 0.)])

    def test_ordinary_no_drop(self):
        net = self.make_network({0: (0, 0), 1: (2, 0), 2: (0, 3), 3: (-4, 0)},
                                [(0, 1), (0, 2), (0, 3)])
        self.assert_geometry(net, [[[0, 0, 0], [2, 0, 0]],
                                  [[0, 0, 0], [0, 3, 0]],
                                  [[0, 0, 0], [-4, 0, 0]]])
        self.assert_crosslinks(net, [(0, 1, [0, 0, 0], 0., 0.),
                                    (0, 2, [0, 0, 0], 0., 0.), (1, 2, [0, 0, 0], 0., 0.)])

    def test_reversed_endpoints_with_dropped_prefix(self):
        net = self.make_network({0: (0, 0), 1: (2, 0), 2: (0, 3), 3: (0, 0)},
                                [(0, 3), (1, 0), (2, 0)])
        self.assert_geometry(net, [[[2, 0, 0], [0, 0, 0]], [[0, 3, 0], [0, 0, 0]]])
        self.assert_crosslinks(net, [(0, 1, [0, 0, 0], 1., 1.)])

    def test_exact_threshold_is_retained(self):
        net = self.make_network({0: (0, 0), 1: (1e-10, 0), 2: (0, 2)},
                                [(0, 1), (0, 2)])
        self.assert_geometry(net, [[[0, 0, 0], [1e-10, 0, 0]],
                                  [[0, 0, 0], [0, 2, 0]]])
        self.assert_crosslinks(net, [(0, 1, [0, 0, 0], 0., 0.)])


if __name__ == "__main__":
    unittest.main()
