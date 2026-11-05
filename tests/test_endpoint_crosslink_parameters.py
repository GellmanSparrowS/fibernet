"""Endpoint data contracts, without graph packages or physical solvers."""
import copy
import unittest

import numpy as np

from fibernet.core.material import Material
from fibernet.core.network import FiberNetwork
from fibernet.core.structure_graph import StructureGraph
from fibernet.gen.unified import _edges_to_network


class EndpointCrosslinkParameterTests(unittest.TestCase):
    def make_fixture(self):
        nodes = {"origin": (0, 0), "east": (2, 0), "north": (0, 3),
                 "coincident": (0, 0), "tiny": (5e-11, 0)}
        edges = [("origin", "coincident"), ("origin", "east"),
                 ("origin", "tiny"), ("east", "origin"),
                 ("origin", "east"), ("north", "origin"), ("east", "east")]
        material = Material()
        net = _edges_to_network(nodes, edges, radius=.23, material=material, dimension=2)
        return net, material

    def signatures(self, net):
        return [(cl.fiber_i, cl.fiber_j, cl.param_i, cl.param_j, tuple(cl.position))
                for cl in net.crosslinks]

    def assert_endpoint_interpolation(self, net):
        for cl in net.crosslinks:
            for fi, param in ((cl.fiber_i, cl.param_i), (cl.fiber_j, cl.param_j)):
                self.assertGreaterEqual(fi, 0)
                self.assertLess(fi, len(net.fibers))
                self.assertIn(param, (0., 1.))
                fiber = net.fibers[fi]
                position = (1. - param) * fiber.start_point + param * fiber.end_point
                np.testing.assert_array_equal(position, cl.position)

    def test_retained_orientation_parallel_edges_and_order(self):
        net, material = self.make_fixture()
        lines = [[[0, 0, 0], [2, 0, 0]], [[2, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [2, 0, 0]], [[0, 3, 0], [0, 0, 0]]]
        self.assertEqual(net.dimension, 2)
        self.assertEqual(len(net.fibers), 4)
        for fiber, line in zip(net.fibers, lines):
            np.testing.assert_array_equal(fiber.centerline, line)
            self.assertEqual(fiber.radius, .23)
            self.assertIs(fiber.material, material)
        origin, east = (0., 0., 0.), (2., 0., 0.)
        self.assertEqual(self.signatures(net), [
            (0, 1, 0., 1., origin), (0, 2, 0., 0., origin),
            (0, 3, 0., 1., origin), (1, 2, 1., 0., origin),
            (1, 3, 1., 1., origin), (2, 3, 0., 1., origin),
            (0, 1, 1., 0., east), (0, 2, 1., 1., east), (1, 2, 0., 1., east)])
        self.assert_endpoint_interpolation(net)

    def test_each_endpoint_orientation(self):
        nodes = {"junction": (0, 0), "left": (-2, 0), "up": (0, 3)}
        for reverse_first in (False, True):
            for reverse_second in (False, True):
                with self.subTest(first=reverse_first, second=reverse_second):
                    first, second = ("junction", "left"), ("junction", "up")
                    if reverse_first:
                        first = first[::-1]
                    if reverse_second:
                        second = second[::-1]
                    net = _edges_to_network(nodes, [first, second], material=Material())
                    self.assertEqual(len(net.crosslinks), 1)
                    cl = net.crosslinks[0]
                    self.assertEqual((cl.param_i, cl.param_j),
                                     (float(reverse_first), float(reverse_second)))
                    self.assert_endpoint_interpolation(net)

    def test_native_dictionary_roundtrip_preserves_endpoint_contract(self):
        net, _ = self.make_fixture()
        restored = FiberNetwork.from_dict(copy.deepcopy(net.to_dict()))
        self.assertEqual(self.signatures(restored), self.signatures(net))
        self.assertEqual(len(restored.fibers), len(net.fibers))
        for before, after in zip(net.fibers, restored.fibers):
            np.testing.assert_array_equal(after.centerline, before.centerline)
            self.assertEqual(after.radius, before.radius)
            self.assertEqual(after.material.to_dict(), before.material.to_dict())
        self.assert_endpoint_interpolation(restored)

    def test_matches_structure_graph_endpoint_convention(self):
        net, material = self.make_fixture()
        graph = StructureGraph(dimension=2)
        origin = graph.add_node([0, 0, 0])
        east = graph.add_node([2, 0, 0])
        north = graph.add_node([0, 3, 0])
        for u, v in [(origin, east), (east, origin), (origin, east), (north, origin)]:
            graph.add_edge(u, v, radius=.23, material=material, allow_parallel=True)
        before = copy.deepcopy(graph.to_dict())
        projected = graph.to_fiber_network()
        self.assertEqual(self.signatures(projected), self.signatures(net))
        self.assert_endpoint_interpolation(projected)
        self.assertEqual(graph.to_dict(), before)

    def test_approximate_endpoint_import_ignores_crosslink_parameters(self):
        net, _ = self.make_fixture()
        graph = StructureGraph.from_fiber_network(net)
        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(graph.num_edges, 4)
        expected = graph.to_dict()
        historical = copy.deepcopy(net)
        for cl in historical.crosslinks:
            cl.param_i = cl.param_j = .5
        self.assertEqual(StructureGraph.from_fiber_network(historical).to_dict(), expected)
        historical.crosslinks.clear()
        self.assertEqual(StructureGraph.from_fiber_network(historical).to_dict(), expected)


if __name__ == "__main__":
    unittest.main()
