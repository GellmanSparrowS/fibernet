"""Finite floating radius admission preserves graph state and compatibility."""
import copy
import unittest

import numpy as np

from fibernet.core.structure_graph import StructureGraph


class TestFiniteRadius(unittest.TestCase):
    def graph(self):
        graph = StructureGraph()
        a = graph.add_node([0, 0])
        b = graph.add_node([1, 0])
        return graph, a, b

    def test_nonfinite_floats_rejected_without_mutation(self):
        for scalar in (float, np.float16, np.float32, np.float64):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(scalar=scalar, value=value):
                    graph, a, b = self.graph()
                    before = copy.deepcopy(graph.to_dict())
                    edge_set = graph._edge_set.copy()
                    with self.assertRaisesRegex(ValueError, 'radius must be finite'):
                        graph.add_edge(a, b, radius=scalar(value))
                    self.assertEqual(graph.to_dict(), before)
                    self.assertEqual(graph._edge_set, edge_set)
                    self.assertEqual(graph.add_edge(a, b), 0)

    def test_from_dict_rejects_without_mutating_input(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            graph, a, b = self.graph()
            edge = graph.add_edge(a, b)
            data = graph.to_dict()
            data['edges'][0]['radius'] = value
            before = copy.deepcopy(data)
            original_edge = data['edges'][0]
            with self.assertRaisesRegex(ValueError, 'radius must be finite'):
                StructureGraph.from_dict(data)
            self.assertIs(data['edges'][0], original_edge)
            self.assertIs(data['edges'][0]['radius'], value)
            # Remove the intentionally non-reflexive NaN before deep comparison.
            actual = copy.deepcopy(data)
            actual['edges'][0]['radius'] = 0
            before['edges'][0]['radius'] = 0
            self.assertEqual(actual, before)
            self.assertEqual(graph.edges[edge].radius, 0.1)

    def test_finite_scalar_identity_is_preserved(self):
        class Uncoercible:
            def __float__(self):
                raise AssertionError('unexpected coercion')

            def __eq__(self, other):
                raise AssertionError('unexpected comparison')

        for radius in (0.25, np.float16(0.25), np.float32(0.25),
                       np.float64(0.25), Uncoercible()):
            graph, a, b = self.graph()
            edge = graph.add_edge(a, b, radius=radius)
            self.assertIs(graph.edges[edge].radius, radius)

    def test_finite_default_zero_negative_values_are_preserved(self):
        graph, a, b = self.graph()
        self.assertEqual(graph.edges[graph.add_edge(a, b)].radius, 0.1)
        for radius in (0.0, -0.0, -0.2):
            edge = graph.add_edge(a, b, radius=radius, allow_parallel=True)
            self.assertIs(graph.edges[edge].radius, radius)
            self.assertEqual(bool(np.signbit(graph.edges[edge].radius)),
                             bool(np.signbit(radius)))

    def test_parallel_curved_edges_preserve_geometry_and_ids(self):
        graph, a, b = self.graph()
        first = graph.add_edge(a, b, radius=0.2)
        points = np.array([[0.5, 0.25, 0.0]])
        second = graph.add_edge(a, b, radius=0.3, internal_points=points,
                                allow_parallel=True)
        self.assertEqual((first, second), (0, 1))
        np.testing.assert_array_equal(graph.edges[second].internal_points, points)
        before = copy.deepcopy(graph.to_dict())
        with self.assertRaisesRegex(ValueError, 'radius must be finite'):
            graph.add_edge(a, b, radius=np.float64('inf'), allow_parallel=True)
        self.assertEqual(graph.to_dict(), before)
        restored = StructureGraph.from_dict(before)
        self.assertEqual(restored.to_dict(), before)

    def test_duplicate_edge_still_ignores_unused_radius(self):
        graph, a, b = self.graph()
        edge = graph.add_edge(a, b, radius=0.2)
        before = copy.deepcopy(graph.to_dict())
        self.assertEqual(graph.add_edge(b, a, radius=float('nan')), edge)
        self.assertEqual(graph.to_dict(), before)

    def test_self_loop_still_ignores_unused_radius(self):
        graph, a, b = self.graph()
        before = copy.deepcopy(graph.to_dict())
        self.assertEqual(graph.add_edge(a, a, radius=float('inf')), -1)
        self.assertEqual(graph.add_edge(999, 999, radius=float('nan')), -1)
        self.assertEqual(graph.to_dict(), before)

    def test_missing_endpoint_error_precedes_radius_validation(self):
        graph, a, b = self.graph()
        before = copy.deepcopy(graph.to_dict())
        with self.assertRaisesRegex(ValueError, 'not in graph'):
            graph.add_edge(a, 999, radius=float('inf'))
        self.assertEqual(graph.to_dict(), before)


if __name__ == '__main__':
    unittest.main()
