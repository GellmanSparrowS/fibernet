"""Displacement size admission and existing finite/ID contracts."""
import copy
import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph

class DisplacementShapeTests(unittest.TestCase):
    def fixture(self):
        g = StructureGraph(dimension=3)
        g.add_node([1., 2., 3.], merge=False)
        g.add_node([4., 5., 6.], merge=False)
        curve = np.array([[2., 3., 4.], [3., 4., 5.]])
        g.add_edge(0, 1, internal_points=curve, segments=7)
        g.add_edge(1, 0, internal_points=curve[::-1].copy(), allow_parallel=True)
        return g

    def state(self, g):
        return (copy.deepcopy(g.to_dict()), copy.deepcopy(g._spatial_hash),
                copy.deepcopy(g._edge_set), g._next_node_id, g._next_edge_id)

    def test_wrong_component_counts_rejected_without_mutation(self):
        for value in (0.5, np.array(0.5), [0.5], [[0.5]], [],
                      [1, 2, 3, 4], [[1, 2], [3, 4]], np.zeros((0, 3))):
            with self.subTest(value=repr(value)):
                g = self.fixture(); before = self.state(g)
                original = g.nodes[0].position
                with self.assertRaisesRegex(ValueError, 'displacement must have 2 or 3 components'):
                    g.displace_node(0, value)
                self.assertEqual(self.state(g), before)
                self.assertIs(g.nodes[0].position, original)

    def test_nested_two_and_three_components_follow_position_flattening(self):
        cases = (([[.5, -.5]], [1.5, 1.5, 3.]),
                 ([[.5], [-.5]], [1.5, 1.5, 3.]),
                 ([[.5, -.5, 1.]], [1.5, 1.5, 4.]),
                 ([[.5], [-.5], [1.]], [1.5, 1.5, 4.]),
                 (np.array([[[.5, -.5, 1.]]]), [1.5, 1.5, 4.]))
        for value, expected in cases:
            with self.subTest(shape=np.shape(value)):
                g = self.fixture(); edges = copy.deepcopy(g.to_dict()['edges'])
                counters = (g._next_node_id, g._next_edge_id)
                g.displace_node(0, value)
                np.testing.assert_array_equal(g.nodes[0].position, expected)
                self.assertEqual(g.to_dict()['edges'], edges)
                self.assertEqual((g._next_node_id, g._next_edge_id), counters)
                self.assertIn(0, g._spatial_hash[g._pos_key(np.array(expected))])
                self.assertNotIn(0, g._spatial_hash.get(g._pos_key(np.array([1., 2., 3.])), []))
                before = self.state(g)
                self.assertEqual(g.add_node(expected), 0)
                self.assertEqual(self.state(g), before)

    def test_flat_vectors_and_existing_numeric_coercion(self):
        for value, expected in (([.5, -.5], [1.5, 1.5, 3.]),
                                ((.5, -.5, 1.), [1.5, 1.5, 4.]),
                                (['0.5', '-0.5'], [1.5, 1.5, 3.])):
            with self.subTest(value=value):
                g = self.fixture(); g.displace_node(np.int64(0), value)
                np.testing.assert_array_equal(g.nodes[0].position, expected)

    def test_nonfinite_and_overflow_keep_graph_unchanged(self):
        for value in ([float('nan'), 0], [float('inf'), 0, 0],
                      [float('-inf'), 0, 0], [1e308, 0, 0]):
            with self.subTest(value=value):
                g = self.fixture(); before = self.state(g)
                with self.assertRaises(ValueError): g.displace_node(0, value)
                self.assertEqual(self.state(g), before)
        # Directly arrange an extreme stored position to reach addition overflow;
        # this test does not claim add_node admits such a hash coordinate.
        g = self.fixture(); g.nodes[0].position[0] = np.finfo(float).max
        before = self.state(g)
        with self.assertRaisesRegex(ValueError, 'finite position'):
            g.displace_node(0, [np.finfo(float).max, 0, 0])
        self.assertEqual(self.state(g), before)

    def test_identifier_error_precedes_displacement_conversion(self):
        class ForbiddenConversion:
            def __array__(self, *args, **kwargs):
                raise AssertionError('displacement converted before ID admission')
        for node_id in (True, 0.0, -1, '0'):
            with self.subTest(node_id=node_id):
                g = self.fixture(); before = self.state(g)
                with self.assertRaises(ValueError): g.displace_node(node_id, ForbiddenConversion())
                self.assertEqual(self.state(g), before)
        g = self.fixture(); before = self.state(g)
        with self.assertRaises(KeyError): g.displace_node(999, ForbiddenConversion())
        self.assertEqual(self.state(g), before)

if __name__ == '__main__': unittest.main()
