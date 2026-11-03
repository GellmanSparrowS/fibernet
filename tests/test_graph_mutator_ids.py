"""Real StructureGraph mutation-ID admission; uses installed NumPy."""
import copy
import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph


class MutatorIDTests(unittest.TestCase):
    def fixture(self):
        g = StructureGraph()
        for p in ((0, 0), (1, 0), (2, 1)):
            g.add_node(p, merge=False)
        curve = np.array([[.3, .2, 0.], [.8, .1, 0.]])
        g.add_edge(0, 1, internal_points=curve, segments=7)
        g.add_edge(1, 0, internal_points=curve[::-1].copy(), allow_parallel=True)
        g.add_edge(1, 2)
        return g

    def state(self, g):
        return (copy.deepcopy(g.to_dict()), copy.deepcopy(g._spatial_hash),
                copy.deepcopy(g._edge_set), g._next_node_id, g._next_edge_id,
                tuple(g.nodes), tuple(g.edges))

    def invoke(self, g, method, value):
        if method in ('remove_node', 'remove_edge'):
            getattr(g, method)(value)
        elif method == 'set_node_positions':
            # Valid first entry proves a later invalid key cannot partly apply.
            g.set_node_positions({2: [7, 8], value: [9, 10]})
        else:
            getattr(g, method)(value, [9, 10])

    def test_invalid_aliases_never_delete_or_move(self):
        for method in ('remove_node', 'remove_edge', 'set_node_position',
                       'set_node_positions', 'displace_node'):
            for value in (True, False, np.bool_(True), 1.0, np.float64(1.0)):
                with self.subTest(method=method, value=repr(value)):
                    g = self.fixture()
                    before = self.state(g)
                    with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                        self.invoke(g, method, value)
                    self.assertEqual(self.state(g), before)

    def test_other_invalid_ids_and_batch_failure_are_transactional(self):
        for method in ('remove_node', 'remove_edge', 'set_node_position',
                       'set_node_positions', 'displace_node'):
            for value in (-1, np.int64(-1), '1', None, float('nan'), float('inf')):
                with self.subTest(method=method, value=repr(value)):
                    g = self.fixture()
                    before = self.state(g)
                    with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                        self.invoke(g, method, value)
                    self.assertEqual(self.state(g), before)

    def test_unhashable_ids_rejected_before_direct_lookup(self):
        for method in ('remove_node', 'remove_edge', 'set_node_position', 'displace_node'):
            for value in ([1], {'id': 1}, np.array([1])):
                with self.subTest(method=method, value=repr(value)):
                    g = self.fixture()
                    before = self.state(g)
                    with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                        self.invoke(g, method, value)
                    self.assertEqual(self.state(g), before)

    def test_valid_unknown_ids_keep_existing_behavior(self):
        for value in (999, np.int64(999), np.uint64(999)):
            g = self.fixture()
            before = self.state(g)
            for method in ('remove_node', 'remove_edge'):
                self.assertIsNone(getattr(g, method)(value))
                self.assertEqual(self.state(g), before)
            for method in ('set_node_position', 'set_node_positions', 'displace_node'):
                with self.assertRaisesRegex(KeyError, 'not found'):
                    self.invoke(g, method, value)
                self.assertEqual(self.state(g), before)

    def test_numpy_integer_moves_keep_lookup_ids_and_curve_records(self):
        g = self.fixture()
        edges = copy.deepcopy(g.to_dict()['edges'])
        counters = (g._next_node_id, g._next_edge_id)
        g.set_node_position(np.int64(1), [4, 5])
        g.displace_node(np.uint32(1), [.5, -.5])
        g.set_node_positions({np.int32(0): [-2, -3], np.uint64(2): [8, 9]})
        np.testing.assert_array_equal(g.nodes[1].position, [4.5, 4.5, 0.])
        self.assertEqual(g.add_node([4.5, 4.5]), 1)
        self.assertEqual(g.add_node([-2, -3]), 0)
        self.assertEqual(g.add_node([8, 9]), 2)
        self.assertTrue(all(type(nid) is int for ids in g._spatial_hash.values() for nid in ids))
        self.assertEqual(g.to_dict()['edges'], edges)
        self.assertEqual((g._next_node_id, g._next_edge_id), counters)

    def test_numpy_integer_removal_retains_parallel_representative_and_counters(self):
        g = self.fixture()
        counters = (g._next_node_id, g._next_edge_id)
        retained = copy.deepcopy(g.to_dict()['edges'][1])
        g.remove_edge(np.int64(0))
        self.assertEqual(tuple(g.edges), (1, 2))
        self.assertEqual(g._edge_set[frozenset((0, 1))], 1)
        self.assertEqual(g.to_dict()['edges'][0], retained)
        self.assertEqual(g.add_edge(0, 1), 1)
        g.remove_node(np.uint64(1))
        self.assertEqual(tuple(g.nodes), (0, 2))
        self.assertEqual(tuple(g.edges), ())
        self.assertEqual(g._edge_set, {})
        self.assertFalse(any(1 in bucket for bucket in g._spatial_hash.values()))
        self.assertEqual((g._next_node_id, g._next_edge_id), counters)

    def test_invalid_batch_key_after_valid_edit_preserves_original_arrays(self):
        g = self.fixture()
        before = self.state(g)
        objects = {i: node.position for i, node in g.nodes.items()}
        with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
            g.set_node_positions({0: [20, 30], 1.0: [40, 50]})
        self.assertEqual(self.state(g), before)
        for i, original in objects.items():
            self.assertIs(g.nodes[i].position, original)


if __name__ == '__main__':
    unittest.main()
