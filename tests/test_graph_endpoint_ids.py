"""Endpoint-ID admission on the real StructureGraph owner; requires NumPy."""
import copy
import json
import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph


class EndpointIDTests(unittest.TestCase):
    def fixture(self):
        graph = StructureGraph()
        for point in ((0, 0), (1, 0), (2, 1)):
            graph.add_node(point, merge=False)
        return graph

    def state(self, graph):
        return (copy.deepcopy(graph.to_dict()), copy.deepcopy(graph._edge_set),
                copy.deepcopy(graph._spatial_hash), graph._next_node_id,
                graph._next_edge_id, tuple(graph.nodes), tuple(graph.edges))

    def test_invalid_endpoint_types_rejected_without_mutation(self):
        values = [True, False, np.bool_(True), 1.0, np.float64(1.0),
                  float('nan'), float('inf'), '1', None, [1], {'id': 1},
                  -1, np.int64(-1)]
        for value in values:
            for slot in (0, 1):
                for existing in (False, True):
                    with self.subTest(value=repr(value), slot=slot, existing=existing):
                        graph = self.fixture()
                        if existing:
                            graph.add_edge(1, 2)
                        before = self.state(graph)
                        endpoints = [1, 2]
                        endpoints[slot] = value
                        with self.assertRaisesRegex(ValueError, 'nonnegative integer node ID'):
                            graph.add_edge(*endpoints)
                        self.assertEqual(self.state(graph), before)

    def test_invalid_self_aliases_are_checked_before_equality(self):
        for value in (True, False, np.bool_(True), 1.0, -1):
            with self.subTest(value=repr(value)):
                graph = self.fixture()
                before = self.state(graph)
                with self.assertRaisesRegex(ValueError, 'nonnegative integer node ID'):
                    graph.add_edge(value, value)
                self.assertEqual(self.state(graph), before)

    def test_numpy_integer_endpoints_are_json_roundtrippable_builtin_ints(self):
        for left, right in ((np.int64(0), np.int32(1)),
                            (np.uint64(1), np.uint32(2))):
            with self.subTest(left=type(left).__name__, right=type(right).__name__):
                graph = self.fixture()
                eid = graph.add_edge(left, right)
                self.assertIs(type(graph.edges[eid].node_i), int)
                self.assertIs(type(graph.edges[eid].node_j), int)
                saved = graph.to_dict()
                loaded = StructureGraph.from_dict(json.loads(json.dumps(saved)))
                self.assertEqual(loaded.to_dict(), saved)

    def test_integer_self_loop_and_unknown_endpoints_preserve_behavior(self):
        graph = self.fixture()
        before = self.state(graph)
        for same in (0, 2, 999, np.int64(1)):
            self.assertEqual(graph.add_edge(same, same), -1)
            self.assertEqual(self.state(graph), before)
        for endpoints in ((0, 999), (999, 0), (np.int64(999), np.int64(0))):
            with self.assertRaisesRegex(ValueError, 'not in graph'):
                graph.add_edge(*endpoints)
            self.assertEqual(self.state(graph), before)

    def test_dedup_reversed_parallel_curves_and_stable_ids(self):
        graph = self.fixture()
        curve = np.array([[.4, .2, 0.], [.7, .1, 0.]])
        first = graph.add_edge(0, 1, radius=.2, internal_points=curve, segments=7,
                               label='original')
        before = self.state(graph)
        self.assertEqual(graph.add_edge(1, 0), first)
        self.assertEqual(graph.add_edge(np.int64(0), np.int64(1)), first)
        self.assertEqual(self.state(graph), before)
        second = graph.add_edge(np.int64(1), np.int64(0), radius=.3,
                                internal_points=curve[::-1].copy(), allow_parallel=True)
        self.assertNotEqual(first, second)
        self.assertEqual((graph.edges[first].node_i, graph.edges[first].node_j), (0, 1))
        self.assertEqual((graph.edges[second].node_i, graph.edges[second].node_j), (1, 0))
        np.testing.assert_array_equal(graph.edges[first].internal_points, curve)
        np.testing.assert_array_equal(graph.edges[second].internal_points, curve[::-1])
        saved = graph.to_dict()
        self.assertEqual(StructureGraph.from_dict(json.loads(json.dumps(saved))).to_dict(), saved)
        graph.remove_edge(first)
        third = graph.add_edge(0, 2)
        self.assertEqual((second, third), (1, 2))
        self.assertEqual(tuple(graph.nodes), (0, 1, 2))
        self.assertEqual(tuple(graph.edges), (1, 2))

    def test_sparse_node_ids_retained(self):
        graph = StructureGraph.from_dict({'nodes': [
            {'id': 7, 'position': [0, 0]}, {'id': 41, 'position': [1, 1]}],
            'edges': [], 'next_node_id': 90, 'next_edge_id': 12})
        eid = graph.add_edge(np.int64(41), np.uint64(7))
        self.assertEqual(eid, 12)
        self.assertEqual((graph.edges[eid].node_i, graph.edges[eid].node_j), (41, 7))
        self.assertEqual(graph._next_node_id, 90)
        saved = graph.to_dict()
        self.assertEqual(StructureGraph.from_dict(json.loads(json.dumps(saved))).to_dict(), saved)


if __name__ == '__main__':
    unittest.main()
