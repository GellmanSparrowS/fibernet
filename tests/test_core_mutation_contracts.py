"""Regression tests for graph position edits and fiber removal."""
import copy
import itertools
import json
import unittest

import numpy as np

from fibernet import Crosslink, Fiber, FiberNetwork, Material, StructureGraph


def sparse_graph():
    graph = StructureGraph(dimension=3, box_size=[4., 4., 4.])
    for point in ([99, 99], [0, 0], [88, 88], [2, 0], [77, 77]):
        graph.add_node(point, merge=False)
    for node_id in (0, 2, 4):
        graph.remove_node(node_id)
    graph.metadata['units'] = {'length': 'mm'}
    graph.nodes[1].metadata['nested'] = {'tags': ['a']}
    for radius, points in ((.1, np.array([[1., .2, 0.]])),
                           (.2, np.empty((0, 3)))):
        graph.add_edge(1, 3, radius=radius, internal_points=points,
                       material=Material(name=str(radius)), allow_parallel=True)
    return graph


def graph_state(graph):
    return copy.deepcopy((graph.to_dict(), graph._spatial_hash,
                          list(graph.nodes), list(graph.edges), graph._edge_set))


def ray_network(count=5):
    network = FiberNetwork(dimension=3)
    for i in range(count):
        network.add_fiber(Fiber.straight([0, 0, 0], [i + 1, 1, 0],
                                        radius=.1 + i / 100, segments=1,
                                        material=Material(name=f'm{i}')))
    for k, (i, j) in enumerate(itertools.combinations(range(count), 2)):
        if k % 2:
            i, j = j, i
        network.add_crosslink(Crosslink(i, j, 0., 0., np.zeros(3),
                                        crosslink_type='pinned' if k % 2 else 'welded',
                                        strength=10. + k, stiffness=100. + k))
    return network


def crosslink_fields(link):
    return (link.param_i, link.param_j, id(link.position), link.position.tolist(),
            link.crosslink_type, link.strength, link.stiffness)


class PositionEdits(unittest.TestCase):
    def test_bad_absolute_size_or_value_is_atomic(self):
        bad_values = (0., np.array(1.), [], [1.], [1, 2, 3, 4],
                      np.ones((2, 2)), np.ones((1, 1, 4)),
                      [np.nan, 0], [0, np.inf, 0], [np.finfo(float).max, 0])
        for value in bad_values:
            with self.subTest(value=repr(value)):
                graph = sparse_graph()
                before = graph_state(graph)
                with self.assertRaises(ValueError):
                    graph.set_node_position(1, value)
                self.assertEqual(graph_state(graph), before)
                self.assertEqual(graph.add_node([0, 0]), 1)
                self.assertEqual(graph_state(graph), before)

    def test_late_bad_batch_entry_preserves_everything(self):
        for value in ([1, 2, 3, 4], np.array(2.), [1], np.ones((2, 2)),
                      [0, np.nan], [np.finfo(float).max, 0]):
            with self.subTest(value=repr(value)):
                graph = sparse_graph()
                before = graph_state(graph)
                with self.assertRaises(ValueError):
                    graph.set_node_positions({1: [10, 20], 3: value})
                self.assertEqual(graph_state(graph), before)
                self.assertEqual(graph.add_node([0, 0]), 1)
                self.assertEqual(graph.add_node([2, 0]), 3)
                self.assertEqual(graph_state(graph), before)

    def test_unknown_late_node_keeps_batch_atomic(self):
        graph = sparse_graph()
        before = graph_state(graph)
        with self.assertRaises(KeyError):
            graph.set_node_positions({1: [10, 20], 999: [4, 5]})
        self.assertEqual(graph_state(graph), before)

    def test_valid_forms_match_add_node_and_detach(self):
        values = ([1, 2], (1, 2, 3), np.array([1, 2], dtype=np.float32),
                  np.array([1, 2, 3], dtype=np.float32),
                  np.array([[1, 2]]), np.array([[1], [2]]),
                  np.array([[1, 2, 3]]), np.array([[1], [2], [3]]),
                  np.array([[[1, 2, 3]]]))
        for dimension, value in itertools.product((2, 3), values):
            with self.subTest(dimension=dimension, shape=np.asarray(value).shape):
                control = StructureGraph(dimension=dimension)
                control_id = control.add_node(value)
                expected = control.nodes[control_id].position.copy()
                graph = StructureGraph(dimension=dimension)
                node_id = graph.add_node([0, 0])
                argument = np.array(value, copy=True)
                graph.set_node_position(node_id, argument)
                np.testing.assert_array_equal(graph.nodes[node_id].position, expected)
                self.assertEqual(graph.nodes[node_id].position.shape, (3,))
                argument[...] = 999
                np.testing.assert_array_equal(graph.nodes[node_id].position, expected)
                self.assertEqual(graph.add_node(expected), node_id)
                restored = StructureGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
                self.assertEqual(restored.to_dict(), graph.to_dict())
                self.assertEqual(graph.to_numpy()[0].shape, (1, 3))

    def test_batch_swap_keeps_identity_curves_and_hash(self):
        graph = sparse_graph()
        before = graph.to_dict()
        graph.set_node_positions({1: [2, 0], 3: [[0, 0, 0]]})
        self.assertEqual(graph.add_node([2, 0]), 1)
        self.assertEqual(graph.add_node([0, 0]), 3)
        self.assertEqual(list(graph.nodes), [1, 3])
        after = graph.to_dict()
        for name in ('edges', 'metadata', 'next_node_id', 'next_edge_id'):
            self.assertEqual(before[name], after[name])
        self.assertEqual(StructureGraph.from_dict(after).to_dict(), after)
        self.assertEqual(graph.add_node([9, 8]), 5)

    def test_legacy_displacement_broadcast_and_invalid_displacement(self):
        graph = sparse_graph()
        graph.displace_node(1, [.25])
        np.testing.assert_array_equal(graph.nodes[1].position, [.25, .25, .25])
        before = graph_state(graph)
        with self.assertRaises(ValueError):
            graph.displace_node(1, [1, 2, 3, 4])
        self.assertEqual(graph_state(graph), before)


class FiberRemoval(unittest.TestCase):
    def check_removal(self, network, index):
        old_fibers = list(network.fibers)
        old_links = list(network.crosslinks)
        records = [(link, link.fiber_i, link.fiber_j, crosslink_fields(link))
                   for link in old_links]
        network.remove_fiber(index)
        expected_fibers = old_fibers[:index] + old_fibers[index + 1:]
        self.assertEqual([id(f) for f in network.fibers], [id(f) for f in expected_fibers])
        self.assertEqual([f.fiber_id for f in network.fibers], list(range(len(expected_fibers))))
        retained = [item for item in records if index not in item[1:3]]
        self.assertEqual([id(link) for link in network.crosslinks], [id(item[0]) for item in retained])
        matrix = np.zeros((len(expected_fibers), len(expected_fibers)), dtype=int)
        for link, old_i, old_j, fields in retained:
            i, j = old_i - (old_i > index), old_j - (old_j > index)
            self.assertEqual((link.fiber_i, link.fiber_j), (i, j))
            self.assertEqual(crosslink_fields(link), fields)
            self.assertIs(network.fibers[i], old_fibers[old_i])
            self.assertIs(network.fibers[j], old_fibers[old_j])
            matrix[i, j] += 1
            matrix[j, i] += 1
        np.testing.assert_array_equal(network.connectivity_matrix(), matrix)
        for link, _, _, fields in records:
            self.assertEqual(crosslink_fields(link), fields)
        return network

    def test_remove_first_middle_last_preserves_surviving_associations(self):
        for index in (0, 2, 4):
            with self.subTest(index=index):
                self.check_removal(ray_network(), index)

    def test_repeated_removal_keeps_associations(self):
        network = ray_network()
        for index in (0, 2, 1, 0, 0):
            with self.subTest(index=index, count=network.num_fibers):
                self.check_removal(network, index)
        self.assertEqual(network.num_fibers, 0)
        self.assertEqual(network.num_crosslinks, 0)

    def test_real_chain_from_canonical_bridge_remains_valid(self):
        graph = StructureGraph()
        graph.add_polyline([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        network = graph.to_fiber_network()
        self.assertTrue(network.validate()['valid'])
        network.remove_fiber(0)
        self.assertEqual([(c.fiber_i, c.fiber_j) for c in network.crosslinks], [(0, 1), (1, 2)])
        self.assertTrue(network.validate()['valid'])
        for link in network.crosslinks:
            np.testing.assert_array_equal(network.fibers[link.fiber_i].end_point, link.position)
            np.testing.assert_array_equal(network.fibers[link.fiber_j].start_point, link.position)
        network.connectivity_matrix()
        restored = FiberNetwork.from_dict(json.loads(json.dumps(network.to_dict())))
        self.assertEqual(restored.to_dict(), network.to_dict())
        np.testing.assert_array_equal(restored.connectivity_matrix(), network.connectivity_matrix())

    def test_in_bounds_wrong_association_is_prevented(self):
        graph = StructureGraph()
        graph.add_polyline([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        network = graph.to_fiber_network()
        network.crosslinks = [network.crosslinks[1]]
        self.check_removal(network, 0)
        self.assertEqual((network.crosslinks[0].fiber_i, network.crosslinks[0].fiber_j), (0, 1))

    def test_duplicate_crosslink_reference_is_not_double_shifted(self):
        network = ray_network()
        link = next(link for link in network.crosslinks if {link.fiber_i, link.fiber_j} == {3, 4})
        network.crosslinks = [link, link]
        self.check_removal(network, 0)
        self.assertIs(network.crosslinks[0], network.crosslinks[1])
        self.assertEqual({link.fiber_i, link.fiber_j}, {2, 3})

    def test_invalid_index_noop_and_singleton(self):
        network = ray_network()
        before = copy.deepcopy(network.to_dict())
        for index in (-1, 5, 99):
            network.remove_fiber(index)
            self.assertEqual(network.to_dict(), before)
        empty = FiberNetwork()
        empty.remove_fiber(0)
        self.assertEqual(empty.num_fibers, 0)
        single = ray_network(1)
        self.check_removal(single, 0)


if __name__ == '__main__':
    unittest.main()
