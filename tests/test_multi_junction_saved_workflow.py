"""Saved exact-weld geometry and ownership, without a mechanical solver.

Legacy JSON deliberately preserves its existing Infinity and load-time ID
renumbering contracts. Only this fixture's native output is asserted finite.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fibernet.core.fiber import Fiber
from fibernet.core.material import Material
from fibernet.core.network import Crosslink, FiberNetwork
from fibernet.core.single_junction_import import convert_exact_welded_crosslinks
from fibernet.core.structure_graph import StructureGraph


def saved_design():
    network = FiberNetwork(
        dimension=2, box_size=np.array([16., 12., 0.]),
        metadata={'nested': {'labels': ['source']}})
    endpoints = [([-4, 0, 0], [4, 0, 0]),
                 ([-2, -2, 0], [-2, 2, 0]),
                 ([2, -2, 0], [2, 2, 0]),
                 ([-3, -1, 0], [-1, 1, 0]),
                 ([5, 0, 0], [5, 2, 0])]
    for i, ((a, b), label) in enumerate(zip(endpoints, [81, 42, 81, -9, 1000])):
        network.fibers.append(Fiber(
            np.linspace(a, b, 5), fiber_id=label, radius=(i + 1) / 10,
            material=Material(name='fiber-' + str(i),
                              extra={'nested': {'labels': ['material-' + str(i)]}}),
            segments=7))
    network.crosslinks = [
        Crosslink(2, 0, .5, .75, np.array([2., 0., 0.])),
        Crosslink(0, 1, .25, .5, np.array([-2., 0., 0.])),
        Crosslink(1, 3, .5, .5, np.array([-2., 0., 0.]))]
    return network


def components(graph):
    neighbors = {node: set() for node in graph.nodes}
    for edge in graph.edges.values():
        neighbors[edge.node_i].add(edge.node_j)
        neighbors[edge.node_j].add(edge.node_i)
    unseen, groups = set(neighbors), []
    while unseen:
        pending, found = [next(iter(unseen))], set()
        while pending:
            node = pending.pop()
            if node not in found:
                found.add(node)
                pending.extend(neighbors[node] - found)
        unseen -= found
        groups.append(found)
    return groups


class MultiJunctionSavedWorkflow(unittest.TestCase):
    def assert_joint_mapping(self, source, graph, report):
        self.assertEqual((graph.num_nodes, graph.num_edges), (12, 10))
        groups = components(graph)
        self.assertEqual(sorted(map(len, groups)), [2, 10])
        self.assertEqual(graph.dimension, source.dimension)
        self.assertEqual(graph.metadata, source.metadata)
        np.testing.assert_array_equal(graph.box_size, source.box_size)
        ids = report['crosslink_junction_node_ids']
        self.assertEqual(len(ids), 3)
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(ids[1], ids[2])
        for link_index, link in enumerate(source.crosslinks):
            node_id = ids[link_index]
            np.testing.assert_array_equal(graph.nodes[node_id].position, link.position)
            for fiber_index, parameter in [(link.fiber_i, link.param_i),
                                           (link.fiber_j, link.param_j)]:
                entry = report['fibers'][fiber_index]
                offset = entry['parameters'].index(parameter)
                self.assertEqual(entry['junction_node_ids'][offset], node_id)
        all_edges = []
        for index, fiber in enumerate(source.fibers):
            entry = report['fibers'][index]
            params = sorted({getattr(link, side) for link in source.crosslinks
                             for i, side in [(link.fiber_i, 'param_i'),
                                             (link.fiber_j, 'param_j')]
                             if i == index})
            self.assertEqual(entry['source_fiber_id'], fiber.fiber_id)
            self.assertEqual(entry['parameters'], params)
            nodes = entry['junction_node_ids']
            self.assertEqual(len(nodes), len(params))
            for parameter, node in zip(params, nodes):
                np.testing.assert_array_equal(
                    graph.nodes[node].position,
                    fiber.centerline[0] * (1 - parameter) +
                    fiber.centerline[-1] * parameter)
            bounds = [0.] + params + [1.]
            self.assertEqual(len(entry['edge_ids']), len(bounds) - 1)
            all_edges.extend(entry['edge_ids'])
            for j, edge_id in enumerate(entry['edge_ids']):
                edge = graph.edges[edge_id]
                for node, parameter in [(edge.node_i, bounds[j]),
                                        (edge.node_j, bounds[j + 1])]:
                    np.testing.assert_array_equal(
                        graph.nodes[node].position,
                        fiber.centerline[0] * (1 - parameter) +
                        fiber.centerline[-1] * parameter)
                if j:
                    self.assertEqual(edge.node_i, nodes[j - 1])
                if j < len(nodes):
                    self.assertEqual(edge.node_j, nodes[j])
                self.assertEqual(edge.radius, fiber.radius)
                self.assertEqual(edge.material.to_dict(), fiber.material.to_dict())
                self.assertEqual(edge.segments, 4)
                self.assertIsNone(edge.internal_points)
        self.assertEqual(sorted(all_edges), sorted(graph.edges))
        isolated = report['fibers'][4]
        self.assertEqual(isolated['parameters'], [])
        edge = graph.edges[isolated['edge_ids'][0]]
        self.assertIn({edge.node_i, edge.node_j}, groups)

    def test_saved_legacy_to_native_keeps_joint_mapping(self):
        source = saved_design()
        # This fixture contains only ordinary finite scalars/arrays and welded
        # infinities; deepcopy avoids FiberNetwork.to_dict's metadata alias.
        source_before = copy.deepcopy(source.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / 'legacy.json'
            native_path = Path(directory) / 'native.json'
            source.save_json(str(legacy_path))
            legacy_bytes = legacy_path.read_bytes()
            self.assertIn(b'Infinity', legacy_bytes)
            loaded = FiberNetwork.load_json(str(legacy_path))
            self.assertEqual([f.fiber_id for f in source.fibers], [81, 42, 81, -9, 1000])
            self.assertEqual([f.fiber_id for f in loaded.fibers], list(range(5)))
            loaded_before = copy.deepcopy(loaded.to_dict())
            graph, report = convert_exact_welded_crosslinks(loaded)
            self.assert_joint_mapping(loaded, graph, report)
            self.assertEqual(loaded.to_dict(), loaded_before)
            graph_before = graph.to_dict()
            graph.save_json(str(native_path))

            def reject_constant(value):
                raise AssertionError('Unexpected native JSON constant: ' + value)

            json.loads(native_path.read_bytes(), parse_constant=reject_constant)
            restored = StructureGraph.load_json(str(native_path))
            self.assertEqual(restored.to_dict(), graph_before)
            self.assertEqual(graph.to_dict(), graph_before)
            self.assert_joint_mapping(loaded, restored, report)
            self.assertEqual(legacy_path.read_bytes(), legacy_bytes)
            self.assertEqual(source.to_dict(), source_before)
            self.assertEqual(loaded.to_dict(), loaded_before)

    def test_nested_ownership_is_independent_in_both_directions(self):
        source = saved_design()
        before = copy.deepcopy(source.to_dict())
        graph, report = convert_exact_welded_crosslinks(source)
        self.assertEqual(source.to_dict(), before)
        children = [graph.edges[e].material for e in report['fibers'][0]['edge_ids']]
        self.assertEqual(len({id(x) for x in children}), len(children))
        self.assertTrue(all(x is not source.fibers[0].material for x in children))
        graph.metadata['nested']['labels'].append('graph edit')
        graph.box_size[0] = 99.
        graph.nodes[next(iter(graph.nodes))].position[0] = 77.
        children[0].extra['nested']['labels'].append('child edit')
        self.assertEqual(source.to_dict(), before)
        self.assertEqual(children[1].extra['nested']['labels'], ['material-0'])
        self.assertEqual(children[2].extra['nested']['labels'], ['material-0'])
        after_graph_edits = graph.to_dict()
        source.metadata['nested']['labels'].append('source edit')
        source.box_size[1] = 88.
        source.fibers[0].material.extra['nested']['labels'].append('source edit')
        source.fibers[0].centerline[0, 0] = -99.
        self.assertEqual(graph.to_dict(), after_graph_edits)


if __name__ == '__main__':
    unittest.main()
