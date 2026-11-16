"""Strict explicit multi-weld geometry/topology conversion."""
import copy
import unittest
import numpy as np
from fibernet.core.fiber import Fiber, CrossSection
from fibernet.core.material import Material
from fibernet.core.network import Crosslink, FiberNetwork
from fibernet.core.single_junction_import import convert_exact_welded_crosslinks


def straight(a, b, fiber_id, radius):
    return Fiber(np.array([a, b], dtype=float), fiber_id=fiber_id, radius=radius,
                 material=Material(name='m' + str(fiber_id)))


def fixture():
    net = FiberNetwork(dimension=2, box_size=np.array([8., 8., 1.]),
                       metadata={'experiment': ['source']})
    net.fibers = [straight([-2, 0, 0], [2, 0, 0], 81, .2),
                  straight([-1, -1, 0], [-1, 1, 0], 42, .3),
                  straight([1, -1, 0], [1, 1, 0], 73, .4)]
    # Reversed order tests stable source mapping and sorting along fiber 0.
    net.crosslinks = [Crosslink(2, 0, .5, .75, np.array([1., 0., 0.])),
                      Crosslink(0, 1, .25, .5, np.array([-1., 0., 0.]))]
    return net


class MultiJunction(unittest.TestCase):
    def test_two_junctions_one_fiber_and_detached_materials(self):
        net = fixture(); net.fibers[2].fiber_id = 81
        before = copy.deepcopy(net.to_dict())
        graph, report = convert_exact_welded_crosslinks(net)
        self.assertEqual((graph.num_nodes, graph.num_edges), (8, 7))
        first, second = report['crosslink_junction_node_ids']
        self.assertNotEqual(first, second)
        self.assertEqual(report['fibers'][0], {'source_fiber_id': 81,
                         'parameters': [.25, .75], 'junction_node_ids': [second, first],
                         'edge_ids': [0, 1, 2]})
        self.assertEqual(report['fibers'][1]['junction_node_ids'], [second])
        self.assertEqual(report['fibers'][2]['junction_node_ids'], [first])
        self.assertEqual(report['fibers'][2]['source_fiber_id'], 81)
        self.assertNotEqual(report['fibers'][0]['edge_ids'], report['fibers'][2]['edge_ids'])
        self.assertEqual([graph.edges[i].material.name for i in range(7)],
                         ['m81'] * 3 + ['m42'] * 2 + ['m73'] * 2)
        self.assertEqual([graph.edges[i].radius for i in range(7)],
                         [.2] * 3 + [.3] * 2 + [.4] * 2)
        self.assertTrue(all(graph.edges[i].segments == 4 for i in range(7)))
        self.assertIsNot(graph.edges[0].material, net.fibers[0].material)
        graph.metadata['experiment'].append('changed')
        self.assertEqual(net.metadata['experiment'], ['source'])
        self.assertEqual(net.to_dict(), before)
        graph2, report2 = convert_exact_welded_crosslinks(net)
        self.assertEqual(report, report2)
        self.assertEqual(graph.edge_array().tolist(), graph2.edge_array().tolist())

    def test_transitive_weld_and_disjoint_same_coordinate(self):
        net = FiberNetwork(dimension=3)
        net.fibers = [straight([-1,0,0], [1,0,0], 0, .1),
                      straight([0,-1,0], [0,1,0], 1, .1),
                      straight([0,0,-1], [0,0,1], 2, .1),
                      straight([-1,-1,0], [1,1,0], 3, .1)]
        zero = np.zeros(3)
        net.crosslinks = [Crosslink(0, 1, .5, .5, zero),
                          Crosslink(1, 2, .5, .5, zero)]
        graph, report = convert_exact_welded_crosslinks(net)
        self.assertEqual(report['crosslink_junction_node_ids'][0],
                         report['crosslink_junction_node_ids'][1])
        junction = report['crosslink_junction_node_ids'][0]
        for index in (0, 1, 2):
            edges = [graph.edges[eid] for eid in report['fibers'][index]['edge_ids']]
            self.assertEqual(sum(junction in (edge.node_i, edge.node_j) for edge in edges), 2)
        self.assertEqual((graph.num_nodes, graph.num_edges), (9, 7))
        self.assertEqual(report['fibers'][3]['parameters'], [])
        net.crosslinks = [Crosslink(0, 1, .5, .5, zero),
                          Crosslink(2, 3, .5, .5, zero)]
        graph, report = convert_exact_welded_crosslinks(net)
        a, b = report['crosslink_junction_node_ids']
        self.assertNotEqual(a, b)
        np.testing.assert_array_equal(graph.nodes[a].position, graph.nodes[b].position)
        self.assertEqual((graph.num_nodes, graph.num_edges), (10, 8))
        groups = []
        for pair in ((0, 1), (2, 3)):
            nodes = set()
            for index in pair:
                for eid in report['fibers'][index]['edge_ids']:
                    edge = graph.edges[eid]
                    nodes.update((edge.node_i, edge.node_j))
            groups.append(nodes)
        self.assertFalse(groups[0] & groups[1])

    def test_rejects_duplicate_invalid_or_nonexact_links(self):
        def duplicate(n): n.crosslinks.append(copy.deepcopy(n.crosslinks[0]))
        def reversed_duplicate(n): n.crosslinks.append(Crosslink(0, 2, .75, .5, np.array([1., 0., 0.])))
        def nearby(n): n.crosslinks[0].position = np.array([1., 1e-9, 0.])
        def endpoint(n): n.crosslinks[0].param_i = 0.
        def bad_ref(n): n.crosslinks[0].fiber_i = 8
        def self_link(n): n.crosslinks[0].fiber_i = n.crosslinks[0].fiber_j
        def pinned(n): n.crosslinks[0].crosslink_type = 'pinned'
        def finite(n): n.crosslinks[0].stiffness = 1.
        def curved(n): n.fibers[0].centerline = np.array([[-2.,0,0], [0,.1,0], [2,0,0]])
        def noncircular(n): n.fibers[0].cross_section = CrossSection.RECTANGULAR
        def periodic(n): n.periodic = True
        def no_links(n): n.crosslinks = []
        def coincident_end(n): n.fibers[2].centerline[-1] = n.fibers[1].centerline[-1].copy()
        for mutate in (duplicate, reversed_duplicate, nearby, endpoint, bad_ref, self_link,
                       pinned, finite, curved, noncircular, periodic, no_links, coincident_end):
            with self.subTest(case=mutate.__name__):
                net = fixture(); mutate(net)
                with self.assertRaises((ValueError, TypeError)):
                    convert_exact_welded_crosslinks(net)

    def test_resource_bounds_and_endpoint_at_unrelated_weld(self):
        net = fixture(); net.box_size = [8., 8.]
        graph, _ = convert_exact_welded_crosslinks(net)
        np.testing.assert_array_equal(graph.box_size, [8, 8, 0])
        net = fixture(); net.fibers = net.fibers * 400
        with self.assertRaisesRegex(ValueError, 'fiber count'):
            convert_exact_welded_crosslinks(net)
        net = fixture(); net.crosslinks = net.crosslinks * 2049
        with self.assertRaisesRegex(ValueError, 'crosslink count'):
            convert_exact_welded_crosslinks(net)
        net = fixture(); net.fibers[0].centerline = np.zeros((10001, 3))
        with self.assertRaisesRegex(ValueError, 'sample limit'):
            convert_exact_welded_crosslinks(net)
        net = fixture(); net.fibers.append(straight([1,0,0], [3,3,0], 99, .1))
        with self.assertRaisesRegex(ValueError, 'coinciding with any fiber endpoint'):
            convert_exact_welded_crosslinks(net)

    def test_same_fiber_same_position_with_distinct_parameters_rejected(self):
        x = float(2**53)
        net = FiberNetwork(dimension=2)
        net.fibers = [straight([x,0,0], [x+8,0,0], 0, .1),
                      straight([x+4,-1,0], [x+4,1,0], 1, .1),
                      straight([x+4,-2,0], [x+4,2,0], 2, .1)]
        net.crosslinks = [Crosslink(0,1,.4,.5,np.array([x+4,0,0])),
                          Crosslink(0,2,.41,.5,np.array([x+4,0,0]))]
        with self.assertRaisesRegex(ValueError, 'multiple parameters'):
            convert_exact_welded_crosslinks(net)


if __name__ == '__main__': unittest.main()
