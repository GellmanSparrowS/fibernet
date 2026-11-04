import copy
import unittest
import numpy as np
from fibernet.core.fiber import Fiber, CrossSection
from fibernet.core.material import Material
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.single_junction_import import convert_single_welded_crosslink


def fixture():
    n = FiberNetwork(dimension=2, box_size=np.array([4., 4., 1.]), metadata={'label': ['source']})
    for a, b, radius in (([-1., 0., 0.], [1., 0., 0.], .2), ([0., -1., 0.], [0., 1., 0.], .3)):
        n.add_fiber(Fiber(np.linspace(a, b, 3), radius=radius, material=Material(name='fiber')))
    n.fibers[0].fiber_id = 81
    n.fibers[1].fiber_id = 42
    n.add_crosslink(Crosslink(1, 0, .5, .5, np.zeros(3)))
    return n

class SingleJunction(unittest.TestCase):
    def test_exact_crossing_has_one_shared_node_and_mapping(self):
        net = fixture()
        before = copy.deepcopy(net.to_dict())
        graph, report = convert_single_welded_crosslink(net)
        self.assertEqual((graph.num_nodes, graph.num_edges), (5, 4))
        self.assertEqual(report['fibers'][0]['source_fiber_id'], 81)
        self.assertEqual(report['fibers'][1]['source_fiber_id'], 42)
        self.assertEqual([report['fibers'][i]['edge_ids'] for i in range(2)], [[0, 1], [2, 3]])
        junction = report['junction_node_id']
        self.assertEqual(sum(junction in (e.node_i, e.node_j) for e in graph.edges.values()), 4)
        self.assertEqual([e.radius for e in graph.edges.values()], [.2, .2, .3, .3])
        for edge in graph.edges.values(): self.assertIsNot(edge.material, net.fibers[0].material)
        graph.metadata['label'].append('changed')
        self.assertEqual(net.metadata['label'], ['source'])
        self.assertEqual(net.to_dict(), before)
        self.assertEqual(StructureGraph.from_fiber_network(net).num_edges, 2)

    def test_three_dimensional_reversed_indices_and_material_mapping(self):
        net = FiberNetwork(dimension=3)
        net.add_fiber(Fiber(np.array([[-1., 0., 2.], [3., 0., 2.]]), radius=.2, material=Material(name='first')))
        net.add_fiber(Fiber(np.array([[0., -3., 2.], [0., 1., 2.]]), radius=.4, material=Material(name='second')))
        net.fibers[0].fiber_id, net.fibers[1].fiber_id = 27, 93
        net.add_crosslink(Crosslink(1, 0, .75, .25, np.array([0., 0., 2.])))
        graph, report = convert_single_welded_crosslink(net)
        self.assertEqual(report['fibers'][0], {'source_fiber_id': 27, 'edge_ids': [0, 1]})
        self.assertEqual(report['fibers'][1], {'source_fiber_id': 93, 'edge_ids': [2, 3]})
        self.assertEqual([graph.edges[i].material.name for i in range(4)], ['first', 'first', 'second', 'second'])
        np.testing.assert_array_equal(graph.nodes[report['junction_node_id']].position, [0, 0, 2])
        self.assertEqual(sum(report['junction_node_id'] in (e.node_i, e.node_j) for e in graph.edges.values()), 4)
        for i in range(4): self.assertIsNot(graph.edges[i].material, net.fibers[i // 2].material)

    def test_masked_complex_and_nonintegral_dimension_are_rejected(self):
        for target, value in (
            ('centerline', np.ma.array([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], mask=False)),
            ('centerline', np.array([[-1.+0.j, 0, 0], [0, 0, 0], [1, 0, 0]])),
            ('position', np.ma.array([0., 0., 0.], mask=[False, True, False])),
            ('position', [np.ma.masked, 0., 0.]),
            ('position', np.array([0.+0.j, 0, 0])),
            ('position', [2**53 + 1, 0, 0]),
            ('box', np.ma.array([4., 4., 1.], mask=False)),
            ('box', np.array([4.+0.j, 4, 1])),
            ('box', [2**53 + 1, 4.0, 1.0]),
            ('box', [True, 4.0, 1.0]),
            ('centerline', [[-1, 0.0, 0], [0, 0, 0], [2**53 + 1, 0, 0]]),
            ('dimension', 2.0),
            ('dimension', True),
        ):
            with self.subTest(target=target, value=value):
                net = fixture()
                if target == 'centerline': net.fibers[0].centerline = value
                elif target == 'position': net.crosslinks[0].position = value
                elif target == 'box': net.box_size = value
                else: net.dimension = value
                with self.assertRaises(ValueError): convert_single_welded_crosslink(net)

    def test_rejects_ambiguous_or_lossy_cases(self):
        def nearby(n): n.crosslinks[0].position = np.array([0., 1e-9, 0.])
        def curve(n): n.fibers[0].centerline[1, 1] = .1
        def near_other_fiber(n): n.fibers[1].centerline[:, 0] += 1e-9
        def material(n): n.fibers[0].cross_section = CrossSection.RECTANGULAR
        def material_subclass(n):
            class CustomMaterial(Material): pass
            n.fibers[0].material = CustomMaterial()
        def connection(n): n.crosslinks[0].crosslink_type = 'pinned'
        def finite(n): n.crosslinks[0].stiffness = 10.
        def endpoint(n): n.crosslinks[0].param_i = 0.
        def extra(n): n.crosslinks.append(copy.deepcopy(n.crosslinks[0]))
        def periodic(n): n.periodic = True
        def refs(n): n.crosslinks[0].fiber_i = 8
        for mutate in (nearby, curve, near_other_fiber, material, material_subclass, connection, finite, endpoint, extra, periodic, refs):
            with self.subTest(mutate=mutate.__name__):
                net = fixture(); mutate(net)
                with self.assertRaises((ValueError, TypeError)):
                    convert_single_welded_crosslink(net)

if __name__ == '__main__': unittest.main()
