import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.nodal_controls import prepare_taichi_schedule_by_id
from fibernet.sim.accelerated import _prepare_displacement_schedule


class ScheduleByIdTests(unittest.TestCase):
    def setUp(self):
        self.g = StructureGraph(dimension=3)
        ids = [self.g.add_node([i, 0, 0]) for i in range(4)]
        self.g.remove_node(ids[1])
        self.g.add_edge(ids[0], ids[2])
        self.g.add_edge(ids[0], ids[2], allow_parallel=True)
        node = self.g._nodes.pop(ids[0]); self.g._nodes[ids[0]] = node
        self.ids = list(self.g.nodes)

    def test_row_mapping_validation_and_copies(self):
        vector = [1., 2., 3.]
        p = prepare_taichi_schedule_by_id(self.g, {self.ids[2]: [(4, vector), (0, [0, 0, 0])], self.ids[0]: []})
        vector[0] = 99
        got = p.taichi_kwargs(self.g)['displacement_schedule']
        self.assertEqual(list(got), [2]); self.assertEqual([x[0] for x in got[2]], [0., 4.])
        np.testing.assert_array_equal(got[2][1][1], [1, 2, 3])
        got[2][1][1][0] = 77
        self.assertEqual(p.taichi_kwargs(self.g)['displacement_schedule'][2][1][1][0], 1)
        direct = _prepare_displacement_schedule({2: [(4, [1, 2, 3]), (0, [0, 0, 0])]}, 3, 3)
        np.testing.assert_array_equal(got[2][0][1], direct[2][0][1])

    def test_reject_bad_ids_and_waypoints(self):
        for bad in ({99: [(0, [0, 0, 0])]}, {True: []}, {1.0: []},
                    {self.ids[0]: [(0, [0, 0, 0]), (0, [1, 0, 0])]},
                    {self.ids[0]: [(-1, [0, 0, 0])]}, {self.ids[0]: [(0, [1, 2])] }):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare_taichi_schedule_by_id(self.g, bad)

    def test_graph_mutation_and_reorder_rejected(self):
        p = prepare_taichi_schedule_by_id(self.g, {self.ids[0]: [(0, [0, 0, 0])]})
        node = self.g._nodes.pop(self.ids[0]); self.g._nodes[self.ids[0]] = node
        with self.assertRaises(ValueError): p.taichi_kwargs(self.g)
        p = prepare_taichi_schedule_by_id(self.g, {})
        self.g.nodes[self.ids[0]].position[0] += 0.25
        with self.assertRaises(ValueError): p.taichi_kwargs(self.g)

if __name__ == '__main__': unittest.main()
