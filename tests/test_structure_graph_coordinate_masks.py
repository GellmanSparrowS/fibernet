import copy
import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph

class CoordinateMasks(unittest.TestCase):
    def graph(self):
        g = StructureGraph()
        g.add_node([0, 0]); g.add_node([20, 0]); g.add_edge(0, 1)
        return g

    def snapshot(self, g):
        return copy.deepcopy((g.to_dict(), g._spatial_hash, g._edge_set))

    def masked_inputs(self):
        masked = np.ma.array([7., 9.], mask=[False, True])
        boxed = np.empty(1, dtype=object); boxed[0] = masked
        return [masked, [masked], boxed]

    def test_mask_rejection_before_mutation(self):
        for operation in ('add', 'set', 'displace'):
            for value in self.masked_inputs():
                with self.subTest(operation=operation, form=type(value).__name__):
                    g = self.graph(); before = self.snapshot(g)
                    with self.assertRaisesRegex(ValueError, 'masked'):
                        if operation == 'add': g.add_node(value)
                        elif operation == 'set': g.set_node_position(0, value)
                        else: g.displace_node(0, value)
                    self.assertEqual(self.snapshot(g), before)

    def test_batch_mask_failure_is_atomic(self):
        for value in self.masked_inputs():
            with self.subTest(form=type(value).__name__):
                g = self.graph(); before = self.snapshot(g)
                with self.assertRaisesRegex(ValueError, 'masked'):
                    g.set_node_positions({0: [1, 2], 1: value})
                self.assertEqual(self.snapshot(g), before)
                self.assertEqual(g.add_node([0, 0]), 0)
                self.assertEqual(g.add_node([20, 0]), 1)

    def test_unmasked_inputs_keep_flattening_and_ownership(self):
        for value in ([7.,9.], (7.,9.), np.array([7.,9.]),
                      np.array([[7.,9.]]), np.ma.array([7.,9.], mask=False),
                      [np.ma.array([7.,9.], mask=False)]):
            with self.subTest(form=type(value).__name__):
                g=self.graph(); n=g.add_node(value)
                np.testing.assert_array_equal(g.nodes[n].position,[7,9,0])
                g.set_node_position(0,value)
                np.testing.assert_array_equal(g.nodes[0].position,[7,9,0])
                g.displace_node(0,value)
                np.testing.assert_array_equal(g.nodes[0].position,[14,18,0])
                g.set_node_positions({0:value,1:[30,0]})
                np.testing.assert_array_equal(g.nodes[0].position,[7,9,0])
                self.assertEqual(g.add_node([7,9]), n)
        source=np.array([4.,5.]);g=self.graph();n=g.add_node(source)
        source[:]=100
        np.testing.assert_array_equal(g.nodes[n].position,[4,5,0])

    def test_polyline_mask_rejection_before_any_node_is_added(self):
        for value in self.masked_inputs():
            with self.subTest(form=type(value).__name__):
                g=self.graph();before=self.snapshot(g)
                with self.assertRaisesRegex(ValueError, 'masked'):
                    g.add_polyline(iter([[40,0],value]))
                self.assertEqual(self.snapshot(g),before)
        g=StructureGraph()
        edges=g.add_polyline(iter([np.ma.array([0.,0.],mask=False),[1.,0.]]))
        self.assertEqual(len(edges),1)
        np.testing.assert_array_equal(g.node_positions(),[[0,0,0],[1,0,0]])

    def test_unmasked_displacement_broadcast_and_invalid_batch(self):
        g=self.graph(); g.displace_node(0,[2])
        np.testing.assert_array_equal(g.nodes[0].position,[2,2,2])
        before=self.snapshot(g)
        with self.assertRaises(ValueError):g.set_node_positions({0:[8,9],1:[np.nan,0]})
        self.assertEqual(self.snapshot(g),before)

if __name__ == '__main__': unittest.main()
