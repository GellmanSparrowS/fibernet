"""Periodic cell inputs are owned by the graph, not by callers."""
import unittest

import numpy as np

from fibernet.core.structure_graph import StructureGraph


class TestStructureBoxOwnership(unittest.TestCase):
    def test_constructor_detaches_float64_array(self):
        source = np.array([3., 4., 5.])
        graph = StructureGraph(dimension=3, box_size=source)
        source[:] = -1
        np.testing.assert_array_equal(graph.box_size, [3., 4., 5.])

    def test_setter_detaches_strided_view(self):
        source = np.arange(6., dtype=np.float64)
        graph = StructureGraph(box_size=[1., 2.])
        graph.box_size = source[::2]
        source[:] = -1
        np.testing.assert_array_equal(graph.box_size, [0., 2., 4.])

    def test_two_graphs_own_separate_inputs(self):
        source = np.array([3., 4., 5.])
        first = StructureGraph(box_size=source)
        second = StructureGraph()
        second.box_size = source
        # The public setter is the supported way to change a cell.
        first.box_size = [6., 7., 8.]
        source[:] = 0
        np.testing.assert_array_equal(first.box_size, [6., 7., 8.])
        np.testing.assert_array_equal(second.box_size, [3., 4., 5.])

    def test_getter_remains_detached(self):
        graph = StructureGraph(box_size=[3., 4., 5.])
        returned = graph.box_size
        returned[:] = 0
        np.testing.assert_array_equal(graph.box_size, [3., 4., 5.])

    def test_existing_coercion_and_two_dimensional_padding(self):
        for source in ([3, 4], (3, 4), np.array([3, 4], dtype=np.int32),
                       np.array([3., 4.], dtype=np.float32)):
            for construct in (True, False):
                with self.subTest(source=type(source), construct=construct):
                    graph = StructureGraph(box_size=source if construct else None)
                    if not construct:
                        graph.box_size = source
                    np.testing.assert_array_equal(graph.box_size, [3., 4., 0.])
                    self.assertEqual(graph.box_size.dtype, np.dtype('float64'))

    def test_readonly_input_and_none(self):
        source = np.array([3., 4., 5.])
        source.flags.writeable = False
        graph = StructureGraph(box_size=source)
        graph.box_size = source
        self.assertTrue(graph.box_size.flags.writeable)
        np.testing.assert_array_equal(source, [3., 4., 5.])
        graph.box_size = None
        self.assertIsNone(graph.box_size)
        self.assertIsNone(StructureGraph().box_size)

    def test_failed_setter_preserves_existing_box(self):
        graph = StructureGraph(box_size=[3., 4., 5.])
        with self.assertRaises((ValueError, TypeError)):
            graph.box_size = ['not a coordinate']
        np.testing.assert_array_equal(graph.box_size, [3., 4., 5.])


if __name__ == '__main__':
    unittest.main()
