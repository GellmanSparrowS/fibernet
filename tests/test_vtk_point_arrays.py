"""Legacy VTK point and cell attribute shape regressions."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.io.vtk import to_vtk


def network(n):
    net = FiberNetwork()
    net.add_fiber(Fiber(centerline=np.array([[float(i), 0., 0.] for i in range(n)])))
    return net


class VTKArrays(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'network.vtk'

    def test_three_point_scalar_and_three_point_vector(self):
        to_vtk(network(3), self.path, point_data={
            'temperature': [1., 2., 3.],
            'displacement': [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]},
            cell_data={'stress': [4., 5.]})
        text = self.path.read_text()
        self.assertIn('POINT_DATA 3\nSCALARS temperature float 1\nLOOKUP_TABLE default\n'
                      '1.000000e+00\n2.000000e+00\n3.000000e+00\n', text)
        self.assertIn('VECTORS displacement float\n1.000000e+00 0.000000e+00 0.000000e+00\n'
                      '0.000000e+00 2.000000e+00 0.000000e+00\n'
                      '0.000000e+00 0.000000e+00 3.000000e+00\n', text)
        self.assertIn('CELL_DATA 2\nSCALARS stress float 1\nLOOKUP_TABLE default\n'
                      '4.000000e+00\n5.000000e+00\n', text)

    def test_four_point_vector(self):
        vectors = np.arange(12).reshape(4, 3)
        to_vtk(network(4), self.path, point_data={'velocity': vectors})
        text = self.path.read_text()
        self.assertIn('POINT_DATA 4\nVECTORS velocity float\n', text)
        self.assertIn('9.000000e+00 1.000000e+01 1.100000e+01\n', text)

    def test_multiple_fiber_point_and_segment_order(self):
        net = network(3)
        net.add_fiber(Fiber(centerline=np.array([[8., 0., 0.], [9., 0., 0.]])))
        to_vtk(net, self.path, point_data={'s': [10, 11, 12, 13, 14]},
               cell_data={'s': [20, 21, 22]})
        text = self.path.read_text()
        self.assertIn('LINES 3 9\n2 0 1\n2 1 2\n2 3 4\n', text)
        self.assertIn('POINT_DATA 5\nSCALARS s float 1\nLOOKUP_TABLE default\n'
                      '1.000000e+01\n1.100000e+01\n1.200000e+01\n'
                      '1.300000e+01\n1.400000e+01\n', text)
        self.assertIn('CELL_DATA 3\nSCALARS s float 1\nLOOKUP_TABLE default\n'
                      '2.000000e+01\n2.100000e+01\n2.200000e+01\n', text)

    def test_masked_values_rejected_before_open(self):
        for kwargs in (
            {'point_data': {'s': np.ma.array([1., 2., 3., 4.], mask=[0, 1, 0, 0])}},
            {'point_data': {'v': [[1., 2., 3.], [1., np.ma.masked, 3.],
                                  [1., 2., 3.], [1., 2., 3.]]}},
            {'cell_data': {'s': [np.ma.array([1., 2., 3.], mask=[0, 0, 1])]}},
        ):
            with self.subTest(kwargs=kwargs):
                self.path.write_text('keep')
                with self.assertRaisesRegex(ValueError, 'masked values'):
                    to_vtk(network(4), self.path, **kwargs)
                self.assertEqual(self.path.read_text(), 'keep')

    def test_empty_geometry_arrays(self):
        to_vtk(FiberNetwork(), self.path,
               point_data={'scalar': np.empty(0), 'vector': np.empty((0, 3))},
               cell_data={'scalar': np.empty(0)})
        text = self.path.read_text()
        self.assertIn('POINT_DATA 0\nSCALARS scalar float 1', text)
        self.assertIn('VECTORS vector float', text)
        self.assertIn('CELL_DATA 0\nSCALARS scalar float 1', text)

    def test_wrong_shapes_do_not_open_output(self):
        for kwargs in ({'point_data': {'v': np.zeros((4, 2))}},
                       {'point_data': {'s': np.zeros(3)}},
                       {'cell_data': {'s': np.zeros(2)}},
                       {'cell_data': {'v': np.zeros((3, 3))}}):
            with self.subTest(kwargs=kwargs):
                self.path.write_text('keep')
                with self.assertRaises(ValueError):
                    to_vtk(network(4), self.path, **kwargs)
                self.assertEqual(self.path.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
