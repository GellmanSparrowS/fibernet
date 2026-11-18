"""VTP fields follow sampled points and whole-fiber polylines."""
from collections import deque, UserList
from array import array
from enum import Enum, IntEnum
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from fibernet.api import export
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.io.vtk import to_vtk_xml


def network():
    net = FiberNetwork()
    for points in ([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
                   [[2., 0., 0.], [2., 1., 0.]]):
        net.add_fiber(Fiber(centerline=np.array(points)))
    return net


class VTPResults(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'result.vtp'

    def piece(self):
        return ET.parse(self.path).getroot().find('./PolyData/Piece')

    def test_public_export_keeps_fields_and_polyline_association(self):
        name = 'temperature&<>"\''
        vectors = np.arange(15, dtype=float).reshape(5, 3) / 7
        export(network(), str(self.path), point_data={name: np.arange(5.), 'u': vectors},
               cell_data={'stress': [4., 7.]})
        piece = self.piece()
        self.assertEqual(piece.attrib, {'NumberOfPoints': '5', 'NumberOfLines': '2'})
        arrays = piece.findall('./PointData/DataArray')
        self.assertEqual([a.get('Name') for a in arrays], [name, 'u'])
        self.assertEqual(piece.find('PointData').get('Scalars'), name)
        self.assertEqual(piece.find('PointData').get('Vectors'), 'u')
        np.testing.assert_array_equal(np.fromstring(arrays[1].text, sep=' ').reshape(5, 3), vectors)
        self.assertEqual(arrays[1].get('NumberOfComponents'), '3')
        self.assertEqual(piece.find('./CellData/DataArray').text.split(), ['4', '7'])
        self.assertEqual(piece.find('./Lines/DataArray[@Name="offsets"]').text.split(), ['3', '5'])

    def test_integer_precision_is_not_lost_to_numpy_sequence_promotion(self):
        fields = {
            'unsigned': [2**64-1, 2**63+1, 2**53+1, 0, 1],
            'signed': np.array([-2**63, -1, 0, 2**53+1, 2**63-1], dtype=np.int64),
            'mixed_numpy_ints': [np.int64(2**53+1), np.uint64(7), 0, 1, 2],
            'flag': [True, False, True, False, True],
        }
        to_vtk_xml(network(), self.path, point_data=fields)
        arrays = {a.get('Name'): a for a in self.piece().findall('./PointData/DataArray')}
        for name, expected in fields.items():
            with self.subTest(name=name):
                self.assertEqual([int(v) for v in arrays[name].text.split()], [int(v) for v in expected])
        self.assertEqual(arrays['unsigned'].get('type'), 'UInt64')
        self.assertEqual(arrays['signed'].get('type'), 'Int64')
        self.assertEqual(arrays['flag'].get('type'), 'UInt8')

    def test_standard_sequences_and_string_enum_names(self):
        class Field(str, Enum):
            force = 'force_N'
        for sequence in (deque, UserList):
            with self.subTest(sequence=sequence):
                to_vtk_xml(network(), self.path,
                           point_data={Field.force: sequence([2**63+1, 7, 0, 1, 2])})
                array = self.piece().find('./PointData/DataArray')
                self.assertEqual(array.get('Name'), 'force_N')
                self.assertEqual(array.get('type'), 'UInt64')
                self.assertEqual(int(array.text.split()[0]), 2**63+1)

    def test_typed_buffers_and_integer_subclasses(self):
        class IntValue(IntEnum):
            one = 1
            def __float__(self):
                return 0.
        for values in (array('Q', [1, 2, 3, 4, 5]), memoryview(array('Q', [1, 2, 3, 4, 5]))):
            to_vtk_xml(network(), self.path, point_data={'s': values})
            self.assertEqual(self.piece().find('./PointData/DataArray').get('type'), 'UInt64')
        to_vtk_xml(network(), self.path, point_data={'s': [IntValue.one, .5, 0, 0, 0]})
        self.assertEqual(self.piece().find('./PointData/DataArray').text.split()[:2], ['1', '0.5'])

    def test_float64_extremes_and_signed_zero(self):
        values = np.array([np.nextafter(0., 1.), np.finfo(float).max, -0., -1e100, 1/3])
        to_vtk_xml(network(), self.path, point_data={'s': values})
        array = self.piece().find('./PointData/DataArray')
        result = np.fromstring(array.text, sep=' ')
        np.testing.assert_array_equal(result.view(np.uint64), values.view(np.uint64))
        self.assertEqual(array.get('type'), 'Float64')

    def test_bad_fields_preserve_existing_destination(self):
        invalid = [
            {'point_data': {'s': [1, 2]}},
            {'point_data': {'s': np.ones((5, 2))}},
            {'cell_data': {'s': [1, 2, 3]}},
            {'cell_data': {'v': np.ones((2, 3))}},
            {'point_data': {'s': [2**53+1, 0., 1., 2., 3.]}},
            {'point_data': {'s': [-1, 2**64-1, 0, 1, 2]}},
            {'point_data': {'s': [2**64, 0, 1, 2, 3]}},
            {'point_data': {'s': np.array([1, 2, 3, 4, 5], dtype=object)}},
            {'point_data': {'s': [1j]*5}},
            {'point_data': {'s': [float('nan')]*5}},
            {'point_data': {'s': [float('inf')]*5}},
            {'point_data': {'s': np.ma.array(np.ones(5), mask=[1, 0, 0, 0, 0])}},
            {'point_data': {'bad name': np.ones(5)}},
            {'point_data': {'bad%name': np.ones(5)}},
            {'point_data': []},
        ]
        if np.dtype(np.longdouble).itemsize > 8:
            invalid.append({'point_data': {'s': np.ones(5, dtype=np.longdouble)}})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                self.path.write_text('keep this result')
                with self.assertRaises(ValueError):
                    export(network(), str(self.path), **kwargs)
                self.assertEqual(self.path.read_text(), 'keep this result')
        self.path.write_text('keep this result')
        with self.assertRaises(TypeError):
            export(network(), str(self.path), unknown_field=[1, 2])
        self.assertEqual(self.path.read_text(), 'keep this result')

    def test_empty_arrays_and_default_geometry(self):
        to_vtk_xml(FiberNetwork(), self.path, point_data={'u': np.empty((0, 3))}, cell_data={'s': []})
        self.assertEqual(self.piece().get('NumberOfPoints'), '0')
        self.assertEqual(self.piece().find('./PointData/DataArray').get('NumberOfComponents'), '3')
        to_vtk_xml(network(), self.path)
        default = self.path.read_bytes()
        to_vtk_xml(network(), self.path, point_data={}, cell_data={})
        self.assertEqual(self.path.read_bytes(), default)
        self.assertIsNone(self.piece().find('PointData'))


if __name__ == '__main__':
    unittest.main()
