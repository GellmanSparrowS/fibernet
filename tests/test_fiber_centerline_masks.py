"""Admission of observed geometry at the legacy public Fiber constructor."""

from collections import UserList
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from fibernet import Fiber, FiberNetwork
from fibernet.api import export


class FiberCenterlineMasks(unittest.TestCase):
    def test_active_missing_coordinates_are_refused_before_coercion(self):
        first = [0., 0., 0.]
        missing = np.ma.array([9., 0., 0.], mask=[True, False, False])
        object_rows = np.array([first, [9., 0., 0.]], dtype=object)
        object_rows[1, 0] = np.ma.masked
        cases = {
            'observed-row mask': np.ma.array(
                [first, [9., 0., 0.]], mask=[[0, 0, 0], [1, 0, 0]]),
            'flattened endpoints': np.ma.array(
                [0., 0., 0., 9., 0., 0.], mask=[0, 0, 0, 1, 0, 0]),
            'list': [first, missing],
            'tuple': (first, missing),
            'UserList': UserList([first, missing]),
            'nested UserList scalar': [first, UserList([np.ma.masked, 0., 0.])],
            'object array scalar': object_rows,
        }
        for form, centerline in cases.items():
            with self.subTest(form=form):
                with self.assertRaisesRegex(ValueError, 'centerline.*masked'):
                    Fiber(centerline)

    def test_fully_observed_wrappers_keep_normal_coordinates(self):
        expected = np.array([[0., 0., 0.], [1., 0., 0.]])
        row = np.ma.array(expected[1], mask=False)
        for centerline in (
                expected, expected.tolist(), memoryview(expected),
                np.ma.array(expected, mask=False),
                [expected[0], row], UserList([expected[0], row])):
            with self.subTest(form=type(centerline)):
                fiber = Fiber(centerline)
                self.assertEqual(type(fiber.centerline), np.ndarray)
                self.assertEqual(fiber.centerline.dtype, np.float64)
                np.testing.assert_array_equal(fiber.centerline, expected)
        endpoints = np.ma.array(expected.ravel(), mask=False)
        np.testing.assert_array_equal(Fiber(endpoints, segments=2).centerline,
                                      [[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]])

    def test_numpy_array_protocol_keeps_precedence_without_sequence_traversal(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]])

        class NoSequenceAccess(Sequence):
            def __len__(self):
                raise AssertionError("conversion must not ask for sequence length")

            def __getitem__(self, index):
                raise AssertionError("conversion must not read a sequence entry")

        class DuckArray:
            def __array__(self, dtype=None, copy=None):
                return np.asarray(points, dtype=dtype)

        class DualArray(NoSequenceAccess, DuckArray):
            pass

        class DualInterface(NoSequenceAccess):
            reads = 0

            @property
            def __array_interface__(self):
                self.reads += 1
                return points.__array_interface__

        class DynamicArray(NoSequenceAccess):
            def __getattr__(self, name):
                if name == '__array__':
                    return DuckArray().__array__
                raise AttributeError(name)

        for form in (DuckArray, DualArray, DualInterface, DynamicArray):
            with self.subTest(form=form):
                numpy_input, fiber_input = form(), form()
                expected = np.asarray(numpy_input, dtype=np.float64)
                np.testing.assert_array_equal(Fiber(fiber_input).centerline, expected)
                if isinstance(numpy_input, DualInterface):
                    self.assertEqual(fiber_input.reads, numpy_input.reads)

    def test_public_legacy_vtp_output_uses_observed_points_unchanged(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]])
        with TemporaryDirectory() as tmp:
            a, b = [Path(tmp) / name for name in ('plain.vtp', 'observed.vtp')]
            export(FiberNetwork(fibers=[Fiber(points)]), str(a))
            export(FiberNetwork(fibers=[Fiber(np.ma.array(points, mask=False))]), str(b))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            actual = ET.parse(b).find('.//Points/DataArray').text.split()
            np.testing.assert_array_equal(list(map(float, actual)), points.ravel())
