"""Binary STL representation checks without native solid generation.

Run: python -m unittest discover -s tests -p test_solid_export_precision.py
"""
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

# Load the standalone exporter, without importing optional simulation engines.
SOURCE = Path(__file__).resolve().parents[1] / 'fibernet/gen/solid_export.py'
spec = importlib.util.spec_from_file_location('binary_stl_export_under_test', SOURCE)
exporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)


def tetra(vertices=None):
    if vertices is None:
        vertices = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    return exporter.FiberSolid(np.array(vertices, dtype=np.float64),
                              np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
                              np.zeros((0, 3)), .1, exporter.PrintSettings())


class BinaryStlTests(unittest.TestCase):
    def refused(self, solid, message):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mesh.stl'
            path.write_bytes(b'existing destination')
            with patch.object(exporter, '_owned_export_file', side_effect=AssertionError('staged too soon')):
                with self.assertRaisesRegex(ValueError, message):
                    exporter.export_solid(path, solid)
            self.assertEqual(path.read_bytes(), b'existing destination')
            self.assertEqual(list(Path(folder).iterdir()), [path])
            path.unlink()
            with self.assertRaises(ValueError):
                exporter.export_solid(path, solid)
            self.assertFalse(path.exists())

    def test_round5_misses_binary32_collision(self):
        solid = tetra([[100.0000049, 0., 0.], [100.0000051, 0., 0.],
                       [100., 1., 0.], [100., 0., 1.]])
        self.assertEqual(len(np.unique(np.round(solid.vertices, 5), axis=0)), 4)
        self.assertGreater(solid.validate(), 0)
        self.refused(solid, 'binary STL serialization.*distinct vertices')

    def test_collinearity_without_coordinate_collision(self):
        solid = tetra([[0., 0., 0.], [1e20, 0., 0.], [0., 1e20, 0.], [2e20, 0., 1e-46]])
        self.assertEqual(len(np.unique(solid.vertices.astype('<f4'), axis=0)), 4)
        self.assertGreater(solid.validate(), 0)
        self.refused(solid, 'binary STL serialization.*zero area')

    def test_welded_edge_closure(self):
        a = tetra()
        b = tetra([[1e-9, 0., 0.], [1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
        # Translation makes both near-edge endpoints round to the same positions.
        a.vertices += 100
        b.vertices += 100
        a.vertices = np.vstack((a.vertices, b.vertices))
        a.faces = np.vstack((a.faces, b.faces + 4))
        self.assertGreater(a.validate(), 0)
        self.refused(a, 'binary STL serialization.*closed two-manifold')

    def test_volume_loss_without_collapsed_faces(self):
        solid = tetra([[0., 0., 0.], [1e20, 0., 0.], [0., 1e20, 0.], [1e20, 1e20, 1e-46]])
        self.assertGreater(solid.validate(), 0)
        self.refused(solid, 'binary STL serialization.*no volume')

    def test_disconnected_closed_components_allowed(self):
        solid = tetra()
        solid.vertices = np.vstack((solid.vertices, solid.vertices + 3))
        solid.faces = np.vstack((solid.faces, solid.faces + 4))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'two.stl'
            exporter.export_solid(path, solid)
            self.assertEqual(path.stat().st_size, 84 + 8 * 50)

    def test_overflow(self):
        solid = tetra()
        solid.vertices *= 1e39
        self.refused(solid, 'binary STL coordinates overflow')

    def test_invalid_source(self):
        for value in (float('nan'), float('inf')):
            solid = tetra()
            solid.vertices[0, 0] = value
            self.refused(solid, 'finite')
        for faces in (np.array([[0, 1, 9]]), np.array([[0., 1., 2.]]),
                      np.array([[0, 0, 1]]), np.array([[0, 1, 2]])):
            solid = tetra()
            solid.faces = faces
            self.refused(solid, 'indices|distinct|closed')

    def test_valid_bytes_and_signed_zero(self):
        solid = tetra()
        solid.vertices[0, 0] = -0.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mesh.STL'
            exporter.export_solid(path, solid)
            data = path.read_bytes()
            self.assertEqual(len(data), 84 + 50 * 4)
            self.assertEqual(struct.unpack('<I', data[80:84])[0], 4)
            records = np.frombuffer(data[84:], dtype=[('normal', '<f4', (3,)),
                                    ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
            self.assertEqual(records['vertices'].tobytes(), solid.vertices[solid.faces].astype('<f4').tobytes())
            self.assertTrue(np.isfinite(records['normal']).all())
            positions = records['vertices'].astype(np.float64)
            expected = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
            expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-15)
            np.testing.assert_array_equal(records['normal'], expected.astype('<f4'))
            self.assertTrue(np.all(records['attribute'] == 0))

    def test_large_finite_normals(self):
        solid = tetra()
        solid.vertices *= 1e30
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mesh.stl'
            exporter.export_solid(path, solid)
            data = path.read_bytes()
            normal = struct.unpack('<3f', data[84:96])
            self.assertTrue(np.isfinite(normal).all())
            self.assertAlmostEqual(np.linalg.norm(normal), 1.)

    def test_3mf_does_not_use_binary32_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mesh.3mf'
            with patch.object(exporter, '_binary_stl_vertices', side_effect=AssertionError('STL-only')):
                exporter.export_solid(path, tetra())
            self.assertTrue(path.read_bytes().startswith(b'PK'))


if __name__ == '__main__':
    unittest.main()
