"""Binary STL representation checks without native solid generation.

Run: python -m unittest discover -s tests -p test_solid_export_precision.py
"""
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET
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


class ThreeMfPrecisionTests(unittest.TestCase):
    NS = '{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}'

    def roundtrip(self, solid):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mesh.3mf'
            exporter.export_solid(path, solid)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()),
                                 {'[Content_Types].xml', '_rels/.rels', '3D/3dmodel.model'})
                self.assertIsNone(archive.testzip())
                model = ET.fromstring(archive.read('3D/3dmodel.model'))
                relationships = ET.fromstring(archive.read('_rels/.rels'))
                self.assertEqual(relationships[0].get('Target'), '/3D/3dmodel.model')
                types = ET.fromstring(archive.read('[Content_Types].xml'))
                self.assertTrue(any(item.get('Extension') == 'model' and
                                    item.get('ContentType') ==
                                    'application/vnd.ms-package.3dmanufacturing-3dmodel+xml'
                                    for item in types))
        ns = self.NS
        self.assertEqual(model.get('unit'), 'millimeter')
        objects = model.findall(ns + 'resources/' + ns + 'object')
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].get('id'), '1')
        self.assertEqual(objects[0].get('type'), 'model')
        items = model.findall(ns + 'build/' + ns + 'item')
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].get('objectid'), '1')
        self.assertIsNone(items[0].get('transform'))
        self.assertEqual(model.findall('.//' + ns + 'metadata'), [])
        vertices = objects[0].findall(ns + 'mesh/' + ns + 'vertices/' + ns + 'vertex')
        faces = objects[0].findall(ns + 'mesh/' + ns + 'triangles/' + ns + 'triangle')
        coordinates = np.array([[float(v.get(k)) for k in ('x', 'y', 'z')]
                                for v in vertices], dtype=np.float64)
        indices = np.array([[int(f.get(k)) for k in ('v1', 'v2', 'v3')]
                            for f in faces], dtype=np.int64)
        np.testing.assert_array_equal(coordinates.view(np.uint64),
                                      solid.vertices.astype(np.float64).view(np.uint64))
        np.testing.assert_array_equal(indices, solid.faces)
        return coordinates, indices, vertices

    def test_binary64_distinct_vertices_collide_under_nine_digits(self):
        solid = tetra([[1.0000000001, 0., 0.], [1.0000000002, 0., 0.],
                       [1., 100., 0.], [1., 0., 100.]])
        self.assertEqual('%.9g' % solid.vertices[0, 0],
                         '%.9g' % solid.vertices[1, 0])
        coordinates, _, _ = self.roundtrip(solid)
        self.assertNotEqual(coordinates[0, 0], coordinates[1, 0])

    def test_binary64_signed_zero_exponents_and_adjacent_values(self):
        solid = tetra([[0., -0., 1e-100], [np.nextafter(1., 2.), 0., 0.],
                       [0., 1., 0.], [0., 0., np.pi]])
        _, _, vertices = self.roundtrip(solid)
        self.assertIn('e', vertices[0].get('z'))
        self.assertEqual(vertices[0].get('y'), '-0')

    def test_narrow_float_and_exact_integer_storage(self):
        for dtype in (np.float16, np.float32, np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                solid = tetra()
                solid.vertices = solid.vertices.astype(dtype)
                self.roundtrip(solid)

    def test_existing_negative_volume_orientation_semantics(self):
        solid = tetra()
        solid.faces = solid.faces[:, [0, 2, 1]]
        original = solid.faces.copy()
        _, indices, _ = self.roundtrip(solid)
        np.testing.assert_array_equal(indices, original[:, [0, 2, 1]])

    def test_two_disconnected_index_components_remain_separate(self):
        first, second = tetra(), tetra()
        second.vertices += 3.
        solid = exporter.FiberSolid(np.vstack([first.vertices, second.vertices]),
                                    np.vstack([first.faces, second.faces + 4]),
                                    np.zeros((0, 3)), .1, exporter.PrintSettings())
        _, indices, _ = self.roundtrip(solid)
        adjacency = {i: set() for i in range(len(solid.vertices))}
        for face in indices:
            for vertex in face:
                adjacency[int(vertex)].update(map(int, face))
        components = []
        unseen = set(adjacency)
        while unseen:
            pending = [min(unseen)]
            component = set()
            while pending:
                vertex = pending.pop()
                if vertex in component:
                    continue
                component.add(vertex)
                pending.extend(adjacency[vertex] - component)
            unseen -= component
            components.append(component)
        self.assertEqual(components, [set(range(4)), set(range(4, 8))])


if __name__ == '__main__':
    unittest.main()
