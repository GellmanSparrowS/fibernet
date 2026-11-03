import tempfile
import unittest
from pathlib import Path
import numpy as np
from fibernet.io.xyz import from_xyz, to_xyz
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork

class TestStrictXYZ(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'points.xyz'
    def read(self, data, **kwargs):
        self.path.write_text(data)
        return from_xyz(self.path, **kwargs)
    def test_valid_first_frame_and_empty(self):
        data = '2\ncomment\nC 1 2 3 extra\nO -1 0 4\n1\nnext\nC 9 9 9\n'
        np.testing.assert_array_equal(self.read(data, strict=True), [[1,2,3],[-1,0,4]])
        self.assertEqual(self.read('0\n\n', strict=True).shape, (0,3))
    def test_invalid(self):
        for data in ('', '1\n', '-1\ncomment\n', 'bad\ncomment\n',
                     '2\ncomment\nC 1 2 3\n', '1\ncomment\nC 1 2\n',
                     '1\ncomment\nC invalid 2 3\n', '1\ncomment\nC nan 2 3\n',
                     '1\ncomment\nC inf 2 3\n'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.read(data, strict=True)
    def test_default_unchanged(self):
        data = '3\ncomment\nC 1 2 3\nC 1 2\n'
        np.testing.assert_array_equal(self.read(data), [[1,2,3]])
        with self.assertRaises(ValueError): self.read(data, strict=True)
    def test_real_export_roundtrip(self):
        fiber = Fiber([[0., 0., 0.], [1., 0., 0.]], radius=.05)
        to_xyz(FiberNetwork(fibers=[fiber]), self.path, bead_spacing=2.)
        np.testing.assert_array_equal(from_xyz(self.path, strict=True), fiber.centerline)

if __name__ == '__main__': unittest.main()
