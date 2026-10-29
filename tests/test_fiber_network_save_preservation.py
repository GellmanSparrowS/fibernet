"""Legacy network serialization-failure preservation, not full I/O atomicity."""
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fibernet import Fiber, FiberNetwork


class FiberNetworkSavePreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'network.json'
        self.network = FiberNetwork(fibers=[Fiber(centerline=np.array([[0., 0., 0.], [1., 0., 0.]]), radius=.2)])

    def test_normal_formatted_bytes_and_roundtrip_unchanged(self):
        data = self.network.to_dict()
        self.network.save_json(self.path)
        reference = self.root / 'legacy-reference.json'
        # Use the original writer's platform encoding/newline behavior, then
        # compare raw bytes: this must not normalize away any format change.
        with open(reference, 'w') as stream:
            json.dump(data, stream, indent=2)
        self.assertEqual(self.path.read_bytes(), reference.read_bytes())
        self.assertFalse(self.path.read_bytes().endswith(b'\n'))
        self.assertEqual(FiberNetwork.load_json(self.path).to_dict(), data)

    def test_existing_good_file_preserved_on_metadata_serialization_failure(self):
        self.network.save_json(self.path)
        before = self.path.read_bytes()
        self.network.metadata['unsupported'] = np.array([1])
        with self.assertRaisesRegex(TypeError, 'not JSON serializable'):
            self.network.save_json(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(FiberNetwork.load_json(self.path).num_fibers, 1)

    def test_new_file_absent_on_metadata_serialization_failure(self):
        self.network.metadata['unsupported'] = np.array([1])
        with self.assertRaises(TypeError):
            self.network.save_json(self.path)
        self.assertFalse(self.path.exists())

    def test_existing_file_preserved_on_cyclic_metadata(self):
        self.path.write_bytes(b'previous-good-file')
        cyclic = []
        cyclic.append(cyclic)
        self.network.metadata['cycle'] = cyclic
        with self.assertRaises(ValueError):
            self.network.save_json(self.path)
        self.assertEqual(self.path.read_bytes(), b'previous-good-file')

    def test_parent_directory_policy_unchanged(self):
        path = self.root / 'missing-parent' / 'network.json'
        with self.assertRaises(FileNotFoundError):
            self.network.save_json(path)
        self.assertFalse(path.parent.exists())

    def test_existing_inode_permissions_and_symlink_behavior_unchanged(self):
        self.path.write_bytes(b'old')
        self.path.chmod(0o640)
        before = self.path.stat()
        link = self.root / 'link.json'
        link.symlink_to(self.path)
        self.network.save_json(link)
        after = self.path.stat()
        self.assertTrue(link.is_symlink())
        self.assertEqual(before.st_ino, after.st_ino)
        self.assertEqual(stat.S_IMODE(after.st_mode), stat.S_IMODE(before.st_mode))
        self.assertEqual(FiberNetwork.load_json(self.path).num_fibers, 1)

    def test_write_failure_still_propagates_not_claimed_atomic(self):
        class BrokenOutput:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def write(self, value):
                raise OSError('simulated write failure')
        with patch('builtins.open', return_value=BrokenOutput()):
            with self.assertRaisesRegex(OSError, 'simulated write failure'):
                self.network.save_json(self.path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
