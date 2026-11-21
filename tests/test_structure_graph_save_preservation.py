"""Serialization-failure preservation; this is not full I/O atomicity."""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fibernet import Material, StructureGraph


class SavePreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'graph.json'
        self.graph = StructureGraph()
        a = self.graph.add_node([0., 0.])
        b = self.graph.add_node([1., 0.])
        self.graph.add_edge(a, b)

    def test_normal_formatted_bytes_and_roundtrip_unchanged(self):
        data = self.graph.to_dict()
        self.graph.save_json(self.path)
        reference = self.root / 'legacy-reference.json'
        # Use the original writer's platform encoding/newline behavior, then
        # compare raw bytes: this must not normalize away any format change.
        with open(reference, 'w') as stream:
            json.dump(data, stream, indent=2)
        self.assertEqual(self.path.read_bytes(), reference.read_bytes())
        self.assertFalse(self.path.read_bytes().endswith(b'\n'))
        self.assertEqual(StructureGraph.load_json(self.path).to_dict(), data)

    def test_numpy_segment_counts_match_native_integer_json(self):
        graph = StructureGraph.from_dict({
            'dimension': 3,
            'nodes': [{'id': 4, 'position': [0., 0., 0.]},
                      {'id': 23, 'position': [1., 0., 0.]}],
            'edges': [{'id': 11, 'node_i': 4, 'node_j': 23,
                       'radius': 0.1, 'segments': 3},
                      {'id': 49, 'node_i': 23, 'node_j': 4,
                       'radius': 0.2, 'segments': 2**53 + 1}],
        })
        unsigned = np.uint64(2**53 + 1)
        # Use public ingestion with a parallel edge as well as existing IDs.
        extra = graph.add_edge(4, 23, radius=0.3, segments=np.int64(5),
                               allow_parallel=True)
        expected = graph.to_dict()
        expected['edges'][-1]['segments'] = 5
        reference = self.root / 'legacy-reference.json'
        with open(reference, 'w') as stream:
            json.dump(expected, stream, indent=2)
        graph.edges[11].segments = np.int64(3)
        graph.edges[49].segments = unsigned

        graph.save_json(self.path)

        self.assertEqual(self.path.read_bytes(), reference.read_bytes())
        restored = StructureGraph.load_json(self.path)
        self.assertEqual(restored.to_dict(), expected)
        self.assertIs(type(restored.edges[49].segments), int)
        self.assertIs(type(graph.edges[11].segments), np.int64)
        self.assertIs(type(graph.edges[extra].segments), np.int64)
        self.assertIs(graph.edges[49].segments, unsigned)
        self.assertIs(type(graph.to_dict()['edges'][0]['segments']), int)
        # Detached records used in graph identity hashes match after reloading.
        self.assertEqual(json.dumps(graph.to_dict(), sort_keys=True, allow_nan=False),
                         json.dumps(restored.to_dict(), sort_keys=True, allow_nan=False))

    def test_only_supported_counts_in_the_dedicated_field_are_coerced(self):
        self.path.write_bytes(b'previous-good-file')
        edge = self.graph.edges[0]
        for value in (np.bool_(True), np.timedelta64(7, 'ns'),
                      np.int64(0), np.int64(-1), np.uint64(2**63)):
            with self.subTest(value=value):
                edge.segments = value
                with self.assertRaisesRegex(TypeError, 'not JSON serializable'):
                    self.graph.save_json(self.path)
                self.assertEqual(self.path.read_bytes(), b'previous-good-file')
                self.assertIs(edge.segments, value)
        edge.segments = True
        self.graph.save_json(self.path)
        self.assertIs(json.loads(self.path.read_text())['edges'][0]['segments'], True)
        previous = self.path.read_bytes()
        edge.segments = np.int64(3)
        edge.metadata['segments'] = np.int64(5)
        with self.assertRaisesRegex(TypeError, 'not JSON serializable'):
            self.graph.save_json(self.path)
        self.assertEqual(self.path.read_bytes(), previous)

    def test_existing_good_file_unchanged_on_metadata_serialization_failure(self):
        self.graph.save_json(self.path)
        before = self.path.read_bytes()
        self.graph.metadata['unsupported'] = np.array([1, 2])
        with self.assertRaisesRegex(TypeError, 'not JSON serializable'):
            self.graph.save_json(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(StructureGraph.load_json(self.path).num_edges, 1)

    def test_new_file_absent_on_metadata_serialization_failure(self):
        self.graph.nodes[0].metadata['unsupported'] = np.array([1])
        with self.assertRaises(TypeError):
            self.graph.save_json(self.path)
        self.assertFalse(self.path.exists())

    def test_existing_file_unchanged_when_to_dict_rejects_material(self):
        self.path.write_bytes(b'previous-good-file')
        self.graph.edges[0].material = object()
        with self.assertRaisesRegex(ValueError, 'Material instance'):
            self.graph.save_json(self.path)
        self.assertEqual(self.path.read_bytes(), b'previous-good-file')

    def test_material_error_does_not_create_new_file(self):
        self.graph.edges[0].material = object()
        with self.assertRaises(ValueError):
            self.graph.save_json(self.path)
        self.assertFalse(self.path.exists())

    def test_parent_directory_policy_unchanged(self):
        path = self.root / 'missing-parent' / 'graph.json'
        with self.assertRaises(FileNotFoundError):
            self.graph.save_json(path)
        self.assertFalse(path.parent.exists())

    def test_existing_inode_permissions_and_symlink_behavior_unchanged(self):
        self.path.write_bytes(b'old')
        self.path.chmod(0o640)
        before = self.path.stat()
        link = self.root / 'link.json'
        link.symlink_to(self.path)
        self.graph.save_json(link)
        after = self.path.stat()
        self.assertTrue(link.is_symlink())
        self.assertEqual(before.st_ino, after.st_ino)
        self.assertEqual(stat.S_IMODE(after.st_mode), stat.S_IMODE(before.st_mode))
        self.assertEqual(StructureGraph.load_json(self.path).num_edges, 1)

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
                self.graph.save_json(self.path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
