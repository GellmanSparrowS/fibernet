"""Focused regression for duplicate JSON keys introduced by key coercion."""
import tempfile
import unittest
from pathlib import Path

from fibernet.core.structure_graph import StructureGraph


class TestStructureGraphJsonSaveKeys(unittest.TestCase):
    def graph(self):
        graph = StructureGraph(dimension=2)
        first = graph.add_node([0, 0])
        second = graph.add_node([1, 0])
        edge = graph.add_edge(first, second)
        return graph, first, edge

    def test_collision_does_not_overwrite_existing_file(self):
        for scope in ('graph', 'node', 'edge'):
            for nested in (False, True):
                with self.subTest(scope=scope, nested=nested), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'graph.json'
                    graph, node, edge = self.graph()
                    graph.save_json(path)
                    original = path.read_bytes()
                    metadata = {'graph': graph.metadata,
                                'node': graph.nodes[node].metadata,
                                'edge': graph.edges[edge].metadata}[scope]
                    collision = {1: 'integer-key', '1': 'string-key'}
                    metadata.update({'nested': [collision]} if nested else collision)
                    with self.assertRaisesRegex(ValueError, 'duplicate JSON object key'):
                        graph.save_json(path)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(StructureGraph.load_json(path).to_dict(),
                                     self.graph()[0].to_dict())

    def test_collision_does_not_create_new_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json'
            graph, _, _ = self.graph()
            graph.metadata.update({1: 'integer-key', '1': 'string-key'})
            with self.assertRaisesRegex(ValueError, 'duplicate JSON object key'):
                graph.save_json(path)
            self.assertFalse(path.exists())

    def test_valid_metadata_roundtrip_preserves_existing_key_coercion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json'
            graph, node, edge = self.graph()
            graph.metadata.update({'name': 'valid', 'nested': [{'flag': True, 'value': None}],
                                   1: 'integer-key'})
            graph.nodes[node].metadata['label'] = 'start'
            graph.edges[edge].metadata['tags'] = ['beam', 2]
            graph.save_json(path)
            loaded = StructureGraph.load_json(path)
            expected = graph.to_dict()
            expected['metadata'] = dict(expected['metadata'])
            expected['metadata']['1'] = expected['metadata'].pop(1)
            self.assertEqual(loaded.to_dict(), expected)


if __name__ == '__main__':
    unittest.main()
