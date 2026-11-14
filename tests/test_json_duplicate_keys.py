"""Native JSON readers reject ambiguous keys before constructing an object."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fibernet._json import unique_object_pairs
from fibernet import api, easy
from fibernet.core.network import FiberNetwork
from fibernet.core.structure_graph import StructureGraph
from fibernet.utils.config import ExperimentConfig


class NativeJsonDuplicateKeyTests(unittest.TestCase):
    routes = ('config', 'graph', 'network', 'api', 'batch_bytes')

    def invoke(self, route, raw, directory):
        if route == 'batch_bytes':
            return easy._batch_graph_from_bytes(raw)
        path = Path(directory) / 'document.json'
        path.write_bytes(raw)
        readers = {'config': ExperimentConfig.from_json,
                   'graph': StructureGraph.load_json,
                   'network': FiberNetwork.load_json,
                   'api': api.load}
        try:
            return readers[route](path)
        finally:
            self.assertEqual(path.read_bytes(), raw)

    def test_duplicates_reject_before_any_constructor(self):
        cases = (
            ('{"x":1,"x":2}', 'x'),
            ('{"x":1,"x":1}', 'x'),
            ('{"metadata":{"tag":1,"tag":2}}', 'tag'),
            ('{"metadata":{"rows":[{"tag":1,"tag":1}]}}', 'tag'),
            ('{"a":1,"\\u0061":2}', 'a'),
        )
        with tempfile.TemporaryDirectory() as directory:
            for route in self.routes:
                for text, key in cases:
                    with self.subTest(route=route, document=text):
                        with patch.object(ExperimentConfig, 'from_dict') as config_ctor, \
                             patch.object(StructureGraph, 'from_dict') as graph_ctor, \
                             patch.object(FiberNetwork, 'from_dict') as network_ctor:
                            for constructor in (config_ctor, graph_ctor, network_ctor):
                                constructor.side_effect = AssertionError('constructor reached')
                            with self.assertRaises(ValueError) as caught:
                                self.invoke(route, text.encode('utf-8'), directory)
                            self.assertIs(type(caught.exception), ValueError)
                            self.assertEqual(str(caught.exception), f'duplicate JSON object key: {key!r}')
                            for constructor in (config_ctor, graph_ctor, network_ctor):
                                constructor.assert_not_called()

    def test_valid_legacy_graph_preserves_ids_parallel_orientation_and_curve(self):
        record = {'nodes': [{'id': 5, 'position': [0, 0, 0]},
                            {'id': 9, 'position': [1, 0, 0]}],
                  'edges': [{'id': 11, 'node_i': 5, 'node_j': 9, 'radius': .1,
                             'internal_points': [[.5, .2, 0]]},
                            {'id': 14, 'node_i': 9, 'node_j': 5, 'radius': .2}],
                  'metadata': {'first': {'tag': 1}, 'second': {'tag': 2}}}
        raw = json.dumps(record).encode('utf-8')
        expected = StructureGraph.from_dict(record).to_dict()
        with tempfile.TemporaryDirectory() as directory:
            for route in ('graph', 'api', 'batch_bytes'):
                with self.subTest(route=route):
                    result = self.invoke(route, raw, directory)
                    self.assertIs(type(result), StructureGraph)
                    self.assertEqual(result.to_dict(), expected)
                    self.assertEqual(sorted(result.nodes), [5, 9])
                    self.assertEqual(sorted(result.edges), [11, 14])
                    self.assertEqual((result.edges[14].node_i, result.edges[14].node_j), (9, 5))
                    self.assertEqual(result.edges[11].internal_points.tolist(), [[.5, .2, 0]])
                    self.assertEqual(result.to_dict()['next_node_id'], 10)
                    self.assertEqual(result.to_dict()['next_edge_id'], 15)

    def test_valid_config_and_legacy_network_dispatch(self):
        config = {'experiment': {'name': 'example'}, 'network': {'parameters': {'cell_size': .01}},
                  'metadata': {'first': {'tag': 1}, 'second': {'tag': 2}}}
        network = {'fibers': [], 'crosslinks': [], 'dimension': 2,
                   'metadata': {'first': {'tag': 1}, 'second': {'tag': 2}}}
        with tempfile.TemporaryDirectory() as directory:
            loaded = self.invoke('config', json.dumps(config).encode(), directory)
            self.assertEqual(loaded.experiment, config['experiment'])
            self.assertEqual(loaded.network, config['network'])
            self.assertEqual(loaded.metadata['first'], {'tag': 1})
            self.assertEqual(loaded.metadata['source_file'], str((Path(directory) / 'document.json').absolute()))
            for route in ('network', 'api'):
                with self.subTest(route=route):
                    loaded = self.invoke(route, json.dumps(network).encode(), directory)
                    self.assertIs(type(loaded), FiberNetwork)
                    self.assertEqual(loaded.to_dict(), FiberNetwork.from_dict(network).to_dict())

    def test_malformed_json_and_missing_files_keep_error_types(self):
        with tempfile.TemporaryDirectory() as directory:
            for route in self.routes:
                with self.subTest(route=route, error='syntax'):
                    with self.assertRaises(json.JSONDecodeError):
                        self.invoke(route, b'{"x":', directory)
            missing = Path(directory) / 'missing.json'
            for reader in (ExperimentConfig.from_json, StructureGraph.load_json,
                           FiberNetwork.load_json, api.load):
                with self.subTest(reader=reader.__qualname__, error='missing'):
                    with self.assertRaises(FileNotFoundError):
                        reader(missing)

    def test_pairs_hook_preserves_plain_dict_order_and_decoder_constants(self):
        self.assertIn('fibernet/_json.py', easy._BATCH_SOURCE_FILES)
        value = json.loads('{"z":0,"a":{"x":1},"b":{"x":2}}',
                           object_pairs_hook=unique_object_pairs)
        self.assertIs(type(value), dict)
        self.assertEqual(list(value), ['z', 'a', 'b'])
        self.assertEqual(value['a'], {'x': 1})
        self.assertEqual(value['b'], {'x': 2})
        # Duplicate-key validation does not add a nonfinite-number policy.
        value = json.loads('{"value":Infinity}', object_pairs_hook=unique_object_pairs)
        self.assertEqual(value['value'], float('inf'))
        self.assertEqual(json.loads('{}', object_pairs_hook=unique_object_pairs), {})


if __name__ == '__main__':
    unittest.main()
