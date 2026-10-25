"""Provider-free native JSON dispatch and export-selection regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fibernet import Fiber, FiberNetwork, Material, StructureGraph, api


class NativeIOTests(unittest.TestCase):
    def test_import_helpers_are_private(self):
        self.assertNotIn("json", vars(api))
        self.assertNotIn("StructureGraph", vars(api))
        self.assertIs(api._StructureGraph, StructureGraph)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'native.json'

    def write_json(self, value):
        self.path.write_text(json.dumps(value))
        return self.path

    def graph(self):
        graph = StructureGraph(dimension=2, box_size=[4., 5.])
        graph.metadata.update(length_unit='mm', coordinate_frame='lab', nested={'a': [1, 2]})
        gone = graph.add_node([-1., 0.])
        a = graph.add_node([0., 0.], boundary=(True, False, False, False, False, False))
        b = graph.add_node([2., 0.])
        graph.remove_node(gone)
        deleted = graph.add_edge(a, b)
        graph.remove_edge(deleted)
        material = Material(name='test', youngs_modulus=1234., density=99., extra={'units': 'SI'})
        graph.add_edge(a, b, radius=.25, material=material,
                       internal_points=np.array([[1., .1, 0.]]), segments=7, provenance='first')
        graph.add_edge(b, a, radius=.15, material=material, allow_parallel=True, provenance='second')
        return graph

    def test_graph_roundtrip_preserves_all_native_fields(self):
        graph = self.graph()
        expected = graph.to_dict()
        self.assertEqual(api.export(graph, self.path), self.path)
        loaded = api.load(self.path)
        self.assertIsInstance(loaded, StructureGraph)
        self.assertEqual(loaded.to_dict(), expected)
        self.assertEqual(loaded.num_edges, 2)
        self.assertEqual(loaded.add_node([3., 0.]), expected['next_node_id'])
        self.assertEqual(loaded.add_edge(1, 2, allow_parallel=True), expected['next_edge_id'])
        loaded.metadata['nested']['a'].append(3)
        self.assertEqual(graph.metadata['nested']['a'], [1, 2])

    def test_graph_legacy_versions(self):
        for version in (None, '3.0', '3.1'):
            data = self.graph().to_dict()
            if version is None:
                del data['version']
            else:
                data['version'] = version
            loaded = api.load(self.write_json(data))
            self.assertEqual(list(loaded.nodes), [1, 2])
            self.assertEqual(list(loaded.edges), [1, 2])

    def test_empty_graph_is_not_empty_network(self):
        api.export(StructureGraph(), self.path)
        loaded = api.load(self.path)
        self.assertIsInstance(loaded, StructureGraph)
        self.assertEqual(loaded.num_nodes, 0)

    def test_fiber_network_roundtrip(self):
        fiber = Fiber(centerline=np.array([[0., 0., 0.], [1., 0., 0.]]), radius=.2)
        network = FiberNetwork(fibers=[fiber], dimension=3, metadata={'length_unit': 'm'})
        expected = network.to_dict()
        api.export(network, self.path)
        loaded = api.load(self.path)
        self.assertIsInstance(loaded, FiberNetwork)
        self.assertEqual(loaded.to_dict(), expected)

    def test_empty_and_versionless_legacy_network(self):
        data = FiberNetwork().to_dict()
        for versioned in (True, False):
            if not versioned:
                del data['version']
                del data['crosslinks']
            loaded = api.load(self.write_json(data))
            self.assertIsInstance(loaded, FiberNetwork)
            self.assertEqual(loaded.num_fibers, 0)

    def test_mixed_schema_rejected(self):
        for payload in ({'nodes': [], 'edges': [], 'fibers': []},
                        {'nodes': [], 'edges': [], 'crosslinks': []}):
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                api.load(self.write_json(payload))

    def test_unknown_shape_rejected(self):
        for payload in ({}, {'version': '3.1'}, [], None, {'unrelated': [1]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                api.load(self.write_json(payload))

    def test_malformed_arrays_rejected(self):
        for payload in ({'nodes': [], 'edges': None}, {'nodes': []}, {'edges': []},
                        {'fibers': None}, {'crosslinks': []}, {'fibers': [], 'crosslinks': {}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                api.load(self.write_json(payload))

    def test_graph_unknown_version_uses_native_reader_policy(self):
        payload = {'version': '99', 'nodes': [], 'edges': []}
        with self.assertRaisesRegex(ValueError, 'unsupported.*schema version'):
            api.load(self.write_json(payload))

    def test_legacy_network_header_extensions_follow_native_reader(self):
        for version in ('0.0.1', '0.1.0', 'legacy-external'):
            payload = FiberNetwork().to_dict()
            payload.update(version=version, schema='application/1', schema_version='v2')
            expected = FiberNetwork.from_dict(payload)
            loaded = api.load(self.write_json(payload))
            self.assertIsInstance(loaded, FiberNetwork)
            self.assertEqual(loaded.to_dict(), expected.to_dict())

    def test_graph_header_extensions_follow_native_reader(self):
        payload = self.graph().to_dict()
        payload.update(schema='application/1', schema_version='v2')
        expected = StructureGraph.from_dict(payload)
        loaded = api.load(self.write_json(payload))
        self.assertEqual(loaded.to_dict(), expected.to_dict())

    def test_nested_metadata_schema_remains_ordinary_data(self):
        for value in (StructureGraph(), FiberNetwork()):
            value.metadata.update(schema='application/1', schema_version='v2')
            api.export(value, self.path)
            loaded = api.load(self.path)
            self.assertIs(type(loaded), type(value))
            self.assertEqual(loaded.metadata, value.metadata)

    def test_graph_validation_remains_authoritative(self):
        data = self.graph().to_dict()
        data['edges'][1]['id'] = data['edges'][0]['id']
        with self.assertRaisesRegex(ValueError, 'duplicate edge ID'):
            api.load(self.write_json(data))

    def test_unknown_export_extension_does_not_create_or_overwrite(self):
        for name in ('mesh.stl', 'mesh.vtu', 'typo.josn', 'no_suffix'):
            target = self.root / name
            with self.assertRaisesRegex(ValueError, 'Cannot infer export format'):
                api.export(FiberNetwork(), target)
            self.assertFalse(target.exists())
            target.write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError, 'Cannot infer export format'):
                api.export(self.graph(), target)
            self.assertEqual(target.read_bytes(), b'keep')

    def test_unknown_load_extension_rejected_before_open(self):
        with self.assertRaisesRegex(ValueError, 'Cannot infer load format'):
            api.load(self.root / 'absent.josn')

    def test_explicit_json_overrides_extension(self):
        target = self.root / 'custom.datax'
        api.export(self.graph(), target, format='json')
        self.assertIsInstance(api.load(target, format='json'), StructureGraph)

    def test_non_json_graph_export_fails_before_writing(self):
        for suffix in ('vtk', 'vtp', 'pdb', 'xyz', 'lammps', 'msh'):
            target = self.root / ('graph.' + suffix)
            with self.assertRaisesRegex(ValueError, 'StructureGraph supports only JSON'):
                api.export(self.graph(), target)
            self.assertFalse(target.exists())

    def test_invalid_export_input_is_actionable(self):
        with self.assertRaisesRegex(TypeError, 'FiberNetwork or StructureGraph'):
            api.export(object(), self.path)
        self.assertFalse(self.path.exists())

    def test_unknown_explicit_formats_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown format'):
            api.export(FiberNetwork(), self.path, format='nonsense')
        with self.assertRaisesRegex(ValueError, 'Cannot load format'):
            api.load(self.path, format='nonsense')
        self.assertFalse(self.path.exists())

    def test_existing_non_json_readers_still_dispatch(self):
        marker = object()
        with patch('fibernet.io.pdb.from_pdb', return_value=marker) as reader:
            self.assertIs(api.load('existing.pdb'), marker)
            reader.assert_called_once_with('existing.pdb')
        with patch('fibernet.io.lammps.from_lammps', return_value=marker) as reader:
            self.assertIs(api.load('existing.data'), marker)
            reader.assert_called_once_with('existing.data')


if __name__ == '__main__':
    unittest.main(verbosity=2)
