"""Typed loader refuses only explicit foreign graph arrays; no schema rewrite."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from fibernet import Crosslink, Fiber, FiberNetwork, StructureGraph, api


class TypedLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'native.json'

    def write(self, payload):
        with open(self.path, 'w') as stream:
            json.dump(payload, stream, indent=2)
        return self.path

    def graph(self):
        graph = StructureGraph(dimension=2)
        graph.metadata.update(length_unit='mm', coordinate_frame='lab')
        unused = graph.add_node([-1., 0.])
        a = graph.add_node([0., 0.])
        b = graph.add_node([1., 0.])
        graph.remove_node(unused)
        graph.add_edge(a, b, radius=.1)
        graph.add_edge(b, a, radius=.2, allow_parallel=True)
        return graph

    def network(self):
        fibers = [Fiber(centerline=np.array([[0., 0., 0.], [1., 0., 0.]]), radius=.1),
                  Fiber(centerline=np.array([[0., 0., 0.], [0., 1., 0.]]), radius=.2)]
        crosslink = Crosslink(fiber_i=0, fiber_j=1, param_i=0., param_j=0.,
                              position=np.array([0., 0., 0.]))
        network = FiberNetwork(dimension=2,
                               metadata={'length_unit': 'mm', 'schema': 'application/7'})
        for fiber in fibers:
            network.add_fiber(fiber)
        network.add_crosslink(crosslink)
        return network

    def test_direct_from_dict_rejects_graph_without_mutating_input(self):
        payload = self.graph().to_dict()
        before = copy.deepcopy(payload)
        with self.assertRaisesRegex(ValueError, r'StructureGraph.*FiberNetwork.*api.load'):
            FiberNetwork.from_dict(payload)
        self.assertEqual(payload, before)

    def test_load_json_rejects_graph_without_changing_file(self):
        self.graph().save_json(self.path)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, r'StructureGraph.load_json'):
            FiberNetwork.load_json(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_empty_and_versionless_graph_records_are_still_foreign(self):
        for version in (None, '3.0', '3.1', 'external-header'):
            payload = StructureGraph().to_dict()
            if version is None:
                del payload['version']
            else:
                payload['version'] = version
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    FiberNetwork.from_dict(payload)
                with self.assertRaises(ValueError):
                    FiberNetwork.load_json(self.write(payload))

    def test_rejection_precedes_network_construction(self):
        class NeverConstruct(FiberNetwork):
            def __init__(self, *args, **kwargs):
                raise AssertionError('foreign record reached constructor')
        with self.assertRaisesRegex(ValueError, 'StructureGraph'):
            NeverConstruct.from_dict(self.graph().to_dict())

    def test_native_empty_and_historical_empty_records_preserved(self):
        for payload in ({}, {'fibers': []}, {'crosslinks': []}, FiberNetwork().to_dict()):
            with self.subTest(payload=payload):
                loaded = FiberNetwork.from_dict(payload)
                self.assertEqual((loaded.num_fibers, loaded.num_crosslinks), (0, 0))
                self.assertIsInstance(FiberNetwork.load_json(self.write(payload)), FiberNetwork)

    def test_normal_legacy_network_roundtrip(self):
        network = self.network()
        data = network.to_dict()
        self.assertEqual(FiberNetwork.from_dict(data).to_dict(), data)
        network.save_json(self.path)
        self.assertEqual(FiberNetwork.load_json(self.path).to_dict(), data)

    def test_legacy_versions_and_header_extensions_remain_tolerated(self):
        for version in (None, '0.0.1', '0.1.0', '3.1', 'external-header'):
            data = self.network().to_dict()
            if version is None:
                del data['version']
            else:
                data['version'] = version
            data.update(schema='external/1', schema_version='unknown', unrelated={'v': 1})
            with self.subTest(version=version):
                loaded = FiberNetwork.from_dict(data)
                self.assertEqual((loaded.num_fibers, loaded.num_crosslinks), (2, 1))
                self.assertEqual(FiberNetwork.load_json(self.write(data)).to_dict(), loaded.to_dict())

    def test_explicit_legacy_family_wins_in_typed_reader_as_before(self):
        for marker in ('fibers', 'crosslinks'):
            data = {'nodes': [], 'edges': [], marker: [], 'version': 'legacy'}
            with self.subTest(marker=marker):
                loaded = FiberNetwork.load_json(self.write(data))
                self.assertEqual((loaded.num_fibers, loaded.num_crosslinks), (0, 0))
        data = self.network().to_dict()
        data.update(nodes=[{'external': 'header'}], edges=[])
        self.assertEqual(FiberNetwork.from_dict(data).num_fibers, 2)

    def test_partial_or_nonarray_unknown_headers_not_newly_classified(self):
        for data in ({'nodes': []}, {'edges': []}, {'nodes': 3, 'edges': 2},
                     {'nodes': [], 'edges': None}, {'version': '3.1', 'schema': 'x'}):
            with self.subTest(data=data):
                self.assertEqual(FiberNetwork.from_dict(data).num_fibers, 0)

    def test_api_load_still_dispatches_graph_with_full_identity(self):
        graph = self.graph()
        graph.save_json(self.path)
        loaded = api.load(self.path)
        self.assertIsInstance(loaded, StructureGraph)
        self.assertEqual(loaded.to_dict(), graph.to_dict())
        self.assertEqual((loaded.num_nodes, loaded.num_edges), (2, 2))

    def test_api_load_still_loads_network_and_rejects_ambiguous_shape(self):
        data = self.network().to_dict()
        loaded = api.load(self.write(data))
        self.assertIsInstance(loaded, FiberNetwork)
        self.assertEqual(loaded.to_dict(), data)
        data.update(nodes=[], edges=[])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            api.load(self.write(data))


if __name__ == '__main__':
    unittest.main(verbosity=2)
