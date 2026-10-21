"""Preserve native curve shapes and multigraph records across serialization."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph
try:
    import networkx as nx
except ImportError:
    nx = None
RECORD = json.loads((Path(__file__).parent / 'fixtures' / 'empty_curve_roundtrip.json').read_text(encoding='utf-8'))

class Roundtrip(unittest.TestCase):

    def setUp(self):
        self.payload = copy.deepcopy(RECORD)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def assert_graph(self, g):
        self.assertEqual(g.to_dict(), self.payload)
        self.assertEqual(list(g.nodes), [4, 9])
        self.assertEqual(list(g.edges), [12, 13])
        self.assertEqual((g._next_node_id, g._next_edge_id), (22, 30))
        self.assertEqual((g.edges[13].node_i, g.edges[13].node_j), (9, 4))
        self.assertEqual(g.edges[13].internal_points.shape, (0, 3))
        self.assertEqual(np.vstack([g.nodes[9].position, g.edges[13].internal_points, g.nodes[4].position]).shape, (2, 3))
        self.assertEqual(g.edges[12].internal_points.shape, (1, 3))
        self.assertEqual(g.edges[12].material.name, 'custom')
        self.assertEqual(g.edges[12].material.extra, {'phase': 'a'})
        self.assertEqual(g._metadata['units'], {'length': 'm'})

    def test_dict_empty_curve_and_complete_record(self):
        self.assert_graph(StructureGraph.from_dict(self.payload))

    def test_json_empty_curve_and_complete_record(self):
        g = StructureGraph.from_dict(self.payload)
        path = self.root / 'graph.json'
        g.save_json(str(path))
        self.assert_graph(StructureGraph.load_json(str(path)))

    def test_malformed_shapes_rejected_both_routes(self):
        for value in [1, [1, 2, 3], [[1, 2]], [[1, 2, 3, 4]], [[[1, 2, 3]]], [[]], [[1, 2, 3], [4, 5]]]:
            for route in ['dict', 'json']:
                with self.subTest(value=value, route=route):
                    p = copy.deepcopy(self.payload)
                    p['edges'][1]['internal_points'] = value
                    with self.assertRaises(ValueError):
                        if route == 'dict':
                            StructureGraph.from_dict(p)
                        else:
                            path = self.root / 'invalid.json'
                            path.write_text(json.dumps(p))
                            StructureGraph.load_json(str(path))

    def test_null_and_nonempty_unchanged(self):
        self.payload['edges'][1]['internal_points'] = None
        g = StructureGraph.from_dict(self.payload)
        self.assertIsNone(g.edges[13].internal_points)
        self.assertEqual(g.to_dict(), self.payload)

    def test_duplicate_ids_rejected(self):
        for kind in ['nodes', 'edges']:
            with self.subTest(kind=kind):
                p = copy.deepcopy(self.payload)
                p[kind].append(copy.deepcopy(p[kind][0]))
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    StructureGraph.from_dict(p)

@unittest.skipIf(nx is None, 'NetworkX is optional')
class NativeInterop(unittest.TestCase):

    def setUp(self):
        self.record = copy.deepcopy(RECORD)
        self.g = StructureGraph.from_dict(self.record)
        self.g.edges[13].internal_points = np.empty((0, 3))
        self.g._nodes = {k: self.g.nodes[k] for k in [9, 4]}
        self.g._edges = {k: self.g.edges[k] for k in [13, 12]}

    def test_native_complete_roundtrip(self):
        nxg = self.g.to_networkx_lossless()
        self.assertIsInstance(nxg, nx.MultiGraph)
        self.assertEqual(nxg.number_of_edges(4, 9), 2)
        self.assertEqual(set(nxg[4][9]), {12, 13})
        h = StructureGraph.from_networkx_lossless(nxg)
        self.assertEqual(h.to_dict(), self.record)
        self.assertEqual(list(h.nodes), [9, 4])
        self.assertEqual(list(h.edges), [13, 12])
        self.assertEqual(h.edges[13].internal_points.shape, (0, 3))
        self.assertEqual(h.edges[12].internal_points.shape, (1, 3))
        self.assertEqual((h.edges[13].node_i, h.edges[13].node_j), (9, 4))
        self.assertEqual(h._metadata['units'], {'length': 'm'})
        self.assertEqual(h._metadata['provenance'], {'generator': 'roundtrip_test'})
        self.assertEqual(h.edges[12].material.extra, {'phase': 'a'})
        self.assertEqual((h._next_node_id, h._next_edge_id), (22, 30))

    def test_detached_export_import(self):
        nxg = self.g.to_networkx_lossless()
        nxg.graph['_fibernet']['metadata']['units']['length'] = 'mm'
        nxg[4][9][12]['_fibernet']['material']['extra']['phase'] = 'b'
        self.assertEqual(self.g._metadata['units']['length'], 'm')
        self.assertEqual(self.g.edges[12].material.extra['phase'], 'a')
        h = StructureGraph.from_networkx_lossless(nxg)
        h._metadata['units']['length'] = 'cm'
        h.edges[12].material.extra['phase'] = 'c'
        self.assertEqual(nxg.graph['_fibernet']['metadata']['units']['length'], 'mm')
        self.assertEqual(nxg[4][9][12]['_fibernet']['material']['extra']['phase'], 'b')

    def test_unsupported_edits_rejected(self):

        def relabel(G):
            return nx.relabel_nodes(G, {4: 99}, copy=True)

        def remove(G):
            G.remove_edge(4, 9, 12)
            return G

        def extra(G):
            G.nodes[4]['arbitrary'] = 'unsupported'
            return G

        def directed(G):
            return nx.MultiDiGraph(G)

        def simple(G):
            return nx.Graph(G)

        def selfloop(G):
            r = copy.deepcopy(G[4][9][12]['_fibernet'])
            r['node_i'] = r['node_j'] = 4
            G.remove_edge(4, 9, 12)
            G.add_edge(4, 4, key=12, _fibernet=r)
            return G
        for label, mutate in [('relabel', relabel), ('topology_delete', remove), ('unknown_attribute', extra), ('directed', directed), ('simple', simple), ('selfloop', selfloop)]:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    StructureGraph.from_networkx_lossless(mutate(self.g.to_networkx_lossless()))

    def test_duplicate_global_edge_id_rejected(self):
        G = self.g.to_networkx_lossless()
        new = copy.deepcopy(G.nodes[9]['_fibernet'])
        new['id'] = 20
        new['position'] = [3.0, 0.0, 0.0]
        G.add_node(20, _fibernet=new)
        G.graph['_fibernet']['node_order'].append(20)
        edge = copy.deepcopy(G[4][9][12]['_fibernet'])
        edge['node_j'] = 20
        G.add_edge(4, 20, key=12, _fibernet=edge)
        with self.assertRaisesRegex(ValueError, 'globally unique'):
            StructureGraph.from_networkx_lossless(G)

    def test_legacy_projection_not_lossless(self):
        G = self.g.to_networkx()
        self.assertIsInstance(G, nx.MultiGraph)
        self.assertEqual(G.number_of_edges(), 2)
        h = StructureGraph.from_networkx(G, dimension=3)
        self.assertEqual(h.num_edges, 2)
        self.assertNotEqual(h.to_dict(), self.record)
        self.assertEqual(list(h.nodes), [0, 1])
        self.assertEqual(h._metadata, {})
        self.assertTrue(all((e.internal_points is None for e in h.edges.values())))
        self.assertTrue(all((e.material.name != 'custom' for e in h.edges.values())))

    def test_json_then_native_empty_curve(self):
        h = StructureGraph.from_dict(self.record)
        G = h.to_networkx_lossless()
        j = StructureGraph.from_networkx_lossless(G)
        self.assertEqual(j.edges[13].internal_points.shape, (0, 3))
        self.assertEqual(j.to_dict(), self.record)
