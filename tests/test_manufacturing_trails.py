"""Small real NumPy/StructureGraph/NetworkX tests; no solver or training runs.

Run from any source checkout: python -m unittest discover -s tests
-p test_manufacturing_trails.py. Uses only public dependencies and local fixtures;
no private baseline, runtime or evidence paths are required.
"""
from collections import Counter
from dataclasses import FrozenInstanceError, replace
import csv
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from fibernet import (StructureGraph, ManufacturingTrails, manufacturing_trails_from_graph,
                      export_mapped_trails, manufacturing_from_graph, export_mapped_route,
                      export_route)
from fibernet.gen import manufacturing_trails as module
from fibernet.gen.manufacturing_bridge import _digest, _snapshot_record
from fibernet.ml.graph_data import graph_batch


def graph(points, connections, *, node_ids=None, edge_ids=None):
    g = StructureGraph(dimension=3)
    nodes = [g.add_node(p, merge=False) for p in points]
    for a, b in connections:
        g.add_edge(nodes[a], nodes[b], allow_parallel=True)
    if node_ids is not None or edge_ids is not None:
        record = g.to_dict()
        node_ids = list(range(len(points))) if node_ids is None else node_ids
        edge_ids = list(range(len(connections))) if edge_ids is None else edge_ids
        for row, node in enumerate(record['nodes']):
            node['id'] = node_ids[row]
        for row, edge in enumerate(record['edges']):
            edge.update(id=edge_ids[row], node_i=node_ids[edge['node_i']], node_j=node_ids[edge['node_j']])
        record['next_node_id'], record['next_edge_id'] = max(node_ids) + 1, max(edge_ids) + 1
        g = StructureGraph.from_dict(record)
    return g


def adapt(g, unit='mm'):
    return manufacturing_trails_from_graph(g, graph_id='fixture', length_unit=unit, coordinate_frame='lab_xyz')


def state(g):
    return _digest(_snapshot_record(graph_batch([g], graph_ids=['fixture'], length_unit='mm',
                                               coordinate_frame='lab_xyz', edge_features=('radius',),
                                               include_provenance=True)))


def symlink_or_skip(case, link, target):
    """Skip only when the host cannot create the requested test symlink."""
    try:
        link.symlink_to(target)
    except NotImplementedError as error:
        case.skipTest('symlink creation is unsupported: ' + str(error))
    except OSError as error:
        if (error.errno in {errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP}
                or getattr(error, 'winerror', None) == 1314):
            case.skipTest('symlink creation is unavailable or not permitted: ' + str(error))
        raise


class TestManufacturingTrails(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='manufacturing-trails-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def export(self, trails, snapshot, stem='route', **kwargs):
        return export_mapped_trails(self.directory / (stem + '.csv'), trails, snapshot,
                                   sidecar_path=self.directory / (stem + '.json'), **kwargs)

    def verify(self, g, strokes, closed):
        before = state(g)
        trails, snapshot = adapt(g)
        self.assertEqual(state(g), before)
        snap_before = _digest(_snapshot_record(snapshot))
        route_before = _digest(module._trails_record(trails))
        self.assertEqual(trails.stroke_count, strokes)
        self.assertEqual(trails.closed.tolist(), closed)
        self.assertEqual(len(trails.route_nodes), len(g.edges) + strokes)
        self.assertEqual(Counter(trails.route_edges.tolist()), Counter(range(len(g.edges))))
        independent = nx.MultiGraph()
        independent.add_nodes_from(range(len(snapshot.node_ids)))
        for row, (a, b) in enumerate(snapshot.edge_index.T.tolist()):
            independent.add_edge(a, b, key=row)
        self.assertTrue(nx.is_connected(independent))
        odd = sorted(node for node, degree in independent.degree() if degree % 2)
        self.assertEqual(strokes, max(1, len(odd) // 2))
        endpoints = []
        seen = []
        for stroke in range(strokes):
            ns = trails.route_nodes[trails.stroke_node_ptr[stroke]:trails.stroke_node_ptr[stroke + 1]].tolist()
            es = trails.route_edges[trails.stroke_edge_ptr[stroke]:trails.stroke_edge_ptr[stroke + 1]].tolist()
            self.assertEqual(len(ns), len(es) + 1)
            self.assertGreater(len(es), 0)
            self.assertEqual(ns[0] == ns[-1], closed[stroke])
            endpoints.extend((ns[0], ns[-1]))
            for a, b, row in zip(ns[:-1], ns[1:], es):
                self.assertTrue(independent.has_edge(a, b, row))
                independent.remove_edge(a, b, row)
                seen.append(row)
        self.assertEqual(independent.number_of_edges(), 0)
        if odd:
            self.assertEqual(sorted(endpoints), odd)
        # Independent NetworkX edge-keyed Euler routines, never nx.eulerize.
        oracle = nx.MultiGraph()
        oracle.add_nodes_from(range(len(snapshot.node_ids)))
        for row, (a, b) in enumerate(snapshot.edge_index.T.tolist()):
            oracle.add_edge(a, b, key=row)
        if len(odd) <= 2:
            route = list(nx.eulerian_path(oracle, keys=True))
            self.assertEqual(Counter(row for a, b, row in route), Counter(seen))
        else:
            for k, (a, b) in enumerate(zip(odd[::2], odd[1::2])):
                oracle.add_edge(a, b, key=len(g.edges) + k)
            route = list(nx.eulerian_circuit(oracle, keys=True))
            self.assertEqual(Counter(row for a, b, row in route if row < len(g.edges)), Counter(seen))
        self.assertEqual(state(g), before)
        again, again_snapshot = adapt(g)
        self.assertEqual(_digest(module._trails_record(again)), route_before)
        self.assertEqual(_digest(_snapshot_record(again_snapshot)), snap_before)
        mapping = self.export(trails, snapshot)
        mapping2 = self.export(again, again_snapshot, 'again')
        self.assertEqual(mapping, mapping2)
        self.assertEqual((self.directory / 'route.csv').read_bytes(), (self.directory / 'again.csv').read_bytes())
        self.assertEqual(state(g), before)
        self.assertEqual(_digest(_snapshot_record(snapshot)), snap_before)
        self.assertEqual(_digest(module._trails_record(trails)), route_before)
        self.assertEqual(mapping['route_csv_sha256'], hashlib.sha256((self.directory / 'route.csv').read_bytes()).hexdigest())
        rows = list(csv.DictReader(io.StringIO((self.directory / 'route.csv').read_text())))
        self.assertEqual(len(rows), len(g.edges) + strokes)
        source_edges = snapshot.edge_index.T.tolist()
        for row, item in zip(rows, mapping['route']):
            self.assertEqual(int(row['node_id']), int(snapshot.node_ids[item['node_row']]))
            if item['step'] == 0:
                self.assertEqual((row['incoming_edge_row'], row['incoming_edge_id'], row['source_edge_direction']), ('-1', '-1', '0'))
                self.assertIsNone(item['incoming_edge_row'])
                self.assertIsNone(item['incoming_edge_id'])
                self.assertIsNone(item['source_edge_direction'])
            else:
                self.assertEqual(int(row['incoming_edge_id']), int(snapshot.edge_ids[item['incoming_edge_row']]))
        return trails, snapshot, mapping

    def test_public_export_identity(self):
        import fibernet
        import fibernet.gen
        for name in ('ManufacturingTrails', 'manufacturing_trails_from_graph', 'export_mapped_trails'):
            self.assertIs(getattr(fibernet, name), getattr(module, name))
            self.assertIs(getattr(fibernet.gen, name), getattr(module, name))
            self.assertIn(name, fibernet.__all__)
            self.assertIn(name, fibernet.gen.__all__)

    def test_symlink_capability_skip_is_narrow(self):
        path = self.directory / 'probe'
        for error in (NotImplementedError('unsupported'), PermissionError(errno.EPERM, 'permission')):
            with patch.object(Path, 'symlink_to', side_effect=error):
                with self.assertRaises(unittest.SkipTest):
                    symlink_or_skip(self, path, self.directory / 'target')
        unexpected = OSError(errno.ENOENT, 'unexpected missing parent')
        with patch.object(Path, 'symlink_to', side_effect=unexpected):
            with self.assertRaises(OSError) as caught:
                symlink_or_skip(self, path, self.directory / 'target')
        self.assertIs(caught.exception, unexpected)

    def test_open_segment(self):
        self.verify(graph([(0., 0., 0.), (1., 2., 3.)], [(0, 1)]), 1, [False])

    def test_bent_three_dimensional_chain(self):
        self.verify(graph([(0., 0., 0.), (1., 2., 3.), (2., -1., 4.), (5., 0., 7.)],
                          [(0, 1), (2, 1), (2, 3)]), 1, [False])

    def test_y_three_dimensional(self):
        self.verify(graph([(0., 0., 0.), (1., 0., 0.), (0., 1., 1.), (0., -1., 1.)],
                          [(0, 1), (0, 2), (0, 3)]), 2, [False, False])

    def test_k4(self):
        self.verify(graph([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
                          [(i, j) for i in range(4) for j in range(i + 1, 4)]), 2, [False, False])

    def test_six_arm_star(self):
        points = [(0., 0., 0.), (1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
                  (0., -1., 0.), (0., 0., 1.), (0., 0., -1.)]
        self.verify(graph(points, [(0, i) for i in range(1, 7)]), 3, [False] * 3)

    def test_cycle_with_branch(self):
        self.verify(graph([(0., 0., 0.), (1., 0., 1.), (0., 1., 1.), (2., 0., 0.)],
                          [(0, 1), (1, 2), (2, 0), (0, 3)]), 1, [False])

    def test_three_parallel_sparse_mixed_direction(self):
        g = graph([(2., 0., 0.), (1., 0., 0.)], [(0, 1), (1, 0), (0, 1)],
                  node_ids=[20, 5], edge_ids=[901, 11, 500])
        t, s, m = self.verify(g, 1, [False])
        self.assertEqual(s.node_ids.tolist(), [5, 20])
        self.assertEqual(s.edge_ids.tolist(), [901, 11, 500])
        self.assertEqual(s.edge_index.T.tolist(), [[1, 0], [0, 1], [1, 0]])
        for index, item in enumerate(m['route']):
            if item['step']:
                old = m['route'][index - 1]['node_row']
                expected = s.edge_index[:, item['incoming_edge_row']].tolist()
                self.assertEqual(item['source_edge_direction'], 1 if expected == [old, item['node_row']] else -1)

    def test_even_ring_legacy_bytes(self):
        g = graph([(0., 0., 0.), (1., 0., 0.), (1., 1., 1.), (0., 1., 1.)],
                  [(0, 1), (1, 2), (2, 3), (3, 0)])
        t, s, _ = self.verify(g, 1, [True])
        expected = ('step,incoming_edge_id,node_id,x_mm,y_mm,z_mm\r\n'
                    '0,-1,0,0.0,0.0,0.0\r\n1,3,3,0.0,1.0,1.0\r\n'
                    '2,2,2,1.0,1.0,1.0\r\n3,1,1,1.0,0.0,0.0\r\n4,0,0,0.0,0.0,0.0\r\n').encode()
        self.legacy_parity(g, t, expected)

    def test_even_multigraph_legacy_bytes(self):
        g = graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1), (1, 0), (0, 1), (1, 0)])
        t, s, _ = self.verify(g, 1, [True])
        expected = ('step,incoming_edge_id,node_id,x_mm,y_mm,z_mm\r\n'
                    '0,-1,0,0.0,0.0,0.0\r\n1,3,1,1.0,0.0,0.0\r\n'
                    '2,2,0,0.0,0.0,0.0\r\n3,1,1,1.0,0.0,0.0\r\n4,0,0,0.0,0.0,0.0\r\n').encode()
        self.legacy_parity(g, t, expected)

    def legacy_parity(self, g, trails, expected):
        network, snapshot = manufacturing_from_graph(g, graph_id='fixture', length_unit='mm', coordinate_frame='lab_xyz')
        np.testing.assert_array_equal(network.route_nodes, trails.route_nodes)
        np.testing.assert_array_equal(network.route_edges, trails.route_edges)
        export_route(self.directory / 'legacy-direct.csv', network, network.positions)
        export_mapped_route(self.directory / 'legacy.csv', network, snapshot, sidecar_path=self.directory / 'legacy.json')
        self.assertEqual((self.directory / 'legacy-direct.csv').read_bytes(), expected)
        self.assertEqual((self.directory / 'legacy.csv').read_bytes(), expected)

    def test_metre_conversion_provenance_and_detachment(self):
        g = graph([(0., 0., 0.), (.001, .002, .003)], [(0, 1), (1, 0), (0, 1)])
        g.metadata.update(length_unit='m', coordinate_frame='lab_xyz', labels=['source'])
        g.edges[0].radius, g.edges[1].radius, g.edges[2].radius = .0001, None, .0003
        g.edges[0].material.extra['tags'] = ['physical']
        g.edges[1].material = None
        g.edges[2].internal_points = np.empty((0, 3), dtype=np.float64)
        t, s = adapt(g, 'm')
        expected_source = _snapshot_record(s)
        m = self.export(t, s, source_input_sha256='a' * 64)
        with (self.directory / 'route.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(any([float(r[k]) for k in ('x_mm', 'y_mm', 'z_mm')] == [1., 2., 3.] for r in rows))
        self.assertEqual(m['source_snapshot'], expected_source)
        self.assertEqual(s.edge_features[:, 0].tolist(), [.0001, 0., .0003])
        self.assertEqual(s.edge_feature_available[:, 0].tolist(), [True, False, True])
        self.assertEqual(s.internal_available.tolist(), [False, False, True])
        self.assertIsNone(s.provenance.edge_material_records[1])
        self.assertEqual(m['source_input_sha256_origin'], 'caller_asserted_not_file_verified')
        g.nodes[1].position[:] = 9.
        g.edges[0].material.extra['tags'].append('changed')
        g.metadata['labels'].append('changed')
        self.assertEqual(_snapshot_record(s), expected_source)
        self.assertEqual(self.export(t, s, 'detached')['source_snapshot'], expected_source)

    def test_result_is_factory_only_and_immutable(self):
        with self.assertRaises(TypeError):
            ManufacturingTrails()
        t, s = adapt(graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)]))
        with self.assertRaises(FrozenInstanceError):
            t.route_nodes = np.array([0, 1])
        for name in module._TRAIL_ARRAYS:
            a = getattr(t, name)
            with self.assertRaises(ValueError):
                a.setflags(write=True)
            with self.assertRaises(ValueError):
                a[0] = 0
        self.assertFalse(np.shares_memory(s.positions, s.edge_index))

    def test_snapshot_mutations_refused_before_writes(self):
        for field in ('positions', 'node_ids', 'edge_ids', 'edge_index', 'edge_features'):
            with self.subTest(field=field):
                t, s = adapt(graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)]))
                a = np.array(getattr(s, field), copy=True)
                a.flat[0] += 1
                a.setflags(write=False)
                with self.assertRaises(ValueError):
                    self.export(t, replace(s, **{field: a}), field)
                self.assertFalse((self.directory / (field + '.csv')).exists())
        with self.assertRaises(ValueError):
            self.export(t, replace(s, coordinate_frame='different'))

    def test_route_mutations_and_malformed_indices_refused(self):
        corruptions = {'route_nodes': [0, 0], 'route_edges': [1], 'stroke_node_ptr': [0, 1],
                       'stroke_edge_ptr': [0, 0], 'closed': [True]}
        for name, value in corruptions.items():
            with self.subTest(field=name):
                t, s = adapt(graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)]))
                object.__setattr__(t, name, module._immutable_array(value, bool if name == 'closed' else np.int64))
                with self.assertRaises(ValueError):
                    self.export(t, s, name)
                # Exercise structural validation independently of the mutation seal.
                object.__setattr__(t, 'route_binding_sha256', _digest(module._trails_record(t)))
                with self.assertRaises(ValueError):
                    self.export(t, s, name)
                self.assertFalse((self.directory / (name + '.csv')).exists())

    def test_empty_disconnected_isolate_and_wrong_type(self):
        cases = [StructureGraph(dimension=3), graph([(0., 0., 0.)], []),
                 graph([(0., 0., 0.), (1., 0., 0.), (2., 0., 0.)], [(0, 1)]),
                 graph([(0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (3., 0., 0.)], [(0, 1), (2, 3)])]
        for g in cases:
            with self.assertRaises(ValueError):
                adapt(g)
        with self.assertRaises(TypeError):
            adapt(nx.Graph())

    def test_invalid_units_frames_curves_and_metadata(self):
        g = graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)])
        for unit in ('cm', '', None):
            with self.assertRaises(ValueError):
                adapt(g, unit)
        for frame in ('', ' ', None):
            with self.assertRaises(ValueError):
                manufacturing_trails_from_graph(g, graph_id='x', length_unit='mm', coordinate_frame=frame)
        for metadata in ({'length_unit': 'm'}, {'nested': {'units': {'radius': 'm'}}}, {'coordinate_frame': 'other'}):
            g.metadata.clear(); g.metadata.update(metadata)
            with self.assertRaises(ValueError):
                adapt(g)
        g.metadata.clear()
        g.edges[0].internal_points = np.array([[.5, 0., 0.]])
        with self.assertRaises(ValueError):
            adapt(g)

    def test_numeric_storage_and_conversion_guards(self):
        for value in (np.array([1., 0., 0.], dtype=np.float32), [2 ** 53 + 1, 0, 0], [float('nan'), 0., 0.]):
            g = graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)])
            g.nodes[1].position = value
            with self.assertRaises(ValueError):
                adapt(g)
        g = graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)])
        g.edges[0].radius = np.float32(.1)
        with self.assertRaises(ValueError):
            adapt(g)
        g.edges[0].radius = .1
        g.nodes[1].position = np.array([1e308, 0., 0.])
        t, s = adapt(g, 'm')
        with self.assertRaises(ValueError):
            self.export(t, s)
        self.assertFalse((self.directory / 'route.csv').exists())

    def test_zero_length_and_stored_self_loop_semantics(self):
        # Topological routing is not a nonzero-length physical validation.
        g = graph([(0., 0., 0.), (0., 0., 0.)], [(0, 1)])
        self.verify(g, 1, [False])
        # add_edge refuses self-loops, but legacy mutable records can contain one.
        g = graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1), (0, 1)])
        g.edges[1].node_j = 0
        t, s = adapt(g)
        self.assertEqual(Counter(t.route_edges.tolist()), Counter([0, 1]))
        self.assertEqual(t.stroke_count, 1)

    def test_fresh_paths_hash_validation_and_same_path(self):
        t, s = adapt(graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)]))
        for bad in ('A' * 64, 'x', 3):
            with self.assertRaises(ValueError):
                self.export(t, s, source_input_sha256=bad)
        path = self.directory / 'same'
        with self.assertRaises(ValueError):
            export_mapped_trails(path, t, s, sidecar_path=path)
        with self.assertRaises(ValueError):
            export_mapped_trails(self.directory / 'absent' / 'r.csv', t, s, sidecar_path=self.directory / 'r.json')
        self.export(t, s)
        csv_before = (self.directory / 'route.csv').read_bytes()
        with self.assertRaises(FileExistsError):
            self.export(t, s)
        self.assertEqual((self.directory / 'route.csv').read_bytes(), csv_before)

    def test_symlink_destinations_refused(self):
        t, s = adapt(graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)]))
        target = self.directory / 'target'; target.write_text('unchanged')
        symlink_or_skip(self, self.directory / 'route.csv', target)
        with self.assertRaises(FileExistsError):
            self.export(t, s)
        self.assertEqual(target.read_text(), 'unchanged')

    def test_sidecar_last_and_failure_evidence(self):
        from fibernet.gen import solid_export
        t, s = adapt(graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)]))
        real_replace, calls = solid_export.atomic_replace, []
        def fail_sidecar(source, target):
            calls.append(Path(target).suffix)
            if Path(target).suffix == '.json':
                self.assertGreater((self.directory / 'route.csv').stat().st_size, 0)
                raise OSError('injected sidecar failure')
            real_replace(source, target)
        with patch.object(solid_export, 'atomic_replace', fail_sidecar):
            with self.assertRaisesRegex(OSError, 'injected'):
                self.export(t, s)
        self.assertEqual(calls, ['.csv', '.json'])
        self.assertGreater((self.directory / 'route.csv').stat().st_size, 0)
        self.assertEqual((self.directory / 'route.json').read_bytes(), b'')
        self.assertFalse(list(self.directory.glob('.*.export-*')))

    def test_duplicate_missing_edges_and_invalid_types(self):
        for values, dtype in (([0, 0], np.int64), ([0., 1.], np.float64), ([0, 2], np.int64)):
            g = graph([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.)], [(0, 1), (1, 2)])
            t, snapshot = adapt(g)
            before = _digest(_snapshot_record(snapshot))
            object.__setattr__(t, 'route_edges', module._immutable_array(values, dtype))
            object.__setattr__(t, 'route_binding_sha256', _digest(module._trails_record(t)))
            with self.assertRaises(ValueError):
                self.export(t, snapshot)
            self.assertEqual(_digest(_snapshot_record(snapshot)), before)
            self.assertFalse((self.directory / 'route.csv').exists())
        with self.assertRaises(ValueError):
            self.export(object.__new__(ManufacturingTrails), snapshot)

    def test_csv_publication_failure_leaves_no_complete_sidecar(self):
        from fibernet.gen import solid_export
        t, snapshot = adapt(graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)]))
        before = _digest(_snapshot_record(snapshot))
        with patch.object(solid_export, 'atomic_replace', side_effect=OSError('injected CSV failure')):
            with self.assertRaisesRegex(OSError, 'injected CSV'):
                self.export(t, snapshot)
        self.assertEqual((self.directory / 'route.csv').read_bytes(), b'')
        self.assertEqual((self.directory / 'route.json').read_bytes(), b'')
        self.assertFalse(list(self.directory.glob('.*.export-*')))
        self.assertEqual(_digest(_snapshot_record(snapshot)), before)

    def test_graph_budget_and_provenance_replacement(self):
        from fibernet.gen import manufacturing
        g = graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)])
        with patch.object(manufacturing, 'MAX_MANUFACTURING_EDGES', 0):
            with self.assertRaises(MemoryError):
                adapt(g)
        t, snapshot = adapt(g)
        changed = replace(snapshot.provenance, edge_material_origin=('changed',))
        with self.assertRaises(ValueError):
            self.export(t, replace(snapshot, provenance=changed))
        with self.assertRaises(ValueError):
            self.export(t, replace(snapshot, graph_ids=('other',)))
        self.assertFalse((self.directory / 'route.csv').exists())

    def test_exclusive_sidecar_race_preserves_foreign_bytes(self):
        t, s = adapt(graph([(0., 0., 0.), (1., 0., 0.)], [(0, 1)]))
        real_reserve = module._reserve
        def reserve(path):
            if path.suffix == '.json':
                path.write_text('foreign')
            return real_reserve(path)
        with patch.object(module, '_reserve', reserve):
            with self.assertRaises(FileExistsError):
                self.export(t, s)
        self.assertEqual((self.directory / 'route.csv').read_bytes(), b'')
        self.assertEqual((self.directory / 'route.json').read_text(), 'foreign')


if __name__ == '__main__':
    unittest.main()
