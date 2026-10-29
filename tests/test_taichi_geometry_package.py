"""Actual core-package geometry contracts, with no native mechanics execution.

Small real StructureGraph/generator/NumPy fixtures and the production SimResult
serializer are used. Engine dispatch is inert; its zero/unit arrays are synthetic,
not mechanical labels. No installed dependency is replaced or faked. Importing
accelerated may import an installed optional Taichi provider; these tests never
initialize or step it. To check the actual package without provider imports, use
a core-only environment with natural Taichi absence and pytest --noconftest,
with plugin autoload disabled to avoid unrelated native test setup.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import fibernet as fn
import fibernet.easy as easy
import fibernet.sim.accelerated as accelerated
import fibernet.sim._taichi_geometry as geometry


class GeometryPackageTests(unittest.TestCase):
    np = np
    fn = fn
    easy = easy
    accelerated = accelerated
    geometry = geometry

    def setUp(self):
        self.assertIs(self.fn.SimResult, self.accelerated.SimResult)
        self.assertIs(self.fn.simulate, self.easy.simulate)
        self.constructor = patch.object(
            self.accelerated.TaichiEngine, '__init__',
            side_effect=AssertionError('Native engine construction is forbidden'))
        self.constructor_call = self.constructor.start()
        self.addCleanup(self.constructor.stop)

    def bounded(self, graph):
        self.assertIsInstance(graph, self.fn.StructureGraph)
        self.assertLessEqual(graph.num_nodes, 64)
        self.assertLessEqual(graph.num_edges, 256)
        for edge in graph.edges.values():
            self.assertTrue(edge.internal_points is None or len(edge.internal_points) <= 4)
        return graph

    def graph(self):
        return self.bounded(self.fn.StructureGraph.from_dict({
            'version': '3.1', 'dimension': 2,
            'metadata': {'fixture': 'geometry contract, not mechanics'},
            'nodes': [
                {'id': 9, 'position': [2., 0., 0.]},
                {'id': 4, 'position': [0., 0., 0.]},
            ],
            'edges': [
                {'id': 13, 'node_i': 9, 'node_j': 4, 'radius': .03,
                 'internal_points': [[1., .25, 0.]], 'metadata': {'fiber': 'offset'}},
                {'id': 12, 'node_i': 4, 'node_j': 9, 'radius': .07,
                 'internal_points': [[1., 0., 0.], [1., 0., 0.]],
                 'metadata': {'fiber': 'duplicate points'}},
                {'id': 6, 'node_i': 9, 'node_j': 4, 'radius': .05,
                 'internal_points': []},
            ],
        }))

    def graph_snapshot(self, graph):
        return (copy.deepcopy(graph.to_dict()), list(graph.nodes), list(graph.edges),
                {eid: None if edge.internal_points is None else
                 (edge.internal_points.shape, edge.internal_points.dtype.str,
                  edge.internal_points.tobytes())
                 for eid, edge in graph.edges.items()})

    def synthetic_result(self, graph, *, include_geometry=True, policy='endpoint_chord'):
        """Construct a transport fixture; never derive or claim a physical result."""
        self.bounded(graph)
        np = self.np
        positions = np.array([node.position for node in graph.nodes.values()], copy=True)
        metadata = {'node_ids': list(graph.nodes), 'edge_ids': list(graph.edges),
                    'validation_fixture': 'synthetic arrays, not mechanics'}
        if include_geometry:
            metadata['taichi_geometry'] = self.geometry.preflight_taichi_geometry(
                graph, geometry_policy=policy)
        return self.fn.SimResult(
            mode='dynamics', energy=0., time_seconds=0.,
            n_nodes=graph.num_nodes, n_edges=graph.num_edges,
            max_force=0., max_stretch=1., mean_stretch=1., std_stretch=0.,
            max_displacement=0., displacements=np.zeros_like(positions),
            deformed_positions=positions.copy(), edge_forces=np.zeros(graph.num_edges),
            edge_stretches=np.ones(graph.num_edges), positions_trajectory=[positions.copy()],
            history=[{'step': 0, 'max_stretch': 1.}], metadata=metadata)

    def test_real_multiedge_strict_and_chord_preserve_graph_and_stable_order(self):
        graph = self.graph()
        self.assertEqual(list(graph.nodes), [9, 4])
        self.assertEqual(list(graph.edges), [13, 12, 6])
        self.assertEqual({frozenset((e.node_i, e.node_j)) for e in graph.edges.values()},
                         {frozenset((9, 4))})
        self.assertEqual(graph.num_edges, 3)
        self.assertEqual(graph.edges[6].internal_points.shape, (0, 3))
        before = self.graph_snapshot(graph)
        references = {eid: (edge, edge.internal_points, edge.material)
                      for eid, edge in graph.edges.items()}
        with self.assertRaises(self.geometry.UnsupportedInternalPointsError) as caught:
            self.geometry.preflight_taichi_geometry(graph)
        self.assertEqual(caught.exception.rejected_edge_ids, (13, 12))
        record = self.geometry.preflight_taichi_geometry(graph, geometry_policy='endpoint_chord')
        self.assertEqual(record['ignored_internal_point_edge_ids'], [13, 12])
        self.assertEqual(record['representation'], 'endpoint_chord')
        self.assertEqual(self.graph_snapshot(graph), before)
        for eid, (edge, points, material) in references.items():
            self.assertIs(graph.edges[eid], edge)
            self.assertIs(graph.edges[eid].internal_points, points)
            self.assertIs(graph.edges[eid].material, material)
        record['ignored_internal_point_edge_ids'].append(999)
        self.assertEqual(self.geometry.preflight_taichi_geometry(
            graph, geometry_policy='endpoint_chord')['ignored_internal_point_edge_ids'], [13, 12])
        self.assertEqual(self.graph_snapshot(graph), before)

    def test_real_graph_json_preserves_multiedges_and_reports_loaded_order(self):
        graph = self.graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json'
            graph.save_json(str(path))
            restored = self.bounded(self.fn.StructureGraph.load_json(str(path)))
        self.assertEqual(restored.to_dict(), graph.to_dict())
        # Graph serialization intentionally sorts IDs; reporting follows the
        # loaded graph's insertion order, not the original in-memory order.
        self.assertEqual(list(restored.edges), [6, 12, 13])
        record = self.geometry.preflight_taichi_geometry(restored, geometry_policy='endpoint_chord')
        self.assertEqual(record['ignored_internal_point_edge_ids'], [12, 13])

    def test_actual_generator_distinguishes_owned_points_from_segmented_nodes(self):
        owned = self.bounded(self.fn.pattern_2d(
            unit='square', box=(2., 2.), grid=(1, 1), n_internal=1, seed=0))
        with self.assertRaises(self.geometry.UnsupportedInternalPointsError):
            self.geometry.preflight_taichi_geometry(owned)
        segmented = self.bounded(self.fn.pattern_2d(
            unit='square', box=(2., 2.), grid=(1, 1), n_internal=0,
            n_pts_per_side=1, point_displacements=[(0., 0.)] * 4, seed=0))
        self.assertGreater(segmented.num_nodes, 4)
        self.assertGreater(segmented.num_edges, 4)
        before = self.graph_snapshot(segmented)
        record = self.geometry.preflight_taichi_geometry(segmented)
        self.assertEqual(record['policy'], 'reject_internal_points')
        self.assertEqual(record['ignored_internal_point_edge_ids'], [])
        self.assertEqual(self.graph_snapshot(segmented), before)

    def test_actual_facade_strict_rejects_before_patched_engine_constructor(self):
        graph = self.graph()
        before = self.graph_snapshot(graph)
        for backend in ('spring', 'taichi'):
            for mode in ('stretch', 'dynamics'):
                with self.subTest(backend=backend, mode=mode):
                    with self.assertRaises(self.geometry.UnsupportedInternalPointsError):
                        self.fn.simulate(graph, backend=backend, mode=mode)
        self.constructor_call.assert_not_called()
        self.assertEqual(self.graph_snapshot(graph), before)

    def test_actual_facade_explicit_chord_dispatch_is_inert(self):
        graph = self.graph()
        before = self.graph_snapshot(graph)
        result = self.synthetic_result(graph)
        calls = []

        class InertEngine:
            def __init__(self, *, arch, num_threads):
                calls.append(('constructor', arch, num_threads))

            def stretch_test(self, value, target_stretch=2., *, geometry_policy):
                calls.append(('stretch', value, geometry_policy))
                return result

            def dynamics(self, value, *, geometry_policy):
                calls.append(('dynamics', value, geometry_policy))
                return result

        with patch.object(self.easy, 'TaichiEngine', InertEngine):
            for backend in ('spring', 'taichi'):
                for mode in ('stretch', 'dynamics'):
                    with self.subTest(backend=backend, mode=mode):
                        calls.clear()
                        actual = self.fn.simulate(graph, backend=backend, mode=mode,
                                                  geometry_policy='endpoint_chord')
                        self.assertIs(actual, result)
                        self.assertEqual(calls[0], ('constructor', 'cpu', 4))
                        self.assertEqual(calls[1], (mode, graph, 'endpoint_chord'))
        self.constructor_call.assert_not_called()
        self.assertEqual(self.graph_snapshot(graph), before)

    def test_actual_result_summary_and_detailed_transport_preserve_geometry(self):
        result = self.synthetic_result(self.graph())
        metadata = copy.deepcopy(result.metadata)
        with tempfile.TemporaryDirectory() as directory:
            for detailed in (False, True):
                with self.subTest(detailed=detailed):
                    path = Path(directory) / ('detailed.json' if detailed else 'summary.json')
                    result.save(str(path), detailed=detailed)
                    payload = json.loads(path.read_text(encoding='utf-8'))
                    restored = self.fn.SimResult.load(str(path))
                    self.assertEqual(payload, result.to_dict(detailed=detailed))
                    self.assertEqual(restored.to_dict(detailed=detailed), payload)
                    self.assertEqual(restored.metadata['taichi_geometry'], metadata['taichi_geometry'])
                    self.assertEqual(restored.metadata['node_ids'], [9, 4])
                    self.assertEqual(restored.metadata['edge_ids'], [13, 12, 6])
                    if detailed:
                        self.np.testing.assert_array_equal(restored.edge_forces, result.edge_forces)
                        self.np.testing.assert_array_equal(restored.deformed_positions, result.deformed_positions)
                    else:
                        self.assertIsNone(restored.edge_forces)
                        self.assertIsNone(restored.deformed_positions)
                    restored.metadata['taichi_geometry']['ignored_internal_point_edge_ids'].append(999)
                    self.assertEqual(result.metadata, metadata)

    def test_actual_legacy_results_do_not_acquire_geometry_provenance(self):
        result = self.synthetic_result(self.graph(), include_geometry=False)
        with tempfile.TemporaryDirectory() as directory:
            for detailed in (False, True):
                with self.subTest(detailed=detailed):
                    path = Path(directory) / ('old-detailed.json' if detailed else 'old-summary.json')
                    result.save(str(path), detailed=detailed)
                    restored = self.fn.SimResult.load(str(path))
                    self.assertNotIn('taichi_geometry', restored.metadata)
                    self.assertNotIn('taichi_geometry', restored.to_dict(detailed=detailed))
                    self.assertEqual(restored.metadata, result.metadata)
            path = Path(directory) / 'old-minimal.json'
            path.write_text(json.dumps({'mode': 'dynamics', 'energy': 0.,
                                        'n_nodes': 0, 'n_edges': 0}), encoding='utf-8')
            self.assertNotIn('taichi_geometry', self.fn.SimResult.load(str(path)).metadata)


if __name__ == '__main__':
    unittest.main()
