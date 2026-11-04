"""Public generation and snapshot geometry acceptance contracts."""
import builtins
import json
import unittest
from unittest.mock import patch

import numpy as np
from fibernet import StructureGraph, pattern_2d, pattern_3d
from fibernet.analysis import SnapshotFeatureExtractor


def graph_with_points(points=None):
    graph = StructureGraph(dimension=2)
    a = graph.add_node([0, 0, 0])
    b = graph.add_node([2, 0, 0])
    graph.add_edge(a, b, internal_points=points)
    return graph


class WorkflowTests(unittest.TestCase):
    custom = dict(points=[(0, 0), (1, 0), (1, 1)], closed=False,
                  box=(1, 1), grid=(1, 1))

    def assert_features_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            with self.subTest(feature=key):
                np.testing.assert_equal(actual[key], expected[key])

    def test_custom_fit_rejects_nondefault_intermediate_node_count(self):
        for count in (1, 2, -1):
            with self.subTest(n_pts_per_side=count):
                with self.assertRaisesRegex(ValueError, "fit_to_box=True.*n_pts_per_side=0"):
                    pattern_2d(**self.custom, fit_to_box=True, n_pts_per_side=count)

    def test_custom_fit_rejects_nonempty_displacements(self):
        for offsets in ([(0, 0.1)], ((0.1, 0),), np.array([[0, 0.1]])):
            with self.subTest(offsets=repr(offsets)):
                with self.assertRaisesRegex(ValueError, "no point_displacements"):
                    pattern_2d(**self.custom, fit_to_box=True, point_displacements=offsets)

    def test_custom_fit_preserves_known_empty_displacement_noops(self):
        for offsets in ([], (), np.empty((0, 2))):
            with self.subTest(offsets=repr(offsets)):
                actual = pattern_2d(**self.custom, fit_to_box=True,
                                    n_pts_per_side=0, point_displacements=offsets)
                expected = pattern_2d(**self.custom, fit_to_box=True)
                self.assertEqual(actual.to_dict(), expected.to_dict())
                self.assertEqual(actual.metadata["pattern"]["n_pts_per_side"], 0)
                self.assertEqual((actual.num_nodes, actual.num_edges), (3, 2))
                self.assertEqual(len(offsets), 0)

    def test_empty_displacements_do_not_bypass_subdivision_guard(self):
        for offsets in ([], (), np.empty((0, 2))):
            with self.subTest(offsets=repr(offsets)):
                with self.assertRaisesRegex(ValueError, "n_pts_per_side=0"):
                    pattern_2d(**self.custom, fit_to_box=True,
                               n_pts_per_side=1, point_displacements=offsets)

    def test_custom_fit_rejects_before_graph_construction(self):
        points = list(self.custom["points"])
        offsets = [(0, 0.1), (0.1, 0)]
        with patch("fibernet.gen.pattern.StructureGraph", wraps=StructureGraph) as constructor:
            with self.assertRaisesRegex(ValueError, "Fit the source points explicitly"):
                pattern_2d(**dict(self.custom, points=points), fit_to_box=True,
                           n_pts_per_side=1, point_displacements=offsets)
            constructor.assert_not_called()
        self.assertEqual(points, self.custom["points"])
        self.assertEqual(offsets, [(0, 0.1), (0.1, 0)])

    def test_accepted_custom_fit_has_truthful_metadata(self):
        actual = pattern_2d(**self.custom, fit_to_box=True)
        np.testing.assert_equal(actual.node_positions(),
                                [[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        self.assertEqual(actual.metadata["pattern"]["n_pts_per_side"], 0)
        self.assertEqual((actual.num_nodes, actual.num_edges), (3, 2))

    def test_unfitted_custom_retains_requested_nodes_and_displacements(self):
        actual = pattern_2d(**self.custom, n_pts_per_side=1,
                            point_displacements=[(0, 0.1), (0.1, 0)])
        self.assertEqual((actual.num_nodes, actual.num_edges), (5, 4))
        self.assertEqual(actual.metadata["pattern"]["n_pts_per_side"], 1)
        np.testing.assert_equal(actual.node_positions(),
                                [[0, 0, 0], [.5, .1, 0], [1, 0, 0],
                                 [1.1, .5, 0], [1, 1, 0]])

    def test_valid_generators_have_expected_topology_and_repeatable_geometry(self):
        cases = [
            (dict(self.custom), (3, 2)),
            (dict(self.custom, fit_to_box=True, n_internal=2), (3, 2)),
            (dict(self.custom, n_pts_per_side=1, seed=7), (5, 4)),
            (dict(unit="square", box=(2, 3), grid=(2, 1), seed=7), (6, 7)),
            (dict(unit="honeycomb", box=(2, 3), grid=(1, 1), seed=7), (6, 6)),
            (dict(unit="square", box=(2, 3), grid=(1, 1), fit_to_box=True,
                  n_pts_per_side=1, seed=7), (8, 8)),
        ]
        for options, counts in cases:
            with self.subTest(options=options):
                graph = pattern_2d(**options)
                self.assertEqual((graph.num_nodes, graph.num_edges), counts)
                self.assertEqual(graph.dimension, 2)
                self.assertTrue(np.isfinite(graph.node_positions()).all())
                self.assertEqual(graph.metadata["pattern"]["n_pts_per_side"],
                                 options.get("n_pts_per_side", 0))
                self.assertEqual(graph.to_dict(), pattern_2d(**options).to_dict())
                if options.get("n_internal") == 2:
                    for edge in graph.edges.values():
                        self.assertEqual(edge.internal_points.shape, (2, 3))
        options = dict(unit="cubic", box=(2, 3, 4), grid=(1, 1, 1), seed=7)
        graph = pattern_3d(**options)
        self.assertEqual((graph.num_nodes, graph.num_edges, graph.dimension), (8, 12, 3))
        self.assertEqual(graph.to_dict(), pattern_3d(**options).to_dict())

    def test_snapshot_rejects_planar_bend(self):
        graph = graph_with_points(np.array([[1., 1., 0.]]))
        with self.assertRaisesRegex(ValueError, "internal_points.*edge IDs.*discretize"):
            SnapshotFeatureExtractor().extract_graph(graph)

    def test_snapshot_rejects_hidden_nonplanar_sample(self):
        graph = graph_with_points(np.array([[1., 0., 1.]]))
        with self.assertRaisesRegex(ValueError, "internal_points.*edge IDs.*discretize"):
            SnapshotFeatureExtractor().extract_graph(graph)

    def test_snapshot_rejects_all_nonempty_internal_samples(self):
        for point in ([1., 0., 0.], [0., 0., 0.], [3., 0., 0.]):
            with self.subTest(point=point):
                with self.assertRaisesRegex(ValueError, "internal_points"):
                    SnapshotFeatureExtractor().extract_graph(
                        graph_with_points(np.array([point])))

    def test_snapshot_rejects_before_projection_without_mutation(self):
        graph = graph_with_points(np.array([[1., 1., 0.]]))
        before = json.dumps(graph.to_dict(), sort_keys=True)
        with patch.object(graph, "to_numpy", wraps=graph.to_numpy) as projection:
            with self.assertRaisesRegex(ValueError, "feature-schema conversion"):
                SnapshotFeatureExtractor().extract_graph(graph)
            projection.assert_not_called()
        self.assertEqual(json.dumps(graph.to_dict(), sort_keys=True), before)

    def test_snapshot_lists_physical_edge_ids_without_collapsing(self):
        graph = graph_with_points(np.array([[1., 1., 0.]]))
        second = graph.add_edge(0, 1, allow_parallel=True,
                                internal_points=np.array([[1., -1., 0.]]))
        with self.assertRaises(ValueError) as caught:
            SnapshotFeatureExtractor().extract_graph(graph)
        self.assertIn(repr((0, second)), str(caught.exception))

    def test_straight_none_empty_and_empty_graph_match_explicit_arrays(self):
        graphs = [StructureGraph(dimension=2), graph_with_points(None),
                  graph_with_points(np.empty((0, 3))), graph_with_points([])]
        for index, graph in enumerate(graphs):
            with self.subTest(case=index):
                extractor = SnapshotFeatureExtractor()
                actual = extractor.extract_graph(graph)
                positions, edges, _ = graph.to_numpy()
                expected = extractor.extract(positions[:, :2], edges)
                self.assert_features_equal(actual, expected)
                self.assertEqual(actual["n_node"], graph.num_nodes)
                self.assertEqual(actual["n_edge"], graph.num_edges)
                self.assertEqual(actual["total_length"], 2.0 if graph.num_edges else 0.0)

    def test_generated_straight_snapshot_matches_explicit_arrays(self):
        graph = pattern_2d(unit="square", box=(2, 2), grid=(1, 1))
        positions, edges, _ = graph.to_numpy()
        for rect in (None, (0, 0, 2, 2)):
            with self.subTest(rect=rect):
                extractor = SnapshotFeatureExtractor()
                actual = extractor.extract_graph(graph, rect=rect)
                expected = extractor.extract(positions[:, :2], edges, rect=rect)
                self.assert_features_equal(actual, expected)
                self.assertEqual((actual["n_node"], actual["n_edge"]), (4, 4))
                self.assertEqual(actual["total_length"], 8.0)
                self.assertEqual(actual["pore_count"], 1)
                self.assertEqual(actual["pore_area_mean"], 4.0)

    def test_legacy_dense_row_adapter_unchanged(self):
        class DenseGraph:
            def node_positions(self):
                return np.array([[0., 0., 0.], [2., 0., 0.]])
            def edge_array(self):
                return np.array([[0, 1]], dtype=int)
        extractor = SnapshotFeatureExtractor()
        graph = DenseGraph()
        actual = extractor.extract_graph(graph)
        expected = extractor.extract(graph.node_positions()[:, :2], graph.edge_array())
        self.assert_features_equal(actual, expected)
        self.assertEqual((actual["n_node"], actual["n_edge"], actual["total_length"]), (2, 1, 2.0))

    def test_explicit_piecewise_straight_graph_is_accepted(self):
        graph = StructureGraph(dimension=2)
        ids = [graph.add_node(point) for point in ([0, 0], [1, 1], [2, 0])]
        graph.add_edge(ids[0], ids[1])
        graph.add_edge(ids[1], ids[2])
        result = SnapshotFeatureExtractor().extract_graph(graph)
        self.assertAlmostEqual(result["total_length"], 2 * np.sqrt(2))
        self.assertEqual((result["n_node"], result["n_edge"]), (3, 2))

    def test_geometry_and_snapshot_do_not_import_simulation_providers(self):
        forbidden = ("taichi", "torch", "lammps", "exudyn", "fibernet.sim")
        original_import = builtins.__import__
        attempts = []

        def checked_import(name, *args, **kwargs):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
                attempts.append(name)
                raise AssertionError("geometry analysis attempted a simulation import: " + name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked_import):
            graph = pattern_2d(**self.custom)
            result = SnapshotFeatureExtractor().extract_graph(graph)
        self.assertEqual(attempts, [])
        self.assertEqual((result["n_node"], result["n_edge"]), (3, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
