"""Stored edge samples survive filling missing StructureGraph samples.

Only graph operations and a provider-free geometry-policy preflight are used.
"""
import copy
import unittest

import numpy as np

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim._taichi_geometry import (
    UnsupportedInternalPointsError,
    preflight_taichi_geometry,
)


def make_graph(empty_straight=False):
    material = Material(
        name="fixture", youngs_modulus=1200.0, poissons_ratio=0.25,
        density=1100.0, extra={"nested": {"labels": ["source"]}},
        stiffness_tensor=np.eye(6),
    ).to_dict()
    second_material = Material(
        name="parallel-fixture", youngs_modulus=2400.0, poissons_ratio=0.2,
        density=900.0, extra={"nested": {"labels": ["parallel"]}},
        stiffness_tensor=2 * np.eye(6),
    ).to_dict()
    return StructureGraph.from_dict({
        "version": "3.1", "dimension": 3, "tolerance": 1e-7,
        "box_size": [3.0, 4.0, 5.0], "next_node_id": 20, "next_edge_id": 30,
        "metadata": {"units": {"length": "m"}, "nested": {"labels": ["source"]}},
        "nodes": [
            {"id": 4, "position": [0.0, 0.0, 0.0],
             "boundary": [True, False, False, False, False, False],
             "metadata": {"nested": {"labels": ["left"]}}},
            {"id": 9, "position": [2.0, 0.0, 0.0],
             "boundary": [False, True, False, False, False, False],
             "metadata": {"nested": {"labels": ["right"]}}},
        ],
        "edges": [
            {"id": 12, "node_i": 4, "node_j": 9, "radius": 0.1,
             "material": copy.deepcopy(material), "segments": 5,
             "internal_points": [[0.5, 1.0, 2.0], [1.5, -1.0, 3.0]],
             "metadata": {"nested": {"labels": ["curved"]}}},
            {"id": 13, "node_i": 9, "node_j": 4, "radius": 0.2,
             "material": second_material, "segments": 7,
             "internal_points": [] if empty_straight else None,
             "metadata": {"nested": {"labels": ["straight"]}}},
        ],
    })


def without_samples(graph):
    record = graph.to_dict()
    for edge in record["edges"]:
        edge.pop("internal_points")
    return record


class StructureGraphDiscretizationTests(unittest.TestCase):
    def test_stored_samples_survive_default_zero_and_one(self):
        for count in (None, 0, 1):
            with self.subTest(count=count):
                graph = make_graph()
                before = copy.deepcopy(graph.to_dict())
                result = (graph.discretize_edges() if count is None
                          else graph.discretize_edges(count))
                self.assertIs(type(result), StructureGraph)
                self.assertIsNot(result, graph)
                np.testing.assert_array_equal(
                    result.edges[12].internal_points,
                    graph.edges[12].internal_points,
                )
                self.assertEqual(graph.to_dict(), before)
                self.assertEqual(
                    result.discretize_edges(0).to_dict(), result.to_dict()
                )

    def test_none_and_empty_edges_keep_existing_linear_sampling(self):
        for empty in (False, True):
            for count in (0, 1, 8):
                with self.subTest(empty=empty, count=count):
                    graph = make_graph(empty_straight=empty)
                    before = copy.deepcopy(graph.to_dict())
                    result = graph.discretize_edges(count)
                    t = np.linspace(0, 1, count + 2)[1:-1, None]
                    expected = (graph.nodes[9].position[None, :] * (1 - t)
                                + graph.nodes[4].position[None, :] * t)
                    np.testing.assert_array_equal(
                        result.edges[13].internal_points, expected
                    )
                    self.assertEqual(
                        result.edges[13].internal_points.shape, (count, 3)
                    )
                    self.assertEqual(graph.to_dict(), before)

    def test_complete_records_and_parallel_identity_are_preserved(self):
        graph = make_graph()
        graph._nodes = {9: graph.nodes[9], 4: graph.nodes[4]}
        graph._edges = {13: graph.edges[13], 12: graph.edges[12]}
        result = graph.discretize_edges()
        self.assertEqual(list(result.nodes), [9, 4])
        self.assertEqual(list(result.edges), [13, 12])
        self.assertEqual((result.num_nodes, result.num_edges), (2, 2))
        self.assertEqual(without_samples(result), without_samples(graph))
        self.assertEqual((result.edges[13].node_i, result.edges[13].node_j), (9, 4))
        self.assertEqual(result.add_node([10, 0, 0], merge=False), 20)
        self.assertEqual(result.add_edge(4, 9, allow_parallel=True), 30)
        self.assertEqual((graph._next_node_id, graph._next_edge_id), (20, 30))

    def test_result_owns_arrays_materials_and_nested_records(self):
        graph = make_graph()
        before = copy.deepcopy(graph.to_dict())
        result = graph.discretize_edges()
        another = graph.discretize_edges()
        for edge_id in graph.edges:
            self.assertIsNot(result.edges[edge_id].material, graph.edges[edge_id].material)
        self.assertFalse(np.shares_memory(
            result.edges[12].internal_points, graph.edges[12].internal_points
        ))
        result.nodes[4].position[0] = 99
        result.edges[12].internal_points[0, 0] = 99
        self.assertFalse(np.shares_memory(result._box_size, graph._box_size))
        result._box_size[0] = 99
        result.metadata["nested"]["labels"].append("result")
        result.nodes[4].metadata["nested"]["labels"].append("result")
        result.edges[12].metadata["nested"]["labels"].append("result")
        result.edges[12].material.extra["nested"]["labels"].append("result")
        result.edges[12].material.stiffness_tensor[0, 0] = 99
        result.edges[12].material.youngs_modulus = 99
        self.assertEqual(graph.to_dict(), before)
        self.assertEqual(another.to_dict(), graph.discretize_edges().to_dict())

    def test_collinear_duplicate_and_backtracking_samples_are_not_simplified(self):
        samples = (
            [[0.5, 0, 0], [1.5, 0, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [[1.5, 0, 0], [0.5, 0, 0]],
        )
        for points in samples:
            for count in (0, 1, 8):
                with self.subTest(points=points, count=count):
                    graph = make_graph()
                    graph.edges[12].internal_points = np.asarray(points, dtype=float)
                    result = graph.discretize_edges(count)
                    np.testing.assert_array_equal(
                        result.edges[12].internal_points,
                        graph.edges[12].internal_points,
                    )

    def test_pure_geometry_policy_still_sees_every_stored_sampled_edge(self):
        for count in (0, 1, 8):
            with self.subTest(count=count):
                result = make_graph().discretize_edges(count)
                expected = (12,) if count == 0 else (12, 13)
                with self.assertRaises(UnsupportedInternalPointsError) as caught:
                    preflight_taichi_geometry(result)
                self.assertEqual(caught.exception.rejected_edge_ids, expected)
                explicit = preflight_taichi_geometry(
                    result, geometry_policy="endpoint_chord"
                )
                self.assertEqual(
                    explicit["ignored_internal_point_edge_ids"], list(expected)
                )
        straight = make_graph()
        straight.edges[12].internal_points = None
        self.assertEqual(
            preflight_taichi_geometry(straight.discretize_edges(0))
            ["ignored_internal_point_edge_ids"], []
        )

    def test_count_expression_remains_per_edge_and_empty_graph_skips_it(self):
        class CountProbe:
            def __init__(self):
                self.calls = 0

            def __add__(self, other):
                self.calls += 1
                return 1 + other

        graph = make_graph()
        graph.edges[13].internal_points = np.array([[1., 0., 0.]])
        probe = CountProbe()
        graph.discretize_edges(probe)
        self.assertEqual(probe.calls, graph.num_edges)
        empty_probe = CountProbe()
        empty = StructureGraph(dimension=3)
        result = empty.discretize_edges(empty_probe)
        self.assertEqual(empty_probe.calls, 0)
        self.assertEqual(result.to_dict(), empty.to_dict())


    def test_atomic_position_edits_compose_with_curve_preserving_fill(self):
        graph = make_graph(empty_straight=True)
        original_edges = copy.deepcopy(graph.to_dict()["edges"])
        left = np.array([[3.0, 1.0]], dtype=np.float32)
        right = np.array([[7.0], [5.0], [2.0]], dtype=np.float32)
        graph.set_node_positions({4: left, 9: right})

        # Existing samples are absolute coordinates, not an inferred
        # displacement field. Endpoint edits do not transport stored samples.
        self.assertEqual(graph.to_dict()["edges"], original_edges)
        self.assertFalse(np.shares_memory(graph.nodes[4].position, left))
        self.assertFalse(np.shares_memory(graph.nodes[9].position, right))
        left[...] = 999.0
        right[...] = 999.0
        for node_id, expected in ((4, [3.0, 1.0, 0.0]), (9, [7.0, 5.0, 2.0])):
            np.testing.assert_array_equal(graph.nodes[node_id].position, expected)
            self.assertEqual(graph.nodes[node_id].position.shape, (3,))
            self.assertEqual(graph.nodes[node_id].position.dtype, np.dtype("float64"))

        source_record = copy.deepcopy(graph.to_dict())
        source_hash = copy.deepcopy(graph._spatial_hash)
        result = graph.discretize_edges(1)
        self.assertIs(type(result), StructureGraph)
        self.assertIsNot(result, graph)
        self.assertEqual((result.num_nodes, result.num_edges), (2, 2))
        self.assertEqual(list(result.nodes), [4, 9])
        self.assertEqual(list(result.edges), [12, 13])
        self.assertEqual((result._next_node_id, result._next_edge_id), (20, 30))
        self.assertEqual((result.edges[13].node_i, result.edges[13].node_j), (9, 4))
        self.assertEqual(without_samples(result), without_samples(graph))
        np.testing.assert_array_equal(
            result.edges[12].internal_points,
            np.array(original_edges[0]["internal_points"]),
        )
        np.testing.assert_array_equal(
            result.edges[13].internal_points, [[5.0, 3.0, 1.0]]
        )
        self.assertEqual(result.edges[13].internal_points.shape, (1, 3))
        self.assertEqual(graph.edges[13].internal_points.shape, (0, 3))
        self.assertFalse(np.shares_memory(
            result.edges[12].internal_points, graph.edges[12].internal_points
        ))
        self.assertFalse(np.shares_memory(result._box_size, graph._box_size))
        for node_id in graph.nodes:
            self.assertFalse(np.shares_memory(
                result.nodes[node_id].position, graph.nodes[node_id].position
            ))
        for edge_id in graph.edges:
            self.assertIsNot(result.edges[edge_id].material, graph.edges[edge_id].material)

        # copy() rebuilds a usable merge lookup from the normalized positions.
        result_record = copy.deepcopy(result.to_dict())
        self.assertEqual(result.add_node([3.0, 1.0, 0.0]), 4)
        self.assertEqual(result.add_node([7.0, 5.0, 2.0]), 9)
        self.assertEqual(result.to_dict(), result_record)

        def state(value):
            return copy.deepcopy((
                value.to_dict(), value._spatial_hash, value._edge_set,
                list(value.nodes), list(value.edges),
            ))

        before_rejected_batch = state(result)
        with self.assertRaises(ValueError):
            result.set_node_positions({4: [[101.0, 201.0]], 9: [[1, 2, 3, 4]]})
        self.assertEqual(state(result), before_rejected_batch)
        self.assertEqual(result.add_node([3.0, 1.0, 0.0]), 4)
        self.assertEqual(result.add_node([7.0, 5.0, 2.0]), 9)
        self.assertEqual(state(result), before_rejected_batch)

        # Mutating the composed result must not change the source graph.
        result.set_node_position(4, [30.0, 10.0, 0.0])
        result.edges[12].internal_points[0, 0] += 10.0
        result.edges[13].internal_points[0, 0] += 10.0
        result._box_size[0] += 10.0
        result.metadata["nested"]["labels"].append("result")
        result.nodes[4].metadata["nested"]["labels"].append("result")
        result.edges[12].metadata["nested"]["labels"].append("result")
        result.edges[12].material.extra["nested"]["labels"].append("result")
        result.edges[12].material.stiffness_tensor[0, 0] += 10.0
        result.edges[13].material.youngs_modulus += 10.0
        self.assertEqual(graph.to_dict(), source_record)
        self.assertEqual(graph._spatial_hash, source_hash)


if __name__ == "__main__":
    unittest.main()

