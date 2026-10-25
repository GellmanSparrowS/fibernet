"""Small API UX regressions: shapes, material ownership, boundary-mode preflight."""
import unittest
from unittest.mock import patch
import numpy as np
import fibernet as fn
import fibernet.materials as materials
import fibernet.gen.pattern as pattern


class GeometryApiUxContractTests(unittest.TestCase):
    def test_copy_helper_is_private(self):
        self.assertNotIn("deepcopy", vars(materials))

    def test_numpy_empty_graph_shapes_and_dtypes(self):
        p, e, r = fn.StructureGraph().to_numpy()
        self.assertEqual((p.shape, e.shape, r.shape), ((0, 3), (0, 2), (0,)))
        self.assertEqual(e.dtype, np.dtype(np.int32))

    def test_numpy_isolate_and_removed_last_edge(self):
        g = fn.StructureGraph()
        a, b = g.add_node((0, 0)), g.add_node((1, 0))
        for state in ("isolates", "last_edge_removed"):
            if state == "last_edge_removed":
                g.remove_edge(g.add_edge(a, b))
            p, e, r = g.to_numpy()
            self.assertEqual((p.shape, e.shape, r.shape), ((2, 3), (0, 2), (0,)))
            self.assertEqual(e.dtype, np.dtype(np.int32))

    def test_numpy_sparse_ids_parallel_rows_unchanged(self):
        g = fn.StructureGraph()
        dead = g.add_node((-1, 0))
        a, b = g.add_node((0, 0)), g.add_node((1, 0))
        g.remove_node(dead)
        first = g.add_edge(a, b, radius=0.1)
        second = g.add_edge(b, a, radius=0.2, allow_parallel=True)
        p, e, r = g.to_numpy()
        np.testing.assert_array_equal(p, [[0, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(e, [[0, 1], [1, 0]])
        np.testing.assert_array_equal(r, [0.1, 0.2])
        self.assertEqual(list(g.edges), [first, second])

    def test_material_independent_scalars_and_shortcut(self):
        first, second = materials.get_material("steel"), materials.get_material("steel")
        self.assertIsNot(first, second)
        old = second.youngs_modulus
        first.youngs_modulus = 123.
        self.assertEqual(materials.get_material("steel").youngs_modulus, old)
        self.assertEqual(materials.m("steel").youngs_modulus, old)

    def test_material_nested_state_and_tensor_detached(self):
        material = fn.Material(name="fixture", extra={"nested": {"values": [1, 2]}},
                               stiffness_tensor=np.eye(6))
        database = {"fixture": material}
        with patch.object(materials, "get_material_database", return_value=database):
            first = materials.get_material("fixture")
            second = materials.get_material("fixture")
            first.extra["nested"]["values"].append(3)
            first.stiffness_tensor[0, 0] = 9
            self.assertEqual(second.extra["nested"]["values"], [1, 2])
            self.assertEqual(database["fixture"].extra["nested"]["values"], [1, 2])
            self.assertEqual(second.stiffness_tensor[0, 0], 1)
            self.assertEqual(database["fixture"].stiffness_tensor[0, 0], 1)

    def test_material_lookup_and_comparison_remain_available(self):
        self.assertEqual(materials.get_material("carbon-fiber").name,
                         materials.get_material("carbon fiber").name)
        self.assertIn("steel", materials.list_materials())
        self.assertIn("youngs_moduli", materials.compare_materials(["steel", "aluminum"]))
        with self.assertRaisesRegex(KeyError, "Available"):
            materials.get_material("not-a-material")

    def test_invalid_boundary_mode_rejected_before_factory(self):
        with patch.dict(pattern._UNIT_FACTORIES, {"square": lambda **kw: self.fail("factory called")}):
            for value in ("typo", "", None, False, ["none"]):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "boundary_mode"):
                        fn.pattern_2d(unit="square", grid=1, boundary_mode=value)

    def test_valid_boundary_modes_keep_default_geometry(self):
        for mode in ("none", "extend", "error"):
            reference = fn.pattern_2d(unit="square", grid=1, seed=23, boundary_mode=mode)
            result = fn.pattern_2d(unit="square", grid=1, seed=23, boundary_mode=mode)
            self.assertEqual((result.num_nodes, result.num_edges),
                             (reference.num_nodes, reference.num_edges))
            np.testing.assert_array_equal(result.node_positions(), reference.node_positions())
            np.testing.assert_array_equal(result.edge_array(), reference.edge_array())

    def test_native_ids_and_multiedge_analysis_roundtrip(self):
        from fibernet.analysis import analyze_topology, analyze_geometry_paths, analyze_spectrum
        g = fn.StructureGraph()
        dead = g.add_node((-1, 0))
        a, b = g.add_node((0, 0)), g.add_node((1, 0))
        g.remove_node(dead)
        g.add_edge(a, b)
        g.add_edge(a, b, allow_parallel=True)
        before = g.to_dict()
        for restored in (fn.StructureGraph.from_dict(before),
                         fn.StructureGraph.from_networkx_lossless(g.to_networkx_lossless())):
            self.assertEqual(sorted(restored.nodes), [a, b])
            self.assertEqual(list(restored.edges), list(g.edges))
            topology = analyze_topology(restored)
            self.assertEqual((topology.n_physical_edges, topology.n_unique_connections), (2, 1))
            self.assertEqual(topology.bridge_edge_ids, ())
            geometry = analyze_geometry_paths(restored, length_unit="m", coordinate_frame="reference",
                                              node_pairs=[(a, b)])
            self.assertEqual(geometry.pair_distance, (1.0,))
            self.assertAlmostEqual(analyze_spectrum(restored).lambda_max, 4.)
            self.assertAlmostEqual(analyze_spectrum(restored, adjacency="unique").lambda_max, 2.)
        self.assertEqual(g.to_dict(), before)

    def test_seeded_2d_and_3d_generation_unchanged(self):
        for builder, unit in ((fn.pattern_2d, "square"), (fn.pattern_3d, "cubic")):
            first = builder(unit=unit, grid=1, n_pts_per_side=1, seed=42)
            second = builder(unit=unit, grid=1, n_pts_per_side=1, seed=42)
            self.assertEqual(first.fingerprint(), second.fingerprint())


if __name__ == "__main__":
    unittest.main()
