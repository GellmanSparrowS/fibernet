"""Prevent a 3D prescribed displacement from constraining a different node."""
import unittest
from unittest.mock import patch

import numpy as np

from fibernet import Material, StructureGraph, pattern_3d
from fibernet.ml.beam_frame_fem import BeamFrameFEM


class FEM3DPrescribedWidthTests(unittest.TestCase):
    def setUp(self):
        # Public cubic unit in metres. Exercise explicit stable IDs distinct
        # from node rows when configuring constraints and interpreting results.
        graph = pattern_3d(unit="cubic", box=(.002, .003, .004), grid=(1, 1, 1),
                           radius=.00006)
        raw = graph.to_dict()
        stable_node = {r["id"]: 100 + 37*r["id"] for r in raw["nodes"]}
        for r in raw["nodes"]:
            r["id"] = stable_node[r["id"]]
        for r in raw["edges"]:
            r["node_i"] = stable_node[r["node_i"]]
            r["node_j"] = stable_node[r["node_j"]]
            r["id"] = 500 + 29*r["id"]
        raw["next_node_id"] = max(r["id"] for r in raw["nodes"]) + 1
        raw["next_edge_id"] = max(r["id"] for r in raw["edges"]) + 1
        self.graph = StructureGraph.from_dict(raw)
        self.ids = list(self.graph.nodes)
        self.edge_ids = list(self.graph.edges)
        row = {nid: i for i, nid in enumerate(self.ids)}
        self.x = np.array([self.graph.nodes[nid].position for nid in self.ids])
        self.edges = np.array([[row[e.node_i], row[e.node_j]] for e in self.graph.edges.values()]).T
        self.radii = np.array([e.radius for e in self.graph.edges.values()])
        self.materials = [Material(youngs_modulus=1e9 if i % 2 else 2e9)
                          for i in range(len(self.edge_ids))]
        self.held = np.flatnonzero(self.x[:, 0] == self.x[:, 0].min()).tolist()
        self.selected, self.following = [i for i in range(len(self.ids)) if i not in self.held][:2]
        self.assertEqual(self.following, self.selected + 1)
        self.assertNotEqual(self.ids[self.following], self.following)
        self.solver = BeamFrameFEM()

    def solve(self, prescribed):
        return self.solver.solve_3d(self.edges, self.x, self.radii, fixed_nodes=self.held,
                                    prescribed_disp=prescribed, edge_materials=self.materials)

    def test_seventh_component_never_prescribes_following_node(self):
        zero = (0.,) * 6
        wrong = 5e-7
        self.assertEqual(self.solve({self.selected: zero})["u"][self.following, 0], 0.)
        self.assertEqual(self.solve({self.selected: zero, self.following: (wrong,)})["u"][self.following, 0], wrong)
        with patch.object(self.solver, "section_properties", side_effect=AssertionError("assembled stiffness")):
            with self.assertRaisesRegex(ValueError, "node row .* at most six"):
                self.solve({self.selected: (*zero, wrong)})

    def test_overwidth_one_shot_iterable_is_not_drained_beyond_seventh(self):
        consumed = []

        def incoming():
            for value in range(7):
                consumed.append(value)
                yield 0.
            raise AssertionError("read past one excessive component")

        with self.assertRaisesRegex(ValueError, "at most six"):
            self.solve({self.selected: incoming()})
        self.assertEqual(consumed, list(range(7)))

    def test_empty_prefix_full_rotation_and_fixed_overlap_keep_meaning(self):
        empty = self.solve({})
        np.testing.assert_array_equal(self.solve({self.selected: iter(())})["u"], empty["u"])
        for values in ((1e-7,), (1e-7, -2e-7, 3e-7),
                       (1e-7, -2e-7, 3e-7, 1e-4, -2e-4, 3e-4)):
            with self.subTest(width=len(values)):
                valid = self.solve({self.selected: iter(values)})
                np.testing.assert_array_equal(valid["u"][self.selected, :len(values)], values)
                converted = self.solver.to_sim_result(valid, self.graph, node_ids=self.ids, edge_ids=self.edge_ids)
                self.assertEqual(converted.metadata["node_ids"], self.ids)
                self.assertEqual(converted.metadata["material_contract"]["edge_ids"], self.edge_ids)
                self.assertEqual(list(converted.metadata["material_contract"]["youngs_modulus"]),
                                 [m.youngs_modulus for m in self.materials])
        overlap = self.solve({self.held[0]: (1e-7,)})
        np.testing.assert_array_equal(overlap["u"][self.held[0]], [1e-7, 0., 0., 0., 0., 0.])
