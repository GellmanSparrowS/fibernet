"""Analytical and boundary checks for the direct 3D thermal beam snapshot."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from fibernet import Material, StructureGraph
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim.thermoelastic_3d import solve_thermoelastic_network_snapshot_3d


_LIMITS = {
    "endpoint_thermal_strain": 1e-4,
    "nodal_rotation": 1e-3,
    "transverse_slope_bound": 1e-3,
    "total_extreme_fiber_strain": 1e-4,
    "endpoint_elastic_extreme_fiber_strain": 1e-4,
    "twist_shear": 1e-4,
}


class Thermoelastic3DPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.length = 0.02
        self.radius = 0.001
        self.youngs = 2e9
        self.alpha = 2e-6
        self.graph = graph = StructureGraph(dimension=3)
        root = graph.add_node((0.0, 0.0, 0.0))
        tip = graph.add_node((self.length, 0.0, 0.0))
        graph.add_edge(root, tip, radius=self.radius,
                       material=Material(youngs_modulus=self.youngs, poissons_ratio=0.3))
        data = graph.to_dict()
        self.root, self.tip, self.edge = 97, 13, 501
        ids = {root: self.root, tip: self.tip}
        for node in data["nodes"]:
            node["id"] = ids[node["id"]]
        edge = data["edges"][0]
        edge.update(id=self.edge, node_i=self.root, node_j=self.tip)
        data.update(next_node_id=98, next_edge_id=502)
        self.graph = StructureGraph.from_dict(data)

    def solve(self, **replace):
        data = {
            "temperature_K_by_node_id": {self.root: 304.0, self.tip: 304.0},
            "stress_free_temperature_K": 300.0,
            "expansion_alpha_per_K_by_edge_id": {self.edge: self.alpha},
            "constraints_by_node_id": {self.root: (0.0,) * 6},
            "length_unit": "m",
            "applicability_limits": _LIMITS,
        }
        return solve_thermoelastic_network_snapshot_3d(self.graph, **(data | replace))

    def test_prescribed_rotation_does_not_lock_unmentioned_transverse_dofs(self):
        L, r, E = self.length, self.radius, self.youngs
        s, theta, F = 4e-7, 1e-5, 1e-4
        result = self.solve(
            constraints_by_node_id={self.root: (0.0,) * 6,
                                    self.tip: (s, None, None, None, theta, None)},
            loads_N_by_node_id={self.tip: (0.0, F, 0.0)},
        )
        A, I = math.pi * r**2, math.pi * r**4 / 4
        elastic_shift = s - L * self.alpha * 4
        axial_force = E * A * elastic_shift / L
        y = F * L**3 / (3 * E * I)
        tip_moment = E * I * theta / L
        tip = result.node_ids.index(self.tip)
        root = result.node_ids.index(self.root)
        edge = result.edge_ids.index(self.edge)
        np.testing.assert_allclose(
            result.displacement_m_and_rad[tip],
            [s, y, -L * theta / 2, 0.0, theta, F * L**2 / (2 * E * I)],
            atol=1e-14, rtol=2e-8,
        )
        np.testing.assert_allclose(
            result.reactions_N_and_Nm[root],
            [-axial_force, -F, 0.0, 0.0, -tip_moment, -F * L],
            atol=1e-11, rtol=2e-8,
        )
        np.testing.assert_allclose(
            result.reactions_N_and_Nm[tip],
            [axial_force, 0.0, 0.0, 0.0, tip_moment, 0.0],
            atol=1e-11, rtol=2e-8,
        )
        np.testing.assert_allclose(result.axial_force_N[edge], axial_force, rtol=2e-8)
        np.testing.assert_allclose(
            result.elastic_energy_J,
            0.5 * (axial_force * elastic_shift + F * y + tip_moment * theta),
            rtol=2e-8, atol=0.0,
        )

    def test_opposite_endpoint_thermal_strains_reject_before_assembly(self):
        # Both local strains violate the ceiling, although their mean is zero.
        with patch.object(BeamFrameFEM, "_assemble_3d_physical_stiffness",
                          side_effect=AssertionError("assembly before endpoint admission")):
            with self.assertRaisesRegex(ValueError, "endpoint_thermal_strain"):
                self.solve(
                    temperature_K_by_node_id={self.root: 289.0, self.tip: 311.0},
                    applicability_limits={**_LIMITS, "endpoint_thermal_strain": 1e-5},
                )

    def test_translation_only_root_leaves_rigid_modes(self):
        # The axial thermal load is compatible with the singular system.
        # A tiny residual by itself would not prove the missing constraints.
        with self.assertRaisesRegex(ValueError, "underconstrained|singular"):
            self.solve(
                constraints_by_node_id={self.root: (0.0, 0.0, 0.0, None, None, None)},
            )

    def test_saved_electrothermal_route_still_rejects_3d(self):
        from fibernet.sim.electrothermal import TemperatureConductanceLaw, solve_electrothermal
        from fibernet.sim.saved_electrothermal import (
            export_saved_electrothermal,
            prepare_saved_electrothermal_temperature,
        )

        node_values = {self.root: 300.0, self.tip: 300.0}
        weights = {self.edge: {self.root: 0.5, self.tip: 0.5}}
        law = TemperatureConductanceLaw(
            {self.edge: 1.0}, {self.edge: 0.0}, {self.edge: 300.0},
            weights, (299.0, 301.0),
        )
        upstream = solve_electrothermal(
            self.graph, law, prescribed_voltage_V={self.root: 0.0, self.tip: 0.0},
            capacity_J_per_K={self.root: 1.0, self.tip: 1.0},
            conductance_W_per_K={self.edge: 1.0}, initial_temperature_K=node_values,
            times_s=(0.0, 0.125), endpoint_heat_weights=weights, length_unit="m",
        )
        self.assertTrue(upstream.completed)
        saved = export_saved_electrothermal(self.graph, upstream, length_unit="m")
        with self.assertRaisesRegex(ValueError, "requires dimension=2"):
            prepare_saved_electrothermal_temperature(self.graph, saved, frame_index=1,
                                                    length_unit="m")


if __name__ == "__main__":
    unittest.main()
