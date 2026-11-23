"""Independent analytic tests for prescribed-temperature Tet4 elasticity."""
from dataclasses import replace
import math
import unittest

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_thermoelastic import solve_prescribed_temperature_tet4


class Tet4ThermoelasticAnalyticTests(unittest.TestCase):
    def setUp(self):
        self.ids = (42, 501, 118, 900)
        self.cell_id, self.material_id = 68, 21
        self.sides_m = (1/16, 1/8, 1/4)
        self.volume = math.prod(self.sides_m) / 6
        self.E = 3 * 2.0**20
        self.alpha = 2.0**-16
        self.reference = 292.0
        self.increments = np.array([2.0, 6.0, 10.0, 14.0])
        self.mesh = Tet4Bundle(
            np.vstack((np.zeros((1, 3)), np.diag(self.sides_m))),
            np.array([[0, 1, 2, 3]]),
            np.array(self.ids, np.uint64),
            np.array([self.cell_id], np.uint64),
            np.array([self.material_id], np.uint64),
            {self.material_id: {"young_modulus_pa": self.E, "poisson_ratio": 0.0}},
            "m",
        )
        # For nu=0, an entirely held affine Tet4 has mean sigma=-E*alpha*mean(T-T0)*I.
        # In an axis-aligned tetrahedron the apex gradient on axis i is 1/L_i.
        mean = float(np.mean(self.increments))
        self.pressure = self.E * self.alpha * mean
        self.equivalent = np.vstack((
            -self.volume * self.pressure / np.asarray(self.sides_m),
            np.diag(self.volume * self.pressure / np.asarray(self.sides_m)),
        ))
        # For uniform volume measure E[N_i^2]=1/10 and E[N_i N_j]=1/20.
        self.variance = float(np.sum((self.increments - mean)**2) / 20)
        self.energy = 1.5 * self.volume * self.E * self.alpha**2 * (mean**2 + self.variance)
        self.mean_only_energy = 1.5 * self.volume * self.E * (self.alpha * mean)**2

    def solve(self, holds, forces):
        return solve_prescribed_temperature_tet4(
            self.mesh,
            temperature_K_by_id=dict(zip(self.ids, self.reference + self.increments)),
            stress_free_temperature_K=self.reference,
            alpha_per_K_by_material={self.material_id: self.alpha},
            fixed_dofs_by_id=holds,
            forces_by_id=forces,
        )

    def assert_thermal_fields(self, result):
        np.testing.assert_array_equal(result.node_ids, self.ids)
        np.testing.assert_array_equal(result.cell_ids, [self.cell_id])
        np.testing.assert_array_equal(result.material_ids, [self.material_id])
        np.testing.assert_array_equal(result.displacement_m, np.zeros((4, 3)))
        np.testing.assert_allclose(result.mean_stress_Pa[0], -self.pressure * np.eye(3),
                                   rtol=2e-12, atol=1e-11)
        self.assertEqual(self.variance, 4.0)
        self.assertGreater(result.elastic_energy_J, self.mean_only_energy)
        np.testing.assert_allclose(result.elastic_energy_J_by_cell, [self.energy], rtol=2e-12)
        np.testing.assert_allclose(result.elastic_energy_J, self.energy, rtol=2e-12)
        self.assertEqual(result.terminal_mechanical_pairing_J, 0.0)
        np.testing.assert_allclose(result.stress_eigenstrain_integral_J,
                                   -2.0 * self.energy, rtol=2e-12)

    def test_blocked_nonuniform_temperature_has_thermal_variance_energy(self):
        holds = {node: ("x", "y", "z") for node in self.ids}
        result = self.solve(holds, {})
        self.assert_thermal_fields(result)
        np.testing.assert_array_equal(result.fixed_mask, np.ones((4, 3), dtype=bool))
        np.testing.assert_allclose(result.reaction_N, -self.equivalent, atol=1e-12)

        # The established static API still interprets the same nodal vectors as
        # mechanical loads; it has no temperature or physical thermal energy.
        static = solve_static_tet4(
            self.mesh, fixed_dofs_by_id=holds,
            forces_by_id=dict(zip(self.ids, map(tuple, self.equivalent))),
        )
        np.testing.assert_array_equal(static.node_ids, result.node_ids)
        np.testing.assert_allclose(static.reaction_N, result.reaction_N, atol=1e-12)
        self.assertEqual(static.strain_energy_J, 0.0)
        self.assertIsNone(static.cell_fields)

    def test_exact_opposed_external_load_keeps_compressed_energy(self):
        a, b, c, _ = self.ids
        holds = {a: ("x", "y", "z"), b: ("y", "z"), c: ("z",)}
        # Sheared dyadic simplex. Its normalized edge matrix has unit
        # diagonal, determinant 1; do not use a rounded near-cancellation
        # from a different geometry as proof of genuinely exact opposition.
        self.mesh = replace(self.mesh, points=np.array([
            [0.0, 0.0, 0.0], [1/8, 0.0, 0.0], [1/16, 1/8, 0.0], [0.0, 1/16, 1/8],
        ]))
        # Solve the barycentric equations directly for this geometry:
        # N3=8z, N2=8y-4z, N1=8x-4y+2z, N0=1-N1-N2-N3.
        gradients = np.array([[-8., -4., -6.], [8., -4., 2.],
                              [0., 8., -4.], [0., 0., 8.]])
        thermal = self.volume * self.pressure * gradients
        np.testing.assert_array_equal(
            thermal, [[-1., -.5, -.75], [1., -.5, .25], [0., 1., -.5], [0., 0., 1.]],
        )
        forces = dict(zip(self.ids, map(tuple, -thermal)))
        result = self.solve(holds, forces)
        self.assert_thermal_fields(result)
        np.testing.assert_array_equal(result.reaction_N, np.zeros((4, 3)))
        np.testing.assert_array_equal(result.fixed_mask, [
            [True, True, True], [False, True, True],
            [False, False, True], [False, False, False],
        ])
        self.assertEqual(result.physical_free_residual_inf_N, 0.0)


    def test_compatible_free_expansion_with_two_material_stiffnesses(self):
        ids = (70, 11, 320, 94, 56)
        points_mm = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.],
        ])
        mesh = Tet4Bundle(
            points_mm,
            np.array([[0, 1, 2, 3], [0, 2, 1, 4]]),
            np.array(ids, np.uint64),
            np.array([460, 12], np.uint64),
            np.array([17, 9], np.uint64),
            {
                17: {"young_modulus_pa": 1e8, "poisson_ratio": 0.20},
                9: {"young_modulus_pa": 3e8, "poisson_ratio": 0.33},
            },
            "mm",
        )
        # Only remove rigid motion. The same free thermal strain in both
        # cells is kinematically compatible even though their stiffnesses differ.
        for dT in (10., -10.):
            with self.subTest(dT=dT):
                result = solve_prescribed_temperature_tet4(
                    mesh, temperature_K_by_id=dict.fromkeys(ids, 300. + dT),
                    stress_free_temperature_K=300.,
                    alpha_per_K_by_material={17: 1e-5, 9: 1e-5},
                    fixed_dofs_by_id={70: ("x", "y", "z"), 11: ("y", "z"), 320: ("z",)},
                    forces_by_id={},
                )
                np.testing.assert_allclose(
                    result.displacement_m, points_mm * (1e-3 * 1e-5 * dT),
                    rtol=3e-13, atol=3e-20,
                )
                np.testing.assert_allclose(result.mean_stress_Pa, 0., rtol=0., atol=2e-10)
                np.testing.assert_allclose(result.reaction_N, 0., rtol=0., atol=1e-15)
                self.assertLess(result.physical_free_residual_inf_N, 1e-15)
                self.assertGreaterEqual(result.elastic_energy_J, 0.)
                self.assertLess(result.elastic_energy_J, 1e-32)
                self.assertEqual(result.terminal_mechanical_pairing_J, 0.)


if __name__ == "__main__":
    unittest.main()
