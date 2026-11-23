"""Analytic and transaction checks for the prescribed-temperature VTU route."""
import math
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
try:
    import meshio
except ImportError:
    meshio = None

from fibernet.io import tet4_results as io
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4_thermoelastic as mechanics


class OnceReadTemperature(dict):
    """A mapping whose items can be read only once by the solver/export."""

    def __init__(self, pairs):
        super().__init__(pairs)
        self.reads = 0

    def items(self):
        self.reads += 1
        if self.reads != 1:
            raise AssertionError("temperatures were read again after the solve")
        return super().items()


class ThermalTet4Vtu(unittest.TestCase):
    def setUp(self):
        self.points = np.vstack((np.zeros((1, 3)), np.diag([1.0, 2.0, 3.0])))
        self.ids = [2**53 + 3, 5, 76, 2**64 - 3]
        self.cell = 2**53 + 19
        self.material = 2**64 - 1
        self.mesh = Tet4Bundle(
            self.points, np.array([[0, 1, 2, 3]]),
            np.array(self.ids, dtype=np.uint64),
            np.array([self.cell], dtype=np.uint64),
            np.array([self.material], dtype=np.uint64),
            {self.material: {"young_modulus_pa": 6e7, "poisson_ratio": 0.}},
            "mm",
        )
        self.temp = [2.123456789, 6.234567891, 10.25, 14.5]
        self.reference = 1000.0
        self.temperature = OnceReadTemperature(zip(reversed(self.ids), reversed(self.temp)))
        self.options = dict(
            temperature_K_by_id=self.temperature,
            stress_free_temperature_K=self.reference,
            alpha_per_K_by_material={self.material: 1e-6},
            forces_by_id={},
            fixed_dofs_by_id=dict.fromkeys(self.ids, ("x", "y", "z")),
        )
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_missing_meshio_refuses_before_consuming_inputs(self):
        with patch.object(io, "_meshio", side_effect=ImportError):
            with self.assertRaisesRegex(ImportError, "optional meshio"):
                io.solve_prescribed_temperature_tet4_to_vtu(self.root / "want.vtu", self.mesh, **self.options)
        self.assertEqual(self.temperature.reads, 0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_target_does_not_read_inputs_or_overwrite(self):
        p = self.root / "prior.VTU"
        p.write_bytes(b"do not replace")
        with patch.object(io, "_meshio", side_effect=AssertionError("provider called first")):
            with self.assertRaises(FileExistsError):
                io.solve_prescribed_temperature_tet4_to_vtu(p, self.mesh, **self.options)
        self.assertEqual(p.read_bytes(), b"do not replace")
        self.assertEqual(self.temperature.reads, 0)
        self.assertEqual(list(self.root.iterdir()), [p])

    @unittest.skipIf(meshio is None, "actual Meshio provider not installed")
    def test_blocked_affine_temperature_exact_provider_round_trip(self):
        # T minus its reference and back is not bit-exact for this example.
        self.assertNotEqual(self.temp[0], (self.temp[0] - self.reference) + self.reference)
        p = self.root / "fields.vtu"
        with patch.object(mechanics, "_solve_prescribed_temperature_tet4_on_validated_mesh",
                          wraps=mechanics._solve_prescribed_temperature_tet4_on_validated_mesh) as solve:
            result = io.solve_prescribed_temperature_tet4_to_vtu(p, self.mesh, **self.options)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(self.temperature.reads, 1)
        cells = meshio.read(p)
        np.testing.assert_array_equal(cells.points, self.points * 1e-3)
        np.testing.assert_array_equal(cells.cells_dict["tetra"], self.mesh.tetrahedra)
        for n, vals in (("node_id", self.mesh.node_ids), ("temperature_K", self.temp),
                        ("displacement_m", np.zeros((4, 3)))):
            np.testing.assert_array_equal(cells.point_data[n], vals)
        for n, vals in (("cell_id", [self.cell]), ("material_id", [self.material])):
            actual = cells.cell_data[n][0]
            self.assertEqual(actual.dtype.kind, "u")
            self.assertEqual(actual.dtype.itemsize, 8)
            np.testing.assert_array_equal(actual, np.array(vals, dtype=np.uint64))
        self.assertEqual(cells.point_data["node_id"].dtype.kind, "u")
        dT = [t - self.reference for t in self.temp]
        mean = math.fsum(dT) / 4
        squared = (math.fsum(dT)**2 + math.fsum(t*t for t in dT)) / 20
        exact_energy_J = 1.5 * 6e7 * 1e-12 * squared * 1e-9  # volume = 1 mm³
        mean_only = 1.5 * 6e7 * 1e-12 * mean**2 * 1e-9
        self.assertGreater(result.elastic_energy_J, mean_only)
        np.testing.assert_allclose(cells.cell_data["elastic_energy_J"][0], [exact_energy_J], rtol=1e-12)
        sigma = -6e7 * 1e-6 * mean * np.eye(3).reshape(1, 9)
        np.testing.assert_allclose(cells.cell_data["mean_stress_Pa"][0], sigma, rtol=1e-12, atol=0.)
        self.assertEqual(list(self.root.iterdir()), [p])

    @unittest.skipIf(meshio is None, "actual Meshio provider not installed")
    def test_invalid_controls_do_not_leave_target(self):
        p = self.root / "invalid.vtu"
        self.options["temperature_K_by_id"] = {0: 312.0}
        with self.assertRaises(ValueError):
            io.solve_prescribed_temperature_tet4_to_vtu(p, self.mesh, **self.options)
        self.assertEqual(list(self.root.iterdir()), [])

    @unittest.skipIf(meshio is None, "actual Meshio provider not installed")
    def test_concurrent_target_is_preserved_and_staging_cleaned(self):
        p = self.root / "race.vtu"
        real_link = os.link

        def concurrent_writer(stage, destination):
            Path(destination).write_bytes(b"other writer won")
            return real_link(stage, destination)

        with patch.object(io.os, "link", side_effect=concurrent_writer):
            with self.assertRaises(FileExistsError):
                io.solve_prescribed_temperature_tet4_to_vtu(p, self.mesh, **self.options)
        self.assertEqual(p.read_bytes(), b"other writer won")
        self.assertEqual(list(self.root.iterdir()), [p])


if __name__ == "__main__":
    unittest.main()
