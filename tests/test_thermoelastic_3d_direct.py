"""Opt-in public import and immutable owned-result contract checks."""
import inspect
from pathlib import Path
import subprocess
import sys
import unittest


class DirectThermoelastic3D(unittest.TestCase):
    def test_new_facade_is_lazy_and_separate(self):
        script = """
import sys
from fibernet.sim import thermoelastic
old_names = tuple(thermoelastic.__all__)
from fibernet.sim.thermoelastic_3d import solve_thermoelastic_network_snapshot_3d
assert 'fibernet.sim._thermoelastic_3d' not in sys.modules
assert 'solve_thermoelastic_network_snapshot_3d' not in old_names
assert tuple(thermoelastic.__all__) == old_names
"""
        subprocess.run([sys.executable, "-B", "-c", script], check=True, timeout=30)

    def test_signature_and_documented_fresh_rod_owned_result(self):
        from fibernet.sim._thermoelastic_3d import _solve_thermoelastic_3d
        from fibernet.sim.thermoelastic_3d import (
            ThermoelasticSnapshot3D,
            solve_thermoelastic_network_snapshot_3d as solve,
        )
        self.assertEqual(
            inspect.signature(solve).parameters,
            inspect.signature(_solve_thermoelastic_3d).parameters,
        )
        doc = Path(__file__).resolve().parents[1] / "docs" / "THERMOELASTIC_3D.md"
        source = doc.read_text(encoding="utf-8").split("```python\n", 1)[1].split("```", 1)[0]
        context = {}
        exec(compile(source, str(doc), "exec"), context)
        r = context["result"]
        self.assertIsInstance(r, ThermoelasticSnapshot3D)
        self.assertEqual(r.node_ids, tuple(context["rod"].nodes))
        self.assertEqual(r.edge_ids, tuple(context["rod"].edges))
        self.assertEqual(len(r.displacement_m_and_rad[0]), 6)
        self.assertAlmostEqual(r.axial_force_N[0], 0.0)
        self.assertAlmostEqual(r.elastic_energy_J, 0.0)
        self.assertEqual(r.applicability.edge_diagnostics[0].edge_id, 30)
        self.assertEqual(dict(r.applicability.edge_diagnostics[0].thermal_strain_by_endpoint_id),
                         {10: 1e-5, 20: 1e-5})

        def immutable(value):
            if type(value) in (str, float, int, type(None)):
                return
            self.assertIsInstance(value, tuple)
            for item in value:
                immutable(item)

        immutable(r)
        with self.assertRaises(AttributeError):
            r.node_ids = ()
        with self.assertRaises(AttributeError):
            r.applicability.edge_diagnostics[0].edge_id = 200
        context["rod"].nodes.clear()
        context["rod"].edges.clear()
        self.assertEqual(r.node_ids, (10, 20))
        self.assertEqual(r.edge_ids, (30,))
        self.assertAlmostEqual(r.displacement_m_and_rad[1][0], 2e-7, places=15)


if __name__ == "__main__":
    unittest.main()
