"""Reproduce load-to-resistance coupling on the complete 3D example."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

_here=Path(__file__).resolve().parent
_example=_here / "mechanoelectric_network_3d.py"
if not _example.exists():
    _example=_here.parent / "examples" / _example.name


class MechanoelectricNetworkExample(unittest.TestCase):
    def run_example(self, strain, noisy_import=False, flags=()):
        command = [sys.executable, str(_example), "--strain", str(strain), *flags]
        if noisy_import:
            # This reproduces an optional import banner without requiring the
            # optional dependency on every test runner.
            shim = """
import builtins
import runpy
import sys
old_import = builtins.__import__
def noisy_import(name, *args, **kwargs):
    if name == "fibernet":
        print("[example test] optional dependency import")
    return old_import(name, *args, **kwargs)
builtins.__import__ = noisy_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
            command[1:1] = ["-c", shim]
        with tempfile.TemporaryDirectory() as cwd:
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            trial = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                                   text=True, timeout=60)
            self.assertEqual(trial.returncode, 0, trial.stderr)
            self.assertFalse(list(Path(cwd).iterdir()))
        try:
            result = json.loads(trial.stdout)
        except json.JSONDecodeError as exc:
            self.fail(f"example stdout is not JSON: {exc}; first line: {trial.stdout[:150]!r}")
        if noisy_import:
            self.assertIn("[example test] optional dependency import", trial.stderr)
        return result

    def test_import_banner_does_not_pollute_json(self):
        actual = self.run_example(.005, noisy_import=True)
        self.assertEqual(actual["member_count"], 360)

    def test_full_3d_geometric_sensitivity(self):
        actual=self.run_example(.005)
        self.assertEqual((actual["node_count"],actual["member_count"]),(91,360))
        self.assertGreater(actual["mechanical_energy_J"],0)
        # Independently assembled dense resistor network on the same mechanical field,
        # recorded to 4 decimal places in percentage change.
        r=(.2652,.0300,.0325)
        for axis,expected in zip("xyz",r):
            case=actual["directions"][axis]
            self.assertAlmostEqual(case["fixed_area"]["change_percent"],expected,places=4)
            self.assertGreater(case["fixed_member_volume"]["change_percent"],
                               case["fixed_area"]["change_percent"])

    def test_same_network_reference_and_compression(self):
        ref=self.run_example(0)
        self.assertEqual(ref["mechanical_energy_J"],0)
        comp=self.run_example(-.005)
        for axis in "xyz":
            baseline=ref["directions"][axis]
            altered=comp["directions"][axis]
            for law in ("fixed_area","fixed_member_volume"):
                self.assertAlmostEqual(baseline[law]["change_percent"],0.,places=10)
                self.assertLess(altered[law]["change_percent"],0.)

    def test_opt_in_affine_report_and_default_are_compatible(self):
        import math
        default = self.run_example(.005)
        audited = self.run_example(.005, flags=("--compare-affine",))
        breakdown = audited.pop("fixed_area_attribution")
        self.assertEqual(audited, default)
        ids = breakdown["edge_ids"]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 360)
        for axis, group in breakdown["directions"].items():
            self.assertGreater(group["affine_change_percent"],
                               audited["directions"][axis]["fixed_area"]["change_percent"])
            for key in ("mechanical_vs_baseline", "mechanical_vs_affine"):
                c = group[key]
                v = c["contributions_ohm"]
                self.assertEqual(len(v), len(ids))
                self.assertAlmostEqual(math.fsum(v), c["direct_delta_ohm"], places=8)
                self.assertAlmostEqual(c["sum_positive_ohm"] + c["sum_negative_ohm"],
                                       c["direct_delta_ohm"], places=8)
                lo, hi = c["bounds_ohm"]
                self.assertLessEqual(lo - 1e-8, c["direct_delta_ohm"])
                self.assertLessEqual(c["direct_delta_ohm"], hi + 1e-8)
        y = breakdown["directions"]["y"]
        self.assertAlmostEqual(y["affine_change_percent"], .0932367631, places=8)
        self.assertAlmostEqual(y["mechanical_vs_baseline"]["sum_negative_ohm"], -.9949959452, places=7)

    def test_reversed_edge_current_keeps_negative_signed_contribution(self):
        import runpy
        from types import SimpleNamespace
        import numpy as np

        attribute = runpy.run_path(str(_example))["_attribution"]
        # Independently solve two simple resistive bridges, with terminals
        # indexed 0 (high) and 3 (low). Stable edge 301 reverses current.
        ends = np.array([(0, 1), (1, 3), (0, 2), (2, 3), (1, 2)])
        ids = np.array([10, 11, 20, 21, 301])
        incidence = np.eye(4)[ends[:, 0]] - np.eye(4)[ends[:, 1]]

        def state(resistance, order):
            conductance = 1. / np.asarray(resistance)
            K = incidence.T @ (conductance[:, None] * incidence)
            v = np.r_[1., np.linalg.solve(K[1:3, 1:3], -K[1:3, 0]), 0.]
            current = conductance * (incidence @ v)
            r = 1. / (K @ v)[0]
            return r, SimpleNamespace(edge_ids=ids[order], endpoints_by_id=ends[order],
                                      current_A=current[order], conductance_S=conductance[order])

        before = state([1., 3., 3., 1., 2.], np.arange(5))
        after = state([3., 1., 1., 3., 2.6], np.array([4, 1, 3, 2, 0]))
        split = attribute(before, after, ids)
        self.assertAlmostEqual(split["contributions_ohm"][-1], -.03260869565217, places=12)
        self.assertAlmostEqual(split["signed_sum_ohm"], after[0] - before[0], places=12)
        after[1].endpoints_by_id[0] = after[1].endpoints_by_id[0][::-1]
        with self.assertRaisesRegex(ValueError, "oriented edge identity"):
            attribute(before, after, ids)


if __name__ == "__main__":
    unittest.main()
