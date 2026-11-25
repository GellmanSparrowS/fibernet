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
    def run_example(self, strain, noisy_import=False):
        command = [sys.executable, str(_example), "--strain", str(strain)]
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


if __name__ == "__main__":
    unittest.main()
