"""Run stiffness and conductivity controls through the documented CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

_here = Path(__file__).resolve().parent
_name = "mechanoelectric_material_controls_3d.py"
_example = _here / _name
if not _example.is_file():
    _example = _here.parent / "examples" / _name


class MechanoelectricMaterialControlExample(unittest.TestCase):
    def test_same_geometry_with_separate_material_controls(self):
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory() as cwd:
            r = subprocess.run([sys.executable, str(_example)], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(list(Path(cwd).iterdir()), [])
        out = json.loads(r.stdout)
        self.assertEqual(out["directions"], ["x", "y", "z"])
        self.assertEqual(out["member_count"], 360)
        displacement = out["same_displacement_change_percent"]
        force = out["same_force_change_percent"]
        for a, b, c, d in zip(displacement["reference_E"], displacement["doubled_E"],
                              force["reference_E"], force["doubled_E"]):
            self.assertAlmostEqual(a, b, places=8)
            self.assertAlmostEqual(a, c, places=8)
            self.assertGreater(c, d)
            self.assertGreater(d, 0.)
        self.assertAlmostEqual(force["reference_E"][0], .26518099, places=7)
        self.assertAlmostEqual(force["doubled_E"][0], .13259340, places=7)
        for ratio in out["uniform_doubled_sigma_loaded_R_ratio"]:
            self.assertAlmostEqual(ratio, .5, places=10)


if __name__ == "__main__":
    unittest.main()
