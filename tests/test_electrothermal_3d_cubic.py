"""Run the spatial DC-heating example and check its output contract."""
from pathlib import Path
import runpy
import unittest

import numpy as np

PATH = Path(__file__).parents[1] / "examples/electrothermal_3d_cubic.py"


class ElectrothermalCubicExampleTest(unittest.TestCase):
    def test_spatial_feedback_and_signed_heat_ledger(self):
        report = runpy.run_path(str(PATH))["run_example"]()
        self.assertEqual((report["nodes"], report["edges"], report["frames"]), (12, 20, 9))
        self.assertEqual((report["time_s"], report["step_s"]), (1., 0.125))
        np.testing.assert_allclose(report["final_plane_mean_temperature_K"],
                                   [300., 300.542542537957, 300.200686619966],
                                   rtol=0, atol=1e-10)
        self.assertLess(report["final_joule_power_W"], report["initial_joule_power_W"])
        self.assertLess(report["cumulative_reservoir_heat_J"], 0.)
        self.assertGreater(report["stored_heat_J"], 0.)
        self.assertLess(abs(report["energy_closure_J"]), 8e-11)
