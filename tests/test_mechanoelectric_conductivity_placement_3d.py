"""Independent use-path checks for the fixed-mechanics conductivity comparison."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "mechanoelectric_conductivity_placement_3d.py"


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_matched_placements_keep_inventory_but_change_reading(phase, tmp_path):
    r = subprocess.run([sys.executable, str(EXAMPLE), "--phase", str(phase)],
                       cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60)
    out = json.loads(r.stdout)
    assert (out["phase"], out["member_count"], out["matched_group_count"]) == (phase, 360, 95)
    assert out["input_strain"] == .005
    assert out["voltage_readout_axis"] == "y"
    assert out["low_high_conductivity_S_per_m"] == [1., 2.]
    rows = [out["placements"][key] for key in ("low", "stable_id", "high")]
    assert len({a["conductive_member_count"] for a in rows}) == 1
    volumes = [a["higher_conductivity_nominal_volume_m3"] for a in rows]
    assert max(volumes) - min(volumes) < 1e-20
    nominal = [a["unloaded_ohm"] for a in rows]
    assert all(3650. < r0 < 3860. for r0 in nominal)
    assert nominal[2] - nominal[0] > 20.
    for law in ("fixed_area", "fixed_member_volume"):
        results = [a["geometry_laws"][law] for a in rows]
        readout = [a["change_percent"] for a in results]
        assert 0. < readout[0] < readout[1] < readout[2] < .13
        assert readout[2] - readout[0] > .022
        for r0, result in zip(nominal, results):
            assert 100*(result["loaded_ohm"]/r0-1) == pytest.approx(result["change_percent"])


def test_invalid_phase_does_not_run_solver(tmp_path):
    r = subprocess.run([sys.executable, str(EXAMPLE), "--phase", "4"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert not r.stdout
