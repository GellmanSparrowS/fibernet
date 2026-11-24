"""The published usage example isolates placement at fixed inventory."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "viscoelastic_material_placement.py"


def test_material_placement_example_preserves_inventory_and_reverses(tmp_path):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
    run = subprocess.run([sys.executable, str(EXAMPLE), "--samples-per-cycle", "64"],
                         cwd=tmp_path, env=env, capture_output=True,
                         encoding="utf-8", check=True, timeout=60)
    report = json.loads(run.stdout)
    assert (report["nodes"], report["edges"], report["samples_per_cycle"]) == (16, 33, 64)
    a, b = report["long_time_inventory"].values()
    assert a == b
    assert (a["horizontal"], a["vertical"]) == (6, 5)
    assert a["volume_m3"] == pytest.approx(2.2116812281272143e-10, rel=1e-10, abs=0)
    groups = list(report["long_time_edge_ids"].values())
    assert len(set(groups[0])) == len(set(groups[1])) == 11
    assert not set(groups[0]).intersection(groups[1])
    lo_high, lo_low, hi_high, hi_low = report["cases"]
    assert lo_high["frequency_Hz"] == lo_low["frequency_Hz"] < hi_high["frequency_Hz"]
    assert hi_high["frequency_Hz"] == hi_low["frequency_Hz"]
    assert lo_high["physical_loss_J"] > lo_low["physical_loss_J"]
    assert hi_high["physical_loss_J"] < hi_low["physical_loss_J"]
    # Independent harmonic frame values; coarse Backward Euler is first order.
    exact = (29.4011391161, 26.5641733385, 38.8820270691, 41.8178191696)
    for row, reference_pJ in zip(report["cases"], exact):
        loss = row["physical_loss_J"]
        assert loss * 1e12 == pytest.approx(reference_pJ, rel=0.05, abs=0)
        assert row["previous_physical_loss_J"] == pytest.approx(loss, rel=5e-4, abs=0)
        assert row["grip_loop_work_J"] == pytest.approx(loss, rel=0.004, abs=0)
        assert row["numerical_loss_J"] > 0
