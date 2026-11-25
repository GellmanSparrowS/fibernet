import builtins
import math
from dataclasses import replace
import runpy
from pathlib import Path

import numpy as np
import pytest

from fibernet.sim.tet4_pore import solve_matched_pore_geometry, solve_x_periodic_stokes

_load = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'examples'
                          / 'tet4_periodic_pore.py'))
bundle = _load['bundle']
settings = dict(viscosity_Pa_s=.001, pressure_gradient_Pa_per_m=500.)


def test_budget_and_stable_ids_are_rejected_before_flow_assembly():
    b = bundle()
    with pytest.raises(ValueError, match='planned work'):
        solve_x_periodic_stokes(b, workspace_limit_bytes=128*1024**2, **settings)
    with pytest.raises(ValueError, match='stable node and cell IDs'):
        solve_matched_pore_geometry(b, replace(b, node_ids=b.node_ids+1), **settings)
    with pytest.raises(ValueError, match='reverses a cell'):
        x = b.points.copy()
        x[:, 0] *= -1.
        solve_matched_pore_geometry(b, replace(b, points=x), **settings)


def test_missing_optional_dependency_can_be_reported(monkeypatch):
    real = builtins.__import__

    def without_provider(name, *args, **kwargs):
        if name == 'skfem' or name.startswith('skfem.'):
            raise ImportError("provider withheld for this test")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_provider)
    with pytest.raises(ImportError, match='pore-flow extra'):
        solve_x_periodic_stokes(bundle(), **settings)


def test_real_provider_signed_flow_and_covariance():
    pytest.importorskip('skfem')
    b = bundle()
    zero_obstacle = solve_x_periodic_stokes(bundle(False), **settings)
    pair = solve_matched_pore_geometry(b, replace(b, points=b.points*1.025), **settings)
    assert 0. < pair.baseline.axial_flow_m3_s < zero_obstacle.axial_flow_m3_s
    # Independent square-duct Poiseuille series (not a saved FE output).
    span_m = 1.4e-3
    odd = sum(math.tanh(k * math.pi / 2) / k**5 for k in range(1, 101, 2))
    exact = (500. * span_m**4 / (12 * .001)) * (1. - 192 * odd / math.pi**5)
    np.testing.assert_allclose(zero_obstacle.axial_flow_m3_s, exact, rtol=3e-3)
    np.testing.assert_allclose(zero_obstacle.cap_flow_m3_s, exact, rtol=3e-3)

    np.testing.assert_allclose(pair.relative_bulk_flow_change, 1.025**4-1, rtol=1e-9)
    np.testing.assert_allclose(pair.relative_cap_flow_change, 1.025**4-1, rtol=1e-9)
    # Fixed drop (0.8 Pa) requires a smaller driving gradient on the longer mesh.
    scaled = replace(b, points=b.points*1.025)
    length_m = np.ptp(scaled.points[:, 0]) * 1e-3
    fixed_drop = solve_x_periodic_stokes(
        scaled, viscosity_Pa_s=.001, pressure_gradient_Pa_per_m=.8/length_m,
    )
    np.testing.assert_allclose(fixed_drop.axial_flow_m3_s/pair.baseline.axial_flow_m3_s,
                               1.025**3, rtol=1e-9)
    np.testing.assert_allclose(np.divide(fixed_drop.cap_flow_m3_s, pair.baseline.cap_flow_m3_s),
                               1.025**3, rtol=1e-9)
    np.testing.assert_allclose(fixed_drop.power_W/pair.baseline.power_W, 1.025**3, rtol=1e-9)
    a = solve_x_periodic_stokes(
        replace(b, points=b.points*.001,length_unit='m'),
        viscosity_Pa_s=.002,
        pressure_gradient_Pa_per_m=-500.,
    )
    np.testing.assert_allclose(a.axial_flow_m3_s, -pair.baseline.axial_flow_m3_s/2, rtol=1e-9)
    np.testing.assert_allclose(a.power_W, pair.baseline.power_W/2, rtol=1e-9)


def test_cap_bijection_is_not_only_a_vertex_count():
    pytest.importorskip("skfem")
    b = bundle()
    x = b.points.copy()
    far = x[:, 0] == x[:, 0].max()
    x[far, 2] += 1e-3
    with pytest.raises(ValueError, match="periodic cap vertices do not pair"):
        solve_x_periodic_stokes(replace(b, points=x), **settings)
