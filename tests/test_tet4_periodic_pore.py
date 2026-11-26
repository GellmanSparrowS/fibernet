import builtins
import math
from dataclasses import replace
import runpy
from pathlib import Path

import numpy as np
import pytest

from fibernet.sim.tet4_pore import (solve_matched_pore_geometry, solve_matched_pore_geometry_cases,
                                    solve_x_periodic_stokes)

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
    pair, shrink = solve_matched_pore_geometry_cases(
        b, (replace(b, points=b.points*1.025), replace(b, points=b.points*.99)), **settings,
    )
    assert shrink.baseline is pair.baseline
    np.testing.assert_allclose(shrink.relative_bulk_flow_change, .99**4-1, rtol=1e-9)
    np.testing.assert_allclose(shrink.relative_cap_flow_change, .99**4-1, rtol=1e-9)
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


def test_conforming_interface_transfers_exactly_and_leaves_box_fixed():
    from examples.tet4_periodic_pore import _scaffold_voxels, _voxel_bundle
    from examples.tet4_solid_to_pore_flow import _boundary_faces, _extend_interface

    solid = _voxel_bundle(_scaffold_voxels(), elastic_modulus=2.e9)
    fluid = bundle()
    d = np.column_stack([solid.points[:, 0] * .002, solid.points[:, 1] * -.001,
                         solid.points[:, 2] * -.001])
    u = _extend_interface(solid, fluid, d)
    surface_ids = np.unique(_boundary_faces(solid.tetrahedra))
    locator = {tuple(row): j for j, row in enumerate(fluid.points)}
    where = [locator[tuple(solid.points[j])] for j in surface_ids]
    np.testing.assert_array_equal(u[where], d[surface_ids])
    outer = np.any((fluid.points == fluid.points.min(axis=0)) |
                   (fluid.points == fluid.points.max(axis=0)), axis=1)
    np.testing.assert_array_equal(u[outer], 0.)
    assert np.count_nonzero(np.linalg.norm(u, axis=1) > 0.) > len(surface_ids)
    with pytest.raises(ValueError, match="no matching pore point"):
        _extend_interface(replace(solid, points=solid.points + .013), fluid, d)


def test_small_loading_changes_flow_with_consistent_cap_readout():
    pytest.importorskip("skfem")
    from examples.tet4_solid_to_pore_flow import demo

    d = demo()
    assert d["solid_tetrahedra"] == 84 and d["pore_tetrahedra"] == 2268
    assert .70 < d["grip_reaction_N"] < .75
    assert 0 < d["loaded_flow_m3_s"] < d["baseline_flow_m3_s"]
    assert -.00070 < d["relative_bulk_flow_change"] < -.00050
    np.testing.assert_allclose(d["relative_end_flow_change"], [-.00063, -.00063], rtol=.08)
    assert .99 < d["minimum_cell_volume_ratio"] < 1 < d["maximum_cell_volume_ratio"] < 1.01


def test_elastic_modulus_comparison_uses_matched_signed_grip_reactions(monkeypatch):
    pytest.importorskip("skfem")
    from examples.tet4_solid_to_pore_flow import compare_materials

    from fibernet.sim import tet4_pore
    raw = bundle()
    original = tet4_pore.solve_x_periodic_stokes
    calls = []

    def counted(candidate, **kw):
        calls.append(np.array_equal(candidate.points, raw.points))
        return original(candidate, **kw)

    monkeypatch.setattr(tet4_pore, "solve_x_periodic_stokes", counted)
    report = compare_materials()
    assert calls == [True, False, False, False, False]
    soft, stiff = report["materials"]
    assert soft["young_modulus_Pa"] == 1e9 and stiff["young_modulus_Pa"] == 2e9
    for result in (soft, stiff):
        np.testing.assert_allclose([x["reaction_N"] for x in result["cases"]], [-.15, .15],
                                   rtol=2e-10, atol=0.)
    np.testing.assert_allclose([x["stroke_m"] for x in stiff["cases"]],
                               np.array([x["stroke_m"] for x in soft["cases"]]) / 2, rtol=2e-10)
    assert soft["relative_bulk_slope_per_N"] < 0
    np.testing.assert_allclose(stiff["relative_bulk_slope_per_N"],
                               .5 * soft["relative_bulk_slope_per_N"], rtol=1e-5)



def test_matched_cases_prevalidate_all_geometries_before_solving(monkeypatch):
    from fibernet.sim import tet4_pore

    def forbidden(*args, **kwargs):
        pytest.fail("matched validation must precede any numeric reference solve")

    monkeypatch.setattr(tet4_pore, "solve_x_periodic_stokes", forbidden)
    b = bundle()
    first = replace(b, points=b.points*1.01)
    malformed = (
        (replace(b, node_ids=b.node_ids+1), "stable node and cell IDs"),
        (replace(b, points=b.points*1.5), "volume-ratio"),
    )
    for second, message in malformed:
        with pytest.raises(ValueError, match=message):
            solve_matched_pore_geometry_cases(b, [first, second], **settings)
    for cases in ([], [first]*9, iter([first])):
        with pytest.raises(ValueError, match="ordered list or tuple"):
            solve_matched_pore_geometry_cases(b, cases, **settings)
