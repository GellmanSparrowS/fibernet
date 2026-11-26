"""Application checks for the synthetic equal-area sensitivity example."""
import numpy as np
import pytest

from examples import tet4_mechanoelectric_patch_screen as example


def test_patch_sensitivity_uses_one_electrical_and_one_adjoint(monkeypatch):
    pytest.importorskip("pyamg")
    calls=[]
    original_dc=example.solve_dc_tet4
    original_adjoint=example._adjoint

    def dc(*a,**k):
        calls.append("dc")
        return original_dc(*a,**k)

    def adjoint(*a,**k):
        calls.append("adjoint")
        return original_adjoint(*a,**k)

    monkeypatch.setattr(example,"solve_dc_tet4",dc)
    monkeypatch.setattr(example,"_adjoint",adjoint)
    s=example.run(refinement=1)
    assert calls==["dc","adjoint"]
    assert s["reference_resistance_ohm"]>0.
    patches=s["patches"]
    assert len(patches)==12 and len({p["patch"] for p in patches})==12
    direction=np.array(s["force_direction"])
    np.testing.assert_allclose(np.linalg.norm(direction),1.,rtol=1e-13)
    previous=np.inf
    for item in patches:
        np.testing.assert_allclose(item["area_mm2"],.0625,rtol=1e-12)
        pos=np.asarray(item["centroid_m"])
        moment=np.cross(pos-np.asarray(s["moment_origin_m"]),direction)
        np.testing.assert_allclose(moment,item["moment_per_N_m"],atol=1e-15)
        slope=item["relative_resistance_slope_per_N"]
        assert np.isfinite(slope) and slope<=previous
        previous=slope
    assert abs(patches[0]["relative_resistance_slope_per_N"]-previous)>1e-6


def test_three_synthetic_patches_agree_with_independent_finite_checks():
    pytest.importorskip("pyamg")
    out=example.run(refinement=1,verify=True)
    checked=[p for p in out["patches"] if "centered_finite_check_per_N" in p]
    assert len(checked)==3
    for p in checked:
        np.testing.assert_allclose(p["relative_resistance_slope_per_N"],
                                   p["centered_finite_check_per_N"],rtol=3e-4,atol=1e-8)
