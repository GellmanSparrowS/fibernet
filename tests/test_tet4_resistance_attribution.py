"""Independent finite-scale reference for the DC attribution."""
from dataclasses import replace
from itertools import permutations, product
from math import log

import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_resistance_attribution import compare_tet4_resistance


def _bar():
    vertices = list(product(range(2), repeat=3))
    rows = {v:i for i,v in enumerate(vertices)}
    axes = np.eye(3, dtype=int)
    cells = [[rows[tuple(v)] for v in np.vstack((np.zeros(3, int), axes[list(p)].cumsum(0)))]
             for p in permutations(range(3))]
    points = np.array(vertices) * [.003, .0005, .0004]
    ids = np.arange(len(points), dtype=np.uint64) * 7 + 2**63
    mesh = Tet4Bundle(points, np.array(cells), ids, np.arange(len(cells)), np.full(len(cells), 7),
                      {7:{"young_modulus_pa":1e9, "poisson_ratio":.28}}, "m")
    return mesh, {int(i): .75 if p[0] else -.25 for i, p in zip(ids, points)}


@pytest.mark.parametrize("solver", ["banded", "cg"])
def test_affine_bar_exact_resistance_with_reordered_unsigned_ids(solver):
    mesh, electrodes = _bar()
    strain = .01
    motion = mesh.points * [strain, -.28 * strain, -.28 * strain]
    a = compare_tet4_resistance(replace(mesh, points=1000*mesh.points, length_unit="mm"),
                                motion[::-1], node_ids=mesh.node_ids[::-1],
                                reference_conductivity_S_per_m_by_material={7:1.6},
                                changed_conductivity_S_per_m_by_material={7:1.6/.965},
                                electrode_voltage_V_by_id=electrodes, solver=solver)
    g = (1.+strain)/(1.-.28*strain)**2
    np.testing.assert_allclose([a.reference_ohm,a.shape_only_ohm,a.material_only_ohm,a.combined_ohm],
                               9375.*np.array([1.,g,.965,g*.965]), rtol=8e-14)
    assert a.shape_log_ratio == pytest.approx(log(g))
    assert a.material_log_ratio == pytest.approx(log(.965))
    assert a.interaction_log_ratio == pytest.approx(0., abs=1e-14)
    assert a.combined_log_ratio == pytest.approx(log(g*.965))


def test_inverted_cell_is_not_reoriented_and_solved():
    mesh, electrodes = _bar()
    motion = mesh.points * [-2.,0.,0.]
    with pytest.raises(ValueError, match="reverses"):
        compare_tet4_resistance(mesh, motion, node_ids=mesh.node_ids,
                                reference_conductivity_S_per_m_by_material={7:1.},
                                changed_conductivity_S_per_m_by_material={7:2.},
                                electrode_voltage_V_by_id=electrodes)


def test_identical_states_reuse_dc_and_give_exact_zeros(monkeypatch):
    from fibernet.sim import tet4_resistance_attribution as module
    orig = module.solve_dc_tet4
    calls = []

    def recording(*args, **kwargs):
        calls.append(kwargs["conductivity_S_per_m_by_material"])
        return orig(*args, **kwargs)

    monkeypatch.setattr(module, "solve_dc_tet4", recording)
    mesh, electrodes = _bar()
    a = compare_tet4_resistance(mesh, np.zeros_like(mesh.points), node_ids=mesh.node_ids,
                                reference_conductivity_S_per_m_by_material={7:1.},
                                changed_conductivity_S_per_m_by_material={7:1.},
                                electrode_voltage_V_by_id=electrodes)
    assert len(calls) == 1
    assert a.reference_ohm == a.shape_only_ohm == a.material_only_ohm == a.combined_ohm
    assert a.shape_log_ratio == a.material_log_ratio == a.interaction_log_ratio == a.combined_log_ratio == 0.
    calls.clear()
    b = compare_tet4_resistance(mesh, np.zeros_like(mesh.points), node_ids=mesh.node_ids,
                                reference_conductivity_S_per_m_by_material={7:1.},
                                changed_conductivity_S_per_m_by_material={7:2.},
                                electrode_voltage_V_by_id=electrodes)
    assert len(calls) == 2
    assert b.reference_ohm == b.shape_only_ohm
    assert b.reference_ohm == pytest.approx(2. * b.combined_ohm, rel=1e-13)
    assert b.material_only_ohm == b.combined_ohm
    assert b.shape_log_ratio == b.interaction_log_ratio == 0.


@pytest.mark.parametrize("case,match", [
    ("ids", "IDs must match"),
    ("voltage", "exactly two"),
    ("conductivity", "invalid changed conductivity"),
    ("shape", "displacement_m"),
])
def test_rejects_bad_input_before_solves(case, match, monkeypatch):
    from fibernet.sim import tet4_resistance_attribution as module
    monkeypatch.setattr(module, "solve_dc_tet4", lambda *a, **k: pytest.fail("invalid data sent to solver"))
    mesh, electrodes = _bar()
    ids = mesh.node_ids
    u = np.zeros_like(mesh.points)
    sigma = {7:2.}
    if case == "ids":
        ids = ids + np.uint64(1)
    elif case == "voltage":
        electrodes[int(ids[0])] = .5
    elif case == "conductivity":
        sigma = {7:0.}
    elif case == "shape":
        u = u[:-1]
    with pytest.raises(ValueError, match=match):
        compare_tet4_resistance(mesh, u, node_ids=ids,
                                reference_conductivity_S_per_m_by_material={7:1.},
                                changed_conductivity_S_per_m_by_material=sigma,
                                electrode_voltage_V_by_id=electrodes)


def test_two_materials_on_fused_three_path_network():
    from fibernet.examples.tet4_mechanoelectric_looped_network import _bundle, _fixture, _lattice
    from fibernet.sim.tet4 import solve_static_tet4
    from fibernet.sim.tet4_dc import solve_dc_tet4
    points, cells = _lattice()
    mesh = _bundle(points, cells, "stiff_half")
    _, grip, fixed, electrodes, _ = _fixture(mesh)
    mech = solve_static_tet4(mesh, forces_by_id={}, fixed_dofs_by_id=fixed,
                             prescribed_displacements_m_by_id={i:{"x":3.e-6} for i in grip})
    ref = {7:1., 19:.7}
    moved = {7:2., 19:.35}
    result = compare_tet4_resistance(mesh, mech.displacement_m, node_ids=mech.node_ids,
                                    reference_conductivity_S_per_m_by_material=ref,
                                    changed_conductivity_S_per_m_by_material=moved,
                                    electrode_voltage_V_by_id=electrodes, solver="cg")
    # Independent DC solutions evaluate voltage drop from Joule power, not the
    # terminal current used in the comparator. Terminal span is exactly 1 V.
    geom = replace(mesh, points=points+1000.*mech.displacement_m)
    expected = []
    for bundle, sig in ((mesh, ref), (geom, ref), (mesh, moved), (geom, moved)):
        current = solve_dc_tet4(bundle, conductivity_S_per_m_by_material=sig,
                               electrode_voltage_V_by_id=electrodes, solver="banded")
        expected.append(1./current.total_joule_power_W)
    np.testing.assert_allclose([result.reference_ohm,result.shape_only_ohm,result.material_only_ohm,
                                result.combined_ohm], expected, rtol=2.e-9)
    assert abs(result.interaction_log_ratio)>1.e-7
    assert result.combined_log_ratio == pytest.approx(
        result.shape_log_ratio+result.material_log_ratio+result.interaction_log_ratio)
