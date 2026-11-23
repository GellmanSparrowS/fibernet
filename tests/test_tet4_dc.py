"""Small independent analytic controls for native solid DC and its thermal handoff."""
from itertools import permutations, product

import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_dc import solve_dc_tet4
from fibernet.sim import tet4_dc_thermal as handoff

_MATERIAL = {1: {'young_modulus_pa': 1e9, 'poisson_ratio': .3}}


def _tet(disconnected=False):
    points = np.vstack([np.zeros(3), np.eye(3)])*.5
    cells = [[0, 1, 2, 3]]
    n = [2**64-1, 7, 2**53+1, 22]
    if disconnected:
        points = np.vstack([points, points+[.75, 0., 0.]])
        points[4:, 0] -= .25  # extent <= 1 m; no shared nodes
        n += [40, 41, 42, 43]
        cells += [[4, 5, 6, 7]]
    return Tet4Bundle(points,cells,n,list(range(len(cells))),[1]*len(cells),_MATERIAL,'m'),n


def test_signed_electrodes_polarity_and_gauge():
    mesh,n = _tet()
    def run(volts):
        return solve_dc_tet4(mesh, conductivity_S_per_m_by_material={1:2.},
                             electrode_voltage_V_by_id=dict(zip(n,volts)))
    base = run([-1.,1.,3.])
    np.testing.assert_array_equal(base.voltage_V, [-1.,1.,3.,-1.])
    np.testing.assert_allclose(base.electrode_current_into_solid_A, [-1.,1/3,2/3,0.], rtol=0.,atol=1e-14)
    np.testing.assert_allclose(base.current_density_A_per_m2, [[-8.,-16.,0.]], rtol=0.,atol=1e-13)
    np.testing.assert_allclose([base.total_joule_power_W,base.terminal_power_W],[10/3,10/3],rtol=1e-13)
    shift = run([95.,97.,99.])
    rev = run([1.,-1.,-3.])
    np.testing.assert_array_equal(shift.centered_voltage_V,base.centered_voltage_V)
    np.testing.assert_allclose(rev.electrode_current_into_solid_A,-base.electrode_current_into_solid_A,atol=1e-14,rtol=0.)
    np.testing.assert_allclose(rev.joule_density_W_per_m3,[160.],rtol=1e-13)
    with pytest.raises(ValueError,match='contrast lost during centering'):
        run([-64.,1.,np.nextafter(1.,2.)])
    equal = run([-64.,1.,1.])
    assert equal.centered_voltage_V[1]==equal.centered_voltage_V[2]==65.


def test_disconnected_components_require_anchor_and_passive_is_exact():
    b,n = _tet(disconnected=True)
    vs = dict(zip(n,[-1.,1.,3.]))
    args = dict(conductivity_S_per_m_by_material={1:2.})
    with pytest.raises(ValueError,match='each DC component needs an electrode'):
        solve_dc_tet4(b,electrode_voltage_V_by_id=vs,**args)
    r = solve_dc_tet4(b,electrode_voltage_V_by_id={**vs,40:-.5},**args)
    np.testing.assert_array_equal(r.voltage_V[4:],[-.5]*4)
    np.testing.assert_array_equal(r.electrode_current_into_solid_A[4:],[0.]*4)
    assert r.joule_power_W_by_cell[1]==0.


def test_owned_one_way_joule_handoff_and_out_of_range_refusal(monkeypatch):
    xyz = np.array(list(product(range(4),range(2),range(2))),dtype=float)
    t = np.array([[4*i,4*i+p,4*i+p+q,4*i+7]
                  for i in range(3) for p,q,_ in permutations((4,2,1))])
    n = np.arange(len(xyz),dtype=np.uint64)+2**53
    c = np.arange(len(t),dtype=np.uint64)+2**62
    materials = np.repeat([1,2,2],6)
    records = {i:dict(_MATERIAL[1]) for i in (1,2)}
    bundle = Tet4Bundle(xyz,t,n,c,materials,records,'mm')
    fixed = list(map(int,np.r_[n[:4],n[-4:]]))
    args = dict(conductivity_S_per_m_by_material={1:2.,2:8.},
                conductivity_W_per_m_K_by_material={1:.25,2:.25},
                prescribed_K_by_id=dict.fromkeys(fixed,300.))
    def electrode(v):
        return dict(zip(fixed,[0.]*4+[v]*4))
    old_thermal = handoff.solve_steady_conduction_tet4_with_source
    old_dc = handoff.solve_dc_tet4
    expected_points = xyz.copy()
    shared = []
    def verify_thermal(owned,**kw):
        assert shared==[owned] and shared[0] is owned
        assert not owned.points.flags.writeable and owned.length_unit=='mm'
        np.testing.assert_array_equal(owned.points,expected_points)
        return old_thermal(owned,**kw)
    def corrupt_external(owned,**kw):
        r=old_dc(owned,**kw)
        shared.append(owned)
        xyz[:]=0.; t[:]=0; n[:]=0; c[:]=0; materials[:]=0; records.clear(); bundle.length_unit='m'
        return r
    monkeypatch.setattr(handoff,'solve_steady_conduction_tet4_with_source',verify_thermal)
    with pytest.raises(ValueError,match='Joule density outside thermal source envelope'):
        handoff.solve_dc_thermal_tet4(bundle,electrode_voltage_V_by_id=electrode(30.),**args)
    assert not shared
    monkeypatch.setattr(handoff,'solve_dc_tet4',corrupt_external)
    r = handoff.solve_dc_thermal_tet4(bundle,electrode_voltage_V_by_id=electrode(.3),**args)
    e,h = r.electrical,r.thermal
    np.testing.assert_array_equal(e.cell_ids,h.cell_ids)
    np.testing.assert_array_equal(e.joule_power_W_by_cell,h.integrated_source_W_by_cell)
    np.testing.assert_allclose([e.terminal_power_W,e.total_joule_power_W,h.total_source_W],[.00012]*3,rtol=1e-12)
    np.testing.assert_allclose([h.heat_into_solid_W[:4].sum(),h.heat_into_solid_W[-4:].sum()],
                               [-8e-5,-4e-5],rtol=1e-12)
    assert abs(h.balance_W)<1e-16
