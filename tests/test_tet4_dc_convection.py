import numpy as np
import pytest
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4 as baseline
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_convection as conduction
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4_with_convection as solve, Tet4DCConvectionResult
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4 as no_robin

def fixture():
    return Tet4Bundle(np.r_[np.zeros((1,3)),np.eye(3)],[[0,1,2,3]],
                      np.array([19,25,39,21]),np.array([807]),np.array([3]),
                      {3:{'young_modulus_pa':1e9,'poisson_ratio':.3}},'mm')

def args():
    return dict(conductivity_S_per_m_by_material={3:2.},electrode_voltage_V_by_id={19:0.,25:.3},
          conductivity_W_per_m_K_by_material={3:.25},prescribed_K_by_id={19:300.,25:300.})

def test_one_owned_mesh_same_exact_composition():
    b=fixture(); a=args(); cv=dict(faces_by_node_id=[(19,39,21)],convection_W_per_m2_K=100.,ambient_K=299.95)
    r=solve(b,**a,**cv)
    assert isinstance(r,Tet4DCConvectionResult)
    assert r.electrical.total_joule_power_W==r.thermal.total_source_W>0.
    np.testing.assert_array_equal(r.electrical.joule_power_W_by_cell,r.thermal.integrated_source_W_by_cell)
    h=conduction(b,**{k:a[k] for k in ('conductivity_W_per_m_K_by_material','prescribed_K_by_id')},
           source_W_per_m3_by_cell=dict(zip(map(int,r.electrical.cell_ids),r.electrical.joule_density_W_per_m3)),**cv)
    np.testing.assert_array_equal(h.temperature_K,r.thermal.temperature_K)
    np.testing.assert_array_equal(h.convective_heat_into_solid_W_by_node,r.thermal.convective_heat_into_solid_W_by_node)
    assert r.thermal.total_convective_heat_into_solid_W<0 and abs(r.thermal.balance_W)<1e-15
    with pytest.raises(ValueError,match='duplicate'):
        solve(b,**a,**(cv|dict(faces_by_node_id=[(19,39,21),(21,19,39)])))

def test_h_zero_and_old_path_parity():
    b=fixture(); a=args()
    p=baseline(b,**a); q=no_robin(b,**a)
    cv=solve(b,**a,faces_by_node_id=[(19,39,21)],convection_W_per_m2_K=0.,ambient_K=299.95)
    for field,value in vars(p.thermal).items():
        if isinstance(value,np.ndarray):
            np.testing.assert_array_equal(value,getattr(q.thermal,field))
            np.testing.assert_array_equal(value,getattr(cv.thermal,field))
        else:
            assert value==getattr(q.thermal,field)==getattr(cv.thermal,field)

def test_mesh_snapshot_shared_after_original_mesh_changes(monkeypatch):
    import fibernet.sim.tet4_dc_thermal as bridge
    b=fixture(); a=args(); before=np.array(b.points,copy=True)
    ref=solve(fixture(),**a,faces_by_node_id=[(19,39,21)],convection_W_per_m2_K=100.,ambient_K=299.95)
    old_dc,old_heat=bridge.solve_dc_tet4,bridge.solve_steady_conduction_tet4_with_convection
    snapshots=[]
    def observe_dc(mesh,**kw):
        snapshots.append(mesh)
        assert mesh is not b
        assert not mesh.points.flags.writeable
        e=old_dc(mesh,**kw)
        b.points[1]+=123.
        b.materials[3]['poisson_ratio']=.99
        return e
    def observe_heat(mesh,**kw):
        assert len(snapshots)==1 and mesh is snapshots[0]
        np.testing.assert_array_equal(mesh.points,before)
        assert mesh.materials[3]['poisson_ratio']==.3
        return old_heat(mesh,**kw)
    monkeypatch.setattr(bridge,'solve_dc_tet4',observe_dc)
    monkeypatch.setattr(bridge,'solve_steady_conduction_tet4_with_convection',observe_heat)
    r=solve(b,**a,faces_by_node_id=[(19,39,21)],convection_W_per_m2_K=100.,ambient_K=299.95)
    np.testing.assert_array_equal(r.thermal.temperature_K,ref.thermal.temperature_K)
    np.testing.assert_array_equal(r.electrical.joule_power_W_by_cell,ref.electrical.joule_power_W_by_cell)

def test_large_joule_source_rejected_before_thermal(monkeypatch):
    import fibernet.sim.tet4_dc_thermal as bridge
    def not_run(*a,**k):
        pytest.fail('thermal solver received a source outside its envelope')
    monkeypatch.setattr(bridge,'solve_steady_conduction_tet4_with_convection',not_run)
    a=args();a['electrode_voltage_V_by_id']={19:0.,25:30.}
    with pytest.raises(ValueError,match='Joule density outside thermal source envelope'):
        solve(fixture(),**a,faces_by_node_id=[(19,39,21)],convection_W_per_m2_K=100.,ambient_K=300.)
