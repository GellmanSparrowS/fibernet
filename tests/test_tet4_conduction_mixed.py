from itertools import product, permutations
from math import fsum
import numpy as np
import pytest
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_mixed_loads as mix
from fibernet.sim.tet4_conduction import _local_mixed_watts as nodal
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_source as oldv
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_surface_flux as olds

def fixture(x, t, unit='m', ids=None):
    n,c=ids if ids is not None else (10+np.arange(len(x))*3, 100+np.arange(len(t))*2)
    b=Tet4Bundle(x,t,n,c,np.ones(len(t),dtype=np.int64),{1:{'young_modulus_pa':1.e9,'poisson_ratio':.3}},unit)
    return b,n,c

def prism():
    x=np.array(list(product((0.,1.),repeat=3)))
    t=np.array([(0,a,a+b,7) for a,b,c in permutations((4,2,1))])
    b,n,c=fixture(x,t)
    args=dict(conductivity_W_per_m_K_by_material={1:1.},prescribed_K_by_id=dict.fromkeys(map(int,n[:4]),300.))
    faces=[tuple(n[idx]) for idx in ([4,5,7],[4,6,7])]
    return b,c,args,faces

def test_equal_component_load_nonzero_field():
    b,c,args,faces=prism()
    r=mix(b,source_W_per_m3_by_cell=dict.fromkeys(map(int,c),6.),faces_by_node_id=faces,inward_flux_W_per_m2=-6.,**args)
    assert (r.total_source_W,r.total_surface_W)==(6.,-6.)
    np.testing.assert_allclose(r.temperature_difference_K,[0.,0.,0.,0.,-48/13,-75/26,-75/26,-63/26],rtol=0.,atol=5e-14)
    np.testing.assert_allclose(r.heat_into_solid_W[:4],[-7/26,-1/52,-1/52,4/13],rtol=0.,atol=5e-14)
    assert abs(r.balance_W)<1e-14 and r.source_pairing_W_K<0<r.surface_pairing_W_K
    assert r.dirichlet_quadratic_W_K==pytest.approx(r.source_pairing_W_K+r.surface_pairing_W_K)
    for a in (r.source_by_node_W,r.surface_by_node_W,r.integrated_source_W_by_cell):
        with pytest.raises(ValueError): a.setflags(write=True)

def test_source_and_surface_zero_paths():
    b,c,args,faces=prism()
    zeros=dict.fromkeys(map(int,c),0.)
    case=dict.fromkeys(map(int,c),6.)
    for table,rates,ref in [(case,0.,oldv(b,source_W_per_m3_by_cell=case,**args)),(zeros,-6.,olds(b,faces_by_node_id=faces,inward_flux_W_per_m2=-6.,**args))]:
        r=mix(b,source_W_per_m3_by_cell=table,faces_by_node_id=faces,inward_flux_W_per_m2=rates,**args)
        np.testing.assert_allclose(r.temperature_difference_K,ref.temperature_difference_K,rtol=0,atol=1e-13)
        np.testing.assert_allclose(r.heat_into_solid_W,ref.heat_into_solid_W,rtol=0,atol=1e-13)

def test_nodewise_cancellation_and_loss():
    assert nodal([(1.,1.,4),(3.,-.25,3)])==0.
    e=2.**-27
    with pytest.raises(ValueError,match='near-opposing'):
        nodal([(1.+e,1.-e,4),(3.,-.25,3)])
    with pytest.raises(ValueError,match='lost a nonzero'):
        nodal([(1.,1.,4),(3.,.25,3),(2.**-104,1.,4)])

def test_boundary_rounding_does_not_make_nonzero_rhs_exactly_cancel():
    b,n,c=fixture(np.r_[np.zeros((1,3)),np.eye(3)],np.array([[0,1,2,3]]))
    with pytest.raises(ValueError,match='near-opposing mixed nodal heat'):
        mix(b,conductivity_W_per_m_K_by_material={1:1.},
            prescribed_K_by_id=dict(zip(map(int,n[:3]),(306.,300.,300.))),
            source_W_per_m3_by_cell={int(c[0]):6.},
            faces_by_node_id=[tuple(map(int,n[[0,1,3]]))],inward_flux_W_per_m2=-7.5)

def test_exact_opposing_boundary_rhs_remains_valid():
    assert nodal([(1.,1.,1),(3.,-1.,3)])==0.

def test_disconnected_components_and_fully_held_surface():
    a=np.r_[np.zeros((1,3)),np.eye(3)]
    x=np.concatenate((a/4.,a/4.+[.5,0.,0.]))
    t=np.array([[0,1,2,3],[4,5,6,7]])
    b,n,c=fixture(x,t)
    kwargs=dict(conductivity_W_per_m_K_by_material={1:1.},
                prescribed_K_by_id=dict(zip(map(int,n[:7]),[300.]*4+[350.]*3)),
                source_W_per_m3_by_cell={int(c[0]):384.,int(c[1]):384.},
                faces_by_node_id=[tuple(map(int,n[[0,1,2]]))],
                inward_flux_W_per_m2=-96.)
    r=mix(b,**kwargs)
    # Each Tet is 1/384 m³, face area 1/32 m², free diagonal 1/24 W/K.
    np.testing.assert_allclose(r.temperature_K,[300.]*4+[350.]*3+[356.],atol=5e-14,rtol=0)
    np.testing.assert_allclose(r.reference_K_by_node,[300.]*4+[350.]*4,rtol=0,atol=0)
    np.testing.assert_allclose(r.source_by_node_W,[.25]*8,atol=1e-15,rtol=0)
    np.testing.assert_allclose(r.surface_by_node_W,[-1.]*3+[0.]*5,atol=1e-15,rtol=0)
    np.testing.assert_allclose(r.heat_into_solid_W,[.75]*3+[-.25,-.5,-.25,-.25,0.],atol=1e-15,rtol=0)
    assert fsum(r.heat_into_solid_W[:4])+fsum(r.source_by_node_W[:4])+fsum(r.surface_by_node_W[:4])==pytest.approx(0.,abs=1e-15)
    assert fsum(r.heat_into_solid_W[4:])+fsum(r.source_by_node_W[4:])==pytest.approx(0.,abs=1e-15)
    assert (r.total_source_W,r.total_surface_W)==(2.,-3.)
    assert abs(r.balance_W)<1e-15
    with pytest.raises(ValueError,match='each conductive component requires'):
        mix(b,**(kwargs|dict(prescribed_K_by_id=dict.fromkeys(map(int,n[:4]),300.))))

    order=np.array([5,3,6,1,7,0,2,4])
    rev=np.argsort(order)
    b2,_,_=fixture(x[order],rev[t[::-1]],ids=(n[order],c[::-1]))
    r2=mix(b2,**(kwargs|dict(faces_by_node_id=[tuple(map(int,n[[2,0,1]]))])))
    np.testing.assert_array_equal(r2.node_ids,n[order])
    for f in ('temperature_K','heat_into_solid_W','source_by_node_W','surface_by_node_W'):
        np.testing.assert_allclose(getattr(r2,f)[rev],getattr(r,f),rtol=0,atol=1e-15)
    np.testing.assert_allclose(r2.integrated_source_W_by_cell[::-1],r.integrated_source_W_by_cell,rtol=0,atol=1e-15)

def test_exact_zero_cannot_hide_same_sign_terms():
    with pytest.raises(ValueError,match='same-sign'):
        nodal([(1.,1.,4),(2**-58,1.,4),(1.,-1.,4),(2**-58,-1.,4)])

def test_effective_nodal_load_and_split_work():
    b,c,args,faces=prism()
    r=mix(b,source_W_per_m3_by_cell=dict.fromkeys(map(int,c),6.),faces_by_node_id=faces,inward_flux_W_per_m2=-6.,**args)
    np.testing.assert_allclose(r.applied_nodal_heat_W,r.source_by_node_W+r.surface_by_node_W,atol=1e-15,rtol=0.)
    assert fsum(r.applied_nodal_heat_W*r.temperature_difference_K)==r.applied_pairing_W_K
    with pytest.raises(ValueError):
        r.applied_nodal_heat_W.setflags(write=True)
