from fractions import Fraction

import numpy as np
import pytest
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_convection as solve
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_source as volume

# Independent rational reference on the unit right tetrahedron.
exact={'zero_h': {'theta': ['0', '1', '2', '3/2'],
            'convection': ['0', '0', '0', '0'],
            'reaction': ['-1', '-1/12', '1/12', '0'],
            'total_convective': '0',
            'convective_pair': '0'},
 'held_face': {'theta': ['0', '1', '2', '3/2'],
               'convection': ['37/6', '6', '35/6', '0'],
               'reaction': ['-43/6', '-73/12', '-23/4', '0'],
               'total_convective': '18',
               'convective_pair': '53/3'},
 'partially_held_hot': {'theta': ['0', '1', '2', '27/2'],
                        'convection': ['17/4', '49/12', '0', '2'],
                        'reaction': ['-29/4', '-25/6', '1/12', '0'],
                        'total_convective': '31/3',
                        'convective_pair': '373/12'},
 'partially_held_cold': {'theta': ['0', '1', '2', '-79/6'],
                         'convection': ['-167/36', '-173/36', '0', '-22/9'],
                         'reaction': ['73/12', '85/18', '1/12', '0'],
                         'total_convective': '-107/9',
                         'convective_pair': '2957/108'}}

def mesh(x,t,length='m'):
    x=np.asarray(x);t=np.asarray(t)
    return Tet4Bundle(x,t,11+np.arange(len(x)),24+np.arange(len(t)),
                      np.ones(len(t),dtype=np.int64),{1:{'young_modulus_pa':1e9,'poisson_ratio':.3}},length)
def right_tet(length='m'):
    return mesh(np.r_[np.zeros((1,3)),np.eye(3)],np.array([[0,1,2,3]]),length)

@pytest.mark.parametrize('label,face,h,ambient',[
    ('zero_h',(11,12,13),0.,310.),('held_face',(11,12,13),4.,310.),
    ('partially_held_hot',(11,12,14),4.,310.),('partially_held_cold',(11,12,14),4.,290.),
])
def test_exact_tet_gauge(label,face,h,ambient):
    b=right_tet()
    base=dict(conductivity_W_per_m_K_by_material={1:1.},prescribed_K_by_id={11:300.,12:301.,13:302.},
              source_W_per_m3_by_cell={24:6.})
    f=lambda seq:[float(Fraction(z)) for z in seq]
    e=exact[label]
    r=solve(b,faces_by_node_id=[face],convection_W_per_m2_K=h,ambient_K=ambient,**base)
    shifted=solve(b,faces_by_node_id=[tuple(reversed(face))],convection_W_per_m2_K=h,ambient_K=ambient+700,
                  **(base|{'prescribed_K_by_id':{k:v+700 for k,v in base['prescribed_K_by_id'].items()}}))
    for field,key in [('temperature_difference_K','theta'),('convective_heat_into_solid_W_by_node','convection'),
                      ('heat_into_solid_W','reaction')]:
        np.testing.assert_allclose(getattr(r,field),f(e[key]),atol=5e-13,rtol=0.)
        np.testing.assert_array_equal(getattr(shifted,field),getattr(r,field))
    assert shifted.balance_W==r.balance_W and abs(r.balance_W)<3e-13
    assert abs(r.quadratic_closure_W_K)<1e-12
    assert r.total_convective_heat_into_solid_W==pytest.approx(float(Fraction(e['total_convective'])))
    assert r.convection_pairing_W_K==pytest.approx(float(Fraction(e['convective_pair'])))
    if h==0:
        legacy=volume(b,**base)
        np.testing.assert_array_equal(legacy.temperature_K,r.temperature_K)
        np.testing.assert_array_equal(legacy.heat_into_solid_W,r.heat_into_solid_W)

def test_adversarial_lost_surface_operator():
    with pytest.raises(ValueError,match='represented contribution'):
        solve(right_tet('mm'),conductivity_W_per_m_K_by_material={1:1e6},
              prescribed_K_by_id={11:300.,12:300.,13:300.},source_W_per_m3_by_cell={24:20000.},
              faces_by_node_id=[(11,12,14)],convection_W_per_m2_K=1e-8,ambient_K=310.)

def test_robin_drive_not_rounded_before_full_rhs():
    base=dict(conductivity_W_per_m_K_by_material={1:1.},prescribed_K_by_id={11:306.,12:300.,13:300.},
              source_W_per_m3_by_cell={24:0.},faces_by_node_id=[(11,12,14)],ambient_K=295.)
    with pytest.raises(ValueError,match='near-opposing'):
        solve(right_tet(),convection_W_per_m2_K=float.fromhex('0x1.d89d89d89d89ep-1'),**base)
    r=solve(right_tet(),convection_W_per_m2_K=.9,**base)
    assert r.temperature_difference_K[-1]==pytest.approx(.10344827586206897)

def test_ambient_subtraction_not_rounded_before_full_rhs():
    x=np.r_[np.zeros((1,3)),np.diag((1.,1.,.75))]
    b=mesh(x,[[0,1,2,3]])
    assert Fraction(1025)-Fraction(float(np.nextafter(1.,2.)))!=Fraction(1024)
    args=dict(conductivity_W_per_m_K_by_material={1:1.},
              source_W_per_m3_by_cell={24:-16384.},faces_by_node_id=[(11,12,13)],
              convection_W_per_m2_K=3.,ambient_K=1025.)
    with pytest.raises(ValueError,match='near-opposing'):
        solve(b,prescribed_K_by_id=dict.fromkeys((11,12,14),float(np.nextafter(1.,2.))),**args)
    exact=solve(b,prescribed_K_by_id=dict.fromkeys((11,12,14),1.),**args)
    np.testing.assert_array_equal(exact.temperature_difference_K,np.zeros(4))
    assert exact.free_residual_max_W==0.

def test_held_face_matches_ambient_despite_off_face_reference():
    x=np.r_[np.zeros((1,3)),np.diag((1.,1.,.75))]
    b=mesh(x,[[0,1,2,3]])
    r=solve(b,conductivity_W_per_m_K_by_material={1:1.},
              source_W_per_m3_by_cell={24:0.},faces_by_node_id=[(11,12,13)],
              convection_W_per_m2_K=3.,ambient_K=1025.,
              prescribed_K_by_id={11:1025.,12:1025.,13:1025.,14:float(np.nextafter(1.,2.))})
    np.testing.assert_array_equal(r.convective_heat_into_solid_W_by_node,np.zeros(4))
    assert r.total_convective_heat_into_solid_W==0.

def test_conductive_boundary_drive_keeps_exact_held_offsets():
    x=np.r_[np.zeros((1,3)),np.diag((1.,1.,.75))]
    b=mesh(x,[[0,1,2,3]])
    def case(cold):
        return solve(b,conductivity_W_per_m_K_by_material={1:1.},
                source_W_per_m3_by_cell={24:-4096.}, faces_by_node_id=[(12,13,14)],
                convection_W_per_m2_K=1.,ambient_K=cold,
                prescribed_K_by_id={11:1025.,13:cold,14:cold})
    with pytest.raises(ValueError,match='near-opposing'):
        case(float(np.nextafter(1.,2.)))
    r=case(1.)
    assert r.temperature_difference_K[1]==0. and r.free_residual_max_W==0.


def test_disconnected_component_heat_closes_separately():
    a=np.r_[np.zeros((1,3)),np.eye(3)]/4.
    x=np.r_[a,a+[.5,0.,0.]]
    t=np.array([[0,1,2,3],[4,5,6,7]])
    n=11+np.arange(8)
    c=np.array([24,26])
    b=Tet4Bundle(x,t,n,c,np.ones(2,dtype=np.int64),{1:{'young_modulus_pa':1e9,'poisson_ratio':.3}},'m')
    args=dict(conductivity_W_per_m_K_by_material={1:1.},prescribed_K_by_id={11:300.,12:300.,13:300.,15:310.,16:310.,17:310.},
              source_W_per_m3_by_cell={24:0.,26:0.},faces_by_node_id=[(11,12,14),(15,16,18)],convection_W_per_m2_K=4.,ambient_K=305.)
    r=solve(b,**args)
    # Free row per right Tet: K_ii=1/24, M_ii=1/48, and the
    # ambient interpolation contributes +/−5·(h A / 3)=+/−5/24 W.
    np.testing.assert_allclose(r.temperature_difference_K[[3,7]],
                               [10./3.,-10./3.],atol=1e-13,rtol=0)
    np.testing.assert_allclose(r.reference_K_by_node,[300.]*4+[310.]*4,atol=0,rtol=0)
    conv=r.convective_heat_into_solid_W_by_node
    cap=r.heat_into_solid_W
    assert conv[:4].sum()>0 and conv[4:].sum()<0
    np.testing.assert_allclose(conv[:4],-conv[4:],atol=1e-14,rtol=0)
    assert abs(sum(cap[:4])+sum(conv[:4]))<1e-14
    assert abs(sum(cap[4:])+sum(conv[4:]))<1e-14
    assert r.total_convective_heat_into_solid_W==pytest.approx(0.,abs=1e-15)
    assert abs(r.balance_W)<1e-14
    with pytest.raises(ValueError,match='each conductive component'):
        solve(b,**(args|dict(prescribed_K_by_id={11:300.,12:300.,13:300.})))
