"""Analytical and integration tests for explicitly selected Tet4 face traction."""
from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction as apply

P=np.array([[0.,0.,0.],[1.,2.,0.],[0.,1.,3.],[0.,0.,1.],[0.,0.,-1.]])
IDS=np.array([2**64-1,7,2**63+7,4,81],dtype=np.uint64)
FACE=tuple(IDS[:3])
MAT={2**64-1:{'young_modulus_pa':1e9,'poisson_ratio':.3}}
T=(800.,-150.,500.)


def bundle(unit='mm', scale=1., second=False):
    cells=[[0,1,2,3]]+([[0,2,1,4]] if second else [])
    return Tet4Bundle(P*scale,cells,IDS,list(range(len(cells))),[2**64-1]*len(cells),MAT,unit)


def test_tilted_exact_resultant_moment_and_reordering():
    mesh=bundle()
    F=apply(mesh,[FACE],traction_Pa=T)
    area=np.sqrt(46)/2*1e-6
    force=np.array(T)*area
    point_keys=[int(i) for i in IDS[:3]]
    values=np.array([F[k] for k in point_keys])
    assert all(type(k) is int for k in F) and set(F)==set(point_keys)
    np.testing.assert_allclose(values,np.tile(force/3,(3,1)),rtol=5e-16)
    np.testing.assert_allclose(values.sum(0),force,rtol=5e-16)
    np.testing.assert_allclose(np.cross(P[:3]*1e-3,values).sum(0),
                               np.cross(P[:3].mean(0)*1e-3,force),rtol=1e-15)
    assert apply(mesh,[(FACE[2],FACE[0],FACE[1])],traction_Pa=T)==F
    new=np.array([2,4,0,1,3])
    remap=replace(mesh,points=mesh.points[new],node_ids=IDS[new],
                  tetrahedra=np.argsort(new)[np.array(mesh.tetrahedra)[:,::-1]])
    assert apply(remap,[FACE],traction_Pa=T)==F
    other=apply(bundle('m',1e-3),[FACE],traction_Pa=T)
    np.testing.assert_allclose(list(other.values()),list(F.values()),rtol=6e-16)


@pytest.mark.parametrize('faces,second,error',[
  ([FACE,FACE[::-1]],False,'duplicate selected'),
  ([FACE],True,'not an exterior'),
  ([(IDS[0],IDS[1],IDS[-1])],False,'not an exterior'),
  ([(IDS[0],IDS[0],IDS[1])],False,'repeated or foreign'),
  ([(float(IDS[0]),IDS[1],IDS[2])],False,'exact nonnegative'),
])
def test_face_rejections(faces,second,error):
    with pytest.raises(ValueError,match=error):
        apply(bundle(second=second),faces,traction_Pa=T)


def test_numeric_domain_never_silently_returns_zero_or_infinite():
    for t in ((1.,2.,float('inf')),(np.uint64(1).view(np.float64),0.,0.),
              (1e-308,0.,0.),(1e-305,0.,0.),(True,0.,0.)):
        with pytest.raises(ValueError):
            apply(bundle(),[FACE],traction_Pa=t)
    with pytest.raises(ValueError,match='area'):
        apply(bundle('m',1e-170),[FACE],traction_Pa=T)
    with pytest.raises(ValueError,match='binary64'):
        apply(bundle('m',1e152),[FACE],traction_Pa=(1e300,0.,0.))
    assert set(apply(bundle(),[FACE],traction_Pa=(0.,0.,0.)).values())=={(0.,0.,0.)}


def test_two_adjacent_faces_add_by_area():
    f2=(IDS[0],IDS[1],IDS[3])
    b=bundle()
    A=apply(b,[FACE],traction_Pa=T)
    B=apply(b,[f2],traction_Pa=T)
    AB=apply(b,[f2,FACE],traction_Pa=T)
    assert AB==apply(b,[FACE,f2],traction_Pa=T)
    for key,v in AB.items():
        np.testing.assert_allclose(v,np.add(A.get(key,(0.,)*3),B.get(key,(0.,)*3)),rtol=7e-16)


def test_welded_quick_matches_native_example():
    path=Path(__file__).resolve().parents[1]/'examples/tet4_static_welded_junction.py'
    demo=runpy.run_path(str(path))
    grid,x,t,_,_,_=demo['make_geometry'](12)
    left,right,_=demo['boundary'](grid,t,12)
    force,fixed,_,pressure=demo['load_vectors'](x,left,right)
    ids=np.arange(len(x),dtype=np.uint64)*19+np.uint64(2**63+23)
    b=Tet4Bundle(x,t,ids,np.arange(len(t),dtype=np.uint64),np.full(len(t),2**64-1,dtype=np.uint64),MAT,'mm')
    q=apply(b,[[ids[i] for i in f] for f in right],traction_Pa=(pressure,0.,0.))
    dense=np.array([q.get(int(i),(0.,0.,0.)) for i in ids])
    np.testing.assert_allclose(dense,force,rtol=8e-16)
    np.testing.assert_allclose(dense.sum(0),[1e-3,0.,0.],rtol=1e-15)
    holds={int(ids[i]):('x','y','z') for i in fixed}
    r=solve_static_tet4(b,forces_by_id=q,fixed_dofs_by_id=holds)
    np.testing.assert_allclose(r.strain_energy_J,1.692609584859141e-12,rtol=2e-11)


def test_exact_cross_detects_skinny_but_represented_triangle():
    eps = 2.**-27
    x = np.array([[0.,0.,0.],[1.+eps,1.,0.],[1.,1.-eps,0.],[0.,0.,1.]])
    # (1+eps)*(1-eps) rounds to 1, but the exact determinant is -2^-54.
    assert not np.cross(x[1],x[2]).any()
    b = Tet4Bundle(x,[[0,1,2,3]],IDS[:4],[11],[2**64-1],MAT,'m')
    f = apply(b,[FACE],traction_Pa=(3.,0.,0.))
    assert set(f.values()) == {(2.**-55,0.,0.)}

def test_contributing_areas_can_sum_above_float_max_when_force_fits():
    scale=1.1e154
    x=np.array([[0.,0.,0.],[scale,0.,0.],[0.,scale,0.],[0.,0.,scale]])
    ids=np.array([2**64-1,2**63+7,19,2**63+3],dtype=np.uint64)
    mesh=Tet4Bundle(x,np.array([[0,1,2,3]]),ids,np.array([51]),np.array([1]),
        {1:{'young_modulus_pa':1e9,'poisson_ratio':.3}},'m')
    faces=[(int(ids[0]),int(ids[i]),int(ids[j])) for i,j in [(1,2),(1,3),(2,3)]]
    forces=apply(mesh,faces,traction_Pa=[1e-101,-1e-101,0.])
    np.testing.assert_allclose(forces[int(ids[0])],[6.05e206,-6.05e206,0.],rtol=4e-15)
    np.testing.assert_allclose(forces[int(ids[2])],[4.033333333333333e206,-4.033333333333333e206,0.],rtol=4e-15)


@pytest.mark.parametrize('selector,why', [
    ([], 'nonempty'), (np.array(0), 'shape'), ([(7, 4)], 'three exact'),
    ([(7, 4, 999)], 'foreign'), (np.array([[7, 4, 81]], dtype=float), 'exact nonnegative'),
    (np.ma.array([[7, 4, 81]], mask=[[False, True, False]]), 'masked'),
])
def test_malformed_selector_refused_before_area(selector, why, monkeypatch):
    import fibernet.sim.tet4_loads as loads
    def forbidden(*args, **kwargs):
        pytest.fail('cannot proceed to numerical area for a bad selector')
    monkeypatch.setattr(loads.math, 'hypot', forbidden)
    with pytest.raises(ValueError, match=why):
        apply(bundle(), selector, traction_Pa=T)


def test_traction_uses_shared_face_scan_exactly_once_without_solving(monkeypatch):
    import fibernet.sim.tet4_loads as loads
    from fibernet.sim import tet4
    original = loads._validate_topology
    calls = []
    def track(x):
        calls.append(x.shape)
        return original(x)
    def forbidden(*args, **kwargs):
        pytest.fail('traction conversion must not assemble stiffness or factor')
    monkeypatch.setattr(loads, '_validate_topology', track)
    monkeypatch.setattr(tet4, '_assemble', forbidden)
    monkeypatch.setattr(tet4.linalg, 'cholesky_banded', forbidden)
    F = apply(bundle(), np.array([FACE], dtype=np.uint64), traction_Pa=T)
    assert calls == [(1, 4)] and len(F) == 3


def test_selector_size_is_checked_before_iteration():
    class Oversized(list):
        def __len__(self):
            return 10000

        def __iter__(self):
            pytest.fail('oversize selector must not be iterated')

    with pytest.raises(ValueError, match='bounded sequence'):
        apply(bundle(), Oversized(), traction_Pa=T)


@pytest.mark.parametrize('multiplier, normal_area', [(1.25, True), (1.0, False)])
def test_normal_range_checked_on_final_area_not_one_cross_component(multiplier, normal_area):
    import math

    s = math.ldexp(multiplier, -511)
    coords = [[0.,0.,0.],[s,-s,0.],[0.,s,-s],[0.,0.,1.]]
    mesh = Tet4Bundle(coords, [[0,1,2,3]], IDS[:4], [17], [2**64-1], MAT, 'm')
    if normal_area:
        # A=sqrt(3)*s²/2, computed without a subnormal intermediate.
        expected = math.ldexp(math.sqrt(3.)*multiplier**2, -1023)
        assert expected > np.finfo(np.float64).tiny
        f = apply(mesh, [FACE], traction_Pa=(3.,0.,0.))
        assert set(f) == set(map(int, FACE))
        for fx, fy, fz in f.values():
            assert math.isclose(fx/expected, 1., rel_tol=4e-15)
            assert (fy, fz) == (0.,0.)
    else:
        # sqrt(3)*2**-1023 < 2**-1022, so the final area itself is subnormal.
        with pytest.raises(ValueError, match='triangle area'):
            apply(mesh, [FACE], traction_Pa=(3.,0.,0.))
