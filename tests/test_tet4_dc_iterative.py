"""Analytic controls and fail-closed checks for opt-in iterative solid DC."""
import numpy as np
import pytest
from itertools import permutations, product
from dataclasses import replace
from fibernet.sim.tet4_dc import solve_dc_tet4
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4_dc as candidate

def cuboid(m):
    coords=np.array(list(product(*(range(i) for i in m))),dtype=float)
    idx={(int(x),int(y),int(z)):i for i,(x,y,z) in enumerate(coords)}
    e=np.eye(3,dtype=int)
    connect=np.array([[idx[tuple(origin+d)] for d in np.vstack((np.zeros(3,dtype=int),np.cumsum(e[list(o)],axis=0)))]
        for origin in map(np.array,product(*(range(i-1) for i in m))) for o in permutations(range(3))])
    ni=np.arange(len(coords),dtype=np.uint64)+2**53+7
    ci=np.arange(len(connect),dtype=np.uint64)+2**62
    case=Tet4Bundle(coords,connect,ni,ci,np.ones(len(ci),dtype=np.uint64),{1:{'young_modulus_pa':1e9,'poisson_ratio':.3}},'mm')
    return case

def kw(b):
    x=b.points[:,0]
    a,z=x.min(),x.max()
    return dict(conductivity_S_per_m_by_material={1:2.5},electrode_voltage_V_by_id={**dict.fromkeys(b.node_ids[x==a],-.7),**dict.fromkeys(b.node_ids[x==z],.5)})

def check(b,r,m):
    x=b.points[:,0]
    current=2.5*1.2*(m[1]-1)*(m[2]-1)*1.e-3/(m[0]-1)
    expected_voltage=1.2*x/(m[0]-1)-.7
    np.testing.assert_allclose(r.voltage_V,expected_voltage,atol=1e-9)
    np.testing.assert_allclose(r.electrode_current_into_solid_A[x==x.max()].sum(),current,rtol=1.e-9)
    np.testing.assert_allclose(r.electrode_current_into_solid_A[x==0].sum(),-current,rtol=1.e-9)
    np.testing.assert_allclose(r.total_joule_power_W,current*1.2,rtol=1.e-9)
    assert r.solver=='cg' and r.iterations>0 and r.factor_entries==0
    assert np.array_equal(r.node_ids,b.node_ids) and np.array_equal(r.cell_ids,b.cell_ids)
    with pytest.raises(ValueError):
        r.node_ids.flags.writeable=True

def test_analytical_cuboid_exceeds_default_point_limit():
    m=(12,12,12); b=cuboid(m)
    with pytest.raises(ValueError,match='DC pilot mesh limits'):
        solve_dc_tet4(b,**kw(b))
    check(b,solve_dc_tet4(b,solver='cg',**kw(b)),m)

def test_cell_node_reversal_does_not_change_result():
    m=(6,4,3); b=cuboid(m)
    n=np.arange(len(b.points)-1,-1,-1); i=np.arange(len(b.tetrahedra)-1,-1,-1)
    q=replace(b,points=b.points[n],node_ids=b.node_ids[n],cell_ids=b.cell_ids[i],material_ids=b.material_ids[i],
              tetrahedra=n[b.tetrahedra[i][:,[2,0,3,1]]])
    check(q,solve_dc_tet4(q,solver='cg',**kw(b)),m)

def test_fail_closed_on_iteration_limit_and_on_false_success(monkeypatch):
    b=cuboid((4,3,3))
    def fake(eq,rhs,**kwargs):
        return np.zeros_like(rhs),11
    monkeypatch.setattr(candidate,'cg',fake)
    with pytest.raises(ValueError,match='did not converge'):
        solve_dc_tet4(b,solver='cg',**kw(b))
    def fake_success(eq,rhs,**kwargs):
        return np.zeros_like(rhs),0
    monkeypatch.setattr(candidate,'cg',fake_success)
    with pytest.raises(ValueError,match='backward residual'):
        solve_dc_tet4(b,solver='cg',**kw(b))

def test_admission_and_strict_topology():
    b=cuboid((4,3,3))
    c=b.tetrahedra.copy(); c[-1]=c[0]
    with pytest.raises(ValueError,match='duplicate Tet4 connectivity'):
        solve_dc_tet4(replace(b,tetrahedra=c),solver='cg',**kw(b))
    for bad in (False, 1, 'anything'):
        with pytest.raises(ValueError,match='banded or cg'):
            solve_dc_tet4(b,solver=bad,**kw(b))

@pytest.mark.parametrize(("cells", "message"), [
    ([[0, 1, 2, 3], [3, 1, 0, 2], [0, 1, 2, 4]],
     "duplicate Tet4 connectivity at row 1"),
    ([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [3, 0, 1, 2]],
     "non-manifold Tet4 face with more than two cells at row 2"),
    ([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]],
     "non-manifold Tet4 face with more than two cells at row 2"),
])
def test_packed_topology_replays_the_first_scalar_diagnostic(cells, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        candidate.core._validate_packed_topology(np.array(cells))


def test_packed_topology_preserves_high_row_bits_without_scalar_fallback(monkeypatch):
    cells = np.array([[1, 2, 3, 4], [3, 2, 1, 5],
                      [32769, 32770, 32771, 32772], [32773, 32771, 32770, 32769]])

    def unexpected(_):
        pytest.fail("valid packed connectivity reached scalar validation")

    monkeypatch.setattr(candidate.core, "_validate_topology", unexpected)
    candidate.core._validate_packed_topology(cells)
    b = cuboid((4, 3, 3))
    check(b, solve_dc_tet4(b, solver="cg", **kw(b)), (4, 3, 3))


def test_electrically_weak_disconnected_body_independent():
    a = cuboid((8,4,4))
    b = cuboid((5,3,3))
    p = np.concatenate([a.points,b.points*0.02+np.array([0.,10.,0.])])
    t = np.concatenate([a.tetrahedra,b.tetrahedra+len(a.points)])
    nodes = np.arange(len(p),dtype=np.uint64)+2**53+1
    ca=np.arange(len(a.points));cb=np.arange(len(a.points),len(p))
    labels=np.repeat([1,2],[len(a.tetrahedra),len(b.tetrahedra)])
    both=replace(a,points=p,tetrahedra=t,node_ids=nodes,
                 cell_ids=np.arange(len(t))+7,material_ids=labels,materials={1:a.materials[1],2:a.materials[1]})
    elect={}
    for rows,low,high in ((ca,-3.,3.),(cb,70.,70.001)):
        x=p[rows,0]
        elect.update(dict.fromkeys(nodes[rows[x==x.min()]],low))
        elect.update(dict.fromkeys(nodes[rows[x==x.max()]],high))
    dc=solve_dc_tet4(both,solver='cg',conductivity_S_per_m_by_material={1:1e5,2:1e-6},electrode_voltage_V_by_id=elect)
    for rows,sigma,delta in ((ca,1e5,6.),(cb,1e-6,70.001-70.)):
        x,y,z=p[rows].T
        ih=rows[x==x.max()]
        want=sigma*delta*np.ptp(y)*np.ptp(z)/np.ptp(x)*1.e-3
        np.testing.assert_allclose(dc.electrode_current_into_solid_A[ih].sum(),want,rtol=1e-8)

def test_legacy_scipy_tolerance_compatibility_path(monkeypatch):
    b=cuboid((4,3,3)); real=candidate.cg
    original_keyword=candidate._CG_RTOL_NAME
    def legacy(A,rhs,*,tol,atol,maxiter,callback):
        return real(A,rhs,atol=atol,maxiter=maxiter,callback=callback,**{original_keyword:tol})
    monkeypatch.setattr(candidate,'_CG_RTOL_NAME','tol')
    monkeypatch.setattr(candidate,'cg',legacy)
    check(b,solve_dc_tet4(b,solver='cg',**kw(b)),(4,3,3))


def test_batched_geometry_matches_scalar_across_batches_and_orientation(monkeypatch):
    monkeypatch.setattr(candidate, "_DC_CELL_BATCH", 2)
    x = np.array([[.001, .002, .003], [.003, .002, .003],
                  [.0015, .004, .0032], [.002, .0025, .005]])
    t = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [2, 3, 0, 1]])
    g, v = candidate._batched_geometry(x, t)
    ref_g, ref_v = zip(*(candidate._checked_geometry_row(x, c, i)
                         for i, c in enumerate(t)))
    np.testing.assert_allclose(g, ref_g, rtol=2e-14, atol=1e-11)
    np.testing.assert_allclose(v, ref_v, rtol=2e-14)


@pytest.mark.parametrize("lengths,message", [
    ((1e-8, 1e-8, 1e-8), "volume"),
    ((1e-3, 1e-3, 1e-14), "ill-conditioned"),
    ((1e-3, 1e-3, 5e-13), "gradient"),
])
def test_batched_geometry_keeps_scalar_invalid_cell_contract(lengths, message):
    x = np.vstack([np.zeros(3), np.diag(lengths)])
    t = np.array([[0, 1, 2, 3]])
    with pytest.raises(ValueError, match=message):
        candidate._batched_geometry(x, t)


def test_geometry_near_volume_cutoff_uses_reference(monkeypatch):
    original = candidate._checked_geometry_row
    calls = []
    def mark(x, cell, row):
        calls.append(row)
        return original(x, cell, row)
    monkeypatch.setattr(candidate, "_checked_geometry_row", mark)
    d = np.cbrt(6e-24) * (1. + 1e-12)
    p = np.vstack([np.zeros(3), np.eye(3) * d])
    _, v = candidate._batched_geometry(p, np.array([[0, 1, 2, 3]]))
    assert calls == [0] and v[0] >= 1e-24
