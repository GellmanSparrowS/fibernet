"""Physics, stable IDs, and admission checks for the explicit Tet4 solver."""
from dataclasses import replace
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/tet4_static_two_material.py'
build_example = runpy.run_path(str(EXAMPLE))['build_example']


def solve(bundle, forces, held):
    return tet4.solve_static_tet4(bundle, forces_by_id=forces, fixed_dofs_by_id=held)


@pytest.mark.parametrize('axis,delta', [(0, .75e-6), (1, 2e-6/3)])
def test_two_material_force_control_against_closed_form(axis, delta):
    bundle, forces, held = build_example(axis)
    x = bundle.points[:, 0]
    points_before = bundle.points.tobytes()
    result = solve(bundle, forces, held)
    assert isinstance(result, tet4.Tet4StaticResult)
    expected = x * delta
    if axis == 0:
        expected = 1e-6*np.minimum(x,.5) + .5e-6*np.maximum(x-.5,0.)
    np.testing.assert_allclose(result.displacement_m[:,0],expected,atol=3e-19,rtol=2e-12)
    np.testing.assert_allclose(result.displacement_m[:,1:],0.,atol=3e-19)
    np.testing.assert_allclose(result.reaction_N.sum(axis=0),[-.001,0.,0.],atol=3e-17)
    work = sum(np.dot(forces.get(int(k),(0.,0.,0.)),u) for k,u in zip(result.node_ids,result.displacement_m))
    np.testing.assert_allclose([result.strain_energy_J,result.half_external_work_J,.5*work],
                              .5e-3*delta,rtol=2e-12)
    assert result.normalized_backward_error<1e-13 and result.relative_energy_work_error<1e-12
    assert bundle.points.tobytes()==points_before and max(result.node_ids)>2**63
    for a in (result.node_ids,result.fixed_mask,result.displacement_m,result.reaction_N):
        assert not a.flags.writeable


def test_mixed_axis_oblique_affine_patch_relabelled_and_reordered():
    x=1e-3*np.vstack([np.zeros(3),np.eye(3)])
    e,nu=17e6,.29
    grad=np.array([[.0008,.0006,-.0004],[0.,-.0003,.0005],[0.,0.,.0002]])
    eps=(grad+grad.T)/2
    dev=eps-np.trace(eps)/3*np.eye(3)
    stress=e/(1+nu)*dev + e/(3*(1-2*nu))*np.trace(eps)*np.eye(3)
    traction=np.zeros_like(x)
    for opposite in range(4):
        face=[n for n in range(4) if n!=opposite]
        a,b,c=x[face]
        area=.5*np.cross(b-a,c-a)
        if area@(x[opposite]-a)>0.:area=-area
        traction[face]+=stress@area/3
    mask=np.array([[True]*3,[False,True,True],[False,False,True],[False]*3])
    f=np.where(mask,[3e-4,-2e-4,1e-4],traction)
    expected_reaction=np.where(mask,traction-f,0.)
    expected_u=x@grad.T
    expected_energy=.5*np.sum(stress*eps)*1e-9/6
    for unit,nodes,order in (
      ('m',[7,1,2**64-1,66],[0,1,2,3]),('mm',[2**63+4,12,2**64-1,81],[1,3,0,2]),
      ('nm',[7,2**63+4,13,2**64-1],[3,2,1,0])):
        ids=np.array(nodes,dtype=np.uint64)
        inv=np.argsort(order)
        b=Tet4Bundle(x[order]*{'m':1.,'mm':1e3,'nm':1e9}[unit], [inv[[3,0,2,1]]],
                     ids[order],[2**64-1],[5],{5:{'young_modulus_pa':e,'poisson_ratio':nu}},unit)
        axes=list('xyz')
        held={int(k):tuple(ax for ax,bind in zip(axes,v) if bind) for k,v in zip(ids,mask) if v.any()}
        result=solve(b,dict(zip(map(int,ids),f)),held)
        np.testing.assert_array_equal(result.node_ids,ids[order])
        np.testing.assert_array_equal(result.fixed_mask[inv],mask)
        np.testing.assert_allclose(result.displacement_m[inv],expected_u,atol=3e-20,rtol=2e-12)
        np.testing.assert_allclose(result.reaction_N[inv],expected_reaction,atol=3e-17,rtol=2e-12)
        np.testing.assert_allclose([result.strain_energy_J,result.half_external_work_J],
                                   expected_energy,rtol=2e-12)
        total=f+result.reaction_N[inv]
        np.testing.assert_allclose(total.sum(axis=0),0.,atol=3e-17)
        np.testing.assert_allclose(np.cross(x,total).sum(axis=0),0.,atol=3e-20)


def test_forces_on_supported_axes_are_reactions_without_work():
    b,_,held=build_example()
    i=next(iter(held))
    reaction=solve(b,{i:(1e-170,0.,0.)},held)
    assert reaction.reaction_N.sum(axis=0)[0]==-1e-170
    assert reaction.strain_energy_J==reaction.half_external_work_J==0.
    np.testing.assert_array_equal(reaction.displacement_m,0.)


@pytest.mark.parametrize('bad,reason', [
    (np.uint64(1).view(np.float64),'normal binary64'),
    (1e-170,'positive work/energy'), (True,'finite binary64'),
    (2**53+1,'exactly representable'),
])
def test_rejects_invalid_or_unrepresentable_free_force(bad,reason):
    b,loads,held=build_example()
    with pytest.raises(ValueError,match=reason):
        solve(b,{next(iter(loads)):(bad,0.,0.)},held)


def test_wider_float_is_rejected_before_narrowing():
    if np.finfo(np.longdouble).minexp>=np.finfo(np.float64).minexp:
        pytest.skip('this platform has no wider longdouble exponent range')
    b,loads,held=build_example()
    with pytest.raises(ValueError,match='exactly representable'):
        solve(b,{next(iter(loads)):np.array([np.longdouble('1e-350'),0,0])},held)


def test_rejects_unknown_id_missing_rigid_modes_and_invalid_axes():
    b,loads,held=build_example()
    i=next(iter(held))
    with pytest.raises(ValueError,match='foreign'):
        solve(b,{2**64-1:(1.,0.,0.)},held)
    with pytest.raises(ValueError,match='underconstrained'):
        solve(b,loads,{i:('x',)})
    with pytest.raises(ValueError,match='distinct'):
        solve(b,loads,{i:('x','x')})


def test_rejects_duplicate_and_nonmanifold_connectivity():
    b,loads,held=build_example()
    copied=replace(b,tetrahedra=np.r_[b.tetrahedra,b.tetrahedra[:1,::-1]],
                   cell_ids=np.r_[b.cell_ids,999],material_ids=np.r_[b.material_ids,b.material_ids[:1]])
    with pytest.raises(ValueError,match='duplicate Tet4'):
        solve(copied,loads,held)
    tetra=Tet4Bundle(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.],[.2,.3,.5]]),
                    [[0,1,2,3],[0,2,1,4],[0,1,2,5]],list(range(6)),[7,8,9],[1,1,1],
                    {1:{'young_modulus_pa':1e6,'poisson_ratio':.3}},'mm')
    with pytest.raises(ValueError,match='non-manifold Tet4'):
        solve(tetra,{}, {})


def test_band_budget_stops_before_assembly(monkeypatch):
    b,loads,held=build_example()
    monkeypatch.setattr(tet4,'_MAX_FACTOR_ENTRIES',50)

    def no_assembly(*args):
        pytest.fail('resource admission happened after assembling the matrix')

    monkeypatch.setattr(tet4,'_assemble',no_assembly)
    with pytest.raises(ValueError,match='topology band exceeds'):
        solve(b,loads,held)


def test_lazy_optional_meshio_and_no_implicit_facade_export():
    code = '''import importlib.abc, sys
class ForbidMeshio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "meshio":
            raise AssertionError("unexpected optional dependency import")
sys.meta_path.insert(0, ForbidMeshio())
import fibernet
import fibernet.sim
assert not hasattr(fibernet, "solve_static_tet4")
assert not hasattr(fibernet.sim, "solve_static_tet4")
from fibernet.sim.tet4 import solve_static_tet4, Tet4StaticResult
assert "meshio" not in sys.modules
'''
    subprocess.run([sys.executable, '-B', '-c', code], check=True)


def test_shifted_factor_only_checks_positive_floor():
    from scipy.sparse import csr_matrix
    a=np.array([[2.,-.25,0.],[-.25,2.,.5],[0.,.5,3.]])
    b=np.array([1.,-2.,3.])
    plan=tet4._BandPlan(np.arange(3),np.arange(3),1,10,6)
    floor=.05  # big enough to expose mistakenly solving the shifted system
    answer=tet4._factor_band(csr_matrix(a),floor,plan)(b)
    np.testing.assert_allclose(answer,np.linalg.solve(a,b),rtol=1e-14)
    assert np.max(np.abs(answer-np.linalg.solve(a-floor*np.eye(3),b)))>.001
    weak=csr_matrix(np.diag([1.,2.,1e-5]))
    with pytest.raises(ValueError,match='shifted band floor rejected'):
        tet4._factor_band(weak,floor,plan)


def test_canonical_mixed_band_plan_is_invariant_to_representable_row_reorder():
    b,_,held=build_example(1)
    rr=np.arange(len(b.points))[::-1]
    co=np.arange(len(b.tetrahedra))[::-1]
    reordered=replace(b,points=b.points[rr]*1e-3,length_unit='m',node_ids=b.node_ids[rr],
                      tetrahedra=np.argsort(rr)[b.tetrahedra[co,::-1]],cell_ids=b.cell_ids[co],
                      material_ids=b.material_ids[co])
    keys=[]
    for example in (b,reordered):
        mask=np.array([[ax in held.get(int(k),()) for ax in 'xyz'] for k in example.node_ids])
        plan=tet4._plan_band(example,mask)
        dof=plan.free[plan.permutation]
        keys.append([(int(example.node_ids[k//3]),int(k%3)) for k in dof])
        assert {(int(example.node_ids[k//3]),int(k%3)) for k in np.flatnonzero(~mask.ravel())}==set(keys[-1])
    assert keys[0]==keys[1]
