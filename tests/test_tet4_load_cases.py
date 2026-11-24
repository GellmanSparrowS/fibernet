from functools import wraps
from pathlib import Path
from typing import Union, get_type_hints
import runpy
import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4
from fibernet.sim.tet4 import solve_static_tet4_load_cases

build=runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples'/'tet4_static_two_material.py'))['build_example']

def test_batch_signature_resolves_runtime_type_hints():
    assert get_type_hints(solve_static_tet4_load_cases)['force_cases'] == Union[list, tuple]


def assert_result(a,b):
    for field in ('node_ids','displacement_m','reaction_N','fixed_mask'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    for field in ('strain_energy_J','half_external_work_J','free_residual_inf_N',
                  'normalized_backward_error','relative_energy_work_error','cell_fields'):
        assert getattr(a,field)==getattr(b,field)

def test_fixed_controls_exact_parity_shared_preparation_distinct_result_ownership(monkeypatch):
    mesh,base,fixed=build(0)
    clamp=next(iter(fixed))
    force_cases=[{},base,{clamp:(-1e-170,0.,0.)},{j:tuple(-3*x for x in v) for j,v in base.items()}]
    expected=tuple(tet4.solve_static_tet4(mesh,forces_by_id=f,fixed_dofs_by_id=fixed) for f in force_cases)
    counts={}
    def watch(owner,name):
        f=getattr(owner,name);counts[name]=0
        @wraps(f)
        def wrapper(*a,**k):
            counts[name]+=1
            return f(*a,**k)
        monkeypatch.setattr(owner,name,wrapper)
    for name in ('_validate_static_mesh','_static_topology_plan','_assemble','_factor_band'):
        watch(tet4,name)
    for name in ('cholesky_banded','cho_solve_banded'):watch(tet4.linalg,name)
    actual=solve_static_tet4_load_cases(mesh,force_cases=force_cases,fixed_dofs_by_id=fixed)
    assert counts=={'_validate_static_mesh':1,'_static_topology_plan':1,'_assemble':1,
                    '_factor_band':1,'cholesky_banded':2,'cho_solve_banded':2}
    assert len(actual)==4 and actual[2].strain_energy_J==0 and actual[2].reaction_N.sum(axis=0)[0]==1e-170
    for a,b in zip(actual,expected): assert_result(a,b)
    for attr in ('node_ids','fixed_mask','displacement_m','reaction_N'):
        a,b=getattr(actual[0],attr),getattr(actual[1],attr)
        assert not a.flags.writeable and not b.flags.writeable and not np.shares_memory(a,b)
    assert not np.shares_memory(actual[0].node_ids,mesh.node_ids)

def test_shared_factor_does_not_mask_tiny_free_load_or_hide_zero_rhs_rank_check(monkeypatch):
    mesh,good,fixed=build(0)
    free=next(iter(good))
    tiny_energy={free:(1e-170,0.,0.)}
    for f in (
        lambda: tet4.solve_static_tet4(mesh,forces_by_id=tiny_energy,fixed_dofs_by_id=fixed),
        lambda: solve_static_tet4_load_cases(mesh,force_cases=[good,tiny_energy],fixed_dofs_by_id=fixed),
    ):
        with pytest.raises(ValueError,match='positive work/energy'):
            f()
    def floor(*args):raise ValueError('probe factor floor on zero RHS')
    monkeypatch.setattr(tet4,'_factor_band',floor)
    with pytest.raises(ValueError,match='probe factor floor'):
        solve_static_tet4_load_cases(mesh,force_cases=[{}],fixed_dofs_by_id=fixed)

def test_count_and_late_invalid_case_fail_before_assembly(monkeypatch):
    mesh,base,fixed=build(0)
    def forbidden(*a,**k): raise AssertionError('invalid case before any numeric assembly')
    monkeypatch.setattr(tet4,'_assemble',forbidden)
    for over in ([], [{}]*9,iter([base]), {'case':base}):
        with pytest.raises(ValueError,match='ordered list or tuple'):
            solve_static_tet4_load_cases(mesh,force_cases=over,fixed_dofs_by_id=fixed)
    nid=next(iter(base))
    for bad,reason in (({1:(1.,0.,0.)},'foreign'),({nid:(2**53+1,0.,0.)},'exactly representable'),
                      ({nid:(np.uint64(1).view(np.float64),0.,0.)},'normal binary64')):
        with pytest.raises(ValueError,match=reason):
            solve_static_tet4_load_cases(mesh,force_cases=[base,bad],fixed_dofs_by_id=fixed)


def face_joined_mesh():
    # The two tetrahedra share one full face, a supported conforming body.
    xyz = .001 * np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],dtype=float)
    ids = np.array([2**63 + 5, 2**53 + 91, 19, 91, 2**64 - 1], dtype=np.uint64)
    mesh = Tet4Bundle(
        xyz, [[0, 1, 2, 3], [0, 1, 2, 4]], ids,
        [2**63+77, 3], [1, 2],
        {1: dict(young_modulus_pa=1e9, poisson_ratio=.25),
         2: dict(young_modulus_pa=2e9, poisson_ratio=.25)}, "m",
    )
    one_held = {int(ids[0]): ('x','y','z')}
    transverse = {int(ids[4]): (0., 2e-4, 0.)}  # finite moment about the held point
    return mesh, ids, one_held, transverse


def test_physical_unrestrained_rotation_rejects_zero_first_rhs_same_as_scalar_without_mocks():
    mesh, _, held, transverse = face_joined_mesh()
    # An infinitesimal rigid rotation about the held origin's x axis is a
    # zero-strain mode, vanishes at that node, and does work under this force.
    u_rigid = np.cross([1.,0.,0.],mesh.points)
    np.testing.assert_array_equal(u_rigid[0],0.)
    assert np.dot(next(iter(transverse.values())),u_rigid[4]) > 0.
    messages=[]
    for force in ({},transverse):
        with pytest.raises(ValueError, match='restraints do not suppress six rigid modes') as exc:
            tet4.solve_static_tet4(mesh, forces_by_id=force, fixed_dofs_by_id=held)
        messages.append(str(exc.value))
    with pytest.raises(ValueError, match='restraints do not suppress six rigid modes') as exc:
        solve_static_tet4_load_cases(mesh, force_cases=[{},transverse], fixed_dofs_by_id=held)
    assert all(message == str(exc.value) for message in messages)


def test_all_held_same_mesh_gives_exact_reactions_without_mocks():
    mesh, ids, _, transverse = face_joined_mesh()
    held = dict.fromkeys(map(int,ids), ('x','y','z'))
    opposite = {int(ids[2]):(1e-170, -1.5e-170, 3e-170)}
    forces = ({},transverse,opposite)
    out = solve_static_tet4_load_cases(mesh, force_cases=forces, fixed_dofs_by_id=held)
    assert len(out) == 3
    for result, force in zip(out,forces):
        assert_result(result, tet4.solve_static_tet4(mesh,forces_by_id=force,fixed_dofs_by_id=held))
        np.testing.assert_array_equal(result.displacement_m,0.)
        np.testing.assert_array_equal(result.reaction_N,[-np.asarray(force.get(int(n),(0.,0.,0.))) for n in ids])
        assert result.fixed_mask.all()
        assert result.half_external_work_J == result.strain_energy_J == 0.


def test_maximum_eight_signed_cases_stable_uint64_order_no_alias_and_same_physics():
    mesh, standard, held = build(0)
    ids = tuple(map(int,mesh.node_ids))
    assert mesh.node_ids.dtype == np.uint64 and max(ids) > 2**63 and min(standard) > 2**53
    scales = (-4., 0., .5, -1., 8., -2., 2., -.125)
    cases = [{k:tuple(s*np.asarray(v)) for k,v in standard.items()} for s in scales]
    results = solve_static_tet4_load_cases(mesh,force_cases=cases,fixed_dofs_by_id=held)
    assert len(results) == 8
    for scale,mapping,got in zip(scales,cases,results):
        assert_result(got,tet4.solve_static_tet4(mesh,forces_by_id=mapping,fixed_dofs_by_id=held))
        assert tuple(map(int,got.node_ids)) == ids
        np.testing.assert_allclose(got.reaction_N.sum(axis=0),[-scale*.001,0.,0.],rtol=1e-12,atol=1e-16)
        np.testing.assert_allclose(got.displacement_m[:,0],
              scale*(1e-6*np.minimum(mesh.points[:,0],.5)+.5e-6*np.maximum(mesh.points[:,0]-.5,0)),rtol=2e-12,atol=1e-19)
        np.testing.assert_allclose(got.strain_energy_J,.5*.001*.75e-6*scale**2,rtol=2e-12,atol=0.)
    for name in ('node_ids','fixed_mask','displacement_m','reaction_N'):
        arrays=[getattr(r,name) for r in results]
        assert all(not a.flags.writeable for a in arrays)
        assert all(not np.shares_memory(a,b) for i,a in enumerate(arrays) for b in arrays[i+1:])
    assert not np.shares_memory(results[0].node_ids,mesh.node_ids)
