"""Numerical parity and admission for iterative solid mechanics."""
from dataclasses import replace
import numpy as np
import pytest

from examples.tet4_mechanoelectric_looped_network import _lattice, _bundle, _fixture
from fibernet.sim.tet4 import solve_static_tet4




@pytest.mark.parametrize('stiffness',['uniform','stiff_half'])
@pytest.mark.parametrize('load',['stroke','force','mixed'])
def test_direct_iterative_parity(stiffness,load):
    pytest.importorskip('pyamg')
    points,cells=_lattice()
    b=_bundle(points,cells,stiffness)
    right, right_ids, fixed, _, _=_fixture(b)
    # On the right cap the fixture already holds y/z. Add a conventional
    # axial drive or a prescribed stroke and a transverse body force.
    body=int(b.node_ids[(points[:,0]>.9)&(points[:,0]<2.1)][0])
    force={int(n):[.01,0.,0.] for n in right_ids} if load=='force' else {}
    if load=='mixed': force[body]=[.006,.013,-.002]
    prescribed={n:{'x':1.5e-6} for n in right_ids} if load in ('stroke','mixed') else None
    kw=dict(forces_by_id=force,fixed_dofs_by_id=fixed,
        prescribed_displacements_m_by_id=prescribed,return_cell_fields=True)
    a=solve_static_tet4(b,**kw)
    x=solve_static_tet4(b,solver='amg-cg',**kw)
    np.testing.assert_array_equal(x.node_ids,a.node_ids)
    np.testing.assert_allclose(x.displacement_m,a.displacement_m,rtol=1e-9,atol=2e-17)
    np.testing.assert_allclose(x.reaction_N,a.reaction_N,rtol=1e-8,atol=2e-11)
    np.testing.assert_allclose(x.cell_fields.stress_Pa,a.cell_fields.stress_Pa,
                               rtol=1e-7,atol=1e-4)
    np.testing.assert_allclose(x.strain_energy_J,a.strain_energy_J,rtol=2e-9)
    np.testing.assert_allclose(x.strain_energy_J,x.half_external_work_J,rtol=1e-9)


def test_insufficient_constraints_and_no_silent_bad_solver():
    p,t=_lattice()
    b=_bundle(p,t,'uniform')
    first=int(b.node_ids[0])
    args=dict(forces_by_id={},fixed_dofs_by_id={first:('x','y','z')})
    with pytest.raises(ValueError,match='six rigid modes'):
        solve_static_tet4(b,solver='amg-cg',**args)
    for bad in (False,1,'approximate'):
        with pytest.raises(ValueError,match='banded or amg-cg'):
            solve_static_tet4(b,solver=bad,**args)


def test_optional_dependency_error_does_not_break_banded(monkeypatch):
    import builtins
    before=builtins.__import__
    def block(name,*args,**kwargs):
        if name=='pyamg':
            raise ImportError('deliberately unavailable backend')
        return before(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',block)
    p,t=_lattice()
    b=_bundle(p,t,'uniform')
    right,ids,fixed,_,_=_fixture(b)
    load={n:{'x':1.5e-6} for n in ids}
    kw=dict(forces_by_id={},fixed_dofs_by_id=fixed,prescribed_displacements_m_by_id=load)
    assert solve_static_tet4(b,**kw).reaction_N[right,0].sum() > 0
    with pytest.raises(ImportError,match='solid-iterative extra'):
        solve_static_tet4(b,solver='amg-cg',**kw)

def test_proportional_sweep_matches_independently_solved_endpoints():
    pytest.importorskip('pyamg')
    from examples.tet4_iterative_mechanoelectric_sweep import run
    from examples.tet4_mechanoelectric_looped_network import _scan
    p,t=_lattice()
    b=_bundle(p,t,'stiff_half')
    out=run(refinement=1)
    for row in out['states']:
        baseline=_scan(b,row['stroke_m']/.003)
        np.testing.assert_allclose([row['resistance_ohm'],row['reaction_N']],baseline,
                                   rtol=1e-9,atol=2e-11)
    resistances=[v['resistance_ohm'] for v in out['states']]
    assert np.all(np.diff(resistances)>0)


@pytest.mark.parametrize('reported_status,message', [
    (17,'did not converge'),
    (0,'backward residual'),
])
def test_iterative_convergence_flag_is_not_enough(monkeypatch,reported_status,message):
    pytest.importorskip('pyamg')
    from fibernet.sim import _tet4_amg
    def failed_or_incorrect(_,rhs,**kwargs):
        return np.zeros_like(rhs),reported_status
    monkeypatch.setattr(_tet4_amg,'cg',failed_or_incorrect)
    points,cells=_lattice()
    b=_bundle(points,cells,'uniform')
    _,grip,fixed,_,_=_fixture(b)
    with pytest.raises(ValueError,match=message):
        solve_static_tet4(b,solver='amg-cg',forces_by_id={},fixed_dofs_by_id=fixed,
                         prescribed_displacements_m_by_id={n:{'x':1.5e-6} for n in grip})


def test_two_face_disconnected_solids_and_lower_material_limit():
    points=np.vstack([np.zeros(3),np.eye(3)])
    elements=np.array([[0,1,2,3],[4,5,6,7]])
    from fibernet.io.tet4_meshio import Tet4Bundle
    b=Tet4Bundle(np.vstack([points,points+[2.,0.,0.]]),elements,
                 np.arange(8),np.arange(2),np.zeros(2,dtype=int),
                 {0:dict(young_modulus_pa=1e9,poisson_ratio=.3)},'mm')
    # Suppress each body's rigid modes so disconnected-body admission,
    # not insufficient supports, is the reason it fails.
    fixed=dict.fromkeys(range(8),('x','y','z'))
    with pytest.raises(ValueError,match='one face-connected solid'):
        solve_static_tet4(b,solver='amg-cg',forces_by_id={},fixed_dofs_by_id=fixed)
    low=Tet4Bundle(points,elements[:1],np.arange(4),np.arange(1),[0],
                   {0:dict(young_modulus_pa=100.,poisson_ratio=.3)},'mm')
    with pytest.raises(ValueError,match='material is outside'):
        solve_static_tet4(low,solver='amg-cg',forces_by_id={},
                         fixed_dofs_by_id={k:fixed[k] for k in range(4)})


def test_mixed_scale_assembly_keeps_scalar_row_fallback(monkeypatch):
    from fibernet.io.tet4_meshio import Tet4Bundle
    from fibernet.sim import _tet4_amg, tet4
    pts = np.vstack([np.zeros(3), np.diag([.001, .002, .0025]), [0, 0, -1e-12]])
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    b = Tet4Bundle(pts, cells, np.arange(5), np.arange(2), [5, 7], {
        5: dict(young_modulus_pa=3e9, poisson_ratio=-.9),
        7: dict(young_modulus_pa=1e9, poisson_ratio=.49)}, 'm')
    mesh = tet4._validate_static_mesh(b)
    old = tet4._assemble(mesh).toarray()
    scalar = tet4._shape_gradients_and_volume_m
    rows = []
    def count(x, t, row):
        rows.append(row)
        return scalar(x, t, row)
    monkeypatch.setattr(tet4, '_shape_gradients_and_volume_m', count)
    new = _tet4_amg.chunked_assembly(mesh).toarray()
    assert rows == [1]
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(new, new.T)


def test_batched_assembly_keeps_original_invalid_row_number():
    from fibernet.io.tet4_meshio import Tet4Bundle
    from fibernet.sim import _tet4_amg, tet4
    pts = np.vstack([np.zeros(3), np.diag([.001, .002, .0025]), [0, 0, -1e-16]])
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    b = Tet4Bundle(pts, cells, np.arange(5), np.arange(2), [1, 1],
                   {1: dict(young_modulus_pa=1e9, poisson_ratio=.3)}, 'm')
    mesh = tet4._validate_static_mesh(b)
    with pytest.raises(ValueError, match='Tet4 row 1: element is too ill-conditioned'):
        _tet4_amg.chunked_assembly(mesh)
