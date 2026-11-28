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


@pytest.mark.parametrize(("options", "mode", "refinement"), [
    ([], "finite", 3),
    (["--refinement", "4"], "finite", 4),
    (["--initial-slope", "--refinement", "5"], "tangent", 5),
])
def test_mechanoelectric_sweep_cli_selects_refinement(monkeypatch, capsys, options, mode, refinement):
    import json
    from fibernet.examples import tet4_iterative_mechanoelectric_sweep as example

    monkeypatch.setattr(example, "run", lambda *, refinement: {"finite": refinement})
    monkeypatch.setattr(example, "run_initial_slope", lambda *, refinement: {"tangent": refinement})
    example.main(options)
    assert json.loads(capsys.readouterr().out) == {mode: refinement}


def test_mechanoelectric_sweep_rejects_mesh_above_demo_budget(monkeypatch, capsys):
    from fibernet.examples import tet4_iterative_mechanoelectric_sweep as example

    def no_solve(**kwargs):
        pytest.fail("invalid refinement reached a solver")

    monkeypatch.setattr(example, "run", no_solve)
    with pytest.raises(SystemExit) as exc:
        example.main(["--refinement", "6"])
    assert exc.value.code == 2
    out = capsys.readouterr()
    assert not out.out and "invalid choice" in out.err
    with pytest.raises(ValueError, match="between 1 and 5"):
        _lattice(refinement=6)


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


@pytest.mark.parametrize('order',[slice(None),slice(None,None,-1)])
def test_packed_face_forest_matches_reference_rigid_check(monkeypatch,order):
    from fibernet.sim import _tet4_amg, tet4
    points,cells=_lattice()
    cells=cells[order,order]
    b=_bundle(points,cells,'uniform')
    right,ids,fixed,_,_=_fixture(b)
    # Regular meshes must not visit the slower validation even for mixed
    # element orientation. Compare against the unchanged reference check.
    def unexpected(_):
        pytest.fail("valid packed topology visited reference validation")
    monkeypatch.setattr(tet4,'_validate_topology',unexpected)
    _,constraints,_=tet4._loads_and_constraints(b,{},fixed,{n:{'x':1.5e-6} for n in ids})
    parents=_tet4_amg.topology_parents(cells)
    assert tet4._check_rigid_modes(b,constraints,parent=parents) == tet4._check_rigid_modes(b,constraints) == 1


@pytest.mark.parametrize('rows,error', [
    ([[0,1,2,3],[3,2,1,4],[1,2,3,0]],'duplicate Tet4 connectivity at row 2'),
    ([[0,1,2,3],[0,1,2,4],[0,1,2,5],[3,2,1,0]],'non-manifold Tet4 face with more than two cells at row 2'),
    ([[59998,59997,59996,59995],[59995,59998,59997,59996]],'duplicate Tet4 connectivity at row 1'),
])
def test_packed_face_validation_keeps_first_reference_error(rows,error):
    from fibernet.sim import _tet4_amg, tet4
    cells=np.asarray(rows,dtype=np.int64)
    with pytest.raises(ValueError) as expected:
        tet4._validate_topology(cells)
    with pytest.raises(ValueError) as actual:
        _tet4_amg.topology_parents(cells)
    assert str(expected.value) == str(actual.value) == error


def test_packed_face_forest_keeps_high_point_bits():
    from fibernet.sim import _tet4_amg
    # Low and high uint16 halves have identical low 15 bits, yet the
    # two families cannot share a face or a tetrahedron.
    cells=np.array([[1,2,3,4],[3,2,1,5],[32769,32770,32771,32772],
                    [32773,32771,32770,32769]],dtype=np.int64)
    assert _tet4_amg.topology_parents(cells) == [0,0,2,2]


def test_initial_geometric_slope_requires_only_reference_dc(monkeypatch):
    pytest.importorskip('pyamg')
    from fibernet.examples import tet4_iterative_mechanoelectric_sweep as example
    curve=example.run(refinement=1)["states"]
    states=[]
    solve=example.solve_dc_tet4
    def counted(mesh, **kwargs):
        states.append(mesh.points.copy())
        return solve(mesh, **kwargs)
    monkeypatch.setattr(example,"solve_dc_tet4",counted)
    initial=example.run_initial_slope(refinement=1)
    assert len(states)==1
    np.testing.assert_array_equal(states[0],_lattice(refinement=1)[0])
    r0=curve[2]["resistance_ohm"]
    np.testing.assert_allclose(initial["resistance_at_zero_ohm"],r0,rtol=1e-10)
    slope=initial["relative_resistance_slope_per_um"]
    assert slope>0
    for minus, plus in [(1,3),(0,4)]:
        hi,lo=curve[plus],curve[minus]
        secant=((hi["resistance_ohm"]-lo["resistance_ohm"]) /
                (r0*(hi["stroke_m"]-lo["stroke_m"])*1e6))
        np.testing.assert_allclose(slope,secant,rtol=5e-5)
    terms=(initial["current_aligned_stretch_per_um"],initial["transverse_contraction_per_um"])
    np.testing.assert_allclose(sum(terms),slope,rtol=1e-13)


def test_current_direction_uses_scaled_norm_with_signed_analytic_strain():
    from examples.tet4_iterative_mechanoelectric_sweep import _power_weighted_strain
    from itertools import permutations
    x=1e-3*np.vstack([np.zeros(3),np.eye(3)])
    rows=np.asarray(list(permutations(range(4))))
    n=np.array([1.,3.,-2.])/np.sqrt(14.)
    j=np.tile([1e-220,1e220,1.,0.],6)[:,None]*n[None,:]
    w=np.where(np.any(j!=0.,axis=1),1.,0.)
    G=1e-3*np.array([[7.,1.,-5.],[2.,-2.,.5],[4.,-1.1,-.8]])
    u=x@G+np.array([1.,-4.,8.])*1e-5
    across=n@G@n-np.trace(G)
    terms=(w.sum()*(n@G@n),w.sum()*across)
    np.testing.assert_allclose(_power_weighted_strain(x,rows,u,j,w),terms,atol=1e-15)
    w[-1]=1
    with pytest.raises(ValueError,match="Zero-current"):
        _power_weighted_strain(x,rows,u,j,w)


@pytest.mark.parametrize("solver", ("banded", "amg-cg"))
def test_prescribed_releases_the_factor_before_element_recovery(monkeypatch, solver):
    from gc import collect
    from weakref import ref
    from fibernet.sim import tet4

    def observe_factor(prepare):
        def factor(*args):
            result = prepare(*args)
            factors.append(ref(result))
            return result
        return factor

    if solver == "amg-cg":
        pytest.importorskip("pyamg")
        from fibernet.sim import _tet4_amg
        module = _tet4_amg
        native_factory = module.factor_constructor

        def provider_factory(*args):
            return observe_factor(native_factory(*args))

        monkeypatch.setattr(module, "factor_constructor", provider_factory)
        recovery_name = "batched_response"
    else:
        module = tet4
        monkeypatch.setattr(module, "_factor_band", observe_factor(module._factor_band))
        from fibernet.sim import _tet4_recovery
        module = _tet4_recovery
        recovery_name = "guarded_banded_response"

    native_recover = getattr(module, recovery_name)

    def observe_recovery(*args):
        collect()
        assert len(factors) == 1 and factors[0]() is None
        touched.append(True)
        return native_recover(*args)

    factors = []
    touched = []
    monkeypatch.setattr(module, recovery_name, observe_recovery)
    points, cells = _lattice()
    bundle = _bundle(points, cells, "stiff_half")
    right, grip, fixed, _, _ = _fixture(bundle)
    r = solve_static_tet4(bundle, forces_by_id={}, fixed_dofs_by_id=fixed, solver=solver,
                          prescribed_displacements_m_by_id={n: {"x": 1.5e-6} for n in grip})
    assert touched == [True] and r.reaction_N[right, 0].sum() > 0
    np.testing.assert_allclose(r.strain_energy_J, r.half_external_work_J, rtol=1e-9)
