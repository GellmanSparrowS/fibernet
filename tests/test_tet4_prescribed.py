"""Independent analytic controls for signed Tet4 displacement constraints."""
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4


def _tet(unit='m', order=(0, 1, 2, 3)):
    ids = np.array([17, 2**63 + 3, 105, 2**64 - 1], dtype=np.uint64)
    x = np.vstack([np.zeros(3), np.eye(3)])
    rows = list(order)
    b = Tet4Bundle(x[rows] * {'m': 1, 'mm': 1000}[unit], [np.argsort(rows)],
                   ids[rows], [99], [7], {7: {'young_modulus_pa': 6., 'poisson_ratio': 0.}}, unit)
    return b, list(map(int, ids)), x


D = 2.**-10


@pytest.mark.parametrize('load_multiplier', [0, 1.25])
@pytest.mark.parametrize('unit,order', [('m', (0, 1, 2, 3)), ('mm', (2, 0, 3, 1))])
def test_fully_prescribed_affine_with_external_force_on_actuated_axis(load_multiplier, unit, order, monkeypatch):
    b, ids, x = _tet(unit, order)
    def forbidden(*a, **k): raise AssertionError('fully held must not factor')
    monkeypatch.setattr(tet4, '_factor_band', forbidden)
    prescribed = {n: {'x': D*x[i, 0]} for i, n in enumerate(ids)}
    fixed = {i: ('y', 'z') for i in ids}
    loads = {ids[1]: (load_multiplier*D, 0., 0.)}
    r = tet4.solve_static_tet4(b, fixed_dofs_by_id=fixed, forces_by_id=loads,
                               prescribed_displacements_m_by_id=prescribed, return_cell_fields=True)
    np.testing.assert_array_equal(r.fixed_mask, True)
    expected = np.zeros((4, 3))
    expected[0, 0] = -D
    expected[1, 0] = (1.-load_multiplier)*D
    np.testing.assert_allclose(r.reaction_N[np.argsort(order)], expected, atol=1e-16)
    np.testing.assert_array_equal(r.displacement_m[:, 0][np.argsort(order)], D*x[:, 0])
    np.testing.assert_allclose(r.cell_fields.stress_Pa[0], np.diag([6*D, 0., 0.]), atol=1e-16)
    assert r.strain_energy_J == pytest.approx(D**2/2)
    assert r.half_external_work_J == pytest.approx(D**2/2)
    assert r.free_residual_inf_N == 0.
    if load_multiplier>1.:
        assert r.reaction_N[np.flatnonzero(r.node_ids==ids[1])[0],0] * D < 0.


def test_exactly_cancelled_effective_rhs_still_factors_and_does_work(monkeypatch):
    b, ids, _ = _tet()
    fixed = {n: ('y', 'z') if i in (0, 1) else ('x','y','z') for i,n in enumerate(ids)}
    old=tet4._factor_band
    calls=[]
    def watch(*a):
        calls.append(a[0].shape)
        factor=old(*a)
        def forbidden(rhs): raise AssertionError('exact zero RHS need not back-substitute')
        return forbidden
    monkeypatch.setattr(tet4, '_factor_band', watch)
    # K_ff=1, K_f0=-1 exactly on this x DOF, and F_f=-D.
    r=tet4.solve_static_tet4(b, fixed_dofs_by_id=fixed, forces_by_id={ids[1]:(-D,0.,0.)},
                            prescribed_displacements_m_by_id={ids[0]:{'x':D}})
    assert calls==[(1,1)]
    np.testing.assert_array_equal(r.displacement_m[:,0],[D,0.,0.,0.])
    np.testing.assert_array_equal(r.reaction_N[:,0],[2*D,0.,-D/2,-D/2])
    # The origin-only drive additionally creates xy and xz engineering shear −D.
    assert r.strain_energy_J==pytest.approx(D**2)
    assert r.half_external_work_J==pytest.approx(D**2)


def test_free_mixed_force_and_prescribed_work_without_cancellation():
    b, ids, _ = _tet()
    fixed = {n: ('y', 'z') if i in (0, 1) else ('x','y','z') for i,n in enumerate(ids)}
    F=-D/4
    r=tet4.solve_static_tet4(b, fixed_dofs_by_id=fixed, forces_by_id={ids[1]:(F,0.,0.)},
                            prescribed_displacements_m_by_id={ids[0]:{'x':D}})
    np.testing.assert_allclose(r.displacement_m[:,0],[D,D+F,0.,0.],atol=1e-16)
    np.testing.assert_allclose(r.reaction_N[:,0],[-F+D,0.,-D/2,-D/2],atol=1e-16)
    assert F*(D+F)<0.
    assert r.strain_energy_J==pytest.approx((F**2+D**2)/2)
    assert r.half_external_work_J==pytest.approx((F**2+D**2)/2)


def test_fully_prescribed_rigid_translation_exactly_zero():
    b, ids, _ = _tet()
    driven={n:{'x':D,'y':-D,'z':D/2} for n in ids}
    r=tet4.solve_static_tet4(b,fixed_dofs_by_id={},forces_by_id={ids[1]:(D,-D,0.)},
                            prescribed_displacements_m_by_id=driven)
    np.testing.assert_array_equal(r.displacement_m,np.tile([D,-D,D/2],(4,1)))
    assert r.strain_energy_J==r.half_external_work_J==0.
    np.testing.assert_array_equal(r.reaction_N[1],[-D,D,0.])


def test_zero_defaults_and_admission_do_not_assemble(monkeypatch):
    b, ids, _ = _tet()
    h={ids[0]:('x','y','z'),ids[1]:('y','z'),ids[2]:('z',)}
    solve=lambda **kw: tet4.solve_static_tet4(b,forces_by_id={ids[1]:(D,0.,0.)},return_cell_fields=True,**kw)
    ordinary=solve(fixed_dofs_by_id=h)
    zero=solve(fixed_dofs_by_id={ids[1]:('y','z'),ids[2]:('z',)},
               prescribed_displacements_m_by_id={ids[0]:dict.fromkeys('xyz',0.)})
    for field in ('displacement_m','reaction_N','fixed_mask','node_ids','strain_energy_J','half_external_work_J'):
        assert np.asarray(getattr(ordinary,field)).tobytes()==np.asarray(getattr(zero,field)).tobytes()
    assert ordinary.cell_fields.stress_Pa.tobytes()==zero.cell_fields.stress_Pa.tobytes()
    def forbidden(*a, **k): raise AssertionError('admission assembles/factors')
    monkeypatch.setattr(tet4,'_assemble',forbidden)
    monkeypatch.setattr(tet4,'_factor_band',forbidden)
    actual=tet4.inspect_static_tet4_admission(b,fixed_dofs_by_id={ids[1]:('y','z'),ids[2]:('z',)},
                prescribed_displacements_m_by_id={ids[0]:{'x':D,'y':0.,'z':0.}})
    assert actual==tet4.inspect_static_tet4_admission(b,fixed_dofs_by_id=h)


class _DuplicateAxes(Mapping):
    def __iter__(self): return iter(['x','x'])
    def __len__(self): return 2
    def __getitem__(self, key): return 0.


@pytest.mark.parametrize('case,reason', [
    ('overlap','overlapping'),('unknown','foreign'),('duplicate','duplicate'),
    ('bool','finite binary64'),('inf','finite'),('quantized','exactly representable'),
    ('subnormal','normal binary64'),('bad_axis','x, y, or z'),('free','underconstrained'),
])
def test_reject_malformed_or_insufficient_driven_constraints(case, reason):
    b, ids, _=_tet()
    h={ids[0]:('x','y','z'),ids[1]:('y','z'),ids[2]:('z',)}
    if case=='free': h={}
    driven={'overlap':{ids[0]:{'x':0.}},'unknown':{10000:{'x':D}},
            'duplicate':{ids[3]:_DuplicateAxes()},'bool':{ids[3]:{'x':True}},
            'inf':{ids[3]:{'x':float('inf')}},'quantized':{ids[3]:{'x':2**53+1}},
            'subnormal':{ids[3]:{'x':np.uint64(1).view(np.float64)}},
            'bad_axis':{ids[3]:{'Z':D}},'free':{ids[3]:{'x':D}}}[case]
    for f,extra in ((tet4.solve_static_tet4,{'forces_by_id':{}}),(tet4.inspect_static_tet4_admission,{})):
        with pytest.raises(ValueError,match=reason):
            f(b,fixed_dofs_by_id=h,prescribed_displacements_m_by_id=driven,**extra)


def test_refuses_unrepresentable_nonzero_energy():
    b, ids, x = _tet()
    h={i:('y','z') for i in ids}
    d={i:{'x':float(1e-165*x[j,0])} for j,i in enumerate(ids)}
    with pytest.raises(ValueError,match='prescribed strain energy'):
        tet4.solve_static_tet4(b, forces_by_id={}, fixed_dofs_by_id=h,
                               prescribed_displacements_m_by_id=d)


def test_large_coincident_applied_load_does_not_cancel_mechanical_work():
    b, ids, x = _tet()
    b=replace(b, points=x, length_unit='mm',
              materials={7:{'young_modulus_pa':1e9,'poisson_ratio':0.}})
    h={i:('y','z') for i in ids}
    d={i:{'x':1e-9*float(x[j,0])} for j,i in enumerate(ids)}
    # The exact physical actuator load for the otherwise uniform affine field
    # is E*(delta/L)*V/L = 1/6000 N, independent of a coincident applied load.
    controlled=[tet4.solve_static_tet4(b,forces_by_id={ids[1]:(f,0.,0.)},
        fixed_dofs_by_id=h,prescribed_displacements_m_by_id=d) for f in (0.,1e6)]
    pure, mixed=controlled
    np.testing.assert_array_equal(pure.displacement_m, mixed.displacement_m)
    assert pure.reaction_N[1,0]==pytest.approx(1./6000.)
    assert mixed.reaction_N[1,0]==pytest.approx(1./6000.-1e6,rel=0.,abs=1e-9)
    for result in controlled:
        assert result.strain_energy_J==pytest.approx(1e-9/12000,rel=1e-12,abs=0.)
        assert result.half_external_work_J==pure.half_external_work_J>0.
        assert result.half_external_work_J==pytest.approx(result.strain_energy_J,rel=1e-12,abs=0.)
