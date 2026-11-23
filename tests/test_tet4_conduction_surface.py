"""Analytical checks for signed, explicitly selected Tet4 surface heat."""
from dataclasses import fields
from itertools import permutations

import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import (
    solve_steady_conduction_tet4 as source_free,
    solve_steady_conduction_tet4_with_surface_flux as surface,
)

MATERIAL = {7: {'young_modulus_pa': 1e9, 'poisson_ratio': .3}}


def _bundle(points, tets, unit):
    ids = tuple(2**63 + 11*i for i in range(len(points)))
    mesh = Tet4Bundle(points, tets, ids, list(range(len(tets))), [7]*len(tets), MATERIAL, unit)
    return mesh, ids


def _solve(bundle, ids, faces, held, rate, k=1.):
    return surface(bundle, conductivity_W_per_m_K_by_material={7:k},
                   prescribed_K_by_id={ids[i]:300. for i in held},
                   faces_by_node_id=[tuple(ids[i] for i in face) for face in faces],
                   inward_flux_W_per_m2=rate)


@pytest.mark.parametrize('sign', [1., -1.])
def test_prism_signed_flux_and_rate_units(sign):
    coords = [(i*3., j*1., k*2.) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    cells = [[0, p[0], p[0]+p[1], 7] for p in permutations((4, 2, 1))]
    mesh, ids = _bundle(coords, cells, 'mm')
    r = _solve(mesh, ids, [(4, 6, 7), (4, 5, 7)], range(4), sign*100., .25)
    np.testing.assert_allclose(r.temperature_K, [300.]*4+[300.+sign*1.2]*4, rtol=0, atol=1e-12)
    cap_weights = np.array([2., 1., 1., 2.]) / 3 * 1e-4
    np.testing.assert_allclose(r.surface_by_node_W, np.r_[np.zeros(4), sign*cap_weights], atol=1e-17, rtol=0)
    np.testing.assert_allclose(r.heat_into_solid_W, np.r_[-sign*cap_weights, np.zeros(4)], atol=1e-17, rtol=0)
    np.testing.assert_allclose(r.flux_W_per_m2, np.tile([-sign*100., 0., 0.], (6, 1)), atol=1e-10, rtol=0)
    assert r.total_surface_W == pytest.approx(sign*2e-4, abs=1e-17)
    assert abs(r.balance_W) < 1e-17
    assert r.surface_pairing_W_K == pytest.approx(.00024, abs=1e-16)
    assert r.dirichlet_quadratic_W_K == pytest.approx(.00024, abs=1e-16)


def test_mixed_and_all_held_reactions_separate_and_zero_field_parity():
    mesh, ids = _bundle([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], [[0,1,2,3]], 'm')
    face = [(0,1,2)]
    r = _solve(mesh, ids, face, (0,1,3), 6.)
    np.testing.assert_array_equal(r.temperature_K, [300.,300.,306.,300.])
    np.testing.assert_array_equal(r.surface_by_node_W, [1.,1.,1.,0.])
    np.testing.assert_array_equal(r.heat_into_solid_W, [-2.,-1.,0.,0.])
    assert r.balance_W == 0. and r.dirichlet_quadratic_W_K == r.surface_pairing_W_K == 6.
    held = _solve(mesh, ids, face, range(4), 6.)
    np.testing.assert_array_equal(held.heat_into_solid_W, -held.surface_by_node_W)
    assert held.total_surface_W == 3. and held.balance_W == held.dirichlet_quadratic_W_K == held.factor_entries == 0
    nothing = _solve(mesh, ids, face, (0,1,3), 0.)
    old = source_free(mesh, conductivity_W_per_m_K_by_material={7:1.},
                      prescribed_K_by_id={ids[i]:300. for i in (0,1,3)})
    for f in fields(old):
        assert np.asarray(getattr(nothing, f.name)).tobytes() == np.asarray(getattr(old, f.name)).tobytes()


def test_reject_interior_and_duplicate_even_with_zero_density():
    mesh, ids = _bundle([[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5],[0,0,-.5]],
                        [[0,1,2,3],[0,2,1,4]], 'm')
    with pytest.raises(ValueError, match='not an exterior'):
        _solve(mesh, ids, [(0,1,2)], range(5), 0.)
    with pytest.raises(ValueError, match='duplicate selected'):
        _solve(mesh, ids, [(0,1,3),(3,0,1)], range(5), 0.)


@pytest.mark.parametrize('flip', [False, True])
def test_same_sign_lumping_admits_resolved_unequal_areas_and_rejects_loss(flip):
    faces = [(0,1,2),(0,4,5)] if not flip else [(5,4,0),(2,1,0)]
    def run(a):
        mesh, ids = _bundle([[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5],[-a,0,0],[0,-a,0],[0,0,-.5]],
                            [[0,1,2,3],[0,4,5,6]], 'm')
        return _solve(mesh, ids, faces, range(7), 6.)
    admitted = run(2.**-15)
    assert admitted.surface_by_node_W[0] == .25 + 2.**-30 and admitted.balance_W == 0.
    with pytest.raises(ValueError, match='lost a nonzero face contribution'):
        run(2.**-30)
