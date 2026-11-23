"""Analytical and admission tests for signed prescribed cellwise sources."""
from itertools import combinations
import math
import numpy as np
import pytest
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4_conduction as conduction

MATERIAL = 2**63 + 7


def _star(two=False):
    x = np.r_[np.zeros((1, 3)), np.eye(3) * 3, [[.75] * 3]]
    t = np.array([(*c, 4) for c in combinations(range(4), 3)])
    ids = np.array([19, 21, 2**53 + 3, 113, 7], dtype=np.uint64)
    cells = np.array([41, 63, 8, 2**53 + 7], dtype=np.uint64)
    if two:
        x = np.vstack((x, x + [0, 8, 0]))
        t = np.vstack((t, t + 5))
        ids = np.r_[ids, ids + 10000]
        cells = np.r_[cells, cells + 10000]
    return Tet4Bundle(x, t, ids, cells, np.full(len(t), MATERIAL, dtype=np.uint64),
                      {MATERIAL: {'young_modulus_pa': 1e8, 'poisson_ratio': .2}}, 'mm')


def _solve(b, source, boundary):
    return conduction.solve_steady_conduction_tet4_with_source(
        b, conductivity_W_per_m_K_by_material={MATERIAL: .25},
        prescribed_K_by_id=boundary, source_W_per_m3_by_cell=source)


def test_star_signed_cell_generation_and_energy():
    b = _star()
    bc = dict.fromkeys(map(int, b.node_ids[:4]), 300.)
    r = _solve(b, dict.fromkeys(map(int, b.cell_ids), 8e6), bc)
    # Parent V=4.5e-9 m³, split equally; nodal center source G/4.
    # k_44=.003 W/K from the affine gradient of the central hat on its
    # four sub-tets, giving (G/4)/k_44=3 K rise above the held corners.
    np.testing.assert_allclose(r.temperature_K, [300]*4 + [303], rtol=0, atol=1e-13)
    np.testing.assert_allclose(r.integrated_source_W_by_cell, .009, rtol=1e-13)
    np.testing.assert_allclose(r.source_by_node_W, [.00675]*4 + [.009], rtol=1e-13)
    np.testing.assert_allclose(r.heat_into_solid_W, [-.009]*4 + [0], atol=1e-17)
    np.testing.assert_allclose([r.total_source_W, r.dirichlet_quadratic_W_K,
                                r.source_pairing_W_K], [.036, .027, .027], rtol=1e-13)
    assert r.terminal_pairing_W_K == 0.
    assert abs(r.balance_W) < 1e-16 and abs(r.quadratic_closure_W_K) < 1e-16
    np.testing.assert_array_equal(r.node_ids, b.node_ids)
    np.testing.assert_array_equal(r.cell_ids, b.cell_ids)
    for a in (r.source_by_node_W, r.integrated_source_W_by_cell):
        with pytest.raises(ValueError):
            a.setflags(write=True)


def test_explicit_zero_retains_source_free_result_bitwise():
    b = _star()
    bc = dict(zip(map(int, b.node_ids[:4]), [300., 310., 320., 340.]))
    old = conduction.solve_steady_conduction_tet4(
        b, conductivity_W_per_m_K_by_material={MATERIAL: .25}, prescribed_K_by_id=bc)
    new = _solve(b, dict.fromkeys(map(int, b.cell_ids), -0.), bc)
    assert type(old) is conduction.Tet4ConductionResult
    assert type(new) is conduction.Tet4SourceConductionResult
    for k, v in vars(old).items():
        right = getattr(new, k)
        assert np.asarray(v).tobytes() == np.asarray(right).tobytes(), k


def test_opposing_sources_balance_separately_on_disconnected_components():
    b = _star(two=True)
    bc = dict.fromkeys(map(int, b.node_ids[:4]), 300.)
    bc.update(dict.fromkeys(map(int, b.node_ids[5:9]), 1500.))
    q = dict(zip(map(int, b.cell_ids), [8e6]*4 + [-8e6]*4))
    r = _solve(b, q, bc)
    np.testing.assert_allclose(r.temperature_K, [300]*4+[303]+[1500]*4+[1497], atol=1e-13, rtol=0)
    np.testing.assert_array_equal(r.reference_K_by_node, [300]*5 + [1500]*5)
    for row, cells, sign in [(slice(0, 5), slice(0, 4), 1), (slice(5, 10), slice(4, 8), -1)]:
        heat = math.fsum(r.heat_into_solid_W[row])
        generation = math.fsum(r.integrated_source_W_by_cell[cells])
        np.testing.assert_allclose([heat, generation], [-.036*sign, .036*sign], atol=1e-16)
        assert abs(heat + generation) < 1e-16
    assert abs(r.total_source_W) < 1e-16 and abs(r.balance_W) < 1e-16
    np.testing.assert_allclose(r.source_pairing_W_K, .054, rtol=1e-13)


@pytest.mark.parametrize('mode,why', [('missing','missing source'), ('foreign','source density'),
                                      ('bool','stable ID'), ('small','below'), ('nonfinite','source density')])
def test_source_map_requires_valid_stable_ids_and_bounded_values(mode, why):
    b = _star()
    q = dict.fromkeys(map(int, b.cell_ids), 0.)
    k = next(iter(q))
    if mode in ('missing', 'foreign', 'bool'):
        del q[k]
    if mode == 'foreign':
        q[2**64 - 1] = 1.
    if mode == 'bool':
        q[True] = 1.
    if mode == 'small':
        q[k] = 1e-12
    if mode == 'nonfinite':
        q[k] = float('inf')
    with pytest.raises(ValueError, match=why):
        _solve(b, q, dict.fromkeys(map(int, b.node_ids[:4]), 300.))


def test_unresolved_represented_rhs_operands_are_not_erased():
    combine = conduction._combine_rhs
    with pytest.raises(ValueError, match='lost a nonzero operand'):
        combine(np.array([1.]), np.array([2.**-54]))
    with pytest.raises(ValueError, match='near-opposing'):
        combine(np.array([1.]), -np.array([np.nextafter(1., 0.)]))
    np.testing.assert_array_equal(combine(np.array([.5, -.25, 0.]), np.array([-.5, .25, 2.])),
                                  [0., 0., 2.])
