"""Analytical/software checks; explicit conductance is not inferred physics."""
import json
from pathlib import Path

import numpy as np
import pytest

from fibernet import StructureGraph
from fibernet.sim.linear_transport import (FloatingComponentError, TransportSolveError,
                                           solve_linear_transport)


def network(endpoints, node_ids=None):
    ids = sorted({i for pair in endpoints for i in pair}) if node_ids is None else node_ids
    payload = {'dimension': 3,
               'nodes': [{'id': nid, 'position': [float(k), float(k % 2), 0.]} for k, nid in enumerate(ids)],
               'edges': [{'id': eid, 'node_i': a, 'node_j': b} for eid, (a, b) in enumerate(endpoints)]}
    return StructureGraph.from_dict(payload)


def solve(g, conductance, bc, **kw):
    return solve_linear_transport(g, conductance, bc, length_unit='m', **kw)


def test_saved_analytic_fixture():
    path = Path(__file__).parents[1] / 'docs/integrations/transport_pilot_fixture.json'
    f = json.loads(path.read_text())
    payload = {'dimension': 3, 'nodes': [{'id': nid, 'position': xyz}
                for nid, xyz in zip(f['node_ids'], f['coordinates_m'])],
               'edges': [{'id': eid, 'node_i': a, 'node_j': b}
                for eid, (a, b) in zip(f['edge_ids'], f['endpoints_by_id'])]}
    g = StructureGraph.from_dict(payload)
    # from_dict inserts in payload order; test explicitly retains it.
    result = solve(g, dict(zip(f['edge_ids'], f['conductance_S'])), [(10, 1), (120, 0)])
    np.testing.assert_array_equal(result.node_ids, f['node_ids'])
    np.testing.assert_array_equal(result.edge_ids, f['edge_ids'])
    np.testing.assert_allclose(result.voltage_V, f['expected_voltage_V'], atol=1e-12)
    np.testing.assert_allclose(result.current_A, f['expected_oriented_current_A'], atol=1e-12)
    np.testing.assert_allclose(result.node_injection_A, [3, 0, 0, -3], atol=1e-12)
    assert result.dissipated_power_W == pytest.approx(3)
    assert result.boundary_power_W == pytest.approx(3)
    assert result.free_node_residual_A < 1e-12


def test_series():
    g = network([(0, 1), (1, 2)])
    r = solve(g, {0: 2, 1: 6}, [(0, 2), (2, 0)])
    np.testing.assert_allclose(r.voltage_V, [2, .5, 0])
    np.testing.assert_allclose(r.current_A, [3, 3])
    assert r.dissipated_power_W == pytest.approx(6)


def test_parallel_members_sum_and_orientation():
    g = network([(10, 30), (30, 10), (10, 30)])
    r = solve(g, {0: 2, 1: 3, 2: 0}, [(10, 1), (30, 0)])
    np.testing.assert_allclose(r.current_A, [2, -3, 0])
    np.testing.assert_allclose(r.edge_power_W, [2, 3, 0])
    np.testing.assert_allclose(r.node_injection_A, [5, -5])
    assert r.dissipated_power_W == pytest.approx(5)


def test_zero_conductance_no_path_is_valid_with_all_components_anchored():
    g = network([(0, 1)])
    r = solve(g, {0: 0}, [(0, 4), (1, -2)])
    np.testing.assert_array_equal(r.voltage_V, [4, -2])
    assert r.current_A[0] == 0 and r.edge_power_W[0] == 0
    assert r.dissipated_power_W == 0


@pytest.mark.parametrize('edges,ids,gs,bc', [
    ([(0, 1)], [0, 1], {0: 0}, [(0, 1)]),
    ([(0, 1), (2, 3)], [0, 1, 2, 3], {0: 1, 1: 1}, [(0, 1)]),
    ([(0, 1)], [0, 1, 7], {0: 1}, [(0, 1)]),
    ([(0, 1)], [0, 1], {0: 1}, []),
])
def test_floating_components_rejected(edges, ids, gs, bc):
    with pytest.raises(FloatingComponentError, match='lack a voltage anchor'):
        solve(network(edges, ids), gs, bc)


def test_separate_anchored_components_and_isolated_node():
    r = solve(network([(0, 1), (2, 3)], [0, 1, 2, 3, 9]),
              {0: 1, 1: 2}, [(0, 2), (2, -1), (9, 7)])
    np.testing.assert_array_equal(r.voltage_V, [2, 2, -1, -1, 7])
    np.testing.assert_array_equal(r.current_A, [0, 0])


def test_singleton_without_edges():
    r = solve(network([], [20]), {}, [(20, 3)])
    assert r.current_A.shape == (0,)
    assert r.endpoints_by_id.shape == (0, 2)
    assert r.voltage_V[0] == 3
    assert r.dissipated_power_W == 0


@pytest.mark.parametrize('value', [-1, np.nan, np.inf, True, '2', 1+2j])
def test_invalid_conductance(value):
    with pytest.raises(ValueError): solve(network([(0, 1)]), {0: value}, [(0, 1)])


@pytest.mark.parametrize('values', [{}, {0: 1, 7: 1}, {True: 1}, {0.0: 1}])
def test_conductance_keys(values):
    with pytest.raises(ValueError): solve(network([(0, 1)]), values, [(0, 1)])


@pytest.mark.parametrize('bc', [[(0, 1), (0, 2)], [(0, np.inf)], [(0, True)],
                               [(True, 1)], [(0.0, 1)], [(99, 0)], None, [2]])
def test_invalid_dirichlet(bc):
    with pytest.raises(ValueError): solve(network([(0, 1)]), {0: 1}, bc)


def test_identical_boundary_duplicates_coalesce():
    r = solve(network([(0, 1)]), {0: 1}, [(0, 1), (0, 1)])
    assert r.prescribed_voltage_V == {0: 1.}
    np.testing.assert_array_equal(r.voltage_V, [1, 1])


def test_permutation_reruns_voltage_scaling_and_nonmutation():
    g = network([(10, 20), (20, 30), (10, 30)])
    before = json.dumps(g.to_dict(), sort_keys=True)
    conductance = {0: 1., 1: 2., 2: 3.}
    meta = {'label': {'values': [1, 2]}}
    a = solve(g, conductance, [(10, 1), (30, 0)], model_metadata=meta)
    b = solve(g, conductance, [(10, 2), (30, 0)])
    c = solve(g, conductance, [(10, 1), (30, 0)])
    np.testing.assert_allclose(b.current_A, a.current_A*2)
    assert b.dissipated_power_W == pytest.approx(a.dissipated_power_W*4)
    np.testing.assert_array_equal(a.current_A, c.current_A)
    assert a.provenance['input_sha256'] == c.provenance['input_sha256']
    assert json.dumps(g.to_dict(), sort_keys=True) == before
    meta['label']['values'][0] = 77
    assert a.provenance['model_metadata']['label']['values'][0] == 1
    with pytest.raises(ValueError): a.voltage_V[0] = 9
    # Storage insertion order changes, physical results retain their stable IDs.
    edge = g.edges.pop(0); g.edges[0] = edge
    d = solve(g, conductance, [(10, 1), (30, 0)])
    assert dict(zip(d.edge_ids, d.current_A)) == pytest.approx(dict(zip(a.edge_ids, a.current_A)))
    encoded = a.to_dict(); encoded['provenance']['model_metadata']['label']['values'][0] = 999
    assert a.provenance['model_metadata']['label']['values'][0] == 1
    json.dumps(encoded, allow_nan=False)


def test_solver_nonfinite_and_conservation_failures(monkeypatch):
    import fibernet.sim.linear_transport as module
    g = network([(0, 1), (1, 2)])
    monkeypatch.setattr(module, 'spsolve', lambda *a, **kw: np.array([np.nan]))
    with pytest.raises(TransportSolveError, match='nonfinite'):
        solve(g, {0: 1, 1: 1}, [(0, 1), (2, 0)])
    monkeypatch.setattr(module, 'spsolve', lambda *a, **kw: np.array([.75]))
    with pytest.raises(TransportSolveError, match='conservation'):
        solve(g, {0: 1, 1: 1}, [(0, 1), (2, 0)])


def test_canonical_square_network_e2e():
    from fibernet import pattern_2d
    g = pattern_2d(unit='square', grid=(2, 2), box=(2., 2.))
    x = {i: float(n.position[0]) for i, n in g.nodes.items()}
    lo, hi = min(x.values()), max(x.values())
    bc = [(i, 1.) for i, v in x.items() if np.isclose(v, lo)]
    bc += [(i, 0.) for i, v in x.items() if np.isclose(v, hi)]
    r = solve(g, {i: 1. for i in g.edges}, bc)
    expected = [(hi-x[i])/(hi-lo) for i in r.node_ids]
    np.testing.assert_allclose(r.voltage_V, expected, atol=1e-10)
    assert abs(r.global_current_balance_A) < 1e-10
    assert r.dissipated_power_W == pytest.approx(r.boundary_power_W)


def test_sparse_rank_warning_is_failure(monkeypatch):
    import warnings
    import fibernet.sim.linear_transport as module
    def bad_solver(*args, **kwargs):
        warnings.warn('synthetic singular matrix', module.MatrixRankWarning)
        return np.array([0.])
    monkeypatch.setattr(module, 'spsolve', bad_solver)
    with pytest.raises(TransportSolveError, match='sparse transport solve failed'):
        solve(network([(0, 1), (1, 2)]), {0: 1, 1: 1}, [(0, 1), (2, 0)])


@pytest.mark.parametrize('kw', [{'rtol': -1}, {'current_atol_A': float('nan')},
                               {'power_atol_W': True}, {'model_metadata': {1: 'bad'}},
                               {'model_metadata': {'bad': float('inf')}}])
def test_invalid_controls(kw):
    with pytest.raises(ValueError):
        solve(network([(0, 1)]), {0: 1}, [(0, 1)], **kw)


def test_geometry_unit_has_no_electrical_rescaling():
    g = network([(0, 1), (1, 2)])
    a = solve_linear_transport(g, {0: 1, 1: 2}, [(0, 1), (2, 0)], length_unit='m')
    for node in g.nodes.values(): node.position *= 1000
    b = solve_linear_transport(g, {0: 1, 1: 2}, [(0, 1), (2, 0)], length_unit='mm')
    np.testing.assert_array_equal(a.voltage_V, b.voltage_V)
    np.testing.assert_array_equal(a.current_A, b.current_A)
    assert b.provenance['length_unit'] == 'mm'
    assert b.provenance['geometry_conversion'] == 'none'
    assert b.provenance['input_sha256'] != a.provenance['input_sha256']


def test_zero_edge_does_not_short_circuit_components():
    g = network([(0, 1), (1, 2), (2, 3)])
    r = solve(g, {0: 1, 1: 0, 2: 2}, [(0, 1), (3, 0)])
    np.testing.assert_array_equal(r.voltage_V, [1, 1, 0, 0])
    np.testing.assert_array_equal(r.current_A, [0, 0, 0])


def test_assembly_overflow_is_explicit_failure():
    g = network([(0, 1), (0, 1)])
    with pytest.raises(TransportSolveError, match='assembly overflowed'):
        solve(g, {0: 1e308, 1: 1e308}, [(0, 1), (1, 0)])


@pytest.mark.parametrize('connected', [False, True])
def test_large_branch_cannot_mask_wrong_small_node(monkeypatch, connected):
    import fibernet.sim.linear_transport as module
    edges = [(0, 1), (2, 3), (3, 4)] if not connected else [(0, 1), (0, 3), (3, 4)]
    bc = [(0, 1), (1, 0), (4, 0)] + ([] if connected else [(2, 1)])
    monkeypatch.setattr(module, 'spsolve', lambda *a, **kw: np.array([.75]))
    with pytest.raises(TransportSolveError, match='node-local'):
        solve(network(edges), {0: 1e20, 1: 1, 2: 1}, bc)


@pytest.mark.parametrize('scale', [1e-20, 1e-150, 1e-200])
def test_default_controls_reject_wrong_tiny_solution(monkeypatch, scale):
    import fibernet.sim.linear_transport as module
    monkeypatch.setattr(module, 'spsolve', lambda *a, **kw: np.array([.75]))
    with pytest.raises(TransportSolveError, match='node-local'):
        solve(network([(0, 1), (1, 2)]), {0: scale, 1: scale}, [(0, 1), (2, 0)])


@pytest.mark.parametrize('g,drop,error', [(1e-300, 1e-100, 'current underflow'),
                                        (1e-200, 1e-100, 'power underflow'),
                                        (1e-310, 1., 'current underflow')])
def test_nonrepresentable_fields_rejected(g, drop, error):
    import struct
    tiny = struct.unpack('>d', bytes.fromhex('0000000000000001'))[0]
    magnitude = struct.unpack('>Q', struct.pack('>d', g))[0] & 0x7fffffffffffffff
    if 0 < magnitude < 0x0010000000000000 and not (tiny > 0. and tiny + tiny > 0.):
        with pytest.raises(ValueError, match='^conductance: nonzero subnormal input requires gradual underflow$'):
            solve(network([(0, 1)]), {0: g}, [(0, drop), (1, 0)])
        return
    with pytest.raises(TransportSolveError, match=error):
        solve(network([(0, 1)]), {0: g}, [(0, drop), (1, 0)])


@pytest.mark.parametrize('scale', [1e-200, 1e-20, 1, 1e200])
def test_clean_uniform_scales_default_controls(scale):
    r = solve(network([(0, 1), (1, 2)]), {0: scale, 1: scale}, [(0, 1), (2, 0)])
    np.testing.assert_allclose(r.voltage_V, [1, .5, 0], atol=0, rtol=1e-12)
    np.testing.assert_allclose(r.current_A, [.5*scale]*2, atol=0, rtol=1e-12)


@pytest.mark.parametrize('bc', [[(0, 3.14)], [(0, 3.14), (3, 3.14)]])
def test_heterogeneous_equipotential_is_exact(bc):
    r = solve(network([(0, 1), (1, 2), (2, 3)]),
              {0: 1e-100, 1: 7., 2: 1e100}, bc)
    np.testing.assert_array_equal(r.voltage_V, [3.14]*4)
    np.testing.assert_array_equal(r.current_A, [0]*3)
    assert r.dissipated_power_W == 0


@pytest.mark.parametrize('offset', [0., 1e10, 1e14])
def test_boundary_power_uses_stable_gauge_shift(offset):
    r = solve(network([(0, 1), (1, 2)]), {0: 1.3, 1: 1.3},
              [(0, offset+1), (2, offset)])
    assert r.dissipated_power_W == pytest.approx(.65, rel=1e-12)
    assert r.boundary_power_W == pytest.approx(.65, rel=1e-12)
    assert r.boundary_power_W == pytest.approx(r.dissipated_power_W, rel=1e-12)
    assert 'reference-shifted' in r.provenance['boundary_power_convention']
