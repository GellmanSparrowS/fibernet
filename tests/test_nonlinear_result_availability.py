"""Availability guards are not nonlinear equilibrium validation."""
import ast
import json
from pathlib import Path
import time
import traceback

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim.accelerated import SimResult
import fibernet.viz.render as render


UNAVAILABLE = {
    'K': 'final_nonlinear_stiffness_not_recovered',
    'reactions': 'final_nonlinear_equilibrium_reactions_not_recovered',
    'node_moment': 'final_total_nodal_moments_not_recovered',
}
APPROXIMATION = 'original_geometry_total_displacement_linear_reference'


def bar(max_iter):
    return BeamFrameFEM(E=1000.).solve_2d_nonlinear(
        np.array([[0], [1]]), np.array([[0., 0.], [1., 0.]]),
        np.array([.05]), prescribed_disp={1: (.1, 0.)}, fixed_nodes=[0],
        n_steps=1, max_iter=max_iter)


def sparse_graph():
    graph = fn.StructureGraph(dimension=2)
    graph.add_node([0, 0, 0]); graph.add_node([1, 0, 0])
    graph.add_edge(0, 1, radius=.05)
    data = graph.to_dict()
    for node in data['nodes']:
        node['id'] = {0: 7, 1: 42}[node['id']]
    data['nodes'].reverse()
    data['edges'][0].update(node_i=7, node_j=42, id=11)
    data.pop('next_node_id'); data.pop('next_edge_id')
    return fn.StructureGraph.from_dict(data)


@pytest.mark.parametrize('max_iter', [1, 2])
def test_raw_bar_fields_never_masquerade_as_final_state(max_iter):
    result = bar(max_iter)
    for field, reason in UNAVAILABLE.items():
        assert result[field] is None
        assert result['field_status'][field] == {
            'status': 'not_supported', 'reason': reason,
            'model': 'incremental_updated_geometry_linear_solves'}
    np.testing.assert_allclose(result['u'][:, 0], [0, .1])
    # A linear-reference check, not a nonlinear equilibrium benchmark.
    np.testing.assert_allclose(result['edge_forces'][0, 0], 1000*np.pi*.05**2*.1)
    for field in ('sigma_axial', 'sigma_bending', 'sigma_total',
                  'moments', 'edge_forces', 'node_stress'):
        assert result['field_status'][field]['status'] == 'approximate'
        assert result['field_status'][field]['approximation'] == APPROXIMATION
    status = result['solver_status']
    assert status['equilibrium'] == 'unverified'
    assert status['update_criterion_is_equilibrium_check'] is False
    assert status['max_iter'] == max_iter
    assert status['history_field_scope']['max_stress'] == 'last_inner_linear_increment'
    step = status['steps'][0]
    assert step['iterations'] == max_iter
    assert step['iteration_limit_reached'] is (max_iter == 1)
    assert step['update_criterion_met'] is (max_iter == 2)
    assert 'converged' not in json.dumps(status)


@pytest.mark.parametrize('detailed', [False, True])
def test_simulate_json_status_and_identity_then_render(tmp_path, monkeypatch, detailed):
    graph = sparse_graph()
    result = fn.simulate(graph, backend='fem', strain=1.1, nonlinear=True, E=1000.)
    path = tmp_path / 'result.json'
    result.save(path, detailed=detailed)
    saved = json.loads(path.read_text())
    restored = SimResult.load(path)
    for current in (result, restored):
        for field, reason in UNAVAILABLE.items():
            assert current.metadata[field] is None
            assert current.metadata['field_status'][field]['reason'] == reason
            assert saved[field] is None
        assert current.metadata['energy_kind'] == 'axial_stress_proxy'
        assert current.metadata['solver_status']['equilibrium'] == 'unverified'
        assert current.metadata['field_status']['beam_resultants']['approximation'] == APPROXIMATION
        assert current.metadata['node_ids'] == [42, 7]
        assert current.metadata['edge_ids'] == [11]
    assert restored.metadata['field_status'] == result.metadata['field_status']
    assert restored.metadata['solver_status'] == result.metadata['solver_status']
    # Summary intentionally omits displacement arrays; only full reload can render.
    if detailed:
        captured = []
        monkeypatch.setattr(render, '_render_3d_segments', lambda ax, g, *a: captured.append(g.copy()))
        monkeypatch.setattr(render, '_render_3d_colored', lambda ax, g, *a: captured.append(g.copy()))
        render.render_deformation_3d(graph, restored)
        np.testing.assert_allclose(captured[-1].nodes[7].position, [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(captured[-1].nodes[42].position, [1.1, 0, 0], atol=1e-12)
        aligned = render._align_result(graph, restored, require_edges=True)
        assert aligned.metadata['reactions'] is None
        assert aligned.metadata['field_status'] == restored.metadata['field_status']
        plt.close('all')


def test_linear_and_legacy_not_retroactively_certified(tmp_path):
    graph = sparse_graph()
    result = fn.simulate(graph, backend='fem', strain=1.1, nonlinear=False, E=1000.)
    np.testing.assert_allclose(result.metadata['reactions'][0, 0], 1000*np.pi*.05**2*.1)
    assert 'solver_status' not in result.metadata
    assert 'field_status' not in result.metadata
    # Historical zeros are retained as data of unknown validity, not reinterpreted.
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps({'mode': 'fem_beam_frame', 'reactions': [[0, 0, 0]]}))
    restored = SimResult.load(path)
    assert restored.metadata['reactions'] == [[0, 0, 0]]
    assert 'solver_status' not in restored.metadata
    assert 'field_status' not in restored.metadata
    raw = bar(1)
    raw.pop('field_status'); raw.pop('solver_status')
    adapted = BeamFrameFEM().to_sim_result(raw)
    assert 'solver_status' not in adapted.metadata
    assert 'field_status' not in adapted.metadata


def test_research_exporter_real_nonlinear_null_path():
    # Compile the actual function without executing module-level output-directory
    # creation or its batch runner. The solver and export function are real.
    path = Path(__file__).parents[1] / 'scripts/deformation_test/run_large_deformation_fem.py'
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'run_fem_2d')
    namespace = dict(np=np, time=time, traceback=traceback, E_MODULUS=1000.,
                     NU=.3, PCT_BOUNDARY=.1, MAX_NODES=15000, NONLINEAR_THRESHOLD=.3)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    exported = namespace['run_fem_2d'](sparse_graph(), 1.5, .05)
    assert exported is not None and exported['use_nonlinear'] is True
    assert json.loads(json.dumps(exported))['reactions'] is None
    # Known out-of-scope exporter gap: null works, explanations are not copied.
    assert 'field_status' not in exported
    assert 'solver_status' not in exported


@pytest.mark.parametrize('name', ['n_steps', 'max_iter'])
@pytest.mark.parametrize('value', [0, -1, 1.5, 2., True, np.bool_(False), '2', None])
def test_invalid_iteration_controls_fail_at_entry(name, value, monkeypatch):
    solver = BeamFrameFEM()
    monkeypatch.setattr(solver, 'solve_2d', lambda *a, **k: pytest.fail('inner solve ran'))
    with pytest.raises(ValueError, match=name):
        solver.solve_2d_nonlinear(None, None, None, {}, **{name: value})


@pytest.mark.parametrize('tol', [-1., np.nan, np.inf, -np.inf, True, np.bool_(False), '1e-6', None])
def test_invalid_tolerance_fails_at_entry(tol):
    with pytest.raises(ValueError, match='tol'):
        BeamFrameFEM().solve_2d_nonlinear(None, None, None, {}, tol=tol)


def test_zero_tolerance_and_numpy_integer_controls():
    result = BeamFrameFEM(E=1000.).solve_2d_nonlinear(
        np.array([[0], [1]]), np.array([[0., 0.], [1., 0.]]), np.array([.05]),
        prescribed_disp={1: (.1, 0.)}, fixed_nodes=[0],
        n_steps=np.int64(1), max_iter=np.int64(2), tol=0.)
    step = result['solver_status']['steps'][0]
    assert step['update_norm'] == step['update_threshold'] == 0.
    assert step['update_criterion_met'] is False
    assert step['iteration_limit_reached'] is True
    assert result['solver_status']['equilibrium'] == 'unverified'
