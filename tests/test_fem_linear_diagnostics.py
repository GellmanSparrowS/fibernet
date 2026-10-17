"""Observational FEM diagnostics preserve physical outputs and provenance."""
import copy
import json

import numpy as np
import pytest

from fibernet import StructureGraph
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim.accelerated import SimResult


COMPONENTS = {2: ['ux', 'uy', 'rz'], 3: ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']}


def inputs(dim, parallel=False):
    return (np.array([[0, 1], [1, 0]]) if parallel else np.array([[0], [1]]),
            np.array([[0., 0., 0.], [1., 0., 0.]])[:, :dim],
            np.array([.05, .1] if parallel else [.05]))


def solve(dim, parallel=False, **kwargs):
    return getattr(BeamFrameFEM(E=1000.), f'solve_{dim}d')(*inputs(dim, parallel), **kwargs)


def assert_diagnostics(raw, dim, free, damping=1e-6):
    diagnostic = raw['numerical_diagnostics']
    assert diagnostic['schema_version'] == 1
    assert diagnostic['model'] == 'linear_beam_frame'
    assert diagnostic['regularization'] == {
        'status': 'available', 'method': 'diagonal_identity', 'coefficient': damping,
        'interpretation': 'numerical_not_physical_support'}
    for claim in ('physical_equilibrium', 'load_compatibility', 'solution_uniqueness'):
        assert diagnostic[claim] == 'not_evaluated'
    residual = diagnostic['physical_free_residual']
    assert residual['status'] == 'available'
    assert residual['definition'] == 'K_physical_u_minus_applied_load'
    assert residual['index_kind'] == 'input_node_row'
    ndof = len(COMPONENTS[dim])
    assert residual['node_rows'] == [int(i // ndof) for i in free]
    assert residual['dof_components'] == [COMPONENTS[dim][i % ndof] for i in free]
    assert residual['values'] == raw['reactions'].ravel()[free].tolist()
    assert len(set(zip(residual['node_rows'], residual['dof_components']))) == len(free)
    json.dumps(diagnostic, allow_nan=False)
    return residual


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('damping', [None, 0., .25])
@pytest.mark.parametrize('prescribed_tip', [False, True])
def test_actual_coefficient_and_physical_free_dof_entries(dim, parallel, damping, prescribed_tip):
    kwargs = {} if damping is None else {'damping': damping}
    coefficient = 1e-6 if damping is None else damping
    forces = np.zeros((2, dim)); forces[:, :2] = [[.02, -.01], [.03, .005]]
    prescribed = {0: (.004, -.002) + (0.,) * (dim - 2)}
    if prescribed_tip:
        prescribed[1] = (.01,) + (0.,) * (dim - 1)
    raw = solve(dim, parallel, forces=forces, fixed_nodes=[0], prescribed_disp=prescribed, **kwargs)
    ndof = len(COMPONENTS[dim]); free = np.arange(ndof + (dim if prescribed_tip else 0), 2 * ndof)
    residual = assert_diagnostics(raw, dim, free, coefficient)
    np.testing.assert_allclose(residual['values'], -coefficient * raw['u'].ravel()[free],
                               rtol=1e-10, atol=1e-12)
    values = list(residual['values'])
    raw['reactions'][:] = 999.
    assert residual['values'] == values  # The diagnostic owns its copied values.


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('kind', ['zero', 'balanced', 'net_force', 'net_couple'])
def test_free_loads_are_observations_without_equilibrium_or_uniqueness_claims(dim, kind):
    forces = np.zeros((2, dim))
    if kind == 'balanced':
        forces[:, 0] = [-.1, .1]
    elif kind == 'net_force':
        forces[1, 0] = .1
    elif kind == 'net_couple':
        forces[:, 1] = [-.1, .1]
    raw = solve(dim, forces=forces, damping=.25)
    free = np.arange(raw['u'].size)
    residual = assert_diagnostics(raw, dim, free, .25)
    np.testing.assert_allclose(residual['values'], -.25 * raw['u'].ravel(), atol=1e-12)
    if kind == 'zero':
        assert residual['values'] == [0.] * len(free)
    elif kind in ('net_force', 'net_couple'):
        assert np.max(np.abs(residual['values'])) > 1e-3


@pytest.mark.parametrize('dim', [2, 3])
def test_fully_constrained_empty_diagnostics_are_available(dim):
    raw = solve(dim, fixed_nodes=[0, 1])
    residual = assert_diagnostics(raw, dim, np.array([], dtype=int))
    assert residual['node_rows'] == residual['dof_components'] == residual['values'] == []


def sparse_graph(dim):
    graph = StructureGraph(dimension=dim)
    graph.add_node([0., 0., 0.]); graph.add_node([1., 0., 0.])
    graph.add_edge(0, 1, radius=.05)
    data = graph.to_dict(); ids = {0: 7, 1: 42}
    for node in data['nodes']:
        node['id'] = ids[node['id']]
    data['nodes'].reverse()
    data['edges'][0].update(id=11, node_i=7, node_j=42)
    data.pop('next_node_id'); data.pop('next_edge_id')
    return StructureGraph.from_dict(data)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('explicit', [False, True])
def test_row_to_stable_id_json_and_renderer_alignment(dim, explicit, tmp_path):
    from fibernet.viz.render import _align_result
    graph = sparse_graph(dim); fem = BeamFrameFEM(E=1000.)
    if explicit:
        pos, elements, radii = graph.to_numpy()
        raw = getattr(fem, f'solve_{dim}d')(elements.T, pos[:, :dim], radii,
                 fixed_nodes=[0], forces=np.array([[0.] * dim, [.1] + [0.] * (dim - 1)]))
        maps = {'node_ids': sorted(graph.nodes), 'edge_ids': list(graph.edges)}
    else:
        inp = fem.graph_to_fem_input(graph, dim=dim)
        raw = getattr(fem, f'solve_{dim}d')(inp['edge_index'], inp['node_pos'], inp['radii'],
                 fixed_nodes=[1], forces=np.array([[.1] + [0.] * (dim - 1), [0.] * dim]))
        maps = {}
    original = copy.deepcopy(raw['numerical_diagnostics'])
    result = fem.to_sim_result(raw, graph, **maps)
    diagnostic = result.metadata['numerical_diagnostics']
    residual = diagnostic['physical_free_residual']
    assert residual['index_kind'] == 'stable_node_id'
    assert 'node_rows' not in residual
    assert residual['node_ids'] == [42] * len(COMPONENTS[dim])
    assert residual['values'] == original['physical_free_residual']['values']
    assert raw['numerical_diagnostics'] == original
    assert diagnostic is not raw['numerical_diagnostics']
    assert fem.to_sim_result(raw).metadata['numerical_diagnostics'] == original
    for detailed in (False, True):
        path = tmp_path / f'{detailed}.json'
        result.save(path, detailed=detailed)
        saved = json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
        restored = SimResult.load(path)
        assert saved['numerical_diagnostics'] == diagnostic
        assert restored.metadata['numerical_diagnostics'] == diagnostic
        aligned = _align_result(graph, restored, require_edges=True)
        assert aligned.metadata['numerical_diagnostics'] == diagnostic
        assert aligned.metadata['numerical_diagnostics'] is not restored.metadata['numerical_diagnostics']
    residual['values'][0] = 123.
    assert raw['numerical_diagnostics'] == original


def test_legacy_absence_is_not_reinterpreted(tmp_path):
    fem = BeamFrameFEM(E=1000.); raw = solve(2, fixed_nodes=[0])
    del raw['numerical_diagnostics']
    assert 'numerical_diagnostics' not in fem.to_sim_result(raw).metadata
    path = tmp_path / 'old.json'
    path.write_text(json.dumps({'mode': 'fem_beam_frame', 'reactions': [[0, 0, 0]]}))
    loaded = SimResult.load(path)
    assert 'numerical_diagnostics' not in loaded.metadata
    loaded.save(path)
    assert 'numerical_diagnostics' not in json.loads(path.read_text())


@pytest.mark.parametrize('field,value', [
    ('node_rows', [-1, 1, 1]), ('node_rows', [2, 1, 1]),
    ('node_rows', [True, 1, 1]), ('node_rows', [1., 1, 1]),
    ('node_rows', [1, 1]), ('node_rows', [[1], 1, 1]),
    ('dof_components', ['ux', 'uy']), ('values', [0., 0.]),
])
def test_malformed_diagnostic_row_records_reject_before_relabeling(field, value):
    graph = sparse_graph(2); fem = BeamFrameFEM(E=1000.)
    raw = solve(2, fixed_nodes=[0])
    raw['numerical_diagnostics']['physical_free_residual'][field] = value
    with pytest.raises(ValueError, match='numerical_diagnostics'):
        fem.to_sim_result(raw, graph, node_ids=sorted(graph.nodes), edge_ids=list(graph.edges))


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_synthetic_nonfinite_free_residual_is_unavailable_json_null(dim, bad):
    ndof = len(COMPONENTS[dim])
    reactions = np.zeros((2, ndof)); reactions[1, 0] = bad
    free = np.arange(ndof, 2 * ndof)
    diagnostic = BeamFrameFEM._linear_numerical_diagnostics(reactions, free, .25)
    residual = diagnostic['physical_free_residual']
    assert residual['status'] == 'unavailable'
    assert residual['reason'] == 'nonfinite_recovered_residual'
    assert residual['values'] is None
    assert residual['node_rows'] == [1] * ndof
    assert residual['dof_components'] == COMPONENTS[dim]
    json.dumps(diagnostic, allow_nan=False)
    # A nonfinite constrained reaction is outside this free-residual vector.
    diagnostic = BeamFrameFEM._linear_numerical_diagnostics(reactions, np.arange(ndof), .25)
    assert diagnostic['physical_free_residual']['status'] == 'available'


@pytest.mark.parametrize('max_iter', [1, 2])
def test_nonlinear_finalization_never_exposes_inner_increment_diagnostics(max_iter, tmp_path):
    fem = BeamFrameFEM(E=1000.)
    raw = fem.solve_2d_nonlinear(*inputs(2), prescribed_disp={1: (.01, .002)},
                               fixed_nodes=[0], n_steps=2, max_iter=max_iter)
    diagnostic = raw['numerical_diagnostics']
    assert diagnostic['model'] == 'incremental_updated_geometry_linear_solves'
    residual = diagnostic['physical_free_residual']
    assert residual['status'] == 'unavailable'
    assert residual['reason'] == 'final_nonlinear_equilibrium_residual_not_recovered'
    assert residual['values'] is None
    assert 'node_rows' not in residual and 'node_ids' not in residual
    result = fem.to_sim_result(raw)
    for detailed in (False, True):
        path = tmp_path / f'nonlinear-{detailed}.json'
        result.save(path, detailed=detailed)
        assert SimResult.load(path).metadata['numerical_diagnostics'] == diagnostic
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_coefficient_is_diagnostic_null_without_a_physical_solve(bad):
    diagnostic = BeamFrameFEM._linear_numerical_diagnostics(
        np.zeros((2, 3)), np.arange(3, 6), bad)
    assert diagnostic['regularization'] == {
        'status': 'unavailable', 'method': 'diagonal_identity', 'coefficient': None,
        'interpretation': 'numerical_not_physical_support',
        'reason': 'nonfinite_regularization_coefficient'}
    assert diagnostic['physical_free_residual']['status'] == 'available'
    assert BeamFrameFEM._copy_numerical_diagnostics(diagnostic, 2, 3) == diagnostic
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize('section,changes', [
    ('regularization', {'status': 'available', 'coefficient': None}),
    ('regularization', {'status': 'available', 'coefficient': np.nan}),
    ('regularization', {'status': 'available', 'reason': 'unexpected'}),
    ('regularization', {'status': 'unavailable', 'coefficient': .25}),
    ('regularization', {'status': 'unavailable', 'coefficient': None}),
    ('regularization', {'status': 'unknown'}),
    ('physical_free_residual', {'status': 'available', 'values': None}),
    ('physical_free_residual', {'status': 'available', 'values': [np.inf, 0., 0.]}),
    ('physical_free_residual', {'status': 'available', 'reason': 'unexpected'}),
    ('physical_free_residual', {'status': 'unavailable', 'values': [0., 0., 0.],
                               'reason': 'nonfinite_recovered_residual'}),
    ('physical_free_residual', {'status': 'unavailable', 'values': None}),
    ('physical_free_residual', {'status': 'unknown'}),
    ('physical_free_residual', {'dof_components': ['ux', 'ux', 'rz']}),
    ('physical_free_residual', {'dof_components': ['ux', 'uz', 'rz']}),
])
def test_adapter_rejects_inconsistent_availability(section, changes):
    raw = solve(2, fixed_nodes=[0])
    raw['numerical_diagnostics'][section].update(changes)
    with pytest.raises(ValueError, match='numerical_diagnostics'):
        BeamFrameFEM().to_sim_result(raw)


@pytest.mark.parametrize('diagnostic', [None, {}, {'schema_version': True}, {'schema_version': 2}])
def test_present_malformed_diagnostics_are_not_treated_as_absent(diagnostic):
    raw = solve(2, fixed_nodes=[0])
    raw['numerical_diagnostics'] = diagnostic
    with pytest.raises(ValueError, match='numerical_diagnostics'):
        BeamFrameFEM().to_sim_result(raw)


def test_legacy_graph_map_cannot_silently_relabel_missing_input_rows():
    raw = solve(2, fixed_nodes=[0])
    graph = sparse_graph(2)
    graph.add_node([2., 0., 0.])
    with pytest.raises(ValueError, match='numerical_diagnostics.*node map'):
        BeamFrameFEM().to_sim_result(raw, graph)


def test_unavailable_synthetic_residual_survives_id_adaptation_and_json(tmp_path):
    raw = solve(2, fixed_nodes=[0])
    reactions = np.zeros((2, 3)); reactions[1, 0] = np.nan
    raw['numerical_diagnostics'] = BeamFrameFEM._linear_numerical_diagnostics(
        reactions, np.arange(3, 6), .25)
    # Only the diagnostic is synthetic; physical arrays remain the valid fixture.
    graph = sparse_graph(2)
    result = BeamFrameFEM().to_sim_result(raw, graph,
                  node_ids=sorted(graph.nodes), edge_ids=list(graph.edges))
    residual = result.metadata['numerical_diagnostics']['physical_free_residual']
    assert residual['node_ids'] == [42, 42, 42]
    assert residual['values'] is None
    path = tmp_path / 'synthetic.json'; result.save(path, detailed=True)
    saved = json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert saved['numerical_diagnostics'] == result.metadata['numerical_diagnostics']
    assert SimResult.load(path).metadata['numerical_diagnostics'] == result.metadata['numerical_diagnostics']


def test_nonlinear_adapter_rejects_leaked_increment_indices():
    raw = BeamFrameFEM(E=1000.).solve_2d_nonlinear(
        *inputs(2), prescribed_disp={1: (.01, .002)}, fixed_nodes=[0], n_steps=1, max_iter=1)
    raw['numerical_diagnostics']['physical_free_residual']['node_rows'] = [1]
    with pytest.raises(ValueError, match='numerical_diagnostics.*without increment indices'):
        BeamFrameFEM().to_sim_result(raw)


@pytest.mark.parametrize('with_rows', [False, True])
def test_raw_stable_id_spoof_is_rejected(with_rows):
    raw = solve(2, fixed_nodes=[0])
    residual = raw['numerical_diagnostics']['physical_free_residual']
    residual.update(index_kind='stable_node_id', node_ids=[42, 42, 42])
    if not with_rows:
        residual.pop('node_rows')
    with pytest.raises(ValueError, match='numerical_diagnostics.*input_node_row'):
        BeamFrameFEM().to_sim_result(raw)


@pytest.mark.parametrize('ndof', [0, 1, 2, 4, 5, 7])
def test_unknown_dof_count_is_not_inferred_as_3d(ndof):
    raw = solve(2, fixed_nodes=[0])
    with pytest.raises(ValueError, match='numerical_diagnostics.*3 or 6 DOFs'):
        BeamFrameFEM._copy_numerical_diagnostics(raw['numerical_diagnostics'], 2, ndof)
    with pytest.raises(ValueError, match='numerical_diagnostics.*3 or 6 DOFs'):
        BeamFrameFEM._linear_numerical_diagnostics(np.zeros((2, ndof)), np.array([], dtype=int), .25)
    raw['u'] = np.zeros((2, ndof))
    with pytest.raises(ValueError, match='numerical_diagnostics'):
        BeamFrameFEM().to_sim_result(raw)


@pytest.mark.parametrize('key,value', [('node_ids', [42]), ('index_kind', 'input_node_row'),
                                      ('dof_components', ['ux'])])
def test_nonlinear_adapter_rejects_every_increment_index_field(key, value):
    raw = BeamFrameFEM(E=1000.).solve_2d_nonlinear(
        *inputs(2), prescribed_disp={1: (.01, .002)}, fixed_nodes=[0], n_steps=1, max_iter=1)
    raw['numerical_diagnostics']['physical_free_residual'][key] = value
    with pytest.raises(ValueError, match='numerical_diagnostics.*without increment indices'):
        BeamFrameFEM().to_sim_result(raw)
