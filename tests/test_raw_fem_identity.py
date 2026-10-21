"""Direct-array FEM provenance through geometry, JSON, and actual rendering."""
import copy
import json

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fibernet import StructureGraph
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim.accelerated import SimResult
import fibernet.viz.render as render


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def graph_fixture(dim=2, parallel=False):
    graph = StructureGraph(dimension=dim)
    for x in (0, 1, 3):
        graph.add_node([x, 0, 0])
    graph.add_edge(0, 1, radius=.05)
    graph.add_edge(1, 2, radius=.1)
    if parallel:
        graph.add_edge(0, 1, radius=.2, allow_parallel=True)
    data = graph.to_dict()
    ids = {0: 7, 1: 42, 2: 91}
    for node in data['nodes']:
        node['id'] = ids[node['id']]
    for edge in data['edges']:
        edge['id'] = 11 + edge['id'] * 8
        edge['node_i'] = ids[edge['node_i']]
        edge['node_j'] = ids[edge['node_j']]
    data['nodes'].reverse()
    data.pop('next_node_id'); data.pop('next_edge_id')
    return StructureGraph.from_dict(data)


def solve(graph, dim=2, permutation=None, deduplicate=False):
    pos, elements, radii = graph.to_numpy()
    edge_ids = list(graph.edges)
    if permutation is not None:
        elements, radii = elements[permutation], radii[permutation]
        edge_ids = [edge_ids[i] for i in permutation]
    solver = BeamFrameFEM(E=1000.)
    raw = getattr(solver, f'solve_{dim}d')(
        elements.T, pos[:, :dim], radii, fixed_nodes=[0],
        prescribed_disp={1: tuple([.1] + [0.] * (dim - 1)),
                         2: tuple([.5] + [0.] * (dim - 1))},
        deduplicate=deduplicate)
    return solver, raw, sorted(graph.nodes), edge_ids


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('permutation', [None, [1, 0]])
def test_array_export_solve_adapt_json_render(dim, permutation, tmp_path, monkeypatch):
    graph = graph_fixture(dim)
    solver, raw, node_ids, edge_ids = solve(graph, dim, permutation)
    graph_before, raw_before = graph.to_dict(), copy.deepcopy(raw)
    result = solver.to_sim_result(raw, graph, node_ids=node_ids, edge_ids=edge_ids)
    assert result.metadata['node_ids'] == [7, 42, 91]
    assert result.metadata['edge_ids'] == edge_ids
    np.testing.assert_allclose(result.deformed_positions[:, 0], [0, 1.1, 3.5])
    stretches = {11: 1.1, 19: 1.2}
    np.testing.assert_allclose(result.edge_stretches, [stretches[e] for e in edge_ids])
    # Independent geometric axial-energy proxy, with distinct lengths and radii.
    expected = .5 * 1000 * (.1**2 * np.pi * .05**2 * 1 +
                            .2**2 * np.pi * .1**2 * 2)
    assert result.energy == pytest.approx(expected)
    captured = []
    monkeypatch.setattr(render, '_render_3d_segments', lambda ax, g, *a: captured.append(g.copy()))
    monkeypatch.setattr(render, '_render_3d_colored', lambda ax, g, *a: captured.append(g.copy()))
    path = tmp_path / 'result.json'
    # Strict JSON encoding is an explicit acceptance check, including metadata.
    json.dumps(result.to_dict(detailed=True), allow_nan=False, default=lambda value: value.tolist())
    result.save(path, detailed=True)
    json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
    restored = SimResult.load(path)
    summary_path = tmp_path / 'summary.json'
    result.save(summary_path)
    summary = SimResult.load(summary_path)
    assert summary.metadata['node_ids'] == node_ids
    assert summary.metadata['edge_ids'] == edge_ids
    for current in (result, restored):
        render.render_deformation_3d(graph, current)
        np.testing.assert_allclose(captured[-1].nodes[7].position, [0, 0, 0])
        np.testing.assert_allclose(captured[-1].nodes[91].position, [3.5, 0, 0])
        render.render_stress_3d(graph, current)
    assert graph.to_dict() == graph_before
    for key in raw:
        if isinstance(raw[key], np.ndarray):
            np.testing.assert_array_equal(raw[key], raw_before[key])
    np.testing.assert_array_equal(raw['K'].toarray(), raw_before['K'].toarray())
    assert node_ids == [7, 42, 91]


@pytest.mark.parametrize('dim', [2, 3])
def test_dedup_selects_input_edge_ids_before_rendering(dim, tmp_path):
    graph = graph_fixture(dim, parallel=True)
    solver, raw, node_ids, edge_ids = solve(graph, dim, [2, 1, 0], deduplicate=True)
    np.testing.assert_array_equal(raw['edge_list'], [0, 1])
    result = solver.to_sim_result(raw, graph, node_ids=node_ids, edge_ids=edge_ids)
    assert result.metadata['edge_ids'] == [27, 19]
    expected = .5 * 1000 * (.1**2 * np.pi * .2**2 + .2**2 * np.pi * .1**2 * 2)
    assert result.energy == pytest.approx(expected)
    path = tmp_path / 'dedup.json'; result.save(path, detailed=True)
    for current in (result, SimResult.load(path)):
        render.render_deformation_3d(graph, current)
        with pytest.raises(ValueError, match='edge_ids coverage'):
            render.render_stress_3d(graph, current)


@pytest.mark.parametrize('axis,bad', [
    ('node_ids', [7, 42]), ('node_ids', [7, 42, 42]),
    ('node_ids', [7, 42, 99]), ('node_ids', [True, 42, 91]),
    ('node_ids', [7., 42, 91]), ('node_ids', [-7, 42, 91]),
    ('node_ids', [[7], [42], [91]]), ('node_ids', 7),
    ('edge_ids', [11]), ('edge_ids', [11, 11]), ('edge_ids', [11, 99]),
    ('edge_ids', [True, 19]), ('edge_ids', [11., 19]),
    ('edge_ids', [-11, 19]), ('edge_ids', [[11], [19]]),
])
def test_invalid_input_maps(axis, bad):
    graph = graph_fixture(); solver, raw, nodes, edges = solve(graph)
    maps = dict(node_ids=nodes, edge_ids=edges); maps[axis] = bad
    with pytest.raises(ValueError, match=axis):
        solver.to_sim_result(raw, graph, **maps)


@pytest.mark.parametrize('maps', [dict(node_ids=[7, 42, 91]), dict(edge_ids=[11, 19])])
def test_both_maps_required(maps):
    graph = graph_fixture(); solver, raw, _, _ = solve(graph)
    with pytest.raises(ValueError, match='both maps and graph'):
        solver.to_sim_result(raw, graph, **maps)


def test_graph_required_and_numpy_integer_ids_accepted():
    graph = graph_fixture(); solver, raw, nodes, edges = solve(graph)
    with pytest.raises(ValueError, match='both maps and graph'):
        solver.to_sim_result(raw, node_ids=nodes, edge_ids=edges)
    result = solver.to_sim_result(raw, graph, node_ids=np.array(nodes), edge_ids=np.array(edges))
    json.dumps(result.to_dict(True), allow_nan=False, default=lambda value: value.tolist())
    assert 'node_ids' not in solver.to_sim_result(raw).metadata


@pytest.mark.parametrize('bad', [[-1, 1], [0, 2], [0, 0], [False, 1], [0., 1.], [[0, 1]], [[0], [1, 2]], 0])
@pytest.mark.parametrize('explicit', [False, True])
def test_retained_rows_validated_before_indexing(bad, explicit):
    graph = graph_fixture(); solver, raw, nodes, edges = solve(graph)
    raw['edge_list'] = bad
    maps = dict(node_ids=nodes, edge_ids=edges) if explicit else {}
    with pytest.raises(ValueError, match='edge_list'):
        solver.to_sim_result(raw, graph, **maps)


@pytest.mark.parametrize('field', ['u', 'n_nodes'])
def test_node_count_mismatch(field):
    graph = graph_fixture(); solver, raw, nodes, edges = solve(graph)
    raw[field] = raw['u'][:2] if field == 'u' else 2
    with pytest.raises(ValueError, match='node_ids length'):
        solver.to_sim_result(raw, graph, node_ids=nodes, edge_ids=edges)


@pytest.mark.parametrize('dim', [2, 3])
def test_legacy_insertion_order_unchanged(dim):
    graph = graph_fixture(dim); solver = BeamFrameFEM(E=1000.)
    inputs = solver.graph_to_fem_input(graph, dim=dim)
    raw = getattr(solver, f'solve_{dim}d')(
        inputs['edge_index'], inputs['node_pos'], inputs['radii'], fixed_nodes=[2],
        prescribed_disp={0: tuple([.5] + [0.] * (dim - 1))})
    legacy = solver.to_sim_result(raw, graph)
    explicit = solver.to_sim_result(raw, graph, node_ids=list(graph.nodes), edge_ids=list(graph.edges))
    assert json.dumps(legacy.to_dict(True), allow_nan=False, default=lambda value: value.tolist()) == json.dumps(explicit.to_dict(True), allow_nan=False, default=lambda value: value.tolist())
    assert legacy.metadata['node_ids'] == [91, 42, 7]
    np.testing.assert_allclose(legacy.deformed_positions[-1], [0, 0, 0])
