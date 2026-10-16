"""Result row identity across real FEM, JSON and visualization.

Hand-constructed fixtures below test rendering only, not numerical accuracy.
"""
import json
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import fibernet as fn
from fibernet.sim.accelerated import SimResult, _graph_to_arrays
from fibernet.ml.beam_frame_fem import BeamFrameFEM
import fibernet.viz.render as render


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def graph_fixture(dim=3, parallel=False):
    graph = fn.StructureGraph(dimension=dim)
    graph.add_node([0, 0, 0]); graph.add_node([1, 0, 0])
    graph.add_edge(0, 1, radius=.05)
    if parallel:
        graph.add_edge(0, 1, radius=.1, allow_parallel=True)
    data = graph.to_dict()
    for node in data['nodes']:
        node['id'] = {0: 7, 1: 42}[node['id']]
    for edge in data['edges']:
        edge['node_i'] = 7; edge['node_j'] = 42
        edge['id'] = 11 + edge['id'] * 8
    data.pop('next_node_id'); data.pop('next_edge_id')
    data['nodes'].reverse()
    return fn.StructureGraph.from_dict(data)


def synthetic(graph):
    return SimResult(displacements=np.array([[.1, 0, 0], [0, 0, 0]]),
                     deformed_positions=np.array([[1.1, 0, 0], [0, 0, 0]]),
                     positions_trajectory=[np.array([[1.1, 0, 0], [0, 0, 0]])],
                     edge_forces=np.arange(graph.num_edges, dtype=float) + 1,
                     edge_stretches=np.ones(graph.num_edges),
                     metadata={'node_ids': [42, 7], 'edge_ids': list(graph.edges)})


@pytest.mark.parametrize('dim', [2, 3])
def test_real_fem_sparse_reversed_reload_fixed_node(dim, tmp_path, monkeypatch):
    graph = graph_fixture(dim)
    result = fn.simulate(graph, backend='fem', strain=1.01, nonlinear=False, E=1000.)
    assert result.metadata['node_ids'] == [42, 7]
    assert result.metadata['edge_ids'] == [11]
    path = tmp_path / 'full.json'; result.save(path, detailed=True)
    restored = SimResult.load(path)
    summary = tmp_path / 'summary.json'; result.save(summary)
    assert SimResult.load(summary).metadata['node_ids'] == [42, 7]
    captured = []
    monkeypatch.setattr(render, '_render_3d_segments', lambda ax, g, *a: captured.append(g.copy()))
    monkeypatch.setattr(render, '_render_3d_colored', lambda ax, g, *a: captured.append(g.copy()))
    for current in (result, restored):
        render.render_deformation_3d(graph, current)
        np.testing.assert_allclose(captured[-1].nodes[7].position, [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(captured[-1].nodes[42].position, [1.01, 0, 0], atol=1e-12)


def test_alignment_reorders_all_fields_without_mutation():
    graph = graph_fixture(parallel=True); result = synthetic(graph)
    result.metadata['edge_ids'].reverse()
    before = json.dumps(result.to_dict(True)); graph_before = graph.to_dict()
    aligned = render._align_result(graph, result, require_edges=True)
    np.testing.assert_array_equal(aligned.edge_forces, [2, 1])
    np.testing.assert_array_equal(aligned.displacements[:, 0], [0, .1])
    np.testing.assert_array_equal(aligned.positions_trajectory[0][:, 0], [0, 1.1])
    aligned.displacements[:] = 99
    assert json.dumps(result.to_dict(True)) == before
    assert graph.to_dict() == graph_before


@pytest.mark.parametrize('axis,bad', [
    ('node_ids', [7, 7]), ('node_ids', [7]), ('node_ids', [7, 99]),
    ('node_ids', [True, 42]), ('node_ids', [-1, 42]), ('node_ids', [7., 42]),
    ('edge_ids', [11, 11]), ('edge_ids', [11, 99]), ('edge_ids', [True, 19]),
])
def test_invalid_maps_fail_before_figure(axis, bad):
    graph = graph_fixture(parallel=True); result = synthetic(graph)
    result.metadata[axis] = bad
    before = plt.get_fignums()
    with pytest.raises(ValueError, match=axis):
        render.render_deformation_3d(graph, result)
    assert plt.get_fignums() == before


@pytest.mark.parametrize('field', ['displacements', 'deformed_positions', 'edge_forces', 'edge_stretches', 'positions_trajectory'])
def test_wrong_array_lengths_rejected(field):
    graph = graph_fixture(parallel=True); result = synthetic(graph)
    if field == 'positions_trajectory':
        result.positions_trajectory = [np.zeros((1, 3))]
    else:
        setattr(result, field, getattr(result, field)[:1])
    with pytest.raises(ValueError, match=field):
        render.render_deformation_3d(graph, result)


def test_edge_colors_follow_ids_with_parallel_fibers():
    graph = graph_fixture(parallel=True); result = synthetic(graph)
    result.metadata['edge_ids'] = [19, 11]
    figure = render.render_stress_3d(graph, result)
    colors = [c.get_colors()[0] for c in figure.axes[0].collections]
    cmap = plt.get_cmap('RdYlBu_r')
    np.testing.assert_allclose(colors, [cmap(1.), cmap(0.)])


def test_actual_fem_dedup_maps_partial_edges_and_refuses_stress():
    graph = graph_fixture(dim=2, parallel=True)
    pos, elements, _, _ = _graph_to_arrays(graph)
    solver = BeamFrameFEM(E=1000.)
    raw = solver.solve_2d(elements.T, pos[:, :2], np.array([.05, .1]),
                          fixed_nodes=[1], prescribed_disp={0: (.01, 0.)},
                          deduplicate=True)
    result = solver.to_sim_result(raw, graph)
    assert result.metadata['edge_ids'] == [11]
    render.render_deformation_3d(graph, result)
    with pytest.raises(ValueError, match='edge_ids coverage'):
        render.render_stress_3d(graph, result)
    with pytest.raises(ValueError, match='edge_ids coverage'):
        render.render_comparison_3d(graph, result)
    assert 'node_ids' not in solver.to_sim_result(raw).metadata


def test_explicit_sparse_2d_maps_and_legacy_fallback():
    graph = graph_fixture(dim=2, parallel=True)
    frames = [np.array([[1.2, 0, 0], [0, 0, 0]])]
    fig = render.render_trajectory(graph, frames, np.array([2., 1.]),
                                    node_ids=[42, 7], edge_ids=[19, 11])
    collection = fig.axes[0].collections[0]
    np.testing.assert_allclose(collection.get_segments()[0], [[0, 0], [1.2, 0]])
    np.testing.assert_array_equal(collection.get_array(), [1, 2])
    fig = render.render_trajectory(graph, [frames[0][::-1]], np.array([1., 2.]))
    np.testing.assert_allclose(fig.axes[0].collections[0].get_segments()[0], [[0, 0], [1.2, 0]])


def test_direct_3d_trajectory_never_inherits_result_map(monkeypatch):
    graph = graph_fixture(); result = synthetic(graph); captured = []
    monkeypatch.setattr(render, '_render_3d_colored_global', lambda ax, g, *a: captured.append(g.copy()))
    direct = [np.array([[0, 0, 0], [1.2, 0, 0]])]
    render.render_trajectory_3d(graph, direct, sim_result=result)
    np.testing.assert_array_equal(captured[-1].nodes[7].position, [0, 0, 0])
    render.render_trajectory_3d(graph, sim_result=result)
    np.testing.assert_array_equal(captured[-1].nodes[7].position, [0, 0, 0])
    with pytest.raises(ValueError, match='explicitly supplied'):
        render.render_trajectory_3d(graph, sim_result=result, node_ids=[42, 7])


def test_historical_metadata_free_contiguous_result():
    graph = fn.StructureGraph(dimension=3)
    graph.add_node([0, 0, 0]); graph.add_node([1, 0, 0]); graph.add_edge(0, 1)
    result = SimResult(displacements=np.zeros((2, 3)), deformed_positions=graph.node_positions(),
                       edge_forces=np.array([1.]), edge_stretches=np.array([1.]))
    for draw in (render.render_deformation_3d, render.render_stress_3d,
                 render.render_comparison_3d, render.render_multi_angle_3d):
        draw(graph, result)


def test_real_spring_kernel_publishes_identity(tmp_path):
    pytest.importorskip('taichi')
    from fibernet.sim.accelerated import TaichiEngine
    graph = graph_fixture(parallel=True)
    result = TaichiEngine(arch='cpu', num_threads=2).dynamics(
        graph, fixed_nodes=[1], displacement_schedule={0: [(0, np.zeros(3)),
        (4, np.array([.01, 0, 0]))]}, stiffness=10., num_steps=4, save_interval=1)
    assert result.metadata['node_ids'] == [42, 7]
    assert result.metadata['edge_ids'] == [11, 19]
    np.testing.assert_allclose(result.displacements[1], 0., atol=1e-12)
    path = tmp_path / 'spring.json'; result.save(path, detailed=True)
    restored = render._align_result(graph, SimResult.load(path), require_edges=True)
    np.testing.assert_allclose(restored.displacements[0], 0., atol=1e-12)
    assert len(restored.positions_trajectory) > 0


def test_explicit_deformation_map_and_historical_insertion_order(monkeypatch):
    colors = []
    original_draw = render.render_graph
    def capture(*args, **kwargs):
        if kwargs.get("color_data") is not None:
            colors.append(kwargs["color_data"].copy())
        return original_draw(*args, **kwargs)
    monkeypatch.setattr(render, "render_graph", capture)
    graph = graph_fixture(dim=2)
    third = graph.add_node([2, 0, 0]); graph.add_edge(42, third)
    mapped = render.render_deformation(graph, graph,
        displacement_data=np.array([[0, 0, 0], [2, 0, 0], [6, 0, 0]]),
        node_ids=[7, 42, third])
    legacy = render.render_deformation(graph, graph,
        displacement_data=np.array([[2, 0, 0], [0, 0, 0], [6, 0, 0]]))
    np.testing.assert_array_equal(colors[0], [1., 4.])
    np.testing.assert_array_equal(colors[0], colors[1])


def test_diagnostic_metadata_axes_are_aligned():
    graph = graph_fixture(parallel=True); result = synthetic(graph)
    result.metadata.update(edge_ids=[19, 11], reactions=[[4, 5, 6], [1, 2, 3]],
        sigma_axial=[20., 10.], sigma_bending=[2., 1.], sigma_total=[22., 11.],
        beam_resultants=[[20., 2., 3.], [10., 1., 2.]])
    aligned = render._align_result(graph, result)
    np.testing.assert_array_equal(aligned.metadata['reactions'][0], [1, 2, 3])
    np.testing.assert_array_equal(aligned.metadata['beam_resultants'][:, 0], [10, 20])
    np.testing.assert_array_equal(aligned.metadata['sigma_axial'], [10, 20])


@pytest.mark.parametrize('bad', [np.nan, np.inf, 'bad', 1 + 2j])
def test_invalid_numeric_fields_rejected(bad):
    graph = graph_fixture(); result = synthetic(graph)
    result.edge_forces = np.array([bad])
    with pytest.raises(ValueError, match='edge_forces.*finite numeric'):
        render.render_deformation_3d(graph, result)


def test_missing_edge_map_and_short_legacy_arrays_fail():
    graph = graph_fixture(parallel=True); result = synthetic(graph)
    result.metadata['edge_ids'] = [11]
    with pytest.raises(ValueError, match='edge_forces.*length'):
        render.render_deformation_3d(graph, result)
    result.metadata = {}; result.displacements = np.zeros((1, 3))
    with pytest.raises(ValueError, match='displacements.*length'):
        render.render_deformation_3d(graph, result)
