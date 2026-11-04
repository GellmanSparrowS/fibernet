"""Runnable public/module/backend journeys; no new solver or graph implementation."""
import csv
import importlib.util

import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim.accelerated import SimResult


def test_quick_api_edit_persist_simulate_analyze_and_render(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    graph = fn.pattern_2d(unit='square', box=(2, 2), grid=(1, 1), radius=.05)
    graph.displace_node(min(graph.nodes), [.01, 0])
    graph.save_json(tmp_path / 'graph.json')
    restored = fn.StructureGraph.load_json(tmp_path / 'graph.json')
    result = fn.simulate(restored, backend='fem', strain=1.01,
                         E=1000., nonlinear=False)
    assert isinstance(result, SimResult)
    assert result.displacements.shape == (restored.num_nodes, 3)
    assert np.isfinite(result.displacements).all()
    result.save(tmp_path / 'result.json', detailed=True)
    replay = SimResult.load(tmp_path / 'result.json')
    np.testing.assert_allclose(replay.displacements, result.displacements)
    extractor = fn.GraphFeatureExtractor(canvas_size=64)
    features = extractor.extract(restored, allow_unavailable=True)
    assert features['n_node'] == restored.num_nodes
    assert features['n_edge'] == restored.num_edges
    figure = fn.show(restored, save_path=str(tmp_path / 'structure.png'))
    assert (tmp_path / 'structure.png').stat().st_size > 100
    plt.close(figure)


def test_module_and_backend_tiers_match_quick_fem():
    graph = fn.pattern_2d(unit='square', box=(2, 2), grid=(1, 1), radius=.05)
    quick = fn.simulate(graph, backend='fem', strain=1.01, E=1000., nonlinear=False)
    fem = BeamFrameFEM(E=1000.)
    module = fem.to_sim_result(fem.stretch_test(graph, target_stretch=1.01,
                                               dim=2, nonlinear=False), graph)
    inp = fem.graph_to_fem_input(graph, dim=2)
    displacement = inp['x_range'] * .01
    raw = fem.solve_2d(inp['edge_index'], inp['node_pos'], inp['radii'],
                       fixed_nodes=inp['boundaries']['left'],
                       prescribed_disp={row: (displacement, 0.)
                                        for row in inp['boundaries']['right']})
    backend = fem.to_sim_result(raw, graph)
    for result in (module, backend):
        np.testing.assert_allclose(result.displacements, quick.displacements,
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result.edge_forces, quick.edge_forces,
                                   rtol=1e-10, atol=1e-12)


def test_manufacturing_module_to_route_export(tmp_path):
    config = fn.PlanarManufacturingConfig(unit='square', grid_x=1, grid_y=1,
                                          n_pts_per_side=1)
    network = fn.compile_planar(config)
    settings = fn.PrintSettings(width=20., depth=20., diameter=1.)
    centers = settings.coordinates(network)
    destination = tmp_path / 'route.csv'
    fn.export_route(destination, network, centers)
    with destination.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(network.edges) + 1
    assert {int(row['incoming_edge_id']) for row in rows[1:]} == set(range(len(network.edges)))
    assert [rows[0][axis] for axis in ('x_mm', 'y_mm', 'z_mm')] == [
        rows[-1][axis] for axis in ('x_mm', 'y_mm', 'z_mm')]


def test_quick_api_errors_name_unsupported_request():
    graph = fn.StructureGraph()
    with pytest.raises(ValueError, match='backend'):
        fn.simulate(graph, backend='unknown')
    with pytest.raises(ValueError, match='stretch'):
        fn.simulate(graph, backend='fem', mode='dynamics')


def test_default_spring_dependency_is_explicit_when_unavailable():
    if importlib.util.find_spec('taichi') is not None:
        pytest.skip('Taichi present; numerical runtime owned by optional-backend validation')
    graph = fn.pattern_2d(unit='square', box=(2, 2), grid=(1, 1))
    with pytest.raises(ImportError, match='[Tt]aichi'):
        fn.simulate(graph, num_steps=1, save_interval=1)
