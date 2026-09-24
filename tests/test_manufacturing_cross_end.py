"""Compare independent library and desktop manufacturing topology inputs."""
import sys
from pathlib import Path

import numpy as np
import pytest

from fibernet.gen.manufacturing import (PlanarManufacturingConfig,
                                       compile_planar, compile_surface,
                                       manufacturable_graph)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FiberScope'))
from fslab.structure import StructureFactory
from examples.manufacturing_workflow import ManufacturingWorkflow


@pytest.mark.parametrize('unit', ['square', 'hexagon', 'ring', 'kagome', 'octagon'])
@pytest.mark.parametrize('rule', ['translate', 'checker_mirror'])
def test_planar_library_matches_app(unit, rule):
    settings = dict(unit=unit, grid_x=2, grid_y=2, n_pts_per_side=2,
                    line_displacements=[[0.02, -0.01], [0.01, 0.03]],
                    perturbation=0.2, seed=23, expansion_rule=rule)
    library = compile_planar(PlanarManufacturingConfig(**settings))
    desktop = compile_planar(StructureFactory(**settings))
    assert np.array_equal(library.edges, desktop.edges)
    assert np.array_equal(library.route_nodes, desktop.route_nodes)
    assert np.array_equal(library.route_edges, desktop.route_edges)
    assert np.allclose(library.reference, desktop.reference, atol=1e-12)
    assert np.allclose(library.positions, desktop.positions, atol=1e-12)
    assert library.health['odd'] == 0
    assert library.health['components'] == 1
    graph = manufacturable_graph(PlanarManufacturingConfig(**settings))
    assert graph.num_nodes == len(library.positions)
    assert graph.num_edges == len(library.edges)
    assert graph.metadata['topology_id'] == library.topology_id


@pytest.mark.parametrize('unit', ['square', 'hexagon', 'kagome'])
def test_surface_library_matches_app(unit):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.2],
                         [0, 1, 0.1], [1, 1, 0.2], [2, 1, 0.4]])
    faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    settings = dict(unit=unit, n_pts_per_side=2, seed=23,
                    line_displacements=[[0.02, -0.01], [0.01, 0.03]])
    library = compile_surface(vertices, faces,
                              PlanarManufacturingConfig(**settings))
    desktop = compile_surface(vertices, faces, StructureFactory(**settings))
    assert np.array_equal(library.edges, desktop.edges)
    assert np.array_equal(library.route_nodes, desktop.route_nodes)
    assert np.allclose(library.positions, desktop.positions, atol=1e-12)
    assert library.health['odd'] == 0


def test_surface_rejects_invalid_indices():
    vertices = np.array([[0, 0, 0], [1, 0, 0],
                         [1, 1, 0], [0, 1, 0]])
    config = PlanarManufacturingConfig(n_pts_per_side=1)
    with pytest.raises(ValueError, match='indices'):
        compile_surface(vertices, [[0, 1, 2, 4]], config)
    with pytest.raises(ValueError, match='indices'):
        compile_surface(vertices, [[0, 1, 1, 3]], config)


def test_documented_manufacturing_workflow(tmp_path):
    summary = ManufacturingWorkflow(tmp_path).run()
    assert summary['planar']['route_closed']
    assert summary['surface']['route_closed']
    for name in ('planar', 'surface'):
        with np.load(tmp_path / (name + '.npz'), allow_pickle=False) as saved:
            assert len(saved['route_edges']) == len(saved['edges'])
