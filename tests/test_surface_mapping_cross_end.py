"""Independent surface-map golden geometry and seam-edge contracts."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from fibernet.gen import MappingConfig, load_obj, map_cells
from examples.surface_mapping_workflow import SurfaceMappingWorkflow

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("unit, expected", [
    ("square", ((2461, 3), (3408, 2), "d66dad1f8c534833", "3e9d2497fa8e1bde")),
    ("hexagon", ((3852, 3), (4704, 2), "34a30d33c3221c40", "db0bbae331ed9aa5")),
])
def test_surface_map_matches_app_and_old_output(unit, expected):
    path = ROOT / "FiberScope" / "assets" / "obj" / "demo_pyramid.obj"
    vertices, faces = load_obj(path)
    spectrum = [[0.02, -0.01], [0.01, 0.03]]
    config = MappingConfig(max_points=100000, max_segments=200000)
    points, edges = map_cells(vertices, faces, spectrum, unit, config)
    assert (points.shape, edges.shape) == expected[:2]
    assert hashlib.sha256(points.tobytes()).hexdigest()[:16] == expected[2]
    assert hashlib.sha256(np.asarray(edges, dtype="<i8").tobytes()).hexdigest()[:16] == expected[3]
    assert config.source_faces == config.mapped_faces == 108


def test_surface_map_rejects_tiny_budget():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=float)
    faces = [[0, 1, 2, 3]] * 20
    with pytest.raises(MemoryError, match="budget exceeded"):
        map_cells(vertices, faces, [], config=MappingConfig(
            max_points=10, max_segments=10))


def test_surface_mapping_public_workflow(tmp_path):
    path = ROOT / "FiberScope" / "assets" / "obj" / "demo_pyramid.obj"
    summary = SurfaceMappingWorkflow(path, tmp_path).run()
    assert summary["mapped_faces"] == 108
    with np.load(tmp_path / "mapped_cells.npz", allow_pickle=False) as saved:
        assert len(saved["points"]) == summary["points"]
        assert len(saved["segments"]) == summary["segments"]


@pytest.mark.parametrize("faces, spectrum", [
    ([[0, 1, 2, 2]], [[0.0, 0.0]]),
    ([[0.0, 1.0, 2.0, 3.0]], [[0.0, 0.0]]),
    ([[0, 1, 2, 3]], [[float("nan"), 0.0]]),
])
def test_surface_mapping_invalid_inputs(faces, spectrum):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=float)
    with pytest.raises(ValueError):
        map_cells(vertices, faces, spectrum)


@pytest.mark.parametrize("edge_rows", [
    [(0, 1), (1, 2)],  # Stable IDs stay within row bounds but select wrong fibers.
    [(0, 1), (1, 2), (2, 3), (3, 0)],  # The last stable ID exceeds row bounds.
])
def test_surface_map_preserves_base_cell_edges_after_node_deletion(edge_rows):
    from fibernet.core.structure_graph import StructureGraph

    graph = StructureGraph(dimension=2)
    first = graph.add_node([0, 0], merge=False)
    removed = graph.add_node([10, 10], merge=False)
    ids = [first] + [graph.add_node(p, merge=False) for p in ([1, 0], [1, 1], [0, 1])]
    graph.remove_node(removed)
    for a, b in edge_rows:
        graph.add_edge(ids[a], ids[b])

    # At unit UV corners, bilinear mapping returns the corresponding 3D corner
    # exactly. Compare geometric fibers independently of output row ordering.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    points, edges = map_cells(vertices, [[0, 1, 2, 3]], [],
                              config=MappingConfig(overscale=1.0),
                              base_graph=graph)
    actual = {frozenset(map(tuple, points[edge])) for edge in edges}
    expected = {frozenset(map(tuple, vertices[[a, b]])) for a, b in edge_rows}
    assert len(edges) == len(expected)
    assert actual == expected
