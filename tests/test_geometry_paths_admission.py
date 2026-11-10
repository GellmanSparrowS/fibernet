"""Geometry analysis must not invent observations during NumPy promotion."""

from dataclasses import asdict

import networkx as nx
import numpy as np
import pytest

from fibernet.analysis import analyze_geometry_paths
from fibernet.core.structure_graph import StructureGraph


def graph():
    return StructureGraph.from_dict(
        {
            "dimension": 3,
            "nodes": [
                {"id": 30, "position": [2, 0, 0]},
                {"id": 10, "position": [0, 0, 0]},
                {"id": 70, "position": [8, 0, 0]},
            ],
            "edges": [
                {"id": 90, "node_i": 10, "node_j": 30},
                {"id": 8, "node_i": 30, "node_j": 10},
            ],
        }
    )


def analyze(g, **kwargs):
    return analyze_geometry_paths(
        g,
        length_unit="mm",
        coordinate_frame="lab",
        node_pairs=[(10, 30), (30, 10), (10, 70)],
        **kwargs,
    )


def masked_position(kind):
    scalar = np.ma.array(9.0, mask=(kind != "unmasked"))
    if kind == "tuple":
        return (scalar, 0.0, 0.0)
    if kind == "object":
        value = np.empty(3, dtype=object)
        value[:] = [scalar, 0.0, 0.0]
        return value
    return [scalar, 0.0, 0.0]


@pytest.mark.parametrize("kind", ["masked", "unmasked", "tuple", "object"])
@pytest.mark.parametrize("node_id", [10, 70])
def test_nested_masked_positions_rejected_even_on_isolates(kind, node_id, monkeypatch):
    g = graph()
    value = masked_position(kind)
    g.nodes[node_id].position = value
    monkeypatch.setattr(
        nx, "shortest_path", lambda *a, **k: pytest.fail("path traversal ran")
    )
    with pytest.raises(ValueError, match="position.*masked"):
        analyze(g)
    assert g.nodes[node_id].position is value


@pytest.mark.parametrize("mask", [[False, True, False], [False] * 3])
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("distance_mode", ["chord", "polyline"])
def test_nested_masked_rows_rejected_before_path_selection(
    mask, container, distance_mode
):
    g = graph()
    # The shorter parallel edge must not hide invalid data on another edge.
    row = np.ma.array([1.0, 9.0, 0.0], mask=mask)
    before_data, before_mask = row.data.copy(), row.mask.copy()
    value = container([row])
    g.edges[90].internal_points = value
    with pytest.raises(ValueError, match="internal_points.*masked"):
        analyze(g, distance_mode=distance_mode)
    assert g.edges[90].internal_points is value
    np.testing.assert_array_equal(row.data, before_data)
    np.testing.assert_array_equal(row.mask, before_mask)


@pytest.mark.parametrize(
    "value",
    [
        [[np.ma.array(1.0, mask=True), 9.0, 0.0]],
        [[np.ma.array(1.0, mask=False), 9.0, 0.0]],
        [np.ma.array([], mask=[])],
        np.ma.empty((0, 3)),
    ],
)
def test_scalar_and_empty_masks_are_not_laundered(value):
    g = graph()
    g.edges[90].internal_points = value
    with pytest.raises(ValueError, match="internal_points.*masked"):
        analyze(g)


@pytest.mark.parametrize(
    "coordinate",
    [True, np.bool_(True), np.array(True), np.array([True, False, False])],
)
@pytest.mark.parametrize("field", ["position", "internal_points"])
def test_boolean_coordinates_rejected_before_numeric_promotion(coordinate, field):
    g = graph()
    row = (
        coordinate
        if isinstance(coordinate, np.ndarray) and coordinate.shape == (3,)
        else [coordinate, 0.0, 0.0]
    )
    owner = g.nodes[10] if field == "position" else g.edges[90]
    value = row if field == "position" else [row]
    setattr(owner, field, value)
    with pytest.raises(ValueError, match="boolean"):
        analyze(g)
    assert getattr(owner, field) is value


@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_numeric_containers_and_explicit_unmasking_preserve_results(dtype, container):
    g = graph()
    g.edges[90].internal_points = np.array([[1, 2, 0]], dtype=dtype)
    expected = asdict(analyze(g))
    for node in g.nodes.values():
        node.position = (
            container(dtype(x) for x in node.position)
            if container is not np.array
            else np.array(node.position, dtype=dtype)
        )
    g.edges[90].internal_points = container([np.array([1, 2, 0], dtype=dtype)])
    stored = [node.position for node in g.nodes.values()]
    internal = g.edges[90].internal_points
    assert asdict(analyze(g)) == expected
    assert all(node.position is value for node, value in zip(g.nodes.values(), stored))
    assert g.edges[90].internal_points is internal
    for node, expected_row in zip(g.nodes.values(), ([2, 0, 0], [0, 0, 0], [8, 0, 0])):
        np.testing.assert_array_equal(node.position, expected_row)
    np.testing.assert_array_equal(internal, [[1, 2, 0]])
    g.edges[90].internal_points = np.ma.array([[1, 2, 0]], mask=False).data.copy()
    assert asdict(analyze(g)) == expected


def test_shared_containers_and_recursive_invalid_inputs_terminate():
    g = graph()
    row = [1.0, 2.0, 0.0]
    g.edges[90].internal_points = [row, row]
    assert analyze(g).polyline_length_by_edge[0] == pytest.approx(2 * np.sqrt(5))
    row.append(row)
    with pytest.raises(ValueError):
        analyze(g)
