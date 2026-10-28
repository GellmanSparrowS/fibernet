"""Spatial welding includes cell corners, and empty arrays retain their axes."""

from itertools import product

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph


@pytest.mark.parametrize("dimension", [2, 3])
def test_welds_across_every_neighboring_hash_cell(dimension):
    tolerance = 1e-3
    for direction in product((-1, 0, 1), repeat=dimension):
        if not any(direction):
            continue
        graph = StructureGraph(dimension=dimension, tolerance=tolerance)
        direction = np.asarray(direction, dtype=float)
        source = direction * (0.49 * tolerance)
        nearby = direction * (0.51 * tolerance)
        assert np.linalg.norm(source - nearby) < tolerance
        node_id = graph.add_node(source)
        assert graph.add_node(nearby) == node_id
        assert graph.num_nodes == 1


@pytest.mark.parametrize("operation", ["copy", "roundtrip", "move"])
def test_neighbor_welding_after_copy_loading_and_position_edits(operation):
    graph = StructureGraph(dimension=3, tolerance=1e-3)
    source = np.array([0.00049, -0.00049, 0.00049])
    nearby = np.array([0.00051, -0.00051, 0.00051])
    node_id = graph.add_node(source if operation != "move" else [1, 2, 3])
    if operation == "copy":
        graph = graph.copy()
    elif operation == "roundtrip":
        graph = StructureGraph.from_dict(graph.to_dict())
    else:
        graph.set_node_position(node_id, source)
    assert graph.add_node(nearby) == node_id
    assert graph.num_nodes == 1


def test_neighborhood_lookup_still_checks_distance_and_explicit_nonmerging():
    graph = StructureGraph(tolerance=1.0)
    a = graph.add_node([0, 0])
    b = graph.add_node([0.9, 0.9])
    assert a != b  # Adjacent hash cells, but farther apart than the tolerance.
    assert graph.add_node([0, 0], merge=False) not in (a, b)
    assert graph.num_nodes == 3


def test_existing_axis_match_keeps_precedence_over_new_diagonal_matches():
    graph = StructureGraph(tolerance=1.0)
    axis = graph.add_node([0.51, 0.49], merge=False)
    graph.add_node([0.51, 0.51], merge=False)
    assert graph.add_node([0.49, 0.49]) == axis


@pytest.mark.parametrize("node_count", [0, 1, 3])
def test_edgeless_numpy_arrays_keep_coordinate_and_endpoint_axes(node_count):
    graph = StructureGraph()
    for index in range(node_count):
        graph.add_node([index, 0])
    positions, edges, radii = graph.to_numpy()
    assert positions.shape == (node_count, 3)
    assert edges.shape == (0, 2)
    assert np.issubdtype(edges.dtype, np.integer)
    assert radii.shape == (0,)


def test_numpy_shape_after_removing_the_last_edge():
    graph = StructureGraph()
    a = graph.add_node([0, 0])
    b = graph.add_node([1, 0])
    edge = graph.add_edge(a, b)
    graph.remove_edge(edge)
    assert graph.to_numpy()[1].shape == (0, 2)
