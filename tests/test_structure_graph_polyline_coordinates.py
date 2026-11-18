"""A rejected polyline coordinate leaves the existing graph ready for reuse."""

import copy

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph


def existing_graph(dimension):
    graph = StructureGraph(dimension=dimension)
    for point in ([99, 99], [0, 0], [88, 88], [2, 0]):
        graph.add_node(point, merge=False)
    graph.remove_node(0)
    graph.remove_node(2)
    graph.add_edge(1, 3, radius=0.2, internal_points=[[1, 0.2, 0]], label="existing")
    graph.nodes[1].metadata["nested"] = {"labels": ["kept"]}
    graph.metadata["units"] = {"length": "mm"}
    return graph


def state(graph):
    return copy.deepcopy((graph.to_dict(), graph._spatial_hash, graph._edge_set))


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("prefix_length", [1, 2])
@pytest.mark.parametrize("bad_point", [
    [np.nan, 0],
    [0, np.inf, 0],
    [-np.inf, 0],
    [np.finfo(float).max, 0],
    [],
    [1],
    [1, 2, 3, 4],
    np.ones((2, 2)),
    np.array(1.0),
])
def test_late_bad_coordinate_preserves_graph_and_future_use(
    dimension, prefix_length, bad_point, tmp_path
):
    graph = existing_graph(dimension)
    before = state(graph)
    node_objects = {nid: id(node) for nid, node in graph.nodes.items()}
    edge_objects = {eid: id(edge) for eid, edge in graph.edges.items()}
    points = [[3, 1], [4, 2]][:prefix_length] + [bad_point]
    with pytest.raises(ValueError):
        graph.add_polyline(iter(points), closed=True)
    assert state(graph) == before
    assert {nid: id(node) for nid, node in graph.nodes.items()} == node_objects
    assert {eid: id(edge) for eid, edge in graph.edges.items()} == edge_objects
    assert graph.add_node([0, 0]) == 1
    assert graph.add_node([2, 0]) == 3
    assert state(graph) == before

    # The next operation must retain both stable-ID allocation and merge lookup.
    assert graph.add_polyline([[2, 0], [4, 0], [4, 3]], closed=True) == [1, 2, 3]
    assert list(graph.nodes) == [1, 3, 4, 5]
    assert graph.add_node([4, 0]) == 4
    assert graph.add_node([4, 3]) == 5
    path = tmp_path / "reused.json"
    graph.save_json(path)
    restored = StructureGraph.load_json(path)
    assert restored.to_dict() == graph.to_dict()
    assert restored.add_node([4, 0]) == 4


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("closed", [False, True])
def test_valid_flattened_and_repeated_points_preserve_identity(dimension, closed):
    graph = existing_graph(dimension)
    source = np.array([[4.0], [3.0], [0.0]])
    points = iter([[[2.0, 0.0]], np.ma.array([4.0, 0.0], mask=False),
                   source, [4.0, 3.0, 0.0]])
    assert graph.add_polyline(points, closed=closed, radius=0.3, label="new") == (
        [1, 2, 3] if closed else [1, 2]
    )
    assert list(graph.nodes) == [1, 3, 4, 5]
    np.testing.assert_array_equal(graph.node_positions(),
                                  [[0, 0, 0], [2, 0, 0], [4, 0, 0], [4, 3, 0]])
    for eid in [1, 2] + ([3] if closed else []):
        assert graph.edges[eid].radius == 0.3
        assert graph.edges[eid].metadata["label"] == "new"
    source[:] = 100
    np.testing.assert_array_equal(graph.nodes[5].position, [4, 3, 0])
    assert graph.add_node([4, 3]) == 5
