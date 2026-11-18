"""Edge-building failures undo only elements allocated by that polyline."""

import copy

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph


def fixture_graph(dimension=2):
    graph = StructureGraph(dimension=dimension)
    for point in ([0, 0], [2, 0], [9, 9]):
        graph.add_node(point, merge=False)
    graph.remove_node(2)  # Leave a stable-ID gap and an existing empty hash bucket.
    graph.add_edge(0, 1, radius=0.1, label="original")
    graph.add_edge(0, 1, radius=0.2, allow_parallel=True, label="parallel")
    graph.metadata["units"] = {"length": "mm"}
    graph.nodes[0].metadata["labels"] = ["original"]
    return graph


def state(graph):
    return copy.deepcopy((graph.to_dict(), graph._spatial_hash, graph._edge_set))


def identities(graph):
    return (
        [id(graph.nodes), id(graph.edges), id(graph._spatial_hash), id(graph._edge_set)],
        {nid: (id(node), id(node.position), id(node.metadata))
         for nid, node in graph.nodes.items()},
        {eid: (id(edge), id(edge.material), id(edge.metadata))
         for eid, edge in graph.edges.items()},
        {key: id(bucket) for key, bucket in graph._spatial_hash.items()},
    )


def assert_reusable(graph, before, before_ids, tmp_path):
    assert state(graph) == before
    assert identities(graph) == before_ids
    assert graph.add_node([0, 0]) == 0
    assert graph.add_node([2, 0]) == 1
    assert state(graph) == before
    assert graph._edge_set[frozenset({0, 1})] == 0
    assert graph.add_polyline([[2, 0], [4, 0], [4, 1]], closed=True) == [2, 3, 4]
    assert list(graph.nodes) == [0, 1, 3, 4]
    assert graph.add_node([4, 0]) == 3
    path = tmp_path / "reused.json"
    graph.save_json(path)
    restored = StructureGraph.load_json(path)
    assert restored.to_dict() == graph.to_dict()
    assert restored.add_node([4, 1]) == 4


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("options,error", [
    ({"radius": np.nan}, ValueError),
    ({"n_internal": 1.5}, TypeError),
])
def test_edge_parameter_failure_restores_state_and_ids(dimension, options, error, tmp_path):
    graph = fixture_graph(dimension)
    before, before_ids = state(graph), identities(graph)
    with pytest.raises(error):
        graph.add_polyline([[0, 0], [3, 0], [4, 1]], **options)
    assert_reusable(graph, before, before_ids, tmp_path)


@pytest.mark.parametrize("allow_parallel", [False, True])
@pytest.mark.parametrize("after_insert", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_later_edge_failure_restores_duplicates_and_parallel_representative(
    allow_parallel, after_insert, closed, monkeypatch, tmp_path
):
    graph = fixture_graph()
    before, before_ids = state(graph), identities(graph)
    original = graph.add_edge
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            if after_insert:
                original(*args, **kwargs)
            raise RuntimeError("injected second-edge failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(graph, "add_edge", fail_second)
        with pytest.raises(RuntimeError, match="second-edge failure"):
            graph.add_polyline([[0, 0], [2, 0], [3, 0], [4, 1]],
                               closed=closed, allow_parallel=allow_parallel)
    assert len(calls) == 2
    assert_reusable(graph, before, before_ids, tmp_path)


def test_hash_rollback_preserves_existing_populated_and_empty_buckets():
    graph = StructureGraph(dimension=3, tolerance=1.0)
    graph.add_node([-.49, -.49, -.49])
    removed = graph.add_node([4, 4, 4])
    graph.remove_node(removed)
    before, before_ids = state(graph), identities(graph)
    # First new point shares an occupied cell but is beyond merge tolerance;
    # second reuses an old empty bucket; third creates a previously absent key.
    with pytest.raises(ValueError, match="radius must be finite"):
        graph.add_polyline([[.49, .49, .49], [4, 4, 4], [6, 4, 4]], radius=np.nan)
    assert state(graph) == before
    assert identities(graph) == before_ids
    assert graph.add_node([-.49, -.49, -.49]) == 0
    assert graph.add_node([4, 4, 4]) == 2


@pytest.mark.parametrize("points,expected", [
    ([], []),
    ([[0, 0]], []),
    ([[0, 0], [0, 0]], []),
    ([[0, 0], [2, 0]], [0]),
])
def test_existing_noop_self_loop_duplicate_admission_is_unchanged(points, expected):
    graph = fixture_graph()
    before = state(graph)
    assert graph.add_polyline(points, radius=np.nan, n_internal=1.5) == expected
    assert state(graph) == before


def test_single_new_vertex_still_succeeds_without_edge_parameter_validation():
    graph = fixture_graph()
    assert graph.add_polyline([[4, 0]], radius=np.nan, n_internal=1.5) == []
    assert graph.add_node([4, 0]) == 3
    assert graph.num_edges == 2
