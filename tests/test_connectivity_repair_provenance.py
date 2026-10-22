"""Pure small-graph provenance checks; no solver, native provider or training."""
import builtins
import copy
import json

import numpy as np
import pytest

from fibernet import Material, StructureGraph
import fibernet.gen.pattern as pattern

KEY = "post_tile_connectivity_repair"


def sparse_graph(points, pairs):
    graph = StructureGraph(dimension=3, box_size=[100., 100., 100.])
    for point in points:
        graph.add_node(point, merge=False, boundary=(True,) * 6, source={"keep": [1]})
    for index, (i, j) in enumerate(pairs):
        graph.add_edge(i, j, radius=.11 + index * .01, allow_parallel=True,
                       material=Material(name=f"material-{index}", youngs_modulus=123. + index),
                       segments=3 + index, source={"keep": [index]})
    # Exercise the native sparse-ID contract, including deleted-ID gaps.
    data = copy.deepcopy(graph.to_dict())
    node_map = {n["id"]: 2 + 6 * n["id"] for n in data["nodes"]}
    for node in data["nodes"]:
        node["id"] = node_map[node["id"]]
    for edge in data["edges"]:
        edge.update(id=3 + 7 * edge["id"], node_i=node_map[edge["node_i"]],
                    node_j=node_map[edge["node_j"]])
    data.update(next_node_id=99, next_edge_id=101)
    data["metadata"] = {"user": {"unchanged": ["a"]}}
    return StructureGraph.from_dict(data)


def triangles(offset=2.):
    graph = sparse_graph(
        [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.],
         [offset, 0., 0.], [offset + 1., 0., 0.], [offset, 1., 0.]],
        [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (1, 0)],
    )
    graph.edges[3].internal_points = np.array([[0., .5, .1]])
    return graph


def without_report(graph):
    value = copy.deepcopy(graph.to_dict())
    value["metadata"].pop(KEY, None)
    for edge in value["edges"]:
        edge["metadata"].pop(KEY, None)
    return value


def assert_original_records_unchanged(before, graph):
    after = graph.to_dict()
    assert after["nodes"] == before["nodes"]
    assert after["next_node_id"] == before["next_node_id"]
    assert after["box_size"] == before["box_size"]
    assert after["dimension"] == before["dimension"]
    assert after["tolerance"] == before["tolerance"]
    edges = {e["id"]: e for e in after["edges"]}
    for edge in before["edges"]:
        assert edges[edge["id"]] == edge
    assert {k: v for k, v in after["metadata"].items() if k != KEY} == before["metadata"]


def assert_report(graph, before_count, after_count, status="completed"):
    report = graph.metadata[KEY]
    assert report["version"] == 1
    assert report["status"] == status
    assert report["components_before"] == before_count
    assert report["components_after"] == after_count
    assert report["coordinate_space"] == "graph_coordinates"
    assert report["length_unit"] is None
    assert report["unit_conversion_applied"] is False
    assert report["distance_comparison"] == "strictly_less_than"
    assert report["max_bridge_dist"] == 3.
    assert report["dangling_max_dist"] == 6.
    assert report["added_node_ids"] == report["merged_node_ids"] == report["removed_edge_ids"] == []
    return report


def test_connected_graph_does_not_enter_dangling_pass():
    graph = sparse_graph([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [(0, 1), (1, 2)])
    before = copy.deepcopy(graph.to_dict())
    assert pattern._post_tile_connectivity_repair(graph) is graph
    assert assert_report(graph, 1, 1, "not_needed")["added_edge_ids"] == []
    assert without_report(graph) == before
    assert all(KEY not in edge.metadata for edge in graph.edges.values())


def test_empty_graph_reports_no_repair():
    graph = StructureGraph(3)
    pattern._post_tile_connectivity_repair(graph)
    assert assert_report(graph, 0, 0, "not_needed")["added_edge_ids"] == []


def test_real_component_bridge_preserves_sparse_physical_identity_and_geometry():
    graph = triangles()
    before = copy.deepcopy(graph.to_dict())
    material = graph.edges[3].material
    median_radius = float(np.median([edge.radius for edge in graph.edges.values()]))
    pattern._post_tile_connectivity_repair(graph)
    report = assert_report(graph, 2, 1)
    assert report["added_edge_ids"] == [101]
    assert graph.to_dict()["next_edge_id"] == 102
    edge = graph.edges[101]
    assert (edge.node_i, edge.node_j) == (14, 20)
    assert edge.material is material
    assert edge.radius == median_radius
    assert edge.metadata[KEY] == {
        "version": 1, "reason": "connect_disconnected_components",
        "distance_at_addition": 1., "distance_limit": 3.,
        "coordinate_space": "graph_coordinates", "length_unit": None,
    }
    assert_original_records_unchanged(before, graph)


def test_dangling_additions_distinguish_duplicate_attempts_and_boundary_nodes():
    # At distance exactly 3, the component bridge fails the strict threshold.
    # Degree-one augmentation at limit 6 then makes four calls, one duplicate.
    graph = sparse_graph([[0, 0, 0], [1, 0, 0], [4, 0, 0], [5, 0, 0]], [(0, 1), (2, 3)])
    before = copy.deepcopy(graph.to_dict())
    pattern._post_tile_connectivity_repair(graph)
    report = assert_report(graph, 2, 1)
    assert report["added_edge_ids"] == [101, 102, 103]
    expected = [(2, 14, 4.), (8, 14, 3.), (20, 8, 4.)]
    for eid, (i, j, distance) in zip(report["added_edge_ids"], expected):
        edge = graph.edges[eid]
        assert (edge.node_i, edge.node_j) == (i, j)
        assert edge.metadata[KEY]["reason"] == "connect_degree_one_node"
        assert edge.metadata[KEY]["distance_at_addition"] == distance
        assert edge.metadata[KEY]["distance_limit"] == 6.
    assert_original_records_unchanged(before, graph)


@pytest.mark.parametrize("offset", [4., 10.])
def test_strict_threshold_and_out_of_range_can_remain_disconnected(offset):
    graph = triangles(offset)
    before = copy.deepcopy(graph.to_dict())
    pattern._post_tile_connectivity_repair(graph)
    assert assert_report(graph, 2, 2)["added_edge_ids"] == []
    assert without_report(graph) == before


def test_custom_limit_uses_raw_coordinates_not_box_scaling():
    graph = triangles()
    before = copy.deepcopy(graph.to_dict())
    pattern._post_tile_connectivity_repair(graph, max_bridge_dist=.75)
    report = graph.metadata[KEY]
    assert (report["max_bridge_dist"], report["dangling_max_dist"]) == (.75, 1.5)
    assert report["added_edge_ids"] == []
    assert report["components_after"] == 2
    assert without_report(graph) == before


def test_save_load_and_copy_preserve_complete_reports_and_ids(tmp_path):
    graph = triangles()
    pattern._post_tile_connectivity_repair(graph)
    record = copy.deepcopy(graph.to_dict())
    assert record["version"] == "3.1"
    json.dumps(record, allow_nan=False)
    path = tmp_path / "graph.json"
    graph.save_json(path)
    restored = StructureGraph.load_json(path)
    assert restored.to_dict() == record
    assert restored.add_node([9, 9, 9], merge=False) == 99
    assert restored.add_edge(2, 8, allow_parallel=True) == 102
    duplicate = graph.copy()
    duplicate.metadata[KEY]["added_edge_ids"].append(999)
    duplicate.edges[101].metadata[KEY]["reason"] = "changed"
    assert graph.to_dict() == record


def test_repeat_is_deterministic_and_latest_invocation_does_not_relabel_old_edges():
    left, right = triangles(), triangles()
    pattern._post_tile_connectivity_repair(left)
    pattern._post_tile_connectivity_repair(right)
    assert left.to_dict() == right.to_dict()
    before = copy.deepcopy(left.to_dict())
    pattern._post_tile_connectivity_repair(left)
    assert assert_report(left, 1, 1, "not_needed")["added_edge_ids"] == []
    assert left.to_dict()["edges"] == before["edges"]
    assert without_report(left) == without_report(right)


@pytest.mark.parametrize("dependency,import_name", [("scipy", "scipy.spatial"), ("networkx", "networkx")])
def test_missing_dependency_is_not_reported_as_success(monkeypatch, dependency, import_name):
    graph = triangles()
    before = copy.deepcopy(graph.to_dict())
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == import_name:
            raise ImportError("unavailable for this test")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    assert pattern._post_tile_connectivity_repair(graph) is graph
    report = graph.metadata[KEY]
    assert report["status"] == "skipped_missing_dependency"
    assert report["missing_dependency"] == dependency
    assert report["components_before"] is report["components_after"] is None
    assert report["added_edge_ids"] == []
    assert without_report(graph) == before


def test_failure_keeps_incomplete_status_and_does_not_invent_additions(monkeypatch):
    graph = triangles()
    before = copy.deepcopy(graph.to_dict())
    def fail(*args, **kwargs):
        raise RuntimeError("injected add failure")
    monkeypatch.setattr(graph, "add_edge", fail)
    with pytest.raises(RuntimeError, match="injected add failure"):
        pattern._post_tile_connectivity_repair(graph)
    report = graph.metadata[KEY]
    assert report["status"] == "incomplete"
    assert report["components_before"] == 2 and report["components_after"] is None
    assert report["added_edge_ids"] == []
    assert without_report(graph) == before


def test_public_hcp_route_really_repairs_and_serializes(monkeypatch, tmp_path):
    captured = []
    repair = pattern._post_tile_connectivity_repair
    def capture(graph, *args, **kwargs):
        captured.append(copy.deepcopy(graph.to_dict()))
        return repair(graph, *args, **kwargs)
    monkeypatch.setattr(pattern, "_post_tile_connectivity_repair", capture)
    graph = pattern.pattern_3d(unit="hcp", box=(2., 2., 2.), grid=(1, 2, 1))
    assert len(captured) == 1
    report = assert_report(graph, 2, 1)
    old = captured[0]
    assert (len(old["nodes"]), len(old["edges"]), graph.num_edges) == (34, 86, 87)
    assert report["added_edge_ids"] == [86]
    assert graph.to_dict()["nodes"] == old["nodes"]
    assert graph.to_dict()["edges"][:86] == old["edges"]
    assert graph.metadata["n_components_post_repair"] == report["components_after"]
    edge = graph.edges[86]
    assert edge.metadata[KEY]["reason"] == "connect_disconnected_components"
    assert edge.metadata[KEY]["distance_at_addition"] == pytest.approx(
        np.linalg.norm(graph.nodes[edge.node_i].position - graph.nodes[edge.node_j].position))
    path = tmp_path / "hcp.json"
    graph.save_json(path)
    assert StructureGraph.load_json(path).to_dict() == graph.to_dict()


def test_non_repair_generators_do_not_claim_coverage():
    graph = pattern.pattern_3d(unit="cubic", box=(2., 2., 2.), grid=1)
    assert KEY not in graph.metadata
    assert all(KEY not in edge.metadata for edge in graph.edges.values())
