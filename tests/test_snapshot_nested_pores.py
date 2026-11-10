"""Analytical bounded-face oracles for disconnected nested centerlines."""

import json

import networkx as nx
import numpy as np
import pytest

from fibernet.analysis import SnapshotFeatureExtractor
from fibernet.core.structure_graph import StructureGraph


def add_polygon(points, edges, polygon):
    first = len(points)
    points.extend(polygon)
    n = len(polygon)
    edges.extend((first + i, first + (i + 1) % n) for i in range(n))


def square(lo, hi):
    x, y = lo
    u, v = hi
    return [(x, y), (u, y), (u, v), (x, v)]


def nested():
    points, edges = [], []
    add_polygon(points, edges, square((0, 0), (4, 4)))
    add_polygon(points, edges, square((1, 1), (3, 3)))
    return np.array(points, float), np.array(edges)


def check(features, areas, perimeters, neighbors, region_area):
    np.testing.assert_allclose(features["pore_areas"], areas, rtol=2e-12, atol=1e-13)
    np.testing.assert_array_equal(features["pore_neighbor"], neighbors)
    assert features["pore_count"] == len(areas)
    assert features["pore_area_mean"] == pytest.approx(np.mean(areas))
    assert features["pore_area_cv"] == pytest.approx(np.std(areas) / np.mean(areas))
    assert features["pore_perim_mean"] == pytest.approx(np.mean(perimeters))
    assert features["pore_neighbor_mean"] == pytest.approx(np.mean(neighbors))
    assert features["largest_pore_ratio"] == pytest.approx(max(areas) / sum(areas))
    assert features["porosity"] == pytest.approx(min(sum(areas) / region_area, 1))


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
@pytest.mark.parametrize("offset", [(0, 0), (16, -32)])
def test_nested_cycles_partition_area_and_share_boundaries(scale, offset):
    points, edges = nested()
    points = (points + offset) * scale
    rect = (np.array([0, 0, 5, 5]) + np.tile(offset, 2)) * scale
    before = points.copy(), edges.copy()
    f = SnapshotFeatureExtractor().extract(points, edges, rect=rect)
    check(
        f,
        np.array([12, 4]) * scale**2,
        np.array([24, 8]) * scale,
        [1, 1],
        25 * scale**2,
    )
    np.testing.assert_array_equal(points, before[0])
    np.testing.assert_array_equal(edges, before[1])


def test_multiple_holes_and_three_nesting_levels():
    points, edges = [], []
    for lo, hi in [
        ((0, 0), (10, 10)),
        ((1, 1), (4, 4)),
        ((2, 2), (3, 3)),
        ((6, 6), (8, 8)),
    ]:
        add_polygon(points, edges, square(lo, hi))
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [87, 8, 1, 4], [60, 16, 4, 8], [2, 2, 1, 1], 100)


def test_subdivided_inner_component_is_subtracted_once():
    points, edges = nested()
    points = np.vstack([points, [2, 1], [2, 3]])
    edges = np.vstack([edges, [8, 9]])
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [12, 2, 2], [24, 6, 6], [2, 2, 2], 16)


def test_concave_parent_does_not_capture_component_in_its_notch():
    points, edges = [], []
    add_polygon(points, edges, [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    add_polygon(points, edges, square((0.2, 0.2), (0.8, 0.8)))
    add_polygon(points, edges, square((2, 2), (3, 3)))
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [6.64, 0.36, 1], [18.4, 2.4, 4], [1, 1, 0], 16)


@pytest.mark.parametrize("tree", ["segment", "branch", "inner", "outside"])
def test_disconnected_trees_contribute_both_sides_only_to_containing_pore(tree):
    points, edges = nested()
    if tree == "segment":
        extra_points, extra_edges, perims = (
            [[0.25, 0.5], [0.75, 0.5]],
            [[8, 9]],
            [25, 8],
        )
    elif tree == "branch":
        extra_points = [[0.25, 0.5], [0.75, 0.5], [0.5, 0.5], [0.5, 0.75]]
        extra_edges, perims = [[8, 9], [10, 11]], [25.5, 8]
    elif tree == "inner":
        extra_points, extra_edges, perims = [[1.5, 2], [2.5, 2]], [[8, 9]], [24, 10]
    else:
        extra_points, extra_edges, perims = [[5, 0], [6, 0]], [[8, 9]], [24, 8]
    points = np.vstack([points, extra_points])
    edges = np.vstack([edges, extra_edges])
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [12, 4], perims, [1, 1], np.prod(np.ptp(points, axis=0)))


def test_connected_annulus_with_bridge_and_inner_tree():
    points, edges = nested()
    # The bridge is traversed twice by the annular face boundary, which is
    # weakly simple. Winding membership must not assign the inner tree to it.
    edges = np.vstack([edges, [0, 4], [8, 9]])
    points = np.vstack([points, [1.5, 2], [2.5, 2]])
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [12, 4], [24 + 2 * np.sqrt(2), 10], [1, 1], 16)


def test_component_with_internal_bridge_has_one_exterior_boundary():
    points, edges = [], []
    for lo, hi in [((0, 0), (10, 10)), ((1, 1), (3, 3)), ((5, 1), (7, 3))]:
        add_polygon(points, edges, square(lo, hi))
    edges.append((5, 8))
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [92, 4, 4], [60, 8, 8], [2, 1, 1], 100)


@pytest.mark.parametrize("seed", range(5))
def test_order_direction_parallel_edges_and_isolates_preserve_face_metrics(seed):
    points, edges = nested()
    expected = SnapshotFeatureExtractor().extract(points, edges)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(points))
    inverse = np.argsort(permutation)
    edges = inverse[edges]
    edges = edges[rng.permutation(len(edges))]
    reverse = rng.random(len(edges)) < 0.5
    edges[reverse] = edges[reverse, ::-1]
    edges = np.vstack([edges, edges[0], edges[3, ::-1]])
    points = np.vstack([points[permutation], [2, 2]])
    f = SnapshotFeatureExtractor().extract(points, edges)
    for key in expected:
        if key.startswith("pore_") or key in ("porosity", "largest_pore_ratio"):
            np.testing.assert_allclose(
                np.sort(np.atleast_1d(f[key])), np.sort(np.atleast_1d(expected[key]))
            )


def test_sparse_structure_graph_json_and_direct_array_routes(tmp_path):
    points, edges = nested()
    graph = StructureGraph()
    for point in points:
        graph.add_node([*point, 0])
    for a, b in edges:
        graph.add_edge(int(a), int(b))
    data = graph.to_dict()
    for node in data["nodes"]:
        node["id"] = 101 + 7 * node["id"]
    for edge in data["edges"]:
        edge["id"] = 503 + 11 * edge["id"]
        edge["node_i"] = 101 + 7 * edge["node_i"]
        edge["node_j"] = 101 + 7 * edge["node_j"]
    data.pop("next_node_id", None)
    data.pop("next_edge_id", None)
    data["nodes"].reverse()
    graph = StructureGraph.from_dict(data)
    before = graph.to_dict()
    graph.save_json(tmp_path / "nested.json")
    saved = StructureGraph.load_json(tmp_path / "nested.json")
    extractor = SnapshotFeatureExtractor()
    f = extractor.extract_graph(saved)
    check(f, [12, 4], [24, 8], [1, 1], 16)
    direct = extractor.extract(points, edges)
    for key in direct:
        np.testing.assert_allclose(f[key], direct[key])
    assert graph.to_dict() == before
    assert saved.to_dict() == before
    # Feature arrays remain caller-owned and can be saved without a new schema.
    result = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in f.items()}
    assert json.loads(json.dumps(result, allow_nan=False))["pore_areas"] == [12, 4]


def test_generated_network_inside_frame_obeys_euler_and_area_partition():
    import fibernet as fn

    graph = fn.pattern_2d(
        unit="honeycomb", grid=(2, 2), box=(2, 2), radius=0.02, seed=23
    )
    positions, elements, _ = graph.to_numpy()
    positions = positions[:, :2] + [2, 2]
    points = positions.tolist()
    edges = elements.tolist()
    add_polygon(points, edges, square((0, 0), (8, 8)))
    topology = nx.Graph()
    topology.add_nodes_from(range(len(points)))
    topology.add_edges_from(edges)
    bounded_faces = (
        topology.number_of_edges()
        - topology.number_of_nodes()
        + nx.number_connected_components(topology)
    )
    f = SnapshotFeatureExtractor().extract(points, edges)
    assert f["pore_count"] == bounded_faces
    assert f["pore_areas"].sum() == pytest.approx(64)
    assert f["porosity"] == pytest.approx(1)
    assert np.all(f["pore_neighbor"] > 0)


def test_tree_roundoff_does_not_hide_slit_or_remove_area():
    points = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.5, 0.5],
            [0.436731312621503, 0.5534509022051359],
            [0.3502981912104216, 0.3484187434081586],
            [0.6457485310934107, 0.1683816859773713],
            [0.42280730777113906, 0.2256853159440465],
            [0.4364343861676986, 0.7837826035116133],
        ]
    )
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]] + [[4, i] for i in range(5, 10)]
    f = SnapshotFeatureExtractor().extract(points, edges)
    perimeter = 4 + 2 * np.linalg.norm(points[5:] - points[4], axis=1).sum()
    check(f, [1], [perimeter], [0], 1)
    assert f["pore_areas"][0] == 1


def test_contact_enabled_preserves_nested_pore_metrics():
    points, edges = nested()
    extractor = SnapshotFeatureExtractor()
    without = extractor.extract(points, edges)
    with_contact = SnapshotFeatureExtractor(include_contact=True).extract(points, edges)
    for key, value in without.items():
        np.testing.assert_array_equal(with_contact[key], value)
    check(with_contact, [12, 4], [24, 8], [1, 1], 16)


@pytest.mark.parametrize("shared_node", [False, True])
def test_point_touching_cycles_have_no_edge_adjacency(shared_node):
    points, edges = [], []
    add_polygon(points, edges, square((0, 0), (2, 2)))
    add_polygon(points, edges, square((2, 2), (3, 3)))
    if shared_node:
        edges = [(2 if a == 4 else a, 2 if b == 4 else b) for a, b in edges]
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [4, 1], [8, 4], [0, 0], 9)


def test_inner_loop_outward_branch_is_part_of_annular_boundary():
    points, edges = nested()
    points = np.vstack([points, [0.5, 1]])
    edges = np.vstack([edges, [4, 8]])
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [12, 4], [25, 8], [1, 1], 16)


def test_loop_in_bridged_annular_face_removes_area_from_that_face_only():
    points, edges = nested()
    points, edges = points.tolist(), edges.tolist()
    edges.append([0, 4])
    add_polygon(points, edges, square((3.25, 1.25), (3.75, 1.75)))
    f = SnapshotFeatureExtractor().extract(points, edges)
    check(f, [11.75, 4, 0.25], [26 + 2 * np.sqrt(2), 8, 2], [2, 1, 1], 16)
