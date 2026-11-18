"""Per-member cylinder estimates retain the radius/length association."""

import math

import networkx as nx
import numpy as np
import pytest

import fibernet as fn
from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D

FIELDS = (
    "volume_fraction",
    "surface_area_approx",
    "specific_surface",
    "sphericity",
    "compactness",
)


def expected_metrics(graph):
    """Independent scalar sums, without extractor conversion or array ordering."""
    volume, area = 0.0, 0.0
    for u, v, data in graph.edges(data=True):
        length = math.dist(graph.nodes[u]["pos"], graph.nodes[v]["pos"])
        radius = float(data.get("radius", 0.05))
        volume += math.pi * radius * radius * length
        area += 2 * math.pi * radius * length
    positions = np.array([d["pos"] for _, d in graph.nodes(data=True)])
    bbox = math.prod(np.ptp(positions, axis=0))
    sphere_area = 4 * math.pi * (3 * volume / (4 * math.pi)) ** (2 / 3)
    return dict(
        volume_fraction=min(volume / max(bbox, 1e-12), 1.0),
        surface_area_approx=area,
        specific_surface=area / max(volume, 1e-12),
        sphericity=sphere_area / max(area, 1e-12),
        compactness=volume / max(area**1.5, 1e-12),
    )


def nonuniform_graph(radii=(0.01, 0.03, 0.06), reverse=False):
    graph = nx.Graph()
    nodes = [(40, (0, 0, 0)), (10, (1, 0, 0)), (90, (1, 2, 0)), (20, (1, 2, 4))]
    graph.add_nodes_from(
        (i, {"pos": p}) for i, p in (nodes[::-1] if reverse else nodes)
    )
    edges = [(40, 10), (10, 90), (90, 20)]
    records = list(zip(edges, radii))
    for (u, v), radius in (records[::-1] if reverse else records):
        data = {} if radius is None else {"radius": radius}
        graph.add_edge(v if reverse else u, u if reverse else v, **data)
    return graph


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "radii",
    [
        (0.01, 0.03, 0.06),
        (0.06, 0.03, 0.01),
        (None, 0.03, None),
        (0.05,) * 3,
        (np.int64(2**32),) * 3,
    ],
)
def test_independent_member_sums(radii, reverse):
    graph = nonuniform_graph(radii, reverse)
    actual = GraphFeatureExtractor3D().extract(graph)
    for name, expected in expected_metrics(graph).items():
        assert actual[name] == pytest.approx(expected, rel=3e-14)
    assert set(actual) == set(GraphFeatureExtractor3D.FEATURE_NAMES)


def test_radius_length_association_is_not_averaged():
    extractor = GraphFeatureExtractor3D()
    thin_long = extractor.extract(nonuniform_graph((0.06, 0.03, 0.01)))
    thick_long = extractor.extract(nonuniform_graph((0.01, 0.03, 0.06)))
    assert thick_long["volume_fraction"] > thin_long["volume_fraction"]
    assert thick_long["surface_area_approx"] > thin_long["surface_area_approx"]


def test_equal_lengths_still_require_mean_square_radius():
    graph = nonuniform_graph()
    for node, pos in zip(graph, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]):
        graph.nodes[node]["pos"] = pos
    actual = GraphFeatureExtractor3D().extract(graph)
    assert actual["volume_fraction"] == pytest.approx(
        math.pi * (0.01**2 + 0.03**2 + 0.06**2)
    )


@pytest.mark.parametrize("unit", ["octet", "bcc", "cubic"])
def test_native_generated_members_and_json_roundtrip(unit, tmp_path):
    graph = fn.pattern_3d(
        unit=unit, box=(10.0, 10.0, 10.0), grid=(1, 1, 1), radius=0.1, n_pts_per_side=0
    )
    graph.save_json(tmp_path / "graph.json")
    restored = fn.StructureGraph.load_json(tmp_path / "graph.json")
    nx_graph = nx.Graph()
    for nid, node in restored.nodes.items():
        nx_graph.add_node(nid, pos=node.position)
    for edge in restored.edges.values():
        nx_graph.add_edge(edge.node_i, edge.node_j, radius=edge.radius)
    for candidate in (graph, restored, nx_graph):
        actual = GraphFeatureExtractor3D().extract(candidate)
        for name, expected in expected_metrics(nx_graph).items():
            assert actual[name] == pytest.approx(expected, rel=3e-14)


@pytest.mark.parametrize("factor", [0.5, 2.0, 3.0])
def test_geometric_similarity(factor):
    graph = nonuniform_graph()
    extractor = GraphFeatureExtractor3D()
    baseline = extractor.extract(graph)
    for _, data in graph.nodes(data=True):
        data["pos"] = np.asarray(data["pos"]) * factor
    for _, _, data in graph.edges(data=True):
        data["radius"] *= factor
    actual = extractor.extract(graph)
    powers = (0, 2, -1, 0, 0)
    for name, power in zip(FIELDS, powers):
        assert actual[name] == pytest.approx(baseline[name] * factor**power, rel=3e-14)


def test_edgeless_positions_retain_zero_estimates():
    graph = nonuniform_graph()
    graph.remove_edges_from(list(graph.edges))
    actual = GraphFeatureExtractor3D().extract(graph)
    assert all(actual[name] == 0.0 for name in FIELDS)


def test_geometry_analysis_does_not_mutate_input():
    graph = nonuniform_graph()
    before = nx.node_link_data(graph)
    GraphFeatureExtractor3D().extract(graph)
    assert nx.node_link_data(graph) == before
