"""The 3D betweenness concentration feature agrees with pairwise inequality."""

from fractions import Fraction
from itertools import combinations

import networkx as nx
import numpy as np
import pytest

import fibernet as fn
from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D


def pairwise_gini(values):
    """Independent definition, without sorting or rank-weighted sums."""
    if not values or not sum(values):
        return 0.0
    return float(
        sum(abs(a - b) for a in values for b in values)
        / (2 * len(values) * sum(values))
    )


def shortest_path_counts(graph):
    """Exact unnormalized counts from exhaustive paths on these small graphs."""
    counts = {frozenset(edge): Fraction(0) for edge in graph.edges}
    for source, target in combinations(graph, 2):
        if not nx.has_path(graph, source, target):
            continue
        paths = list(nx.all_shortest_paths(graph, source, target, weight=None))
        for path in paths:
            for a, b in zip(path, path[1:]):
                counts[frozenset((a, b))] += Fraction(1, len(paths))
    return list(counts.values())


def positioned(graph):
    graph = graph.copy()
    for row, node in enumerate(graph):
        graph.nodes[node]["pos"] = (float(row), float(row**2), float(row % 2))
    return graph


@pytest.mark.parametrize(
    "graph, expected",
    [
        (nx.path_graph(4), Fraction(1, 15)),
        (nx.disjoint_union(nx.path_graph(4), nx.path_graph(2)), Fraction(9, 44)),
        (nx.cycle_graph(5), 0),
        (nx.star_graph(4), 0),
        (nx.complete_graph(4), 0),
        (nx.path_graph(2), 0),
        (nx.empty_graph(4), 0),
        (nx.empty_graph(1), 0),
        (nx.empty_graph(0), 0),
    ],
    ids=[
        "path4",
        "disconnected",
        "cycle",
        "star",
        "complete",
        "edge",
        "edgeless",
        "singleton",
        "empty",
    ],
)
def test_public_values_vector_and_batch_layout(graph, expected):
    graph = positioned(graph)
    before = nx.node_link_data(graph)
    extractor = GraphFeatureExtractor3D()
    name = "edge_betweenness_gini"
    oracle = pairwise_gini(shortest_path_counts(graph))
    assert oracle == pytest.approx(float(expected), abs=1e-15)
    actual = extractor.extract(graph)
    assert actual[name] == pytest.approx(oracle, abs=1e-15)
    assert -1e-15 <= actual[name] <= 1
    assert len(actual) == len(extractor.get_feature_names()) == 60
    assert extractor.get_feature_names()[20:22] == ["edge_betweenness_max", name]
    assert extractor.extract_vector(graph)[21] == pytest.approx(oracle, abs=1e-15)
    assert extractor.batch_extract([graph])[0][name] == pytest.approx(oracle, abs=1e-15)
    matrix = extractor.batch_extract_matrix([graph, graph])
    assert matrix.shape == (2, 60)
    np.testing.assert_allclose(matrix[:, 21], oracle, atol=1e-15)
    assert nx.node_link_data(graph) == before


@pytest.mark.parametrize("seed", range(20))
def test_exact_shortest_path_oracle_and_insertion_order(seed):
    graph = positioned(nx.gnp_random_graph(8, 0.25, seed=seed))
    # Neither arbitrary weights nor sparse labels define this unweighted metric.
    for row, (a, b) in enumerate(graph.edges):
        graph.edges[a, b]["weight"] = 100.0 if row % 2 else 0.001
    graph = nx.relabel_nodes(graph, {i: 101 + 7 * i for i in graph})
    reverse = nx.Graph()
    reverse.add_nodes_from(reversed(list(graph.nodes(data=True))))
    reverse.add_edges_from(
        (b, a, data) for a, b, data in reversed(list(graph.edges(data=True)))
    )
    expected = pairwise_gini(shortest_path_counts(graph))
    extractor = GraphFeatureExtractor3D()
    for candidate in (graph, reverse):
        actual = extractor.extract(candidate)
        assert actual["edge_betweenness_gini"] == pytest.approx(expected, abs=2e-15)
        expected_max = max(nx.edge_betweenness_centrality(graph).values(), default=0)
        assert actual["edge_betweenness_max"] == pytest.approx(expected_max)


def test_structure_graph_saved_workflow_and_sparse_ids(tmp_path):
    graph = fn.StructureGraph(dimension=3)
    for i in range(4):
        graph.add_node((i, i * i, i % 2))
    for i in range(3):
        graph.add_edge(i, i + 1, radius=0.1)
    data = graph.to_dict()
    for node in data["nodes"]:
        node["id"] = 101 + 7 * node["id"]
    for edge in data["edges"]:
        edge["id"] = 503 + 11 * edge["id"]
        edge["node_i"] = 101 + 7 * edge["node_i"]
        edge["node_j"] = 101 + 7 * edge["node_j"]
    data["nodes"].reverse()
    data["edges"].reverse()
    data.pop("next_node_id", None)
    data.pop("next_edge_id", None)
    graph = fn.StructureGraph.from_dict(data)
    graph.save_json(tmp_path / "graph.json")
    restored = fn.StructureGraph.load_json(tmp_path / "graph.json")
    before = restored.to_dict()
    extractor = fn.GraphFeatureExtractor3D()
    for candidate in (graph, restored, restored.to_networkx()):
        assert extractor.extract(candidate)["edge_betweenness_gini"] == pytest.approx(
            1 / 15
        )
    assert restored.to_dict() == before


def test_generated_network_persistence(tmp_path):
    graph = fn.pattern_3d(unit="bcc", grid=(2, 1, 1), box=(4, 2, 2), radius=0.05)
    graph.save_json(tmp_path / "bcc.json")
    restored = fn.StructureGraph.load_json(tmp_path / "bcc.json")
    expected = pairwise_gini(shortest_path_counts(restored.to_networkx()))
    assert expected > 0
    assert GraphFeatureExtractor3D().extract(restored)[
        "edge_betweenness_gini"
    ] == pytest.approx(expected)


def test_fiber_network_keeps_centerline_segment_conversion():
    from fibernet.core.fiber import Fiber
    from fibernet.core.network import FiberNetwork

    network = FiberNetwork(
        fibers=[
            Fiber(centerline=np.array([(0, 0, 0), (1, 0, 0), (1, 2, 0), (1, 2, 3)]))
        ]
    )
    before = network.fibers[0].centerline.copy()
    values = GraphFeatureExtractor3D().extract(network)
    assert values["n_node"] == 4
    assert values["n_edge"] == 3
    assert values["edge_betweenness_gini"] == pytest.approx(1 / 15)
    np.testing.assert_array_equal(network.fibers[0].centerline, before)


@pytest.mark.parametrize("node_count, called", [(499, True), (500, False)])
def test_existing_compute_cutoff_is_unchanged(monkeypatch, node_count, called):
    graph = nx.empty_graph(node_count)
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    calls = []

    def centrality(candidate):
        calls.append(candidate)
        return {(0, 1): 0.5, (1, 2): 2 / 3, (2, 3): 0.5}

    monkeypatch.setattr(nx, "edge_betweenness_centrality", centrality)
    actual = GraphFeatureExtractor3D()._graph_topology(graph)
    assert bool(calls) is called
    assert actual["edge_betweenness_gini"] == pytest.approx(1 / 15 if called else 0)
    assert actual["edge_betweenness_max"] == pytest.approx(2 / 3 if called else 0)


def test_existing_failure_fallback_is_unchanged(monkeypatch):
    def fail(graph):
        raise nx.NetworkXError("unavailable")

    monkeypatch.setattr(nx, "edge_betweenness_centrality", fail)
    actual = GraphFeatureExtractor3D().extract(positioned(nx.path_graph(4)))
    assert actual["edge_betweenness_gini"] == actual["edge_betweenness_max"] == 0
