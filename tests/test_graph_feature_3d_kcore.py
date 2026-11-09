"""Public 3D k-core values use unweighted degree over all graph nodes."""

import networkx as nx
import numpy as np
import pytest

from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D
from fibernet.core.structure_graph import StructureGraph


@pytest.mark.parametrize("use_structure_graph", [False, True])
@pytest.mark.parametrize(
    "node_count, edges, max_core, mean_degree, core_ratio",
    [
        (4, list(nx.complete_graph(4).edges()), 3.0, 3.0, 1.0),
        (3, [(0, 1), (1, 2), (2, 0)], 2.0, 2.0, 1.0),
        (4, [(0, 1), (1, 2), (2, 3)], 1.0, 1.5, 2.0 / 3.0),
        (5, list(nx.complete_graph(4).edges()), 3.0, 2.4, 1.25),
        (4, [], 0.0, 0.0, 0.0),
        (1, [], 0.0, 0.0, 0.0),
        (0, [], 0.0, 0.0, 0.0),
    ],
    ids=["K4", "triangle", "path", "K4-with-isolate", "edgeless", "singleton", "empty"],
)
def test_public_kcore_values_and_vector_order(
    use_structure_graph, node_count, edges, max_core, mean_degree, core_ratio
):
    positions = [(float(i), float(i * i), float(i % 2)) for i in range(node_count)]
    if use_structure_graph:
        graph = StructureGraph(dimension=3)
        for position in positions:
            graph.add_node(position)
        for left, right in edges:
            graph.add_edge(left, right, radius=0.1)
    else:
        graph = nx.Graph()
        for node, position in enumerate(positions):
            graph.add_node(node, pos=position)
        graph.add_edges_from(edges, radius=0.1, weight=17.0)

    extractor = GraphFeatureExtractor3D()
    features = extractor.extract(graph)
    vector = extractor.extract_vector(graph)
    names = extractor.get_feature_names()

    assert len(features) == len(names) == 60
    assert set(features) == set(names)
    assert names[18:20] == ["max_k_core", "kcore_frac"]
    assert vector.shape == (60,)
    for name, expected in (
        ("max_k_core", max_core),
        ("mean_degree", mean_degree),
        ("kcore_frac", core_ratio),
    ):
        assert np.isfinite(features[name])
        assert features[name] == pytest.approx(expected)
        assert vector[names.index(name)] == pytest.approx(expected)

    assert features["n_node"] == node_count
    assert features["n_edge"] == len(edges)
