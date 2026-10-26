"""The historical 94-feature schema must not silently merge physical fibers."""

import networkx as nx
import numpy as np
import pytest

from fibernet.analysis.graph_features import GraphFeatureExtractor
from fibernet.core.structure_graph import StructureGraph


@pytest.mark.parametrize("reverse", [False, True])
def test_physical_parallel_edges_are_rejected_without_modifying_graph(reverse):
    graph = StructureGraph(dimension=2)
    left = graph.add_node([0., 0.])
    right = graph.add_node([1., 0.])
    graph.add_edge(left, right, radius=0.1)
    graph.add_edge(right if reverse else left, left if reverse else right,
                   radius=0.2, allow_parallel=True)
    assert graph.num_edges == 2
    positions = graph.node_positions().copy()
    with pytest.raises(ValueError, match="parallel physical edges"):
        GraphFeatureExtractor(canvas_size=32).extract(graph)
    assert graph.num_edges == 2
    np.testing.assert_array_equal(graph.node_positions(), positions)


@pytest.mark.parametrize("graph_type", [nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph])
def test_unsupported_networkx_topology_has_an_explicit_error(graph_type):
    graph = graph_type()
    graph.add_node(1, pos=(0., 0.))
    graph.add_node(2, pos=(1., 0.))
    graph.add_edge(1, 2)
    with pytest.raises(ValueError, match="simple undirected graph"):
        GraphFeatureExtractor(canvas_size=32).extract(graph)


def test_simple_graph_preserves_named_and_ordered_feature_schema():
    graph = nx.Graph()
    for node, position in enumerate(((0., 0.), (1., 0.), (0., 1.))):
        graph.add_node(node, pos=position)
    graph.add_edges_from(((0, 1), (1, 2), (2, 0)))
    extractor = GraphFeatureExtractor(canvas_size=32)
    values = extractor.extract(graph, allow_unavailable=True)
    vector = extractor.extract_vector(graph, allow_unavailable=True)
    assert len(values) == len(extractor.FEATURE_COLS) == 94
    assert values["n_edge"] == 3
    assert values["total_length"] == pytest.approx(2 + np.sqrt(2))
    np.testing.assert_allclose(vector, [values[key] for key in extractor.FEATURE_COLS])
