"""The 60-feature schema must not silently merge physical fibers."""

import networkx as nx
import numpy as np
import pytest

from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D
from fibernet.core.structure_graph import StructureGraph


@pytest.mark.parametrize("reverse", [False, True])
def test_physical_parallel_edges_are_rejected_without_modifying_graph(reverse):
    graph = StructureGraph(dimension=3)
    left = graph.add_node((0., 0., 0.))
    right = graph.add_node((1., 2., 2.))
    graph.add_edge(left, right, radius=0.1)
    graph.add_edge(right if reverse else left, left if reverse else right,
                   radius=0.2, allow_parallel=True)
    positions = graph.node_positions().copy()
    records = {eid: (edge.node_i, edge.node_j, edge.radius)
               for eid, edge in graph.edges.items()}

    with pytest.raises(ValueError, match="parallel physical edges"):
        GraphFeatureExtractor3D().extract(graph)

    assert graph.num_nodes == 2
    assert graph.num_edges == 2
    assert {eid: (edge.node_i, edge.node_j, edge.radius)
            for eid, edge in graph.edges.items()} == records
    np.testing.assert_array_equal(graph.node_positions(), positions)


@pytest.mark.parametrize("graph_type", [nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph])
def test_unsupported_networkx_topology_has_an_explicit_error(graph_type):
    graph = graph_type()
    graph.add_node(1, pos=(0., 0., 0.))
    graph.add_node(2, pos=(1., 2., 2.))
    graph.add_edge(1, 2)

    with pytest.raises(ValueError, match="simple undirected graph"):
        GraphFeatureExtractor3D().extract(graph)


@pytest.mark.parametrize("use_structure_graph", [False, True])
def test_simple_3d_graph_preserves_named_and_ordered_feature_schema(use_structure_graph):
    positions = ((0., 0., 0.), (1., 0., 0.), (0., 2., 0.), (0., 0., 3.))
    edges = ((0, 1), (1, 2), (2, 0), (0, 3))
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
        graph.add_edges_from(edges, radius=0.1)

    extractor = GraphFeatureExtractor3D()
    values = extractor.extract(graph)
    vector = extractor.extract_vector(graph)

    assert len(values) == len(extractor.FEATURE_NAMES) == 60
    assert set(values) == set(extractor.FEATURE_NAMES)
    assert values["n_node"] == 4
    assert values["n_edge"] == 4
    assert values["total_length"] == pytest.approx(6 + np.sqrt(5))
    assert values["mean_degree"] == pytest.approx(2.)
    np.testing.assert_allclose(vector, [values[key] for key in extractor.FEATURE_NAMES],
                               rtol=1e-12, atol=1e-12)
