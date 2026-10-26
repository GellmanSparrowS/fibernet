"""Connectivity preserves graph semantics while scanning physical edges once."""

import networkx as nx
import pytest

from fibernet.core.structure_graph import StructureGraph


def _oracle(graph):
    network = nx.MultiGraph()
    network.add_nodes_from(graph.nodes)
    network.add_edges_from((edge.node_i, edge.node_j, edge.edge_id)
                           for edge in graph.edges.values())
    return network


@pytest.mark.parametrize("node_count,edge_pairs", [
    (0, []),
    (1, []),
    (4, []),
    (4, [(0, 1), (1, 0), (1, 2)]),
    (4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1)]),
    (6, [(0, 2), (2, 4), (1, 3)]),
])
def test_connectivity_matches_independent_multigraph_components(node_count, edge_pairs):
    graph = StructureGraph()
    for index in range(node_count):
        graph.add_node([index, 0])
    for a, b in edge_pairs:
        graph.add_edge(a, b, allow_parallel=True)
    network = _oracle(graph)
    assert graph.connected_components() == list(nx.connected_components(network))
    expected_connected = nx.is_connected(network) if node_count else True
    assert graph.is_connected() == expected_connected


def test_sparse_ids_and_node_insertion_order_are_preserved():
    graph = StructureGraph.from_dict({
        "nodes": [{"id": node_id, "position": [index, 0]}
                  for index, node_id in enumerate((70, 2, 100, 8, 1))],
        "edges": [{"id": 90, "node_i": 70, "node_j": 8},
                  {"id": 3, "node_i": 8, "node_j": 70},
                  {"id": 999, "node_i": 2, "node_j": 100}],
    })
    expected = [{70, 8}, {2, 100}, {1}]
    assert graph.connected_components() == expected
    assert graph.connected_components() == list(nx.connected_components(_oracle(graph)))
    assert not graph.is_connected()


def test_each_operation_rebuilds_adjacency_after_graph_edits():
    graph = StructureGraph()
    for index in range(4):
        graph.add_node([index, 0])
    first = graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    assert graph.connected_components() == [{0, 1, 2}, {3}]
    assert not graph.is_connected()
    graph.add_edge(2, 3)
    assert graph.is_connected()
    graph.remove_edge(first)
    assert graph.connected_components() == [{0}, {1, 2, 3}]
    graph.remove_node(0)
    assert graph.is_connected()
    graph.remove_node(2)
    assert graph.connected_components() == [{1}, {3}]
    assert not graph.is_connected()


class CountingEdges(dict):
    def __init__(self, entries):
        super().__init__(entries)
        self.scans = 0
        self.visits = 0

    def values(self):
        self.scans += 1
        for edge in super().values():
            self.visits += 1
            yield edge


@pytest.mark.parametrize("operation", ["is_connected", "connected_components"])
@pytest.mark.parametrize("node_count", [16, 128, 512])
def test_connectivity_scans_each_physical_edge_once(operation, node_count):
    graph = StructureGraph()
    for index in range(node_count):
        graph.add_node([index, 0])
    for index in range(node_count - 1):
        graph.add_edge(index, index + 1)
    graph.add_edge(0, 1, allow_parallel=True)
    graph._edges = CountingEdges(graph.edges)
    result = getattr(graph, operation)()
    assert result is True if operation == "is_connected" else result == [set(graph.nodes)]
    assert graph._edges.scans == 1
    assert graph._edges.visits == graph.num_edges
