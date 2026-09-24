"""Parallel fiber contract for StructureGraph and its conversions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fibernet.core.structure_graph import StructureGraph


def test_default_deduplicates():
    graph = StructureGraph(dimension=2)
    a = graph.add_node([0, 0])
    b = graph.add_node([1, 0])
    first = graph.add_edge(a, b)
    assert graph.add_edge(b, a) == first
    assert graph.num_edges == 1


def test_parallel_roundtrip():
    graph = StructureGraph(dimension=2)
    a = graph.add_node([0, 0])
    b = graph.add_node([1, 0])
    first = graph.add_edge(a, b)
    second = graph.add_edge(b, a, allow_parallel=True)
    assert second != first and graph.num_edges == 2
    assert graph.edge_array().tolist() == [[a, b], [b, a]]
    assert graph.copy().edge_array().tolist() == graph.edge_array().tolist()
    assert StructureGraph.from_dict(graph.to_dict()).edge_array().tolist() == graph.edge_array().tolist()
    networkx_graph = graph.to_networkx()
    assert networkx_graph.is_multigraph() and networkx_graph.number_of_edges() == 2
    assert StructureGraph.from_networkx(networkx_graph).num_edges == 2
    graph.remove_edge(first)
    assert graph.add_edge(a, b) == second
    graph.remove_edge(second)
    assert graph.num_edges == 0
    assert graph.add_edge(a, b) >= 0


if __name__ == '__main__':
    test_default_deduplicates()
    test_parallel_roundtrip()
    print('[test_structure_parallel] PASS')
