"""Independent finite multigraph oracles and frozen legacy numerical limits."""
import copy
from dataclasses import FrozenInstanceError

import networkx as nx
import numpy as np
import pytest

from fibernet.analysis import analyze_topology, TopologyMetricsV1
from fibernet.core.structure_graph import StructureGraph


def graph(nodes=(10, 30, 70, 90), edges=((90, 10, 30), (7, 30, 10), (50, 30, 70))):
    return StructureGraph.from_dict({'nodes': [dict(id=n, position=[i, 0, 0])
        for i, n in enumerate(nodes)], 'edges': [dict(id=e, node_i=u, node_j=v)
        for e, u, v in edges]})


def component_count(nodes, edges):
    # Independent small-graph oracle: repeated set closure, no NX algorithm.
    remaining = set(nodes)
    count = 0
    while remaining:
        reached = {next(iter(remaining))}
        previous = -1
        while previous != len(reached):
            previous = len(reached)
            for u, v in edges:
                if u in reached or v in reached:
                    reached.update((u, v))
        remaining.difference_update(reached)
        count += 1
    return count


def test_inventory_and_immutable_version():
    g = graph()
    before = copy.deepcopy(g.to_dict())
    m = analyze_topology(g)
    assert isinstance(m, TopologyMetricsV1)
    assert m.schema == 'fibernet.topology/1'
    assert m.node_ids == (10, 30, 70, 90)
    assert m.edge_ids == (90, 7, 50)
    assert (m.n_nodes, m.n_physical_edges, m.n_unique_connections,
            m.n_components, m.n_isolates) == (4, 3, 2, 2, 1)
    assert m.component_by_node == (10, 10, 10, 90)
    assert m.component_sizes == ((10, 3), (90, 1))
    assert m.degree_physical_by_node == (2, 3, 1, 0)
    assert m.degree_unique_by_node == (1, 2, 1, 0)
    assert m.cycle_rank_physical == 1
    assert m.bridge_edge_ids == (50,)
    assert m.articulation_node_ids == (30,)
    assert m.largest_component_fraction == .75
    assert g.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        m.n_nodes = 8


@pytest.mark.parametrize('edges', [(), ((4, 10, 30),),
    ((4, 10, 30), (9, 30, 10)),
    ((4, 10, 30), (9, 30, 70), (2, 70, 10)),
    ((4, 10, 30), (9, 30, 10), (2, 30, 70), (1, 70, 90)),
    ((4, 10, 30), (9, 30, 70), (2, 70, 10),
     (1, 90, 100), (3, 100, 110), (8, 110, 90))])
def test_bridges_against_independent_single_physical_edge_deletion(edges):
    nodes = (10, 30, 70, 90, 100, 110)
    m = analyze_topology(graph(nodes, edges))
    base = component_count(nodes, [(u, v) for _, u, v in edges])
    expected = tuple(e for e, _, _ in edges if component_count(nodes,
        [(u, v) for other, u, v in edges if other != e]) > base)
    assert m.bridge_edge_ids == expected
    assert m.n_components == base
    assert m.cycle_rank_physical == len(edges) - len(nodes) + base


@pytest.mark.parametrize('nodes', [(), (91,), (70, 10)])
def test_empty_and_isolates(nodes):
    m = analyze_topology(graph(nodes, ()))
    assert m.n_components == len(nodes) == m.n_isolates
    assert m.cycle_rank_physical == 0
    assert m.degree_physical_by_node == (0,) * len(nodes)
    assert m.largest_component_fraction == (1 / len(nodes) if nodes else None)


def test_direction_geometry_material_and_insertion_invariance():
    a = graph()
    b = graph((90, 70, 30, 10), ((50, 70, 30), (7, 10, 30), (90, 30, 10)))
    b.box_size = [100, 100, 100]
    for node in b.nodes.values():
        node.position *= 100
    for edge in b.edges.values():
        edge.radius = 12
        edge.internal_points = np.array([[13., 11., 19.]])
        edge.metadata['weight'] = 123
    x, y = analyze_topology(a), analyze_topology(b)
    for key in x.__dataclass_fields__:
        if key != 'edge_ids':
            assert getattr(x, key) == getattr(y, key)
    assert y.edge_ids == (50, 7, 90)
    a.remove_edge(90)
    m = analyze_topology(a)
    assert m.cycle_rank_physical == 0 and m.n_components == 2
    assert m.bridge_edge_ids == (7, 50)


def test_type_and_corrupt_topology_rejection():
    with pytest.raises(TypeError, match='StructureGraph'):
        analyze_topology(nx.Graph())
    g = graph()
    assert g.add_edge(10, 10) == -1  # Core rejects new self-loops.
    g.edges[90].node_j = 10  # Defensive validation of mutable records.
    with pytest.raises(ValueError, match='self-loops'):
        analyze_topology(g)
    g.edges[90].node_j = 999
    with pytest.raises(ValueError, match='endpoints'):
        analyze_topology(g)
    g = graph()
    g.edges[90].edge_id = 1
    with pytest.raises(ValueError, match='edge IDs'):
        analyze_topology(g)
    g = graph()
    g.nodes[10].node_id = 1
    with pytest.raises(ValueError, match='node IDs'):
        analyze_topology(g)



@pytest.mark.parametrize('bad', [True, 1.0, -1])
def test_record_identity_requires_nonnegative_strict_integer(bad):
    g = graph((0, 1), ((1, 0, 1),))
    g.nodes[1].node_id = bad
    with pytest.raises(ValueError, match='node IDs'):
        analyze_topology(g)
    g = graph((0, 1), ((1, 0, 1),))
    g.edges[1].edge_id = bad
    with pytest.raises(ValueError, match='edge IDs'):
        analyze_topology(g)


def test_negative_mapping_ids_and_endpoints_rejected():
    g = graph()
    node = g.nodes.pop(10)
    node.node_id = -1
    g.nodes[-1] = node
    with pytest.raises(ValueError, match='node IDs'):
        analyze_topology(g)
    g = graph()
    edge = g.edges.pop(90)
    edge.edge_id = -1
    g.edges[-1] = edge
    with pytest.raises(ValueError, match='edge IDs'):
        analyze_topology(g)
    g = graph()
    g.edges[90].node_i = -1
    with pytest.raises(ValueError, match='endpoints'):
        analyze_topology(g)


@pytest.mark.parametrize('bad', [True, 1.0, np.bool_(True)])
def test_equal_looking_endpoint_and_mapping_ids_rejected(bad):
    g = graph((0, 1), ((1, 0, 1),))
    g.edges[1].node_j = bad
    with pytest.raises(ValueError, match='endpoints'):
        analyze_topology(g)
    g = graph((0, 1), ((1, 0, 1),))
    node = g.nodes.pop(1)
    g.nodes[bad] = node
    with pytest.raises(ValueError, match='node IDs'):
        analyze_topology(g)
    g = graph((0, 1), ((1, 0, 1),))
    edge = g.edges.pop(1)
    g.edges[bad] = edge
    with pytest.raises(ValueError, match='edge IDs'):
        analyze_topology(g)


def test_numpy_integral_ids_normalize_to_python_ints():
    g = graph((0, 1), ((1, 0, 1),))
    for nid in list(g.nodes):
        node = g.nodes.pop(nid)
        node.node_id = np.int64(nid)
        g.nodes[np.int64(nid)] = node
    edge = g.edges.pop(1)
    edge.edge_id = np.uint64(1)
    edge.node_i, edge.node_j = np.int32(0), np.uint32(1)
    g.edges[np.uint64(1)] = edge
    m = analyze_topology(g)
    assert m.node_ids == (0, 1) and m.edge_ids == (1,)
    assert all(type(value) is int for value in m.node_ids + m.edge_ids
               + m.component_by_node + m.bridge_edge_ids)
