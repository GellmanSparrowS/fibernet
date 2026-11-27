"""Versioned, geometry-independent topology of explicit physical edges."""
from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Tuple

from fibernet.core.structure_graph import StructureGraph


@dataclass(frozen=True)
class TopologyMetricsV1:
    """Counts and IDs for a finite undirected loop-free physical multigraph.

    Node-aligned tuples use sorted ``node_ids``; edge IDs retain graph insertion
    order. Component labels are their current minimum node ID, not persistent
    identities across edits. No geometric or mechanical validity is certified.
    """

    node_ids: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    component_by_node: Tuple[int, ...]
    component_sizes: Tuple[Tuple[int, int], ...]
    degree_physical_by_node: Tuple[int, ...]
    degree_unique_by_node: Tuple[int, ...]
    n_nodes: int
    n_physical_edges: int
    n_unique_connections: int
    n_components: int
    n_isolates: int
    cycle_rank_physical: int
    bridge_edge_ids: Tuple[int, ...]
    articulation_node_ids: Tuple[int, ...]
    largest_component_fraction: Optional[float]
    schema: str = field(default='fibernet.topology/1', init=False)


def analyze_topology(graph: StructureGraph, *, backend: str = "networkx") -> TopologyMetricsV1:
    """Analyze explicit adjacency without collapsing physical edge identity.

    All counts and degrees are dimensionless. Distances, weights, materials,
    radii, coordinates and periodic equivalences are deliberately not inferred.
    Self-loops, dangling endpoints and inconsistent entity IDs raise ValueError.
    Empty-graph largest-component fraction is undefined (None). The optional
    rustworkx backend computes the same schema; it does not replace StructureGraph.
    """
    if not isinstance(graph, StructureGraph):
        raise TypeError('analyze_topology requires StructureGraph')
    if backend not in ("networkx", "rustworkx"):
        raise ValueError("topology backend must be 'networkx' or 'rustworkx'")

    def valid_id(value):
        return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0

    if any(not valid_id(nid) or not valid_id(node.node_id) or node.node_id != nid
           for nid, node in graph.nodes.items()):
        raise ValueError('node IDs must be integers matching node records')
    node_ids = tuple(sorted(int(nid) for nid in graph.nodes))
    multiplicity = {}
    degree = dict.fromkeys(node_ids, 0)
    pairs = {}
    for eid, edge in graph.edges.items():
        if not valid_id(eid) or not valid_id(edge.edge_id) or edge.edge_id != eid:
            raise ValueError('edge IDs must be integers matching edge records')
        u, v = edge.node_i, edge.node_j
        if not valid_id(u) or not valid_id(v) or u not in degree or v not in degree:
            raise ValueError('edge endpoints must be existing integer node IDs')
        u, v = int(u), int(v)
        if u == v:
            raise ValueError('topology v1 does not support self-loops')
        pair = (min(u, v), max(u, v))
        pairs[eid] = pair
        multiplicity[pair] = multiplicity.get(pair, 0) + 1
        degree[u] += 1
        degree[v] += 1
    edge_ids = tuple(int(eid) for eid in graph.edges)
    if backend == "rustworkx":
        components, simple_bridges, cut_nodes, unique_degree = _rustworkx_topology(node_ids, multiplicity)
    else:
        components, simple_bridges, cut_nodes, unique_degree = _networkx_topology(node_ids, multiplicity)
    labels = {nid: label for label, component in components for nid in component}
    bridges = tuple(eid for eid in edge_ids
                    if pairs[eid] in simple_bridges and multiplicity[pairs[eid]] == 1)
    n, e, c = len(node_ids), len(edge_ids), len(components)
    return TopologyMetricsV1(
        node_ids=node_ids, edge_ids=edge_ids,
        component_by_node=tuple(labels[nid] for nid in node_ids),
        component_sizes=tuple((label, len(component)) for label, component in components),
        degree_physical_by_node=tuple(degree[nid] for nid in node_ids),
        degree_unique_by_node=unique_degree,
        n_nodes=n, n_physical_edges=e, n_unique_connections=len(multiplicity),
        n_components=c, n_isolates=sum(degree[nid] == 0 for nid in node_ids),
        cycle_rank_physical=e - n + c, bridge_edge_ids=bridges,
        articulation_node_ids=cut_nodes,
        largest_component_fraction=max((len(comp) for _, comp in components), default=0) / n if n else None,
    )


def _networkx_topology(node_ids, connections):
    import networkx as nx

    simple = nx.Graph()
    simple.add_nodes_from(node_ids)
    simple.add_edges_from(connections)
    return (sorted((min(c), c) for c in nx.connected_components(simple)),
            {tuple(sorted(pair)) for pair in nx.bridges(simple)},
            tuple(sorted(nx.articulation_points(simple))),
            tuple(simple.degree[nid] for nid in node_ids))


def _rustworkx_topology(node_ids, connections):
    try:
        import rustworkx as rx
    except ImportError as exc:
        raise ImportError("rustworkx topology backend requires the graph-fast extra on a supported Python and platform") from exc
    rows = {nid: i for i, nid in enumerate(node_ids)}
    simple = rx.PyGraph(multigraph=False, node_count_hint=len(node_ids),
                        edge_count_hint=len(connections))
    simple.add_nodes_from([None] * len(node_ids))
    simple.add_edges_from_no_data([(rows[u], rows[v]) for u, v in connections])
    return (sorted((node_ids[min(c)], tuple(node_ids[i] for i in c))
                   for c in rx.connected_components(simple)),
            {tuple(sorted((node_ids[i], node_ids[j]))) for i, j in rx.bridges(simple)},
            tuple(sorted(node_ids[i] for i in rx.articulation_points(simple))),
            tuple(simple.degree(i) for i in range(len(node_ids))))
