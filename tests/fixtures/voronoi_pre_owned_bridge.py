"""Frozen original copy-based bridge; independent oracle, not production code."""
from typing import Optional
import numpy as np
from fibernet import StructureGraph, Material

def _bridge_small_components(
    graph: StructureGraph,
    min_component_size: int = 5,
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> StructureGraph:
    """Connect small isolated components to the largest component via bridge edges.

    After Voronoi clipping, a few edges may end up disconnected at tile
    boundaries. This finds components smaller than ``min_component_size``
    and adds a bridge edge from their nearest node to the main component.
    """
    g = graph.copy()
    if not g.nodes:
        return g

    # Build adjacency
    adj = {nid: set() for nid in g.nodes}
    for e in g.edges.values():
        adj[e.node_i].add(e.node_j)
        adj[e.node_j].add(e.node_i)

    # Find connected components via BFS
    visited = set()
    components = []
    for nid in g.nodes:
        if nid in visited:
            continue
        comp = set()
        queue = [nid]
        visited.add(nid)
        while queue:
            n = queue.pop(0)
            comp.add(n)
            for nb in adj.get(n, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)

    if len(components) <= 1:
        return g  # Already connected

    # Find the main (largest) component
    components.sort(key=len, reverse=True)
    main_comp = components[0]
    main_positions = np.array([g.nodes[nid].position for nid in main_comp])

    for small_comp in components[1:]:
        if len(small_comp) >= min_component_size:
            continue  # Skip large disconnected components (might be intentional)

        # Find the closest pair between small comp and main comp
        best_dist = float("inf")
        best_small_nid = None
        best_main_nid = None

        for snid in small_comp:
            spos = g.nodes[snid].position
            for mnid in main_comp:
                mpos = g.nodes[mnid].position
                d = np.linalg.norm(spos - mpos)
                if d < best_dist:
                    best_dist = d
                    best_small_nid = snid
                    best_main_nid = mnid

        if best_small_nid is not None and best_main_nid is not None:
            g.add_edge(best_small_nid, best_main_nid, radius=radius,
                       material=material or Material())

    return g
