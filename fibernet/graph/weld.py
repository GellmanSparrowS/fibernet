"""
Weld graph operations for fiber networks.

Implements edge intersection detection and node insertion to create
proper weld graphs from overlapping fiber networks.

Uses pure numpy for intersection detection (no Shapely dependency).
"""

import copy
import itertools
from typing import Tuple, Dict, List, Optional

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def _line_segment_intersection(p1, p2, p3, p4, eps=1e-10):
    """Find intersection point of two line segments.
    
    Parameters
    ----------
    p1, p2 : array-like
        Endpoints of first segment.
    p3, p4 : array-like
        Endpoints of second segment.
    eps : float
        Tolerance for parallel lines.
    
    Returns
    -------
    tuple or None
        Intersection point (x, y) if segments cross, None otherwise.
    """
    p1 = np.asarray(p1, dtype=np.float64)[:2]
    p2 = np.asarray(p2, dtype=np.float64)[:2]
    p3 = np.asarray(p3, dtype=np.float64)[:2]
    p4 = np.asarray(p4, dtype=np.float64)[:2]
    
    d1 = p2 - p1
    d2 = p4 - p3
    
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    
    if abs(cross) < eps:
        return None  # Parallel or collinear
    
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    s = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    
    if 0 < t < 1 and 0 < s < 1:
        return p1 + t * d1
    
    return None


def find_intersections(G: "nx.Graph", pos_attr: str = "pos") -> Dict[Tuple, List[Tuple]]:
    """Find all edge-edge intersection points in a graph.
    
    Parameters
    ----------
    G : nx.Graph
        Graph with position attributes on nodes.
    pos_attr : str
        Name of position attribute.
    
    Returns
    -------
    dict
        Mapping from edge (u, v) to list of intersection points [(x, y), ...].
        Only includes intersections between non-adjacent edges.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required: pip install networkx")
    
    pos = nx.get_node_attributes(G, pos_attr)
    edges = list(G.edges())
    intersections = {}
    
    for (u1, v1), (u2, v2) in itertools.combinations(edges, 2):
        # Skip adjacent edges
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
        
        p1 = pos[u1]
        p2 = pos[v1]
        p3 = pos[u2]
        p4 = pos[v2]
        
        ix = _line_segment_intersection(p1, p2, p3, p4)
        if ix is not None:
            ipt = tuple(ix)
            intersections.setdefault((u1, v1), []).append(ipt)
            intersections.setdefault((u2, v2), []).append(ipt)
    
    return intersections


def weld_graph(G: "nx.Graph", 
               pos_attr: str = "pos",
               tolerance: float = 1e-8,
               remove_self_loops: bool = True) -> "nx.Graph":
    """Detect edge intersections and insert weld nodes.
    
    This is the core weld operation. For every pair of crossing edges,
    a new node is inserted at the intersection point, splitting both
    edges into sub-segments. This creates a proper "welded" graph where
    all fiber crossings are connected nodes.
    
    Algorithm:
    1. Find all pairwise edge intersections (non-adjacent edges only)
    2. Insert new nodes at intersection points
    3. Split original edges at intersection points (sorted by projection)
    4. Optionally remove self-loops (zero-length edges)
    
    Parameters
    ----------
    G : nx.Graph
        Input graph with 'pos' attribute on nodes.
    pos_attr : str
        Name of position attribute.
    tolerance : float
        Minimum distance to consider two points as distinct.
    remove_self_loops : bool
        If True, remove edges where both endpoints have same position.
    
    Returns
    -------
    nx.Graph
        Welded graph with intersection nodes inserted.
    
    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_node(1, pos=(10, 10))
    >>> G.add_node(2, pos=(0, 10))
    >>> G.add_node(3, pos=(10, 0))
    >>> G.add_edge(0, 1)  # diagonal
    >>> G.add_edge(2, 3)  # crosses first diagonal
    >>> G_welded = weld_graph(G)
    >>> G_welded.number_of_nodes()  # 5 (4 original + 1 intersection)
    5
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required: pip install networkx")
    
    G = copy.deepcopy(G)
    pos = nx.get_node_attributes(G, pos_attr)
    edges = list(G.edges(data=True))
    
    new_nodes = {}
    intersections = {}
    
    # Step 1: Find all edge-edge intersections
    for (u1, v1, d1), (u2, v2, d2) in itertools.combinations(edges, 2):
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
        
        p1 = np.array(pos[u1])
        p2 = np.array(pos[v1])
        p3 = np.array(pos[u2])
        p4 = np.array(pos[v2])
        
        ix = _line_segment_intersection(p1, p2, p3, p4, eps=tolerance)
        if ix is not None:
            ipt = tuple(ix)
            node_name = f"IX_{ipt[0]:.8f}_{ipt[1]:.8f}"
            if node_name not in G.nodes:
                if len(p1) == 3:
                    new_nodes[node_name] = {pos_attr: (ipt[0], ipt[1], 0.0)}
                else:
                    new_nodes[node_name] = {pos_attr: ipt}
            intersections.setdefault((u1, v1), []).append(ipt)
            intersections.setdefault((u2, v2), []).append(ipt)
    
    # Step 2: Add intersection nodes
    for node_name, attrs in new_nodes.items():
        G.add_node(node_name, **attrs)
    
    # Step 3: Split edges at intersection points
    new_edges = []
    for u, v, data in edges:
        points = intersections.get((u, v), [])
        if not points:
            new_edges.append((u, v, data))
            continue
        
        p_u = np.array(pos[u])[:2]
        p_v = np.array(pos[v])[:2]
        direction = p_v - p_u
        
        # Sort by projection along edge
        def proj_dist(pt):
            pt = np.array(pt)
            return np.dot(pt - p_u, direction) / np.dot(direction, direction)
        
        sorted_points = sorted(points, key=proj_dist)
        
        prev_node = u
        for pt in sorted_points:
            ix_name = f"IX_{pt[0]:.8f}_{pt[1]:.8f}"
            new_edges.append((prev_node, ix_name, {}))
            prev_node = ix_name
        new_edges.append((prev_node, v, {}))
    
    # Step 4: Rebuild graph
    G.remove_edges_from(list(G.edges()))
    for u, v, data in new_edges:
        G.add_edge(u, v, **data)
    
    # Step 5: Remove self-loops
    if remove_self_loops:
        pos_updated = nx.get_node_attributes(G, pos_attr)
        edges_to_remove = []
        for u, v in G.edges():
            pu = np.array(pos_updated[u])
            pv = np.array(pos_updated[v])
            if np.linalg.norm(pu - pv) < tolerance:
                edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)
    
    return G


def merge_coincident_nodes(G: "nx.Graph", 
                          tolerance: float = 0.5,
                          pos_attr: str = "pos") -> "nx.Graph":
    """Merge nodes that are within tolerance distance of each other.
    
    This implements the "weld threshold" behavior where nearby nodes
    are merged into a single node.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph.
    tolerance : float
        Distance threshold for merging.
    pos_attr : str
        Name of position attribute.
    
    Returns
    -------
    nx.Graph
        Graph with coincident nodes merged.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required: pip install networkx")
    
    pos = nx.get_node_attributes(G, pos_attr)
    nodes = list(G.nodes())
    
    # Build merge mapping
    merge_map = {}
    merged = set()
    
    for i, n1 in enumerate(nodes):
        if n1 in merged:
            continue
        p1 = np.array(pos[n1])
        merge_map[n1] = n1
        
        for n2 in nodes[i+1:]:
            if n2 in merged:
                continue
            p2 = np.array(pos[n2])
            if np.linalg.norm(p1 - p2) <= tolerance:
                merge_map[n2] = n1
                merged.add(n2)
    
    # Build new graph
    new_G = nx.Graph()
    for node in G.nodes():
        target = merge_map.get(node, node)
        if target not in new_G.nodes:
            new_G.add_node(target, **G.nodes[node])
    
    for u, v in G.edges():
        new_u = merge_map.get(u, u)
        new_v = merge_map.get(v, v)
        if new_u != new_v:
            new_G.add_edge(new_u, new_v)
    
    return new_G
