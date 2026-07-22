"""
Graph I/O for fiber networks.

Supports:
- NetworkX Graph ↔ FiberNetwork conversion
- JSON format: {nodes: [{id, pos}], links: [{source, target}]}
- NetworkX node_link_data format

The JSON format is compatible with the weld graph format used in
fiber network research workflows.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


def to_networkx(network: FiberNetwork, include_material: bool = True) -> "nx.Graph":
    """Convert FiberNetwork to NetworkX Graph.
    
    Creates a graph where:
    - Nodes = crosslink positions
    - Edges = fiber segments between crosslinks
    
    Parameters
    ----------
    network : FiberNetwork
        Input fiber network.
    include_material : bool
        If True, store material properties as node/edge attributes.
    
    Returns
    -------
    nx.Graph
        Graph with 'pos' node attribute (3D tuple) and optional
        'material' edge attribute.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required: pip install networkx")
    
    G = nx.Graph()
    
    # Add nodes at crosslink positions
    for i, cl in enumerate(network.crosslinks):
        G.add_node(i, pos=tuple(cl.position))
    
    # Build fiber crosslink mapping
    fiber_cls = {}  # fiber_idx -> [(cl_idx, param), ...]
    for cl_idx, cl in enumerate(network.crosslinks):
        for fi, param in [(cl.fiber_i, cl.param_i), (cl.fiber_j, cl.param_j)]:
            if fi not in fiber_cls:
                fiber_cls[fi] = []
            fiber_cls[fi].append((cl_idx, param))
    
    # Sort and add edges along each fiber
    for fi, cls_list in fiber_cls.items():
        cls_list.sort(key=lambda x: x[1])
        fiber = network.fibers[fi]
        
        for k in range(len(cls_list) - 1):
            ni, _ = cls_list[k]
            nj, _ = cls_list[k + 1]
            if ni != nj:
                edge_data = {}
                if include_material and fiber.material:
                    edge_data['youngs_modulus'] = fiber.material.youngs_modulus
                    edge_data['radius'] = fiber.radius
                G.add_edge(ni, nj, **edge_data)
    
    return G


def from_networkx(G: "nx.Graph", 
                  material: Optional[Material] = None,
                  pos_attr: str = "pos") -> FiberNetwork:
    """Convert NetworkX Graph to FiberNetwork.
    
    Interprets the graph as a fiber network where:
    - Each maximal path (sequence of degree-2 nodes) = one Fiber
    - Branch points (degree ≥ 3) and endpoints (degree 1) = Crosslinks
    
    For simple graphs, each edge becomes a fiber segment.
    
    Parameters
    ----------
    G : nx.Graph
        Graph with 'pos' node attribute.
    material : Material, optional
        Default material for all fibers. If None, uses generic polymer.
    pos_attr : str
        Name of the position attribute on nodes.
    
    Returns
    -------
    FiberNetwork
        Converted fiber network.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required: pip install networkx")
    
    if material is None:
        material = Material(name="generic", youngs_modulus=1e9, poissons_ratio=0.3, density=1000)
    
    from fibernet.core.network import Crosslink
    
    # Get node positions
    positions = {}
    for node in G.nodes():
        pos = G.nodes[node].get(pos_attr)
        if pos is None:
            raise ValueError(f"Node {node} missing '{pos_attr}' attribute")
        pos = np.array(pos, dtype=np.float64)
        if len(pos) == 2:
            pos = np.append(pos, 0.0)
        positions[node] = pos
    
    # Build fibers from connected paths
    # Simple approach: each edge → fiber segment, shared endpoints → crosslinks
    fibers = []
    crosslinks = []
    node_to_cl = {}  # node_id -> crosslink_index
    
    # Find nodes that are crosslinks (degree ≠ 2, or all nodes for simple approach)
    cl_nodes = set()
    for node in G.nodes():
        deg = G.degree(node)
        if deg != 2:  # endpoints and branch points
            cl_nodes.add(node)
    
    # For simple graphs, treat all nodes as potential crosslinks
    if not cl_nodes:
        cl_nodes = set(G.nodes())
    
    # Create crosslinks
    for node in cl_nodes:
        cl_idx = len(crosslinks)
        node_to_cl[node] = cl_idx
        crosslinks.append(
            Crosslink(position=positions[node], fiber_i=-1, fiber_j=-1, 
                      param_i=0.0, param_j=0.0)
        )
    
    # Create fibers along edges
    for u, v in G.edges():
        pos_u = positions[u]
        pos_v = positions[v]
        centerline = np.array([pos_u, pos_v])
        
        # Get radius from edge data if available
        radius = G.edges[u, v].get('radius', 0.1)
        E = G.edges[u, v].get('youngs_modulus', material.youngs_modulus)
        
        fiber_mat = Material(name=material.name, youngs_modulus=E, 
                            poissons_ratio=material.poissons_ratio, 
                            density=material.density)
        
        fi = len(fibers)
        fiber = Fiber(centerline=centerline, radius=radius, material=fiber_mat)
        fibers.append(fiber)
        
        # Link crosslinks
        if u in node_to_cl:
            crosslinks[node_to_cl[u]].fiber_i = fi
            crosslinks[node_to_cl[u]].param_i = 0.0
        if v in node_to_cl:
            cl = crosslinks[node_to_cl[v]]
            if cl.fiber_i == -1:
                cl.fiber_i = fi
                cl.param_i = 1.0
            else:
                cl.fiber_j = fi
                cl.param_j = 1.0
    
    # Clean up crosslinks with missing fiber references
    valid_cls = []
    for cl in crosslinks:
        if cl.fiber_i >= 0 and cl.fiber_j >= 0:
            valid_cls.append(cl)
    
    return FiberNetwork(fibers=fibers, crosslinks=valid_cls)


def save_graph_json(network_or_graph: Union[FiberNetwork, "nx.Graph"],
                    path: Union[str, Path],
                    indent: int = 2) -> None:
    """Save fiber network as JSON graph.
    
    Output format:
    {
        "nodes": [{"id": 0, "pos": [x, y, z]}, ...],
        "links": [{"source": 0, "target": 1}, ...]
    }
    
    Parameters
    ----------
    network_or_graph : FiberNetwork or nx.Graph
        Input structure.
    path : str or Path
        Output file path.
    indent : int
        JSON indentation.
    """
    if isinstance(network_or_graph, FiberNetwork):
        G = to_networkx(network_or_graph)
    else:
        G = network_or_graph
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "nodes": [
            {"id": node, "pos": list(G.nodes[node].get("pos", (0, 0, 0)))}
            for node in G.nodes()
        ],
        "links": [
            {"source": u, "target": v}
            for u, v in G.edges()
        ]
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_graph_json(path: Union[str, Path],
                    material: Optional[Material] = None) -> FiberNetwork:
    """Load fiber network from JSON graph.
    
    Accepts formats:
    1. Simple: {"nodes": [{"id", "pos"}], "links": [{"source", "target"}]}
    2. NetworkX: nx.node_link_data output
    
    Parameters
    ----------
    path : str or Path
        Input JSON file.
    material : Material, optional
        Default material for fibers.
    
    Returns
    -------
    FiberNetwork
        Loaded fiber network.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required: pip install networkx")
    
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    
    # Detect format
    if "nodes" in data and "links" in data:
        # Simple format
        G = nx.Graph()
        for node in data["nodes"]:
            pos = node.get("pos", (0, 0, 0))
            if len(pos) == 2:
                pos = list(pos) + [0.0]
            G.add_node(node["id"], pos=tuple(pos))
        for link in data["links"]:
            G.add_edge(link["source"], link["target"])
    else:
        # NetworkX node_link_data format
        G = nx.node_link_graph(data)
    
    return from_networkx(G, material=material)
