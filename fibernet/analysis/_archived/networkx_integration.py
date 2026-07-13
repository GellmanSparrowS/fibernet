"""
NetworkX Integration for Advanced Topology Analysis

Provides integration with networkx library for:
- Advanced graph algorithms
- Community detection
- Centrality measures
- Network visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from fibernet.core.network import FiberNetwork


def to_networkx(network: FiberNetwork, weighted: bool = True):
    """Convert FiberNetwork to networkx Graph.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to convert.
    weighted : bool
        If True, add edge weights based on fiber properties.
    
    Returns
    -------
    networkx.Graph
        NetworkX graph representation.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx required for graph analysis. "
            "Install with: pip install networkx"
        )
    
    G = nx.Graph()
    
    # Add nodes
    node_id = 0
    node_positions = {}
    
    for fiber in network.fibers:
        for pt in fiber.centerline:
            key = tuple(np.round(pt, 6))
            if key not in node_positions:
                G.add_node(node_id, pos=pt)
                node_positions[key] = node_id
                node_id += 1
    
    # Add edges
    for fiber in network.fibers:
        pts = fiber.centerline
        for i in range(len(pts) - 1):
            key1 = tuple(np.round(pts[i], 6))
            key2 = tuple(np.round(pts[i+1], 6))
            
            if key1 in node_positions and key2 in node_positions:
                n1 = node_positions[key1]
                n2 = node_positions[key2]
                
                if weighted:
                    weight = fiber.length / len(pts)
                    G.add_edge(n1, n2, weight=weight, radius=fiber.radius)
                else:
                    G.add_edge(n1, n2)
    
    return G


def compute_centrality(
    network: FiberNetwork,
    centrality_type: str = 'betweenness',
) -> Dict[int, float]:
    """Compute node centrality measures.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    centrality_type : str
        Type of centrality: 'degree', 'betweenness', 'closeness', 'eigenvector'.
    
    Returns
    -------
    Dict[int, float]
        Centrality values for each node.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required")
    
    G = to_networkx(network)
    
    if centrality_type == 'degree':
        return nx.degree_centrality(G)
    elif centrality_type == 'betweenness':
        return nx.betweenness_centrality(G)
    elif centrality_type == 'closeness':
        return nx.closeness_centrality(G)
    elif centrality_type == 'eigenvector':
        try:
            return nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            return {n: 0.0 for n in G.nodes()}
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")


def detect_communities(
    network: FiberNetwork,
    algorithm: str = 'louvain',
) -> Dict[int, int]:
    """Detect communities/clusters in the network.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    algorithm : str
        Community detection algorithm: 'louvain', 'label_propagation', 'greedy'.
    
    Returns
    -------
    Dict[int, int]
        Community labels for each node.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required")
    
    G = to_networkx(network)
    
    if algorithm == 'louvain':
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, seed=42)
            
            # Convert to node->community mapping
            labels = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    labels[node] = comm_id
            return labels
            
        except ImportError:
            # Fallback to label propagation
            algorithm = 'label_propagation'
    
    if algorithm == 'label_propagation':
        from networkx.algorithms.community import label_propagation_communities
        communities = list(label_propagation_communities(G))
        
        labels = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                labels[node] = comm_id
        return labels
    
    elif algorithm == 'greedy':
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        
        labels = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                labels[node] = comm_id
        return labels
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def compute_graph_metrics(network: FiberNetwork) -> Dict[str, float]:
    """Compute comprehensive graph metrics.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    
    Returns
    -------
    Dict[str, float]
        Graph metrics.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required")
    
    G = to_networkx(network)
    
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Connectivity
    metrics['is_connected'] = nx.is_connected(G)
    
    if metrics['is_connected']:
        metrics['diameter'] = nx.diameter(G)
        metrics['average_shortest_path'] = nx.average_shortest_path_length(G)
        metrics['radius'] = nx.radius(G)
        metrics['center_size'] = len(nx.center(G))
    else:
        # Use largest connected component
        if G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc)
            
            metrics['largest_component_size'] = len(largest_cc)
            metrics['num_components'] = nx.number_connected_components(G)
            
            if len(largest_cc) > 1:
                metrics['diameter'] = nx.diameter(G_cc)
                metrics['average_shortest_path'] = nx.average_shortest_path_length(G_cc)
            else:
                metrics['diameter'] = 0
                metrics['average_shortest_path'] = 0
        else:
            metrics['largest_component_size'] = 0
            metrics['num_components'] = 0
            metrics['diameter'] = 0
            metrics['average_shortest_path'] = 0
    
    # Clustering
    metrics['average_clustering'] = nx.average_clustering(G)
    metrics['transitivity'] = nx.transitivity(G)
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    if degrees:
        metrics['mean_degree'] = np.mean(degrees)
        metrics['std_degree'] = np.std(degrees)
        metrics['max_degree'] = max(degrees)
        metrics['min_degree'] = min(degrees)
    
    # Assortativity
    try:
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    except Exception:
        metrics['degree_assortativity'] = 0.0
    
    return metrics


def find_shortest_path(
    network: FiberNetwork,
    source_node: int,
    target_node: int,
) -> Tuple[List[int], float]:
    """Find shortest path between two nodes.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    source_node : int
        Source node ID.
    target_node : int
        Target node ID.
    
    Returns
    -------
    Tuple[List[int], float]
        (path_nodes, path_length)
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required")
    
    G = to_networkx(network, weighted=True)
    
    try:
        path = nx.shortest_path(G, source_node, target_node, weight='weight')
        length = nx.shortest_path_length(G, source_node, target_node, weight='weight')
        return path, length
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], float('inf')


def compute_small_world_metrics(network: FiberNetwork) -> Dict[str, float]:
    """Compute small-world network metrics.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    
    Returns
    -------
    Dict[str, float]
        Small-world metrics including sigma coefficient.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required")
    
    G = to_networkx(network)
    
    if G.number_of_nodes() < 4:
        return {'sigma': 0.0, 'omega': 0.0}
    
    # Compute clustering coefficient and path length
    C = nx.average_clustering(G)
    
    if nx.is_connected(G):
        L = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc)
        L = nx.average_shortest_path_length(G_cc)
    
    # Compare with random graph
    n = G.number_of_nodes()
    k = np.mean([d for n, d in G.degree()])
    
    if k > 0:
        C_random = k / n
        L_random = np.log(n) / np.log(k) if k > 1 else n
    else:
        C_random = 1e-10
        L_random = n
    
    # Small-world coefficient (sigma)
    gamma = C / max(C_random, 1e-10)
    lambda_ = L / max(L_random, 1e-10)
    sigma = gamma / max(lambda_, 1e-10)
    
    # Small-world coefficient (omega)
    omega = (L_random / max(L, 1e-10)) - (C / max(C_random, 1e-10))
    
    return {
        'sigma': float(sigma),
        'omega': float(omega),
        'clustering_coefficient': float(C),
        'average_path_length': float(L),
        'gamma': float(gamma),
        'lambda': float(lambda_),
    }
