"""
Network Topology Analysis Module

Provides graph-theoretic analysis of fiber networks using NetworkX:
- Degree distribution and statistics
- Clustering coefficient
- Betweenness centrality
- Network connectivity
- Shortest paths
- Community detection

References:
- Newman, M.E.J., "Networks: An Introduction", Oxford University Press, 2010
- Barabási, A.L., "Network Science", Cambridge University Press, 2016
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("NetworkX not available. Install with: pip install networkx")

from fibernet.core.network import FiberNetwork


@dataclass
class TopologyResult:
    """Result of topology analysis."""
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    max_degree: int = 0
    min_degree: int = 0
    degree_std: float = 0.0
    clustering_coefficient: float = 0.0
    avg_path_length: float = 0.0
    diameter: int = 0
    num_components: int = 0
    largest_component_size: int = 0
    assortativity: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'density': self.density,
            'avg_degree': self.avg_degree,
            'clustering_coefficient': self.clustering_coefficient,
            'avg_path_length': self.avg_path_length,
            'diameter': self.diameter,
            'num_components': self.num_components,
            'largest_component_size': self.largest_component_size,
            'assortativity': self.assortativity,
        }


@dataclass
class CentralityResult:
    """Result of centrality analysis."""
    degree_centrality: Dict[int, float] = field(default_factory=dict)
    betweenness_centrality: Dict[int, float] = field(default_factory=dict)
    closeness_centrality: Dict[int, float] = field(default_factory=dict)
    eigenvector_centrality: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'degree_centrality': self.degree_centrality,
            'betweenness_centrality': self.betweenness_centrality,
            'closeness_centrality': self.closeness_centrality,
            'eigenvector_centrality': self.eigenvector_centrality,
        }


class TopologyAnalyzer:
    """Analyze network topology using graph theory.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    weighted : bool
        Use weighted edges (by fiber length or conductance).
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.analysis.topology import TopologyAnalyzer
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> topo = TopologyAnalyzer(net)
    >>> result = topo.analyze()
    >>> print(f"Clustering: {result.clustering_coefficient:.3f}")
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        weighted: bool = False,
    ):
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for topology analysis")
        
        self.network = network
        self.weighted = weighted
        self.graph = self._build_graph()
    
    def _build_graph(self) -> nx.Graph:
        """Build NetworkX graph from fiber network.
        
        Strategy: Create nodes at crosslink positions. Each fiber creates
        an edge between the crosslinks it connects.
        """
        G = nx.Graph()
        
        # If no crosslinks, create nodes at fiber endpoints
        if not self.network.crosslinks:
            node_map = {}  # position tuple -> node_id
            node_id = 0
            
            for i, fiber in enumerate(self.network.fibers):
                start = tuple(np.round(fiber.centerline[0], decimals=6))
                end = tuple(np.round(fiber.centerline[-1], decimals=6))
                
                # Add start node
                if start not in node_map:
                    node_map[start] = node_id
                    G.add_node(node_id, position=start, type='endpoint')
                    node_id += 1
                
                # Add end node
                if end not in node_map:
                    node_map[end] = node_id
                    G.add_node(node_id, position=end, type='endpoint')
                    node_id += 1
                
                # Add edge
                weight = fiber.length if self.weighted else 1.0
                G.add_edge(node_map[start], node_map[end], weight=weight, fiber_id=i)
        
        else:
            # Create nodes at crosslinks
            for i, cl in enumerate(self.network.crosslinks):
                G.add_node(i, position=cl.position, type='crosslink')
            
            # Find which crosslinks each fiber connects
            for i, fiber in enumerate(self.network.fibers):
                start = fiber.centerline[0]
                end = fiber.centerline[-1]
                
                # Find closest crosslinks
                start_cl = None
                end_cl = None
                min_dist_start = float('inf')
                min_dist_end = float('inf')
                
                for j, cl in enumerate(self.network.crosslinks):
                    dist_start = np.linalg.norm(start - cl.position)
                    dist_end = np.linalg.norm(end - cl.position)
                    
                    if dist_start < min_dist_start:
                        min_dist_start = dist_start
                        start_cl = j
                    if dist_end < min_dist_end:
                        min_dist_end = dist_end
                        end_cl = j
                
                # Add edge if we found valid crosslinks
                if start_cl is not None and end_cl is not None and start_cl != end_cl:
                    weight = fiber.length if self.weighted else 1.0
                    G.add_edge(start_cl, end_cl, weight=weight, fiber_id=i)
        
        return G
    
    def analyze(self) -> TopologyResult:
        """Perform comprehensive topology analysis.
        
        Returns
        -------
        result : TopologyResult
            Topology analysis results.
        """
        G = self.graph
        
        # Basic statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        degree_std = np.std(degrees) if degrees else 0.0
        
        # Clustering coefficient
        clustering = nx.average_clustering(G)
        
        # Connected components
        components = list(nx.connected_components(G))
        num_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0
        
        # Path statistics (only for largest component)
        if largest_component_size > 1:
            largest_cc = max(components, key=len)
            subgraph = G.subgraph(largest_cc)
            
            if nx.is_connected(subgraph):
                avg_path_length = nx.average_shortest_path_length(subgraph)
                diameter = nx.diameter(subgraph)
            else:
                avg_path_length = 0.0
                diameter = 0
        else:
            avg_path_length = 0.0
            diameter = 0
        
        # Assortativity
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
        except:
            assortativity = 0.0
        
        return TopologyResult(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            avg_degree=avg_degree,
            max_degree=max_degree,
            min_degree=min_degree,
            degree_std=degree_std,
            clustering_coefficient=clustering,
            avg_path_length=avg_path_length,
            diameter=diameter,
            num_components=num_components,
            largest_component_size=largest_component_size,
            assortativity=assortativity,
        )
    
    def compute_centrality(self) -> CentralityResult:
        """Compute node centrality measures.
        
        Returns
        -------
        result : CentralityResult
            Centrality analysis results.
        """
        G = self.graph
        
        # Degree centrality
        degree_cent = nx.degree_centrality(G)
        
        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G)
        except:
            betweenness = {n: 0.0 for n in G.nodes()}
        
        # Closeness centrality
        try:
            closeness = nx.closeness_centrality(G)
        except:
            closeness = {n: 0.0 for n in G.nodes()}
        
        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector = {n: 0.0 for n in G.nodes()}
        
        return CentralityResult(
            degree_centrality=degree_cent,
            betweenness_centrality=betweenness,
            closeness_centrality=closeness,
            eigenvector_centrality=eigenvector,
        )
    
    def degree_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute degree distribution.
        
        Returns
        -------
        degrees : np.ndarray
            Unique degree values.
        counts : np.ndarray
            Count of nodes with each degree.
        """
        degrees = [d for n, d in self.graph.degree()]
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        return unique_degrees, counts
    
    def find_communities(self, method: str = 'louvain') -> Dict[int, int]:
        """Detect communities in the network.
        
        Parameters
        ----------
        method : str
            Community detection method: 'louvain', 'label_propagation'.
        
        Returns
        -------
        communities : dict
            Mapping from node ID to community ID.
        """
        G = self.graph
        
        if method == 'louvain':
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G)
                # Convert to node -> community mapping
                comm_map = {}
                for comm_id, comm in enumerate(communities):
                    for node in comm:
                        comm_map[node] = comm_id
                return comm_map
            except ImportError:
                warnings.warn("Louvain not available, using label propagation")
                method = 'label_propagation'
        
        if method == 'label_propagation':
            try:
                from networkx.algorithms.community import label_propagation_communities
                communities = list(label_propagation_communities(G))
                comm_map = {}
                for comm_id, comm in enumerate(communities):
                    for node in comm:
                        comm_map[node] = comm_id
                return comm_map
            except:
                return {n: 0 for n in G.nodes()}
        
        raise ValueError(f"Unknown method: {method}")
    
    def get_critical_nodes(self, metric: str = 'betweenness', top_k: int = 10) -> List[int]:
        """Find most critical nodes by centrality.
        
        Parameters
        ----------
        metric : str
            Centrality metric: 'betweenness', 'degree', 'closeness'.
        top_k : int
            Number of top nodes to return.
        
        Returns
        -------
        nodes : list
            List of node IDs.
        """
        centrality = self.compute_centrality()
        
        if metric == 'betweenness':
            cent = centrality.betweenness_centrality
        elif metric == 'degree':
            cent = centrality.degree_centrality
        elif metric == 'closeness':
            cent = centrality.closeness_centrality
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Sort by centrality
        sorted_nodes = sorted(cent.items(), key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in sorted_nodes[:top_k]]


    def degree_statistics(self) -> Dict[str, float]:
        """Compute degree statistics (for compatibility).
        
        Returns
        -------
        stats : dict
            Dictionary with 'mean', 'std', 'min', 'max' keys.
        """
        degrees = [d for n, d in self.graph.degree()]
        if not degrees:
            return {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0}
        
        return {
            'mean': float(np.mean(degrees)),
            'std': float(np.std(degrees)),
            'min': int(min(degrees)),
            'max': int(max(degrees)),
        }
    
    def clustering_coefficient(self) -> float:
        """Compute average clustering coefficient (for compatibility).
        
        Returns
        -------
        clustering : float
            Average clustering coefficient.
        """
        return float(nx.average_clustering(self.graph))

    def full_report(self) -> Dict:
        """Generate full topology report (for compatibility).
        
        Returns
        -------
        report : dict
            Comprehensive topology report.
        """
        result = self.analyze()
        centrality = self.compute_centrality()
        
        # Check if graph is connected
        is_connected = nx.is_connected(self.graph)
        
        return {
            'num_fibers': len(self.network.fibers),
            'num_crosslinks': len(self.network.crosslinks),
            'is_connected': is_connected,
            'basic': result.to_dict(),
            'centrality': {
                'degree': centrality.degree_centrality,
                'betweenness': centrality.betweenness_centrality,
                'closeness': centrality.closeness_centrality,
            },
            'degree_distribution': {
                'degrees': self.degree_distribution()[0].tolist(),
                'counts': self.degree_distribution()[1].tolist(),
            },
        }


def analyze_topology(
    network: FiberNetwork,
    weighted: bool = False,
) -> TopologyResult:
    """Convenience function for topology analysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    weighted : bool
        Use weighted edges.
    
    Returns
    -------
    result : TopologyResult
        Topology analysis results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.analysis.topology import analyze_topology
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> result = analyze_topology(net)
    >>> print(f"Clustering: {result.clustering_coefficient:.3f}")
    """
    analyzer = TopologyAnalyzer(network, weighted=weighted)
    return analyzer.analyze()
