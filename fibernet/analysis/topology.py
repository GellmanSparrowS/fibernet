"""
Topology analysis for fiber networks.

Provides:
- Connectivity analysis
- Percolation detection
- Euler characteristics
- Network metrics (degree, clustering, betweenness)
- Pore size distribution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from fibernet.core.network import FiberNetwork

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


def _check_networkx():
    if not HAS_NETWORKX:
        raise ImportError("networkx is required for topology analysis. Install with: pip install networkx")



class TopologyAnalyzer:
    """Analyze topological properties of fiber networks."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self._graph = None
    
    @property
    def graph(self):
        """Get NetworkX graph representation."""
        if self._graph is None:
            self._graph = self.network.to_networkx()
        return self._graph
    
    def degree_statistics(self) -> Dict[str, float]:
        """Compute degree statistics."""
        degrees = self.network.degree_distribution()
        if len(degrees) == 0:
            return {"mean": 0, "std": 0, "max": 0, "min": 0}
        return {
            "mean": float(np.mean(degrees)),
            "std": float(np.std(degrees)),
            "max": int(np.max(degrees)),
            "min": int(np.min(degrees)),
            "median": float(np.median(degrees)),
        }
    
    def clustering_coefficient(self) -> float:
        """Average clustering coefficient of the network."""
        _check_networkx()
        return float(nx.average_clustering(self.graph))
    
    def is_connected(self) -> bool:
        """Check if the fiber network is fully connected."""
        _check_networkx()
        return nx.is_connected(self.graph) if len(self.graph.nodes) > 0 else False
    
    def num_connected_components(self) -> int:
        """Number of connected components."""
        _check_networkx()
        return nx.number_connected_components(self.graph) if len(self.graph.nodes) > 0 else 0
    
    def largest_component_fraction(self) -> float:
        """Fraction of fibers in the largest connected component."""
        _check_networkx()
        if len(self.graph.nodes) == 0:
            return 0.0
        components = list(nx.connected_components(self.graph))
        largest = max(len(c) for c in components)
        return largest / self.network.num_fibers if self.network.num_fibers > 0 else 0.0
    
    def betweenness_centrality(self) -> Dict[int, float]:
        """Betweenness centrality for each fiber node."""
        _check_networkx()
        return nx.betweenness_centrality(self.graph)
    
    def average_path_length(self) -> float:
        """Average shortest path length (over largest component)."""
        _check_networkx()
        if len(self.graph.nodes) == 0:
            return 0.0
        if not nx.is_connected(self.graph):
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
        else:
            subgraph = self.graph
        if len(subgraph.nodes) < 2:
            return 0.0
        return float(nx.average_shortest_path_length(subgraph))
    
    def full_report(self) -> Dict[str, any]:
        """Generate comprehensive topology report."""
        _check_networkx()
        
        report = {
            "num_fibers": self.network.num_fibers,
            "num_crosslinks": self.network.num_crosslinks,
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges),
            "degree_stats": self.degree_statistics(),
            "clustering_coefficient": self.clustering_coefficient(),
            "is_connected": self.is_connected(),
            "num_components": self.num_connected_components(),
            "largest_component_fraction": self.largest_component_fraction(),
        }
        
        if len(self.graph.nodes) > 0:
            report["density"] = nx.density(self.graph)
        
        return report
