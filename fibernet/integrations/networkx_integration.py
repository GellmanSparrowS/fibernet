"""
NetworkX Integration for FiberNet

Provides seamless integration with NetworkX for advanced graph analysis:
- Convert FiberNetwork to NetworkX graph
- Community detection for fiber bundles
- Graph algorithms (shortest path, centrality, clustering)
- Graph visualization with NetworkX layouts
- Import/export from various graph formats

NetworkX is BSD-licensed: https://github.com/networkx/networkx
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings

from fibernet.core.network import FiberNetwork


@dataclass
class GraphAnalysisResult:
    """Result of NetworkX graph analysis."""
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    num_components: int = 0
    largest_component_size: int = 0
    average_clustering: float = 0.0
    average_path_length: float = 0.0
    diameter: int = 0
    communities: List[List[int]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'density': self.density,
            'num_components': self.num_components,
            'largest_component_size': self.largest_component_size,
            'average_clustering': self.average_clustering,
            'average_path_length': self.average_path_length,
            'diameter': self.diameter,
            'num_communities': len(self.communities),
        }


class NetworkXBridge:
    """Bridge between FiberNetwork and NetworkX.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.integrations.networkx_integration import NetworkXBridge
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> bridge = NetworkXBridge(net)
    >>> G = bridge.to_networkx()
    >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        try:
            import networkx as nx
            self.nx = nx
        except ImportError:
            raise ImportError("NetworkX required. Install with: pip install networkx")
    
    def to_networkx(
        self,
        node_type: str = 'fiber',
        include_fiber_attrs: bool = True,
    ):
        """Convert FiberNetwork to NetworkX graph.
        
        Parameters
        ----------
        node_type : str
            Node representation: 'fiber' or 'crosslink'
        include_fiber_attrs : bool
            Include fiber attributes as node/edge attributes.
        
        Returns
        -------
        G : networkx.Graph
            NetworkX graph representation.
        """
        if node_type == 'fiber':
            return self._fiber_graph(include_fiber_attrs)
        elif node_type == 'crosslink':
            return self._crosslink_graph(include_fiber_attrs)
        else:
            raise ValueError(f"Unknown node_type: {node_type}")
    
    def _fiber_graph(self, include_attrs: bool):
        """Create graph with fibers as nodes.
        
        Nodes = fibers, edges = fibers that are close enough to interact.
        """
        G = self.nx.Graph()
        
        # Add fibers as nodes
        for i, fiber in enumerate(self.network.fibers):
            attrs = {}
            if include_attrs:
                attrs = {
                    'length': fiber.length,
                    'radius': fiber.radius,
                    'center': fiber.centerline.mean(axis=0).tolist(),
                    'num_points': fiber.num_points,
                }
            G.add_node(i, **attrs)
        
        # Add edges between fibers that intersect or are close
        for i in range(len(self.network.fibers)):
            for j in range(i + 1, len(self.network.fibers)):
                f1 = self.network.fibers[i]
                f2 = self.network.fibers[j]
                
                # Check minimum distance between centerlines
                min_dist = self._min_distance_between_fibers(f1, f2)
                contact_dist = f1.radius + f2.radius
                
                if min_dist <= contact_dist * 2:  # Within 2x contact distance
                    G.add_edge(i, j, min_distance=min_dist)
        
        return G
    
    def _crosslink_graph(self, include_attrs: bool):
        """Create graph with crosslinks as nodes.
        
        If no crosslinks exist, creates a proximity-based graph.
        """
        if self.network.num_crosslinks > 0:
            return self._crosslink_graph_from_crosslinks(include_attrs)
        else:
            # No crosslinks - use fiber endpoints as nodes
            return self._crosslink_graph_from_endpoints(include_attrs)
    
    def _crosslink_graph_from_crosslinks(self, include_attrs: bool):
        """Create graph from existing crosslinks."""
        G = self.nx.Graph()
        
        # Add crosslinks as nodes
        for i, cl in enumerate(self.network.crosslinks):
            attrs = {'position': cl.position.tolist()}
            G.add_node(i, **attrs)
        
        # Build fiber-to-crosslinks mapping
        fiber_crosslinks = {}
        for cl_idx, cl in enumerate(self.network.crosslinks):
            # Each crosslink connects two fibers
            for fiber_idx in [cl.fiber_i, cl.fiber_j]:
                if fiber_idx not in fiber_crosslinks:
                    fiber_crosslinks[fiber_idx] = []
                fiber_crosslinks[fiber_idx].append(cl_idx)
        
        # Add edges: fibers with multiple crosslinks connect them
        for fiber_idx, cl_indices in fiber_crosslinks.items():
            if len(cl_indices) >= 2:
                fiber = self.network.fibers[fiber_idx]
                # Connect all crosslinks on this fiber
                for i in range(len(cl_indices) - 1):
                    cl_start = cl_indices[i]
                    cl_end = cl_indices[i + 1]
                    attrs = {}
                    if include_attrs:
                        attrs = {'length': fiber.length, 'radius': fiber.radius}
                    G.add_edge(cl_start, cl_end, **attrs)
        
        return G
    
    def _crosslink_graph_from_endpoints(self, include_attrs: bool):
        """Create graph using fiber endpoints as nodes."""
        G = self.nx.Graph()
        
        node_id = 0
        fiber_endpoints = {}
        
        # Add fiber start/end points as nodes
        for i, fiber in enumerate(self.network.fibers):
            start_pt = fiber.start_point
            end_pt = fiber.end_point
            
            # Find or create start node
            start_id = self._find_nearest_node(G, start_pt, threshold=fiber.radius * 2)
            if start_id is None:
                G.add_node(node_id, position=start_pt.tolist())
                start_id = node_id
                node_id += 1
            
            # Find or create end node
            end_id = self._find_nearest_node(G, end_pt, threshold=fiber.radius * 2)
            if end_id is None:
                G.add_node(node_id, position=end_pt.tolist())
                end_id = node_id
                node_id += 1
            
            fiber_endpoints[i] = (start_id, end_id)
            
            # Add edge
            attrs = {}
            if include_attrs:
                attrs = {'length': fiber.length, 'radius': fiber.radius}
            G.add_edge(start_id, end_id, **attrs)
        
        return G
    
    def _find_nearest_node(self, G, point, threshold):
        """Find nearest existing node within threshold distance."""
        best_id = None
        best_dist = threshold
        
        for node_id, data in G.nodes(data=True):
            if 'position' in data:
                pos = np.array(data['position'])
                dist = np.linalg.norm(pos - point)
                if dist < best_dist:
                    best_dist = dist
                    best_id = node_id
        
        return best_id
    
    def _min_distance_between_fibers(self, f1, f2) -> float:
        """Compute minimum distance between two fiber centerlines."""
        cl1 = f1.centerline
        cl2 = f2.centerline
        
        # Use subset of points for efficiency
        step1 = max(1, len(cl1) // 20)
        step2 = max(1, len(cl2) // 20)
        
        pts1 = cl1[::step1]
        pts2 = cl2[::step2]
        
        # Compute pairwise distances
        diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        return float(dists.min())
    
    def analyze(self, G=None) -> GraphAnalysisResult:
        """Perform comprehensive graph analysis.
        
        Parameters
        ----------
        G : networkx.Graph, optional
            Pre-computed graph. If None, creates fiber graph.
        
        Returns
        -------
        result : GraphAnalysisResult
            Analysis results.
        """
        if G is None:
            G = self.to_networkx()
        
        result = GraphAnalysisResult()
        
        result.num_nodes = G.number_of_nodes()
        result.num_edges = G.number_of_edges()
        result.density = self.nx.density(G) if result.num_nodes > 1 else 0.0
        
        # Connected components
        if result.num_nodes > 0:
            components = list(self.nx.connected_components(G))
            result.num_components = len(components)
            result.largest_component_size = max(len(c) for c in components)
        else:
            result.num_components = 0
            result.largest_component_size = 0
        
        # Clustering
        if result.num_nodes > 2:
            result.average_clustering = self.nx.average_clustering(G)
        
        # Path metrics (only for connected graphs)
        if result.num_nodes > 1 and self.nx.is_connected(G):
            result.average_path_length = self.nx.average_shortest_path_length(G)
            result.diameter = self.nx.diameter(G)
        
        # Community detection
        if result.num_edges > 0:
            try:
                communities = self.nx.community.greedy_modularity_communities(G)
                result.communities = [list(c) for c in communities]
            except Exception:
                result.communities = []
        
        return result
    
    def centrality_analysis(self, G=None, method: str = 'betweenness') -> Dict[int, float]:
        """Compute node centrality.
        
        Parameters
        ----------
        G : networkx.Graph, optional
            Pre-computed graph.
        method : str
            Centrality method: 'betweenness', 'closeness', 'degree', 'eigenvector'
        
        Returns
        -------
        centrality : dict
            Node centrality values.
        """
        if G is None:
            G = self.to_networkx()
        
        if G.number_of_nodes() == 0:
            return {}
        
        if method == 'betweenness':
            return self.nx.betweenness_centrality(G)
        elif method == 'closeness':
            return self.nx.closeness_centrality(G)
        elif method == 'degree':
            return dict(G.degree())
        elif method == 'eigenvector':
            try:
                return self.nx.eigenvector_centrality_numpy(G)
            except Exception:
                return {n: 0.0 for n in G.nodes()}
        else:
            raise ValueError(f"Unknown centrality method: {method}")
    
    def shortest_path(self, node1: int, node2: int, G=None) -> List[int]:
        """Find shortest path between two nodes.
        
        Parameters
        ----------
        node1, node2 : int
            Node indices.
        G : networkx.Graph, optional
            Pre-computed graph.
        
        Returns
        -------
        path : list
            List of node indices in path.
        """
        if G is None:
            G = self.to_networkx()
        
        try:
            return self.nx.shortest_path(G, node1, node2)
        except (self.nx.NetworkXNoPath, self.nx.NodeNotFound):
            return []
    
    def detect_communities(self, G=None, algorithm: str = 'greedy') -> List[List[int]]:
        """Detect communities (fiber bundles) in the network.
        
        Parameters
        ----------
        G : networkx.Graph, optional
            Pre-computed graph.
        algorithm : str
            Community detection algorithm: 'greedy', 'label_propagation', 'louvain'
        
        Returns
        -------
        communities : list of list
            List of communities (each is a list of node indices).
        """
        if G is None:
            G = self.to_networkx()
        
        if G.number_of_edges() == 0:
            return []
        
        try:
            if algorithm == 'greedy':
                communities = self.nx.community.greedy_modularity_communities(G)
                return [list(c) for c in communities]
            elif algorithm == 'label_propagation':
                communities = self.nx.community.label_propagation_communities(G)
                return [list(c) for c in communities]
            elif algorithm == 'louvain':
                try:
                    communities = self.nx.community.louvain_communities(G)
                    return [list(c) for c in communities]
                except AttributeError:
                    return self.detect_communities(G, 'greedy')
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        except Exception as e:
            warnings.warn(f"Community detection failed: {e}")
            return []


def analyze_network_topology(network: FiberNetwork, **kwargs) -> GraphAnalysisResult:
    """Convenience function for network topology analysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    
    Returns
    -------
    result : GraphAnalysisResult
        Analysis results.
    """
    bridge = NetworkXBridge(network)
    return bridge.analyze(**kwargs)
