"""
Feature Extraction for Fiber Networks

Converts fiber network structures into feature vectors suitable for ML models.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from fibernet.core.network import FiberNetwork
from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer
from fibernet.analysis.advanced import SpectralAnalyzer, PoreAnalyzer, AnisotropyAnalyzer


class FeatureExtractor:
    """Extract comprehensive feature vectors from fiber networks.
    
    Parameters
    ----------
    include_morphology : bool
        Include morphological features (length, orientation, tortuosity)
    include_topology : bool
        Include topological features (degree, connectivity, components)
    include_spectral : bool
        Include spectral features (eigenvalues, spectral gap)
    include_pore : bool
        Include pore structure features (pore size distribution)
    include_anisotropy : bool
        Include anisotropy features (orientation tensors)
    """
    
    def __init__(
        self,
        include_morphology: bool = True,
        include_topology: bool = True,
        include_spectral: bool = True,
        include_pore: bool = True,
        include_anisotropy: bool = True,
    ):
        self.include_morphology = include_morphology
        self.include_topology = include_topology
        self.include_spectral = include_spectral
        self.include_pore = include_pore
        self.include_anisotropy = include_anisotropy
    
    def extract(self, network: FiberNetwork) -> Dict[str, float]:
        """Extract all features from a fiber network.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        Dict[str, float]
            Feature dictionary with string keys and numeric values
        """
        features = {}
        
        # Basic network properties
        features['num_fibers'] = network.num_fibers
        features['num_crosslinks'] = network.num_crosslinks
        features['dimension'] = network.dimension
        
        # Morphological features
        if self.include_morphology:
            try:
                morph = MorphologyAnalyzer(network)
                report = morph.full_report()
                features['mean_length'] = report.get('mean_length', 0.0)
                features['std_length'] = report.get('std_length', 0.0)
                features['mean_radius'] = report.get('mean_radius', 0.0)
                features['std_radius'] = report.get('std_radius', 0.0)
                features['nematic_order'] = report.get('nematic_order', 0.0)
                features['mean_tortuosity'] = report.get('mean_tortuosity', 1.0)
                features['total_length'] = report.get('total_length', 0.0)
                features['volume_fraction'] = report.get('volume_fraction', 0.0)
            except Exception:
                # Fill with defaults if analysis fails
                features.update({
                    'mean_length': 0.0,
                    'std_length': 0.0,
                    'mean_radius': 0.0,
                    'std_radius': 0.0,
                    'nematic_order': 0.0,
                    'mean_tortuosity': 1.0,
                    'total_length': 0.0,
                    'volume_fraction': 0.0,
                })
        
        # Topological features
        if self.include_topology:
            try:
                topo = TopologyAnalyzer(network)
                report = topo.full_report()
                features['num_nodes'] = report.get('num_nodes', 0)
                features['num_edges'] = report.get('num_edges', 0)
                features['mean_degree'] = report.get('degree_stats', {}).get('mean', 0.0)
                features['std_degree'] = report.get('degree_stats', {}).get('std', 0.0)
                features['max_degree'] = report.get('degree_stats', {}).get('max', 0)
                features['min_degree'] = report.get('degree_stats', {}).get('min', 0)
                features['is_connected'] = 1.0 if report.get('is_connected', False) else 0.0
                features['num_components'] = report.get('num_components', 1)
            except Exception:
                features.update({
                    'num_nodes': 0,
                    'num_edges': 0,
                    'mean_degree': 0.0,
                    'std_degree': 0.0,
                    'max_degree': 0,
                    'min_degree': 0,
                    'is_connected': 0.0,
                    'num_components': 1,
                })
        
        # Spectral features
        if self.include_spectral:
            try:
                spectral = SpectralAnalyzer(network)
                features['spectral_gap'] = spectral.spectral_gap()
                features['spectral_entropy'] = spectral.spectral_entropy()
                # Add first few eigenvalues
                eigenvalues = spectral.eigenvalues()
                for i, eig in enumerate(eigenvalues[:5]):
                    features[f'eigenvalue_{i}'] = eig
            except Exception:
                features.update({
                    'spectral_gap': 0.0,
                    'spectral_entropy': 0.0,
                    'eigenvalue_0': 0.0,
                    'eigenvalue_1': 0.0,
                    'eigenvalue_2': 0.0,
                    'eigenvalue_3': 0.0,
                    'eigenvalue_4': 0.0,
                })
        
        # Pore structure features
        if self.include_pore:
            try:
                pore = PoreAnalyzer(network)
                stats = pore.pore_size_statistics()
                features['mean_pore_size'] = stats.get('mean', 0.0)
                features['std_pore_size'] = stats.get('std', 0.0)
                features['median_pore_size'] = stats.get('median', 0.0)
                features['p95_pore_size'] = stats.get('p95', 0.0)
                features['pore_size_range'] = stats.get('max', 0.0) - stats.get('min', 0.0)
            except Exception:
                features.update({
                    'mean_pore_size': 0.0,
                    'std_pore_size': 0.0,
                    'median_pore_size': 0.0,
                    'p95_pore_size': 0.0,
                    'pore_size_range': 0.0,
                })
        
        # Anisotropy features
        if self.include_anisotropy:
            try:
                aniso = AnisotropyAnalyzer(network)
                features['anisotropy_index'] = aniso.anisotropy_index()
                # Add orientation tensor components
                tensor = aniso.orientation_tensor()
                features['orientation_xx'] = tensor[0, 0]
                features['orientation_yy'] = tensor[1, 1]
                features['orientation_zz'] = tensor[2, 2]
                features['orientation_xy'] = tensor[0, 1]
                features['orientation_xz'] = tensor[0, 2]
                features['orientation_yz'] = tensor[1, 2]
            except Exception:
                features.update({
                    'anisotropy_index': 0.0,
                    'orientation_xx': 0.0,
                    'orientation_yy': 0.0,
                    'orientation_zz': 0.0,
                    'orientation_xy': 0.0,
                    'orientation_xz': 0.0,
                    'orientation_yz': 0.0,
                })
        
        return features
    
    def extract_to_array(self, network: FiberNetwork) -> np.ndarray:
        """Extract features as a numpy array.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        np.ndarray
            1D array of feature values
        """
        features = self.extract(network)
        return np.array(list(features.values()))
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns
        -------
        List[str]
            Feature names in consistent order
        """
        # Create a dummy network to get feature keys
        from fibernet import gen
        dummy = gen.random_straight_2d(num_fibers=5, fiber_length=5, box_size=(20, 20), seed=42)
        features = self.extract(dummy)
        return list(features.keys())


def extract_features(
    network: FiberNetwork,
    as_array: bool = False,
) -> Any:
    """Convenience function to extract features from a network.
    
    Parameters
    ----------
    network : FiberNetwork
        Input fiber network
    as_array : bool
        If True, return numpy array; otherwise return dict
        
    Returns
    -------
    Dict[str, float] or np.ndarray
        Extracted features
    """
    extractor = FeatureExtractor()
    if as_array:
        return extractor.extract_to_array(network)
    else:
        return extractor.extract(network)


class GNNFeatureExtractor:
    """
    Extract graph neural network features from fiber networks.
    
    Converts fiber networks into graph representations suitable for GNN models.
    Supports PyTorch Geometric and DGL formats.
    
    Parameters
    ----------
    node_features : list of str
        Node feature names to extract. Options: 'position', 'degree', 'centrality'
    edge_features : list of str
        Edge feature names to extract. Options: 'length', 'angle', 'weight'
    """
    
    def __init__(
        self,
        node_features: Optional[List[str]] = None,
        edge_features: Optional[List[str]] = None,
    ):
        self.node_features = node_features or ['position', 'degree']
        self.edge_features = edge_features or ['length', 'angle']
    
    def extract_graph(self, network: FiberNetwork) -> Dict[str, Any]:
        """
        Extract graph representation from fiber network.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        Dict[str, Any]
            Graph data with node_features, edge_index, edge_features
        """
        import networkx as nx
        
        # Convert to NetworkX graph
        G = self._to_networkx(network)
        
        # Extract node features
        node_feat_matrix = self._extract_node_features(G, network)
        
        # Extract edge index and features
        edge_index, edge_feat_matrix = self._extract_edge_features(G, network)
        
        return {
            'node_features': node_feat_matrix,
            'edge_index': edge_index,
            'edge_features': edge_feat_matrix,
            'num_nodes': node_feat_matrix.shape[0],
            'num_edges': edge_index.shape[1],
            'graph': G,
        }
    
    def to_pytorch_geometric(self, network: FiberNetwork, label: Optional[float] = None):
        """
        Convert to PyTorch Geometric Data object.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
        label : float, optional
            Graph-level label for supervised learning
            
        Returns
        -------
        torch_geometric.data.Data
            PyTorch Geometric data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch Geometric required. Install with: pip install torch torch-geometric")
        
        graph_data = self.extract_graph(network)
        
        # Convert to tensors
        x = torch.tensor(graph_data['node_features'], dtype=torch.float32)
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph_data['edge_features'], dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.float32)
        
        return data
    
    def to_dgl(self, network: FiberNetwork, label: Optional[float] = None):
        """
        Convert to DGL graph object.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
        label : float, optional
            Graph-level label
            
        Returns
        -------
        dgl.DGLGraph
            DGL graph object
        """
        try:
            import torch
            import dgl
        except ImportError:
            raise ImportError("DGL required. Install with: pip install dgl")
        
        graph_data = self.extract_graph(network)
        
        # Create DGL graph
        edge_index = graph_data['edge_index']
        src = edge_index[0]
        dst = edge_index[1]
        
        g = dgl.graph((src, dst), num_nodes=graph_data['num_nodes'])
        
        # Add node features
        g.ndata['feat'] = torch.tensor(graph_data['node_features'], dtype=torch.float32)
        
        # Add edge features
        g.edata['feat'] = torch.tensor(graph_data['edge_features'], dtype=torch.float32)
        
        # Add label
        if label is not None:
            g.label = torch.tensor([label], dtype=torch.float32)
        
        return g
    
    def _to_networkx(self, network: FiberNetwork):
        """Convert fiber network to NetworkX graph."""
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes (fibers)
        for i, fiber in enumerate(network.fibers):
            pos = (fiber.start_point + fiber.end_point) / 2  # Center position
            G.add_node(i, pos=pos, length=fiber.length, radius=fiber.radius)
        
        # Add edges (crosslinks)
        for crosslink in network.crosslinks:
            i = crosslink.fiber_i
            j = crosslink.fiber_j
            G.add_edge(i, j, position=crosslink.position)
        
        return G
    
    def _extract_node_features(self, G, network: FiberNetwork) -> np.ndarray:
        """Extract node feature matrix."""
        import networkx as nx
        
        num_nodes = G.number_of_nodes()
        features = []
        
        for node in G.nodes():
            node_feat = []
            
            if 'position' in self.node_features:
                pos = G.nodes[node]['pos']
                node_feat.extend(pos)
            
            if 'degree' in self.node_features:
                degree = G.degree(node)
                node_feat.append(degree)
            
            if 'centrality' in self.node_features:
                try:
                    centrality = nx.betweenness_centrality(G)
                    node_feat.append(centrality.get(node, 0.0))
                except:
                    node_feat.append(0.0)
            
            # Add fiber properties
            fiber = network.fibers[node]
            node_feat.extend([fiber.length, fiber.radius])
            
            features.append(node_feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_edge_features(self, G, network: FiberNetwork) -> tuple:
        """Extract edge index and edge feature matrix."""
        edges = list(G.edges())
        
        if len(edges) == 0:
            # No edges
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
        
        # Edge index (source, target)
        edge_index = np.array(edges, dtype=np.int64).T
        
        # Edge features
        edge_features = []
        for u, v in edges:
            edge_feat = []
            
            if 'length' in self.edge_features:
                # Distance between fiber centers
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                dist = np.linalg.norm(np.array(pos_v) - np.array(pos_u))
                edge_feat.append(dist)
            
            if 'angle' in self.edge_features:
                # Angle between fibers
                fiber_u = network.fibers[u]
                fiber_v = network.fibers[v]
                
                dir_u = fiber_u.end_point - fiber_u.start_point
                dir_v = fiber_v.end_point - fiber_v.start_point
                
                dir_u = dir_u / (np.linalg.norm(dir_u) + 1e-10)
                dir_v = dir_v / (np.linalg.norm(dir_v) + 1e-10)
                
                cos_angle = np.clip(np.dot(dir_u, dir_v), -1, 1)
                angle = np.arccos(cos_angle)
                edge_feat.append(angle)
            
            if 'weight' in self.edge_features:
                # Crosslink stiffness or strength
                edge_feat.append(1.0)  # Default weight
            
            edge_features.append(edge_feat)
        
        edge_feat_matrix = np.array(edge_features, dtype=np.float32)
        
        return edge_index, edge_feat_matrix
    
    def create_dataset(
        self,
        networks: List[FiberNetwork],
        labels: Optional[List[float]] = None,
        format: str = 'pyg',
    ) -> Any:
        """
        Create a dataset from multiple networks.
        
        Parameters
        ----------
        networks : list of FiberNetwork
            List of fiber networks
        labels : list of float, optional
            Labels for each network
        format : str
            Output format: 'pyg' (PyTorch Geometric) or 'dgl'
            
        Returns
        -------
        Dataset object
        """
        if format == 'pyg':
            try:
                from torch_geometric.data import InMemoryDataset
            except ImportError:
                raise ImportError("PyTorch Geometric required")
            
            data_list = []
            for i, network in enumerate(networks):
                label = labels[i] if labels is not None else None
                data = self.to_pytorch_geometric(network, label)
                data_list.append(data)
            
            return data_list
        
        elif format == 'dgl':
            try:
                import dgl
            except ImportError:
                raise ImportError("DGL required")
            
            graph_list = []
            for i, network in enumerate(networks):
                label = labels[i] if labels is not None else None
                g = self.to_dgl(network, label)
                graph_list.append(g)
            
            return graph_list
        
        else:
            raise ValueError(f"Unknown format: {format}. Use 'pyg' or 'dgl'")
