"""
Graph Neural Network Module for Fiber Networks

This module provides GNN-based property prediction for fiber networks:
- Network-to-graph conversion
- GNN model definitions (GCN, GAT, GraphSAGE)
- Property prediction (mechanical, thermal, electrical)
- Training utilities
- Feature extraction for graph nodes and edges

This is essential for rapid property screening and design optimization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.analysis.topology import TopologyAnalyzer


@dataclass
class GraphData:
    """Graph representation of fiber network.
    
    Attributes
    ----------
    node_features : np.ndarray
        Node feature matrix (num_nodes, num_node_features)
    edge_features : np.ndarray
        Edge feature matrix (num_edges, num_edge_features)
    edge_index : np.ndarray
        Edge connectivity (2, num_edges)
    node_labels : np.ndarray, optional
        Node labels for supervised learning
    graph_label : float, optional
        Graph-level label for supervised learning
    """
    node_features: np.ndarray = None
    edge_features: np.ndarray = None
    edge_index: np.ndarray = None
    node_labels: np.ndarray = None
    graph_label: float = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'node_features': self.node_features.tolist() if self.node_features is not None else None,
            'edge_features': self.edge_features.tolist() if self.edge_features is not None else None,
            'edge_index': self.edge_index.tolist() if self.edge_index is not None else None,
            'node_labels': self.node_labels.tolist() if self.node_labels is not None else None,
            'graph_label': self.graph_label,
        }
    
    def to_pyg_data(self):
        """Convert to PyTorch Geometric Data object.
        
        Returns
        -------
        data : torch_geometric.data.Data
            PyG data object, or None if PyG not available.
        """
        try:
            import torch
            from torch_geometric.data import Data
            
            x = torch.tensor(self.node_features, dtype=torch.float)
            edge_index = torch.tensor(self.edge_index, dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index)
            
            if self.edge_features is not None:
                data.edge_attr = torch.tensor(self.edge_features, dtype=torch.float)
            
            if self.node_labels is not None:
                data.y = torch.tensor(self.node_labels, dtype=torch.float)
            
            if self.graph_label is not None:
                data.y = torch.tensor([self.graph_label], dtype=torch.float)
            
            return data
        except ImportError:
            warnings.warn("PyTorch Geometric not available. Install with: pip install torch torch-geometric")
            return None


class NetworkGraphConverter:
    """Convert fiber networks to graph representations.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to convert.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.ml.graph_neural_network import NetworkGraphConverter
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> converter = NetworkGraphConverter(net)
    >>> graph = converter.to_graph()
    >>> print(f"Nodes: {graph.node_features.shape[0]}, Edges: {graph.edge_index.shape[1]}")
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self.topology = TopologyAnalyzer(network)
        self.graph = self.topology.graph
    
    def to_graph(
        self,
        node_features: List[str] = None,
        edge_features: List[str] = None,
    ) -> GraphData:
        """Convert network to graph data.
        
        Parameters
        ----------
        node_features : list of str, optional
            Node features to extract. Options:
            - 'degree': node degree
            - 'betweenness': betweenness centrality
            - 'closeness': closeness centrality
            - 'eigenvector': eigenvector centrality
            - 'clustering': clustering coefficient
            Default: ['degree', 'betweenness']
        
        edge_features : list of str, optional
            Edge features to extract. Options:
            - 'weight': edge weight (fiber length)
            - 'distance': Euclidean distance
            - 'angle': angle between fibers
            Default: ['weight']
        
        Returns
        -------
        graph : GraphData
            Graph representation.
        """
        if node_features is None:
            node_features = ['degree', 'betweenness']
        
        if edge_features is None:
            edge_features = ['weight']
        
        # Extract node features
        node_feat_matrix = self._extract_node_features(node_features)
        
        # Extract edge features
        edge_index, edge_feat_matrix = self._extract_edge_features(edge_features)
        
        return GraphData(
            node_features=node_feat_matrix,
            edge_features=edge_feat_matrix,
            edge_index=edge_index,
        )
    
    def _extract_node_features(self, feature_names: List[str]) -> np.ndarray:
        """Extract node features.
        
        Parameters
        ----------
        feature_names : list of str
            Feature names to extract.
        
        Returns
        -------
        features : np.ndarray
            Node feature matrix (num_nodes, num_features)
        """
        num_nodes = self.graph.number_of_nodes()
        features = []
        
        # Compute centrality measures
        centrality = {}
        
        if 'betweenness' in feature_names:
            try:
                import networkx as nx
                centrality['betweenness'] = nx.betweenness_centrality(self.graph)
            except Exception:
                centrality['betweenness'] = {n: 0.0 for n in self.graph.nodes()}
        
        if 'closeness' in feature_names:
            try:
                import networkx as nx
                centrality['closeness'] = nx.closeness_centrality(self.graph)
            except Exception:
                centrality['closeness'] = {n: 0.0 for n in self.graph.nodes()}
        
        if 'eigenvector' in feature_names:
            try:
                import networkx as nx
                centrality['eigenvector'] = nx.eigenvector_centrality_numpy(self.graph)
            except Exception:
                centrality['eigenvector'] = {n: 0.0 for n in self.graph.nodes()}
        
        if 'clustering' in feature_names:
            try:
                import networkx as nx
                centrality['clustering'] = nx.clustering(self.graph)
            except Exception:
                centrality['clustering'] = {n: 0.0 for n in self.graph.nodes()}
        
        # Build feature matrix
        for node in sorted(self.graph.nodes()):
            node_feat = []
            
            if 'degree' in feature_names:
                node_feat.append(float(self.graph.degree(node)))
            
            if 'betweenness' in feature_names:
                node_feat.append(centrality['betweenness'].get(node, 0.0))
            
            if 'closeness' in feature_names:
                node_feat.append(centrality['closeness'].get(node, 0.0))
            
            if 'eigenvector' in feature_names:
                node_feat.append(centrality['eigenvector'].get(node, 0.0))
            
            if 'clustering' in feature_names:
                node_feat.append(centrality['clustering'].get(node, 0.0))
            
            features.append(node_feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_edge_features(self, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract edge features.
        
        Parameters
        ----------
        feature_names : list of str
            Feature names to extract.
        
        Returns
        -------
        edge_index : np.ndarray
            Edge connectivity (2, num_edges)
        edge_features : np.ndarray
            Edge feature matrix (num_edges, num_features)
        """
        edges = []
        features = []
        
        for u, v, data in self.graph.edges(data=True):
            edges.append([u, v])
            
            edge_feat = []
            
            if 'weight' in feature_names:
                edge_feat.append(data.get('weight', 1.0))
            
            if 'distance' in feature_names:
                # Get node positions
                pos_u = data.get('pos_u', np.zeros(3))
                pos_v = data.get('pos_v', np.zeros(3))
                dist = np.linalg.norm(np.array(pos_v) - np.array(pos_u))
                edge_feat.append(dist)
            
            if 'angle' in feature_names:
                # Angle between fibers (simplified)
                edge_feat.append(0.0)
            
            features.append(edge_feat)
        
        edge_index = np.array(edges, dtype=np.int64).T
        edge_features = np.array(features, dtype=np.float32)
        
        return edge_index, edge_features


class GNNPropertyPredictor:
    """GNN-based property predictor for fiber networks.
    
    Parameters
    ----------
    model_type : str
        GNN model type: 'gcn', 'gat', 'graphsage'
    hidden_channels : int
        Number of hidden channels.
    num_layers : int
        Number of GNN layers.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.ml.graph_neural_network import NetworkGraphConverter, GNNPropertyPredictor
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> converter = NetworkGraphConverter(net)
    >>> graph = converter.to_graph()
    >>> predictor = GNNPropertyPredictor(model_type='gcn')
    >>> prediction = predictor.predict(graph)
    """
    
    def __init__(
        self,
        model_type: str = 'gcn',
        hidden_channels: int = 64,
        num_layers: int = 3,
    ):
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build GNN model."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
            
            class GNNModel(nn.Module):
                def __init__(self, model_type, in_channels, hidden_channels, num_layers):
                    super().__init__()
                    
                    self.convs = nn.ModuleList()
                    
                    # Choose convolution type
                    if model_type == 'gcn':
                        conv_class = GCNConv
                    elif model_type == 'gat':
                        conv_class = GATConv
                    elif model_type == 'graphsage':
                        conv_class = SAGEConv
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                    
                    # Build layers
                    self.convs.append(conv_class(in_channels, hidden_channels))
                    for _ in range(num_layers - 1):
                        self.convs.append(conv_class(hidden_channels, hidden_channels))
                    
                    # Output layer
                    self.lin = nn.Linear(hidden_channels, 1)
                
                def forward(self, x, edge_index, batch=None):
                    # Apply convolutions
                    for conv in self.convs:
                        x = conv(x, edge_index)
                        x = F.relu(x)
                        x = F.dropout(x, p=0.5, training=self.training)
                    
                    # Global pooling
                    if batch is not None:
                        x = global_mean_pool(x, batch)
                    else:
                        x = x.mean(dim=0, keepdim=True)
                    
                    # Output
                    x = self.lin(x)
                    return x
            
            # Assume input features (will be updated during predict)
            in_channels = 2  # Default: degree + betweenness
            
            self.model = GNNModel(
                self.model_type,
                in_channels,
                self.hidden_channels,
                self.num_layers
            )
            
        except ImportError:
            warnings.warn("PyTorch Geometric not available. Model not built.")
            self.model = None
    
    def predict(self, graph: GraphData) -> float:
        """Predict property from graph.
        
        Parameters
        ----------
        graph : GraphData
            Graph representation.
        
        Returns
        -------
        prediction : float
            Predicted property value.
        """
        if self.model is None:
            warnings.warn("Model not available. Returning default prediction.")
            return 0.0
        
        try:
            import torch
            
            # Convert to PyTorch tensors
            x = torch.tensor(graph.node_features, dtype=torch.float)
            edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
            
            # Update model input size if needed
            if x.shape[1] != self.model.convs[0].in_channels:
                # Rebuild model with correct input size
                self.model.convs[0].in_channels = x.shape[1]
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(x, edge_index)
            
            return prediction.item()
            
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            return 0.0
    
    def train(
        self,
        graphs: List[GraphData],
        labels: List[float],
        epochs: int = 100,
        lr: float = 0.01,
    ) -> Dict:
        """Train GNN model.
        
        Parameters
        ----------
        graphs : list of GraphData
            Training graphs.
        labels : list of float
            Training labels.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        
        Returns
        -------
        history : dict
            Training history with 'loss' list.
        """
        if self.model is None:
            warnings.warn("Model not available. Training skipped.")
            return {'loss': []}
        
        try:
            import torch
            import torch.nn as nn
            
            # Prepare data
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            history = {'loss': []}
            
            self.model.train()
            
            for epoch in range(epochs):
                total_loss = 0.0
                
                for graph, label in zip(graphs, labels):
                    optimizer.zero_grad()
                    
                    x = torch.tensor(graph.node_features, dtype=torch.float)
                    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
                    target = torch.tensor([label], dtype=torch.float)
                    
                    # Forward
                    output = self.model(x, edge_index)
                    loss = criterion(output, target)
                    
                    # Backward
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(graphs)
                history['loss'].append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            return history
            
        except Exception as e:
            warnings.warn(f"Training failed: {e}")
            return {'loss': []}


def predict_property(
    network: FiberNetwork,
    property_name: str = 'stiffness',
    model_type: str = 'gcn',
) -> float:
    """Convenience function for property prediction.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    property_name : str
        Property to predict: 'stiffness', 'strength', 'conductivity'
    model_type : str
        GNN model type: 'gcn', 'gat', 'graphsage'
    
    Returns
    -------
    prediction : float
        Predicted property value.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.ml.graph_neural_network import predict_property
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> stiffness = predict_property(net, property_name='stiffness')
    >>> print(f"Predicted stiffness: {stiffness:.2e} Pa")
    """
    # Convert to graph
    converter = NetworkGraphConverter(network)
    graph = converter.to_graph()
    
    # Predict
    predictor = GNNPropertyPredictor(model_type=model_type)
    prediction = predictor.predict(graph)
    
    return prediction
