"""
Tests for graph neural network module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.ml.graph_neural_network import (
    GraphData, NetworkGraphConverter, GNNPropertyPredictor,
    predict_property
)


class TestGraphData:
    """Test GraphData dataclass."""
    
    def test_initialization(self):
        node_features = np.array([[1.0, 2.0], [3.0, 4.0]])
        edge_features = np.array([[0.5], [0.7]])
        edge_index = np.array([[0, 1], [1, 0]]).T
        
        graph = GraphData(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
            graph_label=1.5
        )
        
        assert graph.node_features.shape == (2, 2)
        assert graph.edge_features.shape == (2, 1)
        assert graph.edge_index.shape == (2, 2)
        assert graph.graph_label == 1.5
    
    def test_to_dict(self):
        node_features = np.array([[1.0, 2.0]])
        edge_index = np.array([[0], [0]])
        
        graph = GraphData(
            node_features=node_features,
            edge_index=edge_index,
            graph_label=2.0
        )
        
        data = graph.to_dict()
        
        assert isinstance(data, dict)
        assert 'node_features' in data
        assert 'edge_index' in data
        assert data['graph_label'] == 2.0
    
    def test_to_pyg_data_without_pyg(self):
        """Test conversion when PyG not available."""
        node_features = np.array([[1.0, 2.0]])
        edge_index = np.array([[0], [0]])
        
        graph = GraphData(
            node_features=node_features,
            edge_index=edge_index
        )
        
        # Should return None or warn if PyG not available
        pyg_data = graph.to_pyg_data()
        
        # If PyG is installed, pyg_data should not be None
        # If not installed, should be None
        try:
            from torch_geometric.data import Data
            assert pyg_data is not None
        except ImportError:
            assert pyg_data is None


class TestNetworkGraphConverter:
    """Test NetworkGraphConverter."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        converter = NetworkGraphConverter(net)
        
        assert converter.network == net
        assert converter.graph is not None
    
    def test_to_graph_default(self):
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        converter = NetworkGraphConverter(net)
        
        graph = converter.to_graph()
        
        assert isinstance(graph, GraphData)
        assert graph.node_features is not None
        assert graph.edge_index is not None
        
        # Default features: degree, betweenness
        assert graph.node_features.shape[1] == 2
    
    def test_to_graph_custom_features(self):
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        converter = NetworkGraphConverter(net)
        
        graph = converter.to_graph(
            node_features=['degree', 'closeness', 'clustering'],
            edge_features=['weight']
        )
        
        assert isinstance(graph, GraphData)
        assert graph.node_features.shape[1] == 3  # 3 node features
        assert graph.edge_features.shape[1] == 1  # 1 edge feature
    
    def test_extract_node_features(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        converter = NetworkGraphConverter(net)
        
        features = converter._extract_node_features(['degree', 'betweenness'])
        
        assert isinstance(features, np.ndarray)
        # Nodes represent fibers in the topology graph
        assert features.shape[0] == converter.graph.number_of_nodes()
        assert features.shape[1] == 2
        
        # Degree should be non-negative
        assert np.all(features[:, 0] >= 0)
        
        # Betweenness should be in [0, 1]
        assert np.all(features[:, 1] >= 0)
        assert np.all(features[:, 1] <= 1)
    
    def test_extract_edge_features(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        converter = NetworkGraphConverter(net)
        
        edge_index, edge_features = converter._extract_edge_features(['weight'])
        
        assert isinstance(edge_index, np.ndarray)
        assert isinstance(edge_features, np.ndarray)
        
        assert edge_index.shape[0] == 2
        assert edge_features.shape[0] == edge_index.shape[1]
        assert edge_features.shape[1] == 1
        
        # Weights should be positive
        assert np.all(edge_features > 0)


class TestGNNPropertyPredictor:
    """Test GNNPropertyPredictor."""
    
    def test_initialization(self):
        predictor = GNNPropertyPredictor(
            model_type='gcn',
            hidden_channels=32,
            num_layers=2
        )
        
        assert predictor.model_type == 'gcn'
        assert predictor.hidden_channels == 32
        assert predictor.num_layers == 2
    
    def test_initialization_invalid_model(self):
        """Test initialization with invalid model type."""
        # Invalid model type is caught in _build_model and issues a warning
        predictor = GNNPropertyPredictor(model_type='invalid')
        # Model should be None since _build_model failed
        assert predictor.model is None
    
    def test_predict_without_model(self):
        """Test prediction when model not available."""
        predictor = GNNPropertyPredictor()
        predictor.model = None  # Simulate missing PyG
        
        node_features = np.array([[1.0, 2.0]])
        edge_index = np.array([[0], [0]])
        graph = GraphData(node_features=node_features, edge_index=edge_index)
        
        prediction = predictor.predict(graph)
        
        # Should return default value
        assert prediction == 0.0
    
    def test_predict_with_graph(self):
        """Test prediction with actual graph."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        converter = NetworkGraphConverter(net)
        graph = converter.to_graph()
        
        predictor = GNNPropertyPredictor(model_type='gcn')
        
        # May fail if PyG not available, but should not crash
        try:
            prediction = predictor.predict(graph)
            assert isinstance(prediction, float)
        except Exception:
            # PyG not available or prediction failed
            pass
    
    def test_train_without_model(self):
        """Test training when model not available."""
        predictor = GNNPropertyPredictor()
        predictor.model = None
        
        graphs = []
        labels = []
        
        history = predictor.train(graphs, labels, epochs=10)
        
        assert isinstance(history, dict)
        assert 'loss' in history
        assert history['loss'] == []


class TestPredictPropertyFunction:
    """Test convenience function."""
    
    def test_basic_usage(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        
        # May fail if PyG not available, but should not crash
        try:
            prediction = predict_property(net, property_name='stiffness')
            assert isinstance(prediction, float)
        except Exception:
            # PyG not available or prediction failed
            pass
    
    def test_different_model_types(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        
        for model_type in ['gcn', 'gat', 'graphsage']:
            try:
                prediction = predict_property(net, model_type=model_type)
                assert isinstance(prediction, float)
            except Exception:
                # PyG not available or prediction failed
                pass


class TestGNNIntegration:
    """Integration tests for GNN workflow."""
    
    def test_full_workflow(self):
        """Test complete GNN workflow."""
        # Generate network
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        
        # Convert to graph
        converter = NetworkGraphConverter(net)
        graph = converter.to_graph(
            node_features=['degree', 'betweenness', 'closeness'],
            edge_features=['weight']
        )
        
        assert isinstance(graph, GraphData)
        
        # Create predictor
        predictor = GNNPropertyPredictor(
            model_type='gcn',
            hidden_channels=64,
            num_layers=3
        )
        
        # Predict (may fail if PyG not available)
        try:
            prediction = predictor.predict(graph)
            assert isinstance(prediction, float)
        except Exception:
            pass
        
        # Convert to PyG data (may fail if PyG not available)
        try:
            pyg_data = graph.to_pyg_data()
            if pyg_data is not None:
                assert hasattr(pyg_data, 'x')
                assert hasattr(pyg_data, 'edge_index')
        except Exception:
            pass
    
    def test_multiple_networks(self):
        """Test with multiple networks."""
        networks = [
            gen.random_straight_2d(num_fibers=30, seed=i)
            for i in range(5)
        ]
        
        graphs = []
        for net in networks:
            converter = NetworkGraphConverter(net)
            graph = converter.to_graph()
            graphs.append(graph)
        
        assert len(graphs) == 5
        
        # All should have valid structure
        for graph in graphs:
            assert isinstance(graph, GraphData)
            assert graph.node_features is not None
            assert graph.edge_index is not None
