"""Tests for GNN feature extraction."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.ml import GNNFeatureExtractor

pytest.importorskip("sklearn")



class TestGNNFeatureExtractor:
    """Test GNNFeatureExtractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.network = gen.random_straight_2d(
            num_fibers=30,
            fiber_length=10.0,
            box_size=(20, 20),
            seed=42
        )
        self.extractor = GNNFeatureExtractor(
            node_features=['position', 'degree'],
            edge_features=['length', 'angle']
        )
    
    def test_extractor_creation(self):
        """Test creating extractor."""
        extractor = GNNFeatureExtractor()
        assert extractor.node_features == ['position', 'degree']
        assert extractor.edge_features == ['length', 'angle']
    
    def test_extractor_custom_features(self):
        """Test creating extractor with custom features."""
        extractor = GNNFeatureExtractor(
            node_features=['position', 'degree', 'centrality'],
            edge_features=['length', 'angle', 'weight']
        )
        assert 'centrality' in extractor.node_features
        assert 'weight' in extractor.edge_features
    
    def test_extract_graph(self):
        """Test extracting graph representation."""
        graph_data = self.extractor.extract_graph(self.network)
        
        assert 'node_features' in graph_data
        assert 'edge_index' in graph_data
        assert 'edge_features' in graph_data
        assert 'num_nodes' in graph_data
        assert 'num_edges' in graph_data
        
        assert graph_data['num_nodes'] == self.network.num_fibers
        assert graph_data['node_features'].shape[0] == self.network.num_fibers
    
    def test_extract_graph_shapes(self):
        """Test graph data shapes."""
        graph_data = self.extractor.extract_graph(self.network)
        
        # Node features: (num_nodes, num_features)
        assert graph_data['node_features'].ndim == 2
        assert graph_data['node_features'].shape[0] == graph_data['num_nodes']
        
        # Edge index: (2, num_edges)
        assert graph_data['edge_index'].ndim == 2
        assert graph_data['edge_index'].shape[0] == 2
        assert graph_data['edge_index'].shape[1] == graph_data['num_edges']
        
        # Edge features: (num_edges, num_features)
        if graph_data['num_edges'] > 0:
            assert graph_data['edge_features'].ndim == 2
            assert graph_data['edge_features'].shape[0] == graph_data['num_edges']
    
    def test_extract_graph_values(self):
        """Test graph data values are reasonable."""
        graph_data = self.extractor.extract_graph(self.network)
        
        # Node features should be finite
        assert np.all(np.isfinite(graph_data['node_features']))
        
        # Edge features should be finite (if any)
        if graph_data['num_edges'] > 0:
            assert np.all(np.isfinite(graph_data['edge_features']))
        
        # Edge index should be valid
        if graph_data['num_edges'] > 0:
            assert np.all(graph_data['edge_index'] >= 0)
            assert np.all(graph_data['edge_index'] < graph_data['num_nodes'])
    
    def test_to_pytorch_geometric(self):
        """Test conversion to PyTorch Geometric."""
        pytest.importorskip('torch_geometric')
        
        data = self.extractor.to_pytorch_geometric(self.network)
        
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert data.x.shape[0] == self.network.num_fibers
    
    def test_to_pytorch_geometric_with_label(self):
        """Test PyTorch Geometric with label."""
        pytest.importorskip('torch_geometric')
        
        data = self.extractor.to_pytorch_geometric(self.network, label=1.5)
        
        assert hasattr(data, 'y')
        assert data.y.item() == 1.5
    
    def test_to_dgl(self):
        """Test conversion to DGL."""
        pytest.importorskip('dgl')
        
        g = self.extractor.to_dgl(self.network)
        
        assert g.num_nodes() == self.network.num_fibers
        assert 'feat' in g.ndata
    
    def test_to_dgl_with_label(self):
        """Test DGL with label."""
        pytest.importorskip('dgl')
        
        g = self.extractor.to_dgl(self.network, label=2.0)
        
        assert hasattr(g, 'label')
        assert g.label.item() == 2.0
    
    def test_create_dataset_pyg(self):
        """Test creating PyTorch Geometric dataset."""
        pytest.importorskip('torch_geometric')
        
        networks = [
            gen.random_straight_2d(num_fibers=20, fiber_length=8.0, box_size=(15, 15), seed=i)
            for i in range(5)
        ]
        labels = [float(i) for i in range(5)]
        
        dataset = self.extractor.create_dataset(networks, labels, format='pyg')
        
        assert len(dataset) == 5
        for i, data in enumerate(dataset):
            assert hasattr(data, 'y')
            assert data.y.item() == float(i)
    
    def test_create_dataset_dgl(self):
        """Test creating DGL dataset."""
        pytest.importorskip('dgl')
        
        networks = [
            gen.random_straight_2d(num_fibers=20, fiber_length=8.0, box_size=(15, 15), seed=i)
            for i in range(5)
        ]
        labels = [float(i) for i in range(5)]
        
        dataset = self.extractor.create_dataset(networks, labels, format='dgl')
        
        assert len(dataset) == 5
        for i, g in enumerate(dataset):
            assert hasattr(g, 'label')
            assert g.label.item() == float(i)
    
    def test_create_dataset_invalid_format(self):
        """Test invalid format raises error."""
        networks = [self.network]
        
        with pytest.raises(ValueError, match="Unknown format"):
            self.extractor.create_dataset(networks, format='invalid')
    
    def test_node_features_content(self):
        """Test node features contain expected values."""
        extractor = GNNFeatureExtractor(
            node_features=['position', 'degree'],
            edge_features=['length']
        )
        
        graph_data = extractor.extract_graph(self.network)
        
        # Should have position (3D) + degree (1) + fiber properties (length, radius)
        expected_features = 3 + 1 + 2  # position, degree, length, radius
        assert graph_data['node_features'].shape[1] == expected_features
    
    def test_edge_features_content(self):
        """Test edge features contain expected values."""
        extractor = GNNFeatureExtractor(
            node_features=['position'],
            edge_features=['length', 'angle']
        )
        
        graph_data = extractor.extract_graph(self.network)
        
        if graph_data['num_edges'] > 0:
            # Should have length + angle
            assert graph_data['edge_features'].shape[1] == 2
    
    def test_networkx_graph(self):
        """Test NetworkX graph creation."""
        graph_data = self.extractor.extract_graph(self.network)
        
        assert 'graph' in graph_data
        
        G = graph_data['graph']
        assert G.number_of_nodes() == self.network.num_fibers
        assert G.number_of_edges() == self.network.num_crosslinks


class TestGNNFeatureExtractorEdgeCases:
    """Test edge cases for GNNFeatureExtractor."""
    
    def test_empty_network(self):
        """Test with empty network."""
        from fibernet.core import FiberNetwork
        network = FiberNetwork()
        extractor = GNNFeatureExtractor()
        
        graph_data = extractor.extract_graph(network)
        
        assert graph_data['num_nodes'] == 0
        assert graph_data['num_edges'] == 0
    
    def test_single_fiber(self):
        """Test with single fiber."""
        network = gen.random_straight_2d(num_fibers=1, fiber_length=10.0, box_size=(20, 20), seed=42)
        extractor = GNNFeatureExtractor()
        
        graph_data = extractor.extract_graph(network)
        
        assert graph_data['num_nodes'] == 1
    
    def test_no_crosslinks(self):
        """Test with no crosslinks."""
        # Create network with very small fibers (unlikely to crosslink)
        network = gen.random_straight_2d(
            num_fibers=10,
            fiber_length=0.1,
            box_size=(100, 100),
            seed=42
        )
        extractor = GNNFeatureExtractor()
        
        graph_data = extractor.extract_graph(network)
        
        # May have 0 crosslinks
        assert graph_data['num_edges'] >= 0


class TestGNNFeatureExtractor3D:
    """Test GNNFeatureExtractor with 3D networks."""
    
    def test_3d_network(self):
        """Test with 3D network."""
        network = gen.random_straight_3d(
            num_fibers=20,
            fiber_length=10.0,
            box_size=(20, 20, 20),
            seed=42
        )
        extractor = GNNFeatureExtractor()
        
        graph_data = extractor.extract_graph(network)
        
        assert graph_data['num_nodes'] == network.num_fibers
        assert graph_data['node_features'].shape[0] == network.num_fibers
