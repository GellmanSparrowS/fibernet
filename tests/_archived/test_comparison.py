"""Tests for network comparison and similarity analysis."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.analysis.comparison import (
    NetworkFingerprint, NetworkComparator, compare_networks, network_similarity
)


class TestNetworkFingerprint:
    """Test network fingerprint computation."""
    
    def test_fingerprint_shape(self):
        """Test fingerprint has correct shape."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        fp = NetworkFingerprint(net)
        fingerprint = fp.compute_fingerprint()
        assert len(fingerprint) == 10
    
    def test_fingerprint_features(self):
        """Test fingerprint contains expected features."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        fp = NetworkFingerprint(net)
        names = fp.get_feature_names()
        assert 'nematic_order' in names
        assert 'anisotropy_index' in names
        assert 'length_mean' in names
    
    def test_fingerprint_different_networks(self):
        """Test that different networks have different fingerprints."""
        net1 = gen.random_straight_2d(num_fibers=50, seed=42)
        net2 = gen.random_straight_2d(num_fibers=100, seed=43)
        
        fp1 = NetworkFingerprint(net1).compute_fingerprint()
        fp2 = NetworkFingerprint(net2).compute_fingerprint()
        
        # Should be different
        assert not np.allclose(fp1, fp2)
    
    def test_fingerprint_same_network(self):
        """Test that same network has same fingerprint."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        
        fp1 = NetworkFingerprint(net).compute_fingerprint()
        fp2 = NetworkFingerprint(net).compute_fingerprint()
        
        assert np.allclose(fp1, fp2, equal_nan=True)


class TestNetworkComparator:
    """Test network comparison."""
    
    def test_pairwise_distances(self):
        """Test pairwise distance computation."""
        nets = [
            gen.random_straight_2d(num_fibers=50, seed=42),
            gen.random_straight_2d(num_fibers=60, seed=43),
            gen.random_straight_2d(num_fibers=70, seed=44),
        ]
        
        comparator = NetworkComparator(nets)
        distances = comparator.pairwise_distances()
        
        assert distances.shape == (3, 3)
        assert np.all(distances >= 0)
        assert np.allclose(np.diag(distances), 0)
    
    def test_most_similar(self):
        """Test finding most similar networks."""
        nets = [
            gen.random_straight_2d(num_fibers=50, seed=42),
            gen.random_straight_2d(num_fibers=60, seed=43),
            gen.random_straight_2d(num_fibers=70, seed=44),
        ]
        
        comparator = NetworkComparator(nets)
        similar = comparator.most_similar(query_idx=0, top_k=2)
        
        assert len(similar) == 2
        assert all(isinstance(s, tuple) for s in similar)
        assert all(len(s) == 2 for s in similar)
    
    def test_compare_statistics(self):
        """Test statistics comparison."""
        nets = [
            gen.random_straight_2d(num_fibers=50, seed=42),
            gen.random_straight_2d(num_fibers=60, seed=43),
        ]
        
        comparator = NetworkComparator(nets)
        stats = comparator.compare_statistics()
        
        assert 'network_0' in stats
        assert 'network_1' in stats
        assert stats['network_0']['n_fibers'] == 50
        assert stats['network_1']['n_fibers'] == 60


class TestCompareNetworks:
    """Test compare_networks function."""
    
    def test_compare_two_networks(self):
        """Test comparing two networks."""
        nets = [
            gen.random_straight_2d(num_fibers=50, seed=42),
            gen.random_straight_2d(num_fibers=60, seed=43),
        ]
        
        results = compare_networks(nets)
        
        assert 'distances' in results
        assert 'statistics' in results
        assert results['n_networks'] == 2
    
    def test_compare_multiple_networks(self):
        """Test comparing multiple networks."""
        nets = [
            gen.random_straight_2d(num_fibers=50, seed=i)
            for i in range(5)
        ]
        
        results = compare_networks(nets)
        
        assert results['n_networks'] == 5
        assert results['distances'].shape == (5, 5)
    
    def test_compare_single_network_error(self):
        """Test that comparing single network raises error."""
        nets = [gen.random_straight_2d(num_fibers=50, seed=42)]
        
        with pytest.raises(ValueError):
            compare_networks(nets)


class TestNetworkSimilarity:
    """Test network similarity function."""
    
    def test_similarity_same_network(self):
        """Test similarity of network with itself."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        sim = network_similarity(net, net)
        
        # Should be close to 1
        assert sim > 0.9
    
    def test_similarity_different_networks(self):
        """Test similarity of different networks."""
        net1 = gen.random_straight_2d(num_fibers=50, seed=42)
        net2 = gen.random_straight_2d(num_fibers=100, seed=43)
        
        sim = network_similarity(net1, net2)
        
        assert 0 <= sim <= 1
    
    def test_similarity_symmetric(self):
        """Test that similarity is symmetric."""
        net1 = gen.random_straight_2d(num_fibers=50, seed=42)
        net2 = gen.random_straight_2d(num_fibers=60, seed=43)
        
        sim1 = network_similarity(net1, net2)
        sim2 = network_similarity(net2, net1)
        
        assert np.isclose(sim1, sim2)
    
    def test_similarity_different_types(self):
        """Test similarity between different network types."""
        net1 = gen.random_straight_2d(num_fibers=50, seed=42)
        net2 = gen.square_lattice_2d(spacing=2.0, grid_size=(5, 5))
        
        sim = network_similarity(net1, net2)
        
        # Should be less than 1 (different types)
        assert sim < 1.0
        assert sim >= 0
