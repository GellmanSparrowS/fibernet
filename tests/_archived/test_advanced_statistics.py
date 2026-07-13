"""Tests for advanced statistical analysis."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.analysis.spatial import (
    SpatialStatistics, OrientationAnalysis, LengthAnalysis,
    ConnectivityAnalysis, AnisotropyAnalysis, compute_spatial_statistics
)


class TestSpatialStatistics:
    """Test spatial statistics."""
    
    def test_ripley_k_empty(self):
        """Test Ripley's K with empty network."""
        from fibernet.core.network import FiberNetwork
        net = FiberNetwork(dimension=2)
        spatial = SpatialStatistics(net)
        r = np.linspace(0, 10, 10)
        K = spatial.ripley_k(r)
        assert len(K) == len(r)
    
    def test_ripley_k_basic(self):
        """Test Ripley's K with basic network."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        spatial = SpatialStatistics(net)
        r = np.linspace(1, 10, 10)
        K = spatial.ripley_k(r)
        assert len(K) == len(r)
        assert all(K >= 0)
    
    def test_pair_correlation(self):
        """Test pair correlation function."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        spatial = SpatialStatistics(net)
        r = np.linspace(1, 10, 20)
        g = spatial.pair_correlation(r)
        assert len(g) == len(r)
        assert all(g >= 0)
    
    def test_nearest_neighbor_distances(self):
        """Test nearest neighbor distances."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        spatial = SpatialStatistics(net)
        distances = spatial.nearest_neighbor_distances()
        assert len(distances) > 0
        assert all(d > 0 for d in distances)


class TestOrientationAnalysis:
    """Test orientation analysis."""
    
    def test_get_orientations(self):
        """Test getting orientation vectors."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        orient = OrientationAnalysis(net)
        orientations = orient.get_orientations()
        assert len(orientations) == len(net.fibers)
        assert orientations.shape[1] == 3
    
    def test_orientation_histogram_2d(self):
        """Test 2D orientation histogram."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        orient = OrientationAnalysis(net)
        counts, bin_edges = orient.orientation_histogram(n_bins=18, dimension='2d')
        assert len(counts) == 18
        assert len(bin_edges) == 19
        assert sum(counts) == len(net.fibers)
    
    def test_orientation_histogram_3d(self):
        """Test 3D orientation histogram."""
        net = gen.random_straight_3d(num_fibers=50, seed=42)
        orient = OrientationAnalysis(net)
        counts, bin_edges = orient.orientation_histogram(n_bins=18, dimension='3d')
        assert len(counts) == 18
        assert len(bin_edges) == 19
    
    def test_nematic_order_isotropic(self):
        """Test nematic order for isotropic network."""
        net = gen.random_straight_2d(num_fibers=100, seed=42)
        orient = OrientationAnalysis(net)
        S = orient.nematic_order_parameter()
        assert 0 <= S <= 1
        # Random network should have low order
        assert S < 0.3
    
    def test_nematic_order_aligned(self):
        """Test nematic order for aligned network."""
        net = gen.square_lattice_2d(spacing=2.0, grid_size=(5, 5))
        orient = OrientationAnalysis(net)
        S = orient.nematic_order_parameter()
        assert 0 <= S <= 1
    
    def test_mean_orientation(self):
        """Test mean orientation."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        orient = OrientationAnalysis(net)
        mean = orient.mean_orientation()
        assert len(mean) == 3
        assert np.linalg.norm(mean) <= 1.0 + 1e-6


class TestLengthAnalysis:
    """Test length analysis."""
    
    def test_get_lengths(self):
        """Test getting fiber lengths."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        length = LengthAnalysis(net)
        lengths = length.get_lengths()
        assert len(lengths) == len(net.fibers)
        assert all(l > 0 for l in lengths)
    
    def test_length_statistics(self):
        """Test length statistics."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        length = LengthAnalysis(net)
        stats = length.length_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert stats['min'] <= stats['mean'] <= stats['max']
    
    def test_length_histogram(self):
        """Test length histogram."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        length = LengthAnalysis(net)
        counts, bin_edges = length.length_histogram(n_bins=20)
        assert len(counts) == 20
        assert len(bin_edges) == 21
    
    def test_length_histogram_density(self):
        """Test length histogram with density normalization."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        length = LengthAnalysis(net)
        counts, bin_edges = length.length_histogram(n_bins=20, density=True)
        # Check that integral is approximately 1
        dr = bin_edges[1] - bin_edges[0]
        integral = np.sum(counts) * dr
        assert np.isclose(integral, 1.0, atol=0.1)


class TestConnectivityAnalysis:
    """Test connectivity analysis."""
    
    def test_degree_distribution(self):
        """Test degree distribution."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        conn = ConnectivityAnalysis(net)
        degrees = conn.degree_distribution()
        assert isinstance(degrees, dict)
    
    def test_mean_connectivity(self):
        """Test mean connectivity."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        conn = ConnectivityAnalysis(net)
        mean = conn.mean_connectivity()
        assert mean >= 0
    
    def test_connectivity_statistics(self):
        """Test connectivity statistics."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        conn = ConnectivityAnalysis(net)
        stats = conn.connectivity_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats


class TestAnisotropyAnalysis:
    """Test anisotropy analysis."""
    
    def test_fabric_tensor(self):
        """Test fabric tensor computation."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        aniso = AnisotropyAnalysis(net)
        A = aniso.fabric_tensor()
        assert A.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(A, A.T)
    
    def test_anisotropy_index(self):
        """Test anisotropy index."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        aniso = AnisotropyAnalysis(net)
        AI = aniso.anisotropy_index()
        assert 0 <= AI <= 1
    
    def test_principal_directions(self):
        """Test principal directions."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        aniso = AnisotropyAnalysis(net)
        eigenvalues, eigenvectors = aniso.principal_directions()
        assert len(eigenvalues) == 3
        assert eigenvectors.shape == (3, 3)
        # Eigenvalues should be in descending order
        assert eigenvalues[0] >= eigenvalues[1] >= eigenvalues[2]


class TestComputeAllStatistics:
    """Test comprehensive statistics computation."""
    
    def test_compute_all_2d(self):
        """Test computing all statistics for 2D network."""
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        stats = compute_spatial_statistics(net)
        assert 'n_fibers' in stats
        assert 'n_crosslinks' in stats
        assert 'length' in stats
        assert 'nematic_order' in stats
        assert 'connectivity' in stats
        assert 'anisotropy_index' in stats
    
    def test_compute_all_3d(self):
        """Test computing all statistics for 3D network."""
        net = gen.random_straight_3d(num_fibers=50, seed=42)
        stats = compute_spatial_statistics(net)
        assert 'n_fibers' in stats
        assert 'dimension' in stats
        assert stats['dimension'] == 3
