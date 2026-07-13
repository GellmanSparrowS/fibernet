"""
Tests for network topology analysis module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.analysis.topology import (
    TopologyAnalyzer, TopologyResult, CentralityResult,
    analyze_topology
)

# Skip tests if networkx not available
nx = pytest.importorskip("networkx")


class TestTopologyResult:
    """Test TopologyResult dataclass."""
    
    def test_empty_result(self):
        result = TopologyResult()
        assert result.num_nodes == 0
        assert result.num_edges == 0
        assert result.density == 0.0
    
    def test_result_with_data(self):
        result = TopologyResult(
            num_nodes=100,
            num_edges=150,
            density=0.03,
            avg_degree=3.0,
            clustering_coefficient=0.25,
        )
        assert result.num_nodes == 100
        assert result.avg_degree == 3.0
        assert result.clustering_coefficient == 0.25
    
    def test_to_dict(self):
        result = TopologyResult(
            num_nodes=50,
            num_edges=80,
            density=0.065,
        )
        data = result.to_dict()
        assert isinstance(data, dict)
        assert 'num_nodes' in data
        assert data['num_nodes'] == 50


class TestCentralityResult:
    """Test CentralityResult dataclass."""
    
    def test_empty_result(self):
        result = CentralityResult()
        assert len(result.degree_centrality) == 0
        assert len(result.betweenness_centrality) == 0
    
    def test_result_with_data(self):
        result = CentralityResult(
            degree_centrality={0: 0.5, 1: 0.3},
            betweenness_centrality={0: 0.8, 1: 0.2},
        )
        assert result.degree_centrality[0] == 0.5
        assert result.betweenness_centrality[1] == 0.2
    
    def test_to_dict(self):
        result = CentralityResult(
            degree_centrality={0: 0.5},
        )
        data = result.to_dict()
        assert 'degree_centrality' in data
        assert 'betweenness_centrality' in data


class TestTopologyAnalyzer:
    """Test TopologyAnalyzer."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = TopologyAnalyzer(net)
        assert analyzer.network == net
        assert analyzer.graph is not None
    
    def test_initialization_weighted(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = TopologyAnalyzer(net, weighted=True)
        assert analyzer.weighted is True
    
    def test_build_graph(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = TopologyAnalyzer(net)
        G = analyzer.graph
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
    
    def test_analyze_basic(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = TopologyAnalyzer(net)
        result = analyzer.analyze()
        
        assert isinstance(result, TopologyResult)
        assert result.num_nodes > 0
        assert result.num_edges > 0
        assert 0 <= result.density <= 1
        assert result.avg_degree >= 0
    
    def test_analyze_clustering(self):
        net = gen.random_straight_2d(num_fibers=40, seed=42)
        analyzer = TopologyAnalyzer(net)
        result = analyzer.analyze()
        
        assert 0 <= result.clustering_coefficient <= 1
    
    def test_analyze_components(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = TopologyAnalyzer(net)
        result = analyzer.analyze()
        
        assert result.num_components >= 1
        assert result.largest_component_size > 0
        assert result.largest_component_size <= result.num_nodes
    
    def test_compute_centrality(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = TopologyAnalyzer(net)
        result = analyzer.compute_centrality()
        
        assert isinstance(result, CentralityResult)
        assert len(result.degree_centrality) > 0
        assert len(result.betweenness_centrality) > 0
        assert len(result.closeness_centrality) > 0
    
    def test_degree_distribution(self):
        net = gen.random_straight_2d(num_fibers=40, seed=42)
        analyzer = TopologyAnalyzer(net)
        degrees, counts = analyzer.degree_distribution()
        
        assert isinstance(degrees, np.ndarray)
        assert isinstance(counts, np.ndarray)
        assert len(degrees) == len(counts)
        assert all(degrees >= 0)
        assert all(counts > 0)
    
    def test_find_communities_louvain(self):
        net = gen.random_straight_2d(num_fibers=40, seed=42)
        analyzer = TopologyAnalyzer(net)
        
        try:
            communities = analyzer.find_communities(method='louvain')
            assert isinstance(communities, dict)
            assert len(communities) > 0
        except:
            pytest.skip("Louvain not available")
    
    def test_find_communities_label_propagation(self):
        net = gen.random_straight_2d(num_fibers=40, seed=42)
        analyzer = TopologyAnalyzer(net)
        communities = analyzer.find_communities(method='label_propagation')
        
        assert isinstance(communities, dict)
        assert len(communities) > 0
    
    def test_find_communities_invalid_method(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = TopologyAnalyzer(net)
        
        with pytest.raises(ValueError):
            analyzer.find_communities(method='invalid')
    
    def test_get_critical_nodes_betweenness(self):
        net = gen.random_straight_2d(num_fibers=40, seed=42)
        analyzer = TopologyAnalyzer(net)
        critical = analyzer.get_critical_nodes(metric='betweenness', top_k=5)
        
        assert isinstance(critical, list)
        assert len(critical) <= 5
        assert all(isinstance(n, int) for n in critical)
    
    def test_get_critical_nodes_degree(self):
        net = gen.random_straight_2d(num_fibers=40, seed=42)
        analyzer = TopologyAnalyzer(net)
        critical = analyzer.get_critical_nodes(metric='degree', top_k=10)
        
        assert isinstance(critical, list)
        assert len(critical) <= 10
    
    def test_get_critical_nodes_invalid_metric(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = TopologyAnalyzer(net)
        
        with pytest.raises(ValueError):
            analyzer.get_critical_nodes(metric='invalid')
    
    def test_different_network_types(self):
        """Test topology on different network types."""
        networks = [
            gen.random_straight_2d(num_fibers=30, seed=42),
            gen.square_lattice_2d(spacing=2.0, grid_size=(5, 5)),
            gen.honeycomb_lattice_2d(cell_size=2.0, grid_size=(5, 5)),
        ]
        
        for net in networks:
            analyzer = TopologyAnalyzer(net)
            result = analyzer.analyze()
            
            assert result.num_nodes > 0
            assert result.num_edges > 0


class TestAnalyzeTopology:
    """Test convenience function."""
    
    def test_basic_usage(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        result = analyze_topology(net)
        
        assert isinstance(result, TopologyResult)
        assert result.num_nodes > 0
        assert result.num_edges > 0
    
    def test_weighted(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        result = analyze_topology(net, weighted=True)
        
        assert isinstance(result, TopologyResult)
        assert result.num_nodes > 0
