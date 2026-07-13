"""Test percolation analysis module."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.analysis import PercolationAnalyzer, estimate_percolation_threshold


class TestPercolationAnalyzer:
    def test_basic_analysis(self):
        net = gen.random_straight_2d(num_fibers=80, fiber_length=10.0, box_size=(30, 30), radius=0.2, seed=42)
        analyzer = PercolationAnalyzer(net)
        result = analyzer.analyze()
        assert result is not None
        assert isinstance(result.percolates, bool)
        assert result.largest_cluster_size > 0
        assert result.percolation_probability >= 0
        assert result.percolation_probability <= 1

    def test_cluster_analysis(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(40, 40), radius=0.1, seed=123)
        analyzer = PercolationAnalyzer(net)
        clusters = analyzer.cluster_analysis()
        assert 'n_clusters' in clusters
        assert 'cluster_sizes' in clusters
        assert 'largest_cluster' in clusters
        assert 'smallest_cluster' in clusters
        assert clusters['n_clusters'] > 0

    def test_high_density_percolates(self):
        net = gen.random_straight_2d(num_fibers=150, fiber_length=15.0, box_size=(30, 30), radius=0.2, seed=42)
        analyzer = PercolationAnalyzer(net)
        result = analyzer.analyze()
        assert result.percolates is True
        assert result.largest_cluster_size > 50

    def test_low_density_no_percolation(self):
        net = gen.random_straight_2d(num_fibers=10, fiber_length=3.0, box_size=(100, 100), radius=0.05, seed=999)
        analyzer = PercolationAnalyzer(net)
        result = analyzer.analyze()
        # Low density should not percolate
        assert result.largest_cluster_size <= 10

    def test_find_percolating_path(self):
        net = gen.random_straight_2d(num_fibers=100, fiber_length=12.0, box_size=(30, 30), radius=0.2, seed=42)
        analyzer = PercolationAnalyzer(net)
        path = analyzer.find_percolating_path()
        # Dense network should have a path
        if path:
            assert len(path) > 0

    def test_custom_contact_distance(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(30, 30), radius=0.1, seed=42)
        analyzer1 = PercolationAnalyzer(net, contact_distance=0.5)
        analyzer2 = PercolationAnalyzer(net, contact_distance=2.0)
        result1 = analyzer1.analyze()
        result2 = analyzer2.analyze()
        # Larger contact distance should create more connections
        assert result2.largest_cluster_size >= result1.largest_cluster_size

    def test_3d_network(self):
        net = gen.random_straight_3d(num_fibers=100, fiber_length=15.0, box_size=(30, 30, 30), radius=0.2, seed=42)
        analyzer = PercolationAnalyzer(net)
        result = analyzer.analyze()
        assert result is not None
        assert result.largest_cluster_size > 0

    def test_effective_conductivity(self):
        net = gen.random_straight_2d(num_fibers=150, fiber_length=15.0, box_size=(30, 30), radius=0.2, seed=42)
        analyzer = PercolationAnalyzer(net)
        result = analyzer.analyze()
        assert result.effective_conductivity >= 0.0
        if result.percolates:
            assert result.effective_conductivity > 0.0
        else:
            assert result.effective_conductivity == 0.0
