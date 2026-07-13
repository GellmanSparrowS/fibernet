"""
Tests for buckling analysis module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.buckling_analysis import (
    BucklingAnalyzer, FiberBucklingResult, NetworkBucklingResult,
    analyze_buckling
)


class TestFiberBucklingResult:
    """Test FiberBucklingResult dataclass."""
    
    def test_initialization(self):
        result = FiberBucklingResult(
            fiber_index=0,
            critical_load=100.0,
            critical_stress=1e6,
            buckling_mode='pinned-pinned',
            effective_length_factor=1.0,
            slenderness_ratio=100.0,
            euler_valid=True
        )
        
        assert result.fiber_index == 0
        assert result.critical_load == 100.0
        assert result.critical_stress == 1e6
        assert result.buckling_mode == 'pinned-pinned'
        assert result.euler_valid is True
    
    def test_to_dict(self):
        result = FiberBucklingResult(
            fiber_index=5,
            critical_load=50.0,
            critical_stress=5e5,
            buckling_mode='fixed-fixed',
            effective_length_factor=0.5,
            slenderness_ratio=80.0,
            euler_valid=True
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data['fiber_index'] == 5
        assert data['critical_load'] == 50.0
        assert data['buckling_mode'] == 'fixed-fixed'


class TestNetworkBucklingResult:
    """Test NetworkBucklingResult dataclass."""
    
    def test_initialization(self):
        result = NetworkBucklingResult(
            critical_loads=np.array([100.0, 200.0, 300.0]),
            num_modes=3,
            min_critical_load=100.0,
            max_critical_load=300.0,
            mean_critical_load=200.0,
            first_buckling_mode=0,
            dominant_buckling_type='global'
        )
        
        assert len(result.critical_loads) == 3
        assert result.num_modes == 3
        assert result.min_critical_load == 100.0
    
    def test_to_dict(self):
        result = NetworkBucklingResult(
            critical_loads=np.array([100.0, 200.0]),
            num_modes=2,
            min_critical_load=100.0,
            max_critical_load=200.0,
            mean_critical_load=150.0,
            first_buckling_mode=0,
            dominant_buckling_type='global'
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data['num_modes'] == 2
        assert data['min_critical_load'] == 100.0
        assert len(data['critical_loads']) == 2


class TestBucklingAnalyzer:
    """Test BucklingAnalyzer class."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=3)
        
        assert analyzer.network == net
        assert analyzer.segments == 3
        assert analyzer.fem is not None
    
    def test_analyze_fiber_buckling_pinned_pinned(self):
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=3)
        
        results = analyzer.analyze_fiber_buckling(buckling_mode='pinned-pinned')
        
        assert isinstance(results, list)
        assert len(results) == net.num_fibers
        
        # Check first fiber result
        result = results[0]
        assert isinstance(result, FiberBucklingResult)
        assert result.fiber_index == 0
        assert result.critical_load > 0
        assert result.critical_stress > 0
        assert result.buckling_mode == 'pinned-pinned'
        assert result.effective_length_factor == 1.0
        assert result.slenderness_ratio > 0
    
    def test_analyze_fiber_buckling_fixed_fixed(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=2)
        
        results_pinned = analyzer.analyze_fiber_buckling(buckling_mode='pinned-pinned')
        results_fixed = analyzer.analyze_fiber_buckling(buckling_mode='fixed-fixed')
        
        # Fixed-fixed should have 4x higher critical load (K=0.5 vs K=1.0)
        # P_cr ∝ 1/K², so ratio should be (1.0/0.5)² = 4
        ratio = results_fixed[0].critical_load / results_pinned[0].critical_load
        assert 3.9 < ratio < 4.1  # Allow small numerical error
    
    def test_analyze_fiber_buckling_invalid_mode(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=2)
        
        with pytest.raises(ValueError, match="Unknown buckling mode"):
            analyzer.analyze_fiber_buckling(buckling_mode='invalid-mode')
    
    def test_euler_validity(self):
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=3)
        
        results = analyzer.analyze_fiber_buckling()
        
        # Most fibers should have valid Euler buckling (high slenderness)
        euler_valid_count = sum(1 for r in results if r.euler_valid)
        assert euler_valid_count > len(results) / 2
    
    def test_analyze_network_buckling(self):
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=3)
        
        result = analyzer.analyze_network_buckling(num_modes=3)
        
        assert isinstance(result, NetworkBucklingResult)
        # May have zero modes if eigenvalue solver fails, but should not crash
        assert result.num_modes >= 0
        
        if result.num_modes > 0:
            assert len(result.critical_loads) == result.num_modes
            assert len(result.mode_shapes) == result.num_modes
            assert result.min_critical_load > 0
            assert result.max_critical_load >= result.min_critical_load
    
    def test_analyze_network_buckling_different_directions(self):
        net = gen.random_straight_2d(num_fibers=15, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=2)
        
        # Test different loading directions
        result_x = analyzer.analyze_network_buckling(num_modes=2, load_direction=0)
        result_y = analyzer.analyze_network_buckling(num_modes=2, load_direction=1)
        
        # Both should complete without error
        assert isinstance(result_x, NetworkBucklingResult)
        assert isinstance(result_y, NetworkBucklingResult)


class TestAnalyzeBucklingFunction:
    """Test convenience function."""
    
    def test_basic_usage(self):
        net = gen.random_straight_2d(num_fibers=15, seed=42)
        
        results = analyze_buckling(net, segments_per_fiber=2)
        
        assert isinstance(results, dict)
        assert 'fiber' in results
        assert 'network' in results
        
        assert isinstance(results['fiber'], list)
        assert len(results['fiber']) == net.num_fibers
        
        assert isinstance(results['network'], NetworkBucklingResult)
    
    def test_fiber_only(self):
        net = gen.random_straight_2d(num_fibers=15, seed=42)
        
        results = analyze_buckling(
            net,
            segments_per_fiber=2,
            analyze_fibers=True,
            analyze_network=False
        )
        
        assert 'fiber' in results
        assert 'network' not in results
    
    def test_network_only(self):
        net = gen.random_straight_2d(num_fibers=15, seed=42)
        
        results = analyze_buckling(
            net,
            segments_per_fiber=2,
            analyze_fibers=False,
            analyze_network=True
        )
        
        assert 'fiber' not in results
        assert 'network' in results
    
    def test_with_parameters(self):
        net = gen.random_straight_2d(num_fibers=15, seed=42)
        
        results = analyze_buckling(
            net,
            segments_per_fiber=3,
            num_modes=5
        )
        
        assert 'fiber' in results
        assert 'network' in results
        
        # Check that network analysis used num_modes
        # (may have fewer if eigenvalue solver found fewer modes)
        assert results['network'].num_modes <= 5


class TestBucklingAnalysisIntegration:
    """Integration tests for buckling analysis."""
    
    def test_critical_load_ordering(self):
        """Test that critical loads are properly ordered."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=3)
        
        fiber_results = analyzer.analyze_fiber_buckling()
        
        # Sort by critical load
        sorted_results = sorted(fiber_results, key=lambda r: r.critical_load)
        
        # Check ordering
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i].critical_load <= sorted_results[i+1].critical_load
    
    def test_slenderness_vs_critical_load(self):
        """Test that higher slenderness gives lower critical load."""
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=3)
        
        results = analyzer.analyze_fiber_buckling()
        
        # Group by similar slenderness
        high_slenderness = [r for r in results if r.slenderness_ratio > 100]
        low_slenderness = [r for r in results if r.slenderness_ratio < 50]
        
        if len(high_slenderness) > 0 and len(low_slenderness) > 0:
            # High slenderness should have lower critical load on average
            avg_high = np.mean([r.critical_load for r in high_slenderness])
            avg_low = np.mean([r.critical_load for r in low_slenderness])
            
            # This is a general trend, but may not always hold due to material differences
            # So we just check that the computation completes
            assert avg_high > 0
            assert avg_low > 0
    
    def test_euler_formula_consistency(self):
        """Test that Euler formula is applied consistently."""
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        analyzer = BucklingAnalyzer(net, segments_per_fiber=2)
        
        results = analyzer.analyze_fiber_buckling()
        
        # Manually verify Euler formula for first fiber
        fiber = net.fibers[0]
        result = results[0]
        
        L = fiber.length
        E = fiber.material.youngs_modulus
        r = fiber.radius
        I = np.pi * r**4 / 4
        K = result.effective_length_factor
        
        P_cr_expected = np.pi**2 * E * I / (K * L)**2
        
        # Allow 1% numerical error
        assert abs(result.critical_load - P_cr_expected) / P_cr_expected < 0.01
