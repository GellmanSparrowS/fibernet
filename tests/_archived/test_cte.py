"""
Tests for CTE (Coefficient of Thermal Expansion) module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.cte import CTEAnalyzer, CTEResult, compute_cte


class TestCTEResult:
    """Test CTEResult dataclass."""
    
    def test_initialization(self):
        tensor = np.eye(3) * 10e-6
        result = CTEResult(
            cte_tensor=tensor,
            principal_ctes=np.array([10e-6, 10e-6, 10e-6]),
            effective_cte=10e-6,
            anisotropy_ratio=1.0,
            rule_of_mixtures=10e-6
        )
        
        assert result.effective_cte == 10e-6
        assert result.anisotropy_ratio == 1.0
    
    def test_to_dict(self):
        tensor = np.eye(3) * 10e-6
        result = CTEResult(
            cte_tensor=tensor,
            effective_cte=10e-6,
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert 'cte_tensor' in data
        assert data['effective_cte'] == 10e-6


class TestCTEAnalyzer:
    """Test CTEAnalyzer."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = CTEAnalyzer(net)
        
        assert analyzer.network == net
    
    def test_compute_cte(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        analyzer = CTEAnalyzer(net)
        
        result = analyzer.compute_cte()
        
        assert isinstance(result, CTEResult)
        assert result.cte_tensor is not None
        assert result.cte_tensor.shape == (3, 3)
        assert result.principal_ctes is not None
        assert len(result.principal_ctes) == 3
        assert result.effective_cte > 0
        assert result.rule_of_mixtures > 0
    
    def test_cte_tensor_symmetry(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        analyzer = CTEAnalyzer(net)
        
        result = analyzer.compute_cte()
        
        # CTE tensor should be symmetric
        assert np.allclose(result.cte_tensor, result.cte_tensor.T)
    
    def test_anisotropy_ratio(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        analyzer = CTEAnalyzer(net)
        
        result = analyzer.compute_cte()
        
        # Anisotropy ratio should be >= 1
        assert result.anisotropy_ratio >= 1.0
    
    def test_thermal_strain(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        analyzer = CTEAnalyzer(net)
        
        delta_T = 100.0  # 100 K temperature increase
        strain = analyzer.thermal_strain(delta_T)
        
        assert strain.shape == (3, 3)
        # Thermal strain should be positive for positive CTE and delta_T
        assert np.all(np.diag(strain) >= 0)
    
    def test_thermal_strain_with_precomputed_cte(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        analyzer = CTEAnalyzer(net)
        
        cte_result = analyzer.compute_cte()
        strain = analyzer.thermal_strain(50.0, cte_result)
        
        assert strain.shape == (3, 3)
    
    def test_fiber_volume_fraction(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        analyzer = CTEAnalyzer(net)
        
        V_f = analyzer._compute_fiber_volume_fraction()
        
        assert 0 <= V_f <= 1
    
    def test_orientation_tensor(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        analyzer = CTEAnalyzer(net)
        
        a = analyzer._compute_orientation_tensor()
        
        assert a.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(a, a.T)
        # Trace should be 1 (normalized)
        assert abs(np.trace(a) - 1.0) < 0.1
    
    def test_different_material_properties(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        analyzer = CTEAnalyzer(net)
        
        # Low fiber CTE
        result1 = analyzer.compute_cte(fiber_cte=1e-6, matrix_cte=50e-6)
        
        # High fiber CTE
        result2 = analyzer.compute_cte(fiber_cte=20e-6, matrix_cte=50e-6)
        
        # Higher fiber CTE should give higher effective CTE
        assert result2.effective_cte > result1.effective_cte
    
    def test_isotropic_network(self):
        """Test that isotropic network gives near-isotropic CTE."""
        # Large random network should be approximately isotropic
        net = gen.random_straight_2d(num_fibers=200, seed=42)
        analyzer = CTEAnalyzer(net)
        
        result = analyzer.compute_cte()
        
        # For isotropic 2D network, anisotropy should be moderate
        # (not perfectly isotropic due to 2D constraint)
        assert result.anisotropy_ratio < 10.0


class TestComputeCTEFunction:
    """Test convenience function."""
    
    def test_basic_usage(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        
        result = compute_cte(net)
        
        assert isinstance(result, CTEResult)
        assert result.effective_cte > 0
    
    def test_with_parameters(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        
        result = compute_cte(
            net,
            fiber_cte=5e-6,
            matrix_cte=60e-6,
            fiber_modulus=100e9,
            matrix_modulus=5e9
        )
        
        assert isinstance(result, CTEResult)
        assert result.effective_cte > 0


class TestCTEValidation:
    """Validation tests."""
    
    def test_cte_between_bounds(self):
        """CTE should be between fiber and matrix CTE."""
        net = gen.random_straight_2d(num_fibers=100, seed=42)
        
        alpha_f = 5e-6
        alpha_m = 50e-6
        
        result = compute_cte(net, fiber_cte=alpha_f, matrix_cte=alpha_m)
        
        # Effective CTE should be between fiber and matrix
        assert alpha_f <= result.effective_cte <= alpha_m
    
    def test_rule_of_mixtures_consistency(self):
        """Rule of mixtures should be between fiber and matrix CTE."""
        net = gen.random_straight_2d(num_fibers=100, seed=42)
        
        alpha_f = 5e-6
        alpha_m = 50e-6
        
        result = compute_cte(net, fiber_cte=alpha_f, matrix_cte=alpha_m)
        
        assert alpha_f <= result.rule_of_mixtures <= alpha_m
