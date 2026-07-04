"""
Tests for incremental nonlinear FEM module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.incremental_fem import (
    IncrementalFEM, IncrementalResult,
    compute_stress_strain_curve
)


class TestIncrementalFEM:
    """Test incremental FEM solver."""
    
    def test_initialization(self):
        """Test solver initialization."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = IncrementalFEM(net, segments_per_fiber=3)
        
        assert solver.network == net
        assert solver.segments == 3
        assert solver.material_model == 'elastic'
    
    def test_elastic_analysis(self):
        """Test purely elastic material response."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = IncrementalFEM(
            net,
            segments_per_fiber=3,
            material_model='elastic'
        )
        
        result = solver.run_incremental_analysis(
            max_strain=0.01,
            num_increments=10,
            verbose=False
        )
        
        assert isinstance(result, IncrementalResult)
        assert len(result.strain_history) > 0
        assert len(result.stress_history) > 0
        assert len(result.strain_history) == len(result.stress_history)
        
        # Stress should increase with strain (elastic)
        assert result.stress_history[-1] > result.stress_history[0]
    
    def test_elastic_plastic_analysis(self):
        """Test elastic-plastic material response."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = IncrementalFEM(
            net,
            segments_per_fiber=3,
            material_model='elastic_plastic',
            yield_stress=1e8,
            hardening_modulus=1e7
        )
        
        result = solver.run_incremental_analysis(
            max_strain=0.03,
            num_increments=30,
            verbose=False
        )
        
        assert isinstance(result, IncrementalResult)
        assert len(result.strain_history) > 0
        
        # Should have yield point detected
        assert result.yield_strength > 0
    
    def test_damage_analysis(self):
        """Test damage evolution."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = IncrementalFEM(
            net,
            segments_per_fiber=3,
            material_model='damage',
            damage_threshold=0.01,
            failure_strain=0.05
        )
        
        result = solver.run_incremental_analysis(
            max_strain=0.04,
            num_increments=40,
            verbose=False
        )
        
        assert isinstance(result, IncrementalResult)
        
        # Damage should accumulate
        assert result.damage_history[-1] >= result.damage_history[0]
    
    def test_element_failure(self):
        """Test element failure detection."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = IncrementalFEM(
            net,
            segments_per_fiber=3,
            material_model='damage',
            failure_strain=0.02
        )
        
        result = solver.run_incremental_analysis(
            max_strain=0.03,
            num_increments=30,
            verbose=False
        )
        
        # Some elements should fail
        assert len(result.failed_elements) > 0
    
    def test_computed_properties(self):
        """Test computed mechanical properties."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = IncrementalFEM(net, segments_per_fiber=3)
        
        result = solver.run_incremental_analysis(
            max_strain=0.02,
            num_increments=20,
            verbose=False
        )
        
        # Young's modulus should be positive
        assert result.youngs_modulus > 0
        
        # Ultimate strength should be positive
        assert result.ultimate_strength > 0
        
        # Toughness should be positive
        assert result.toughness > 0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = IncrementalFEM(net, segments_per_fiber=3)
        
        result = solver.run_incremental_analysis(
            max_strain=0.01,
            num_increments=10,
            verbose=False
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert 'strain' in data
        assert 'stress' in data
        assert 'youngs_modulus' in data
        assert len(data['strain']) == len(result.strain_history)


class TestComputeStressStrainCurve:
    """Test convenience function."""
    
    def test_basic_usage(self):
        """Test basic function call."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        
        result = compute_stress_strain_curve(
            net,
            max_strain=0.01,
            num_increments=10,
            material_model='elastic'
        )
        
        assert isinstance(result, IncrementalResult)
        assert len(result.strain_history) > 0
    
    def test_with_parameters(self):
        """Test with custom parameters."""
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        
        result = compute_stress_strain_curve(
            net,
            max_strain=0.02,
            num_increments=20,
            material_model='elastic_plastic',
            yield_stress=5e7,
            hardening_modulus=5e6
        )
        
        assert isinstance(result, IncrementalResult)
        assert result.yield_strength > 0


class TestIncrementalResult:
    """Test IncrementalResult dataclass."""
    
    def test_plot_without_matplotlib(self):
        """Test that plot raises ImportError if matplotlib not available."""
        result = IncrementalResult(
            strain_history=np.array([0, 0.01, 0.02]),
            stress_history=np.array([0, 1e6, 2e6])
        )
        
        # This should work if matplotlib is available
        try:
            ax = result.plot()
            assert ax is not None
        except ImportError:
            # Expected if matplotlib not installed
            pass
    
    def test_empty_result(self):
        """Test handling of empty result."""
        result = IncrementalResult()
        
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data['strain'] == []
        assert data['stress'] == []
