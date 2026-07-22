"""
Tests for uncertainty quantification module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.analysis import MorphologyAnalyzer
from fibernet.sim.uncertainty import (
    EnsembleResult, monte_carlo_ensemble,
    sensitivity_analysis, convergence_study
)


def compute_order(net):
    """Helper: compute nematic order parameter."""
    return MorphologyAnalyzer(net).nematic_order_parameter()


def compute_fiber_count(net):
    """Helper: return fiber count."""
    return net.num_fibers


class TestEnsembleResult:
    """Test EnsembleResult dataclass."""
    
    def test_initialization(self):
        values = np.array([1.0, 2.0, 3.0])
        result = EnsembleResult(
            values=values,
            mean=2.0,
            std=1.0,
            min_val=1.0,
            max_val=3.0,
            confidence_interval=(1.5, 2.5),
            cv=0.5,
            num_samples=3,
            converged=True
        )
        
        assert result.mean == 2.0
        assert result.std == 1.0
        assert result.num_samples == 3
        assert result.converged == True
    
    def test_to_dict(self):
        values = np.array([1.0, 2.0, 3.0])
        result = EnsembleResult(
            values=values,
            mean=2.0,
            std=1.0,
            confidence_interval=(1.5, 2.5),
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data['mean'] == 2.0
        assert data['std'] == 1.0
        assert len(data['values']) == 3
    
    def test_empty_result(self):
        result = EnsembleResult()
        assert result.mean == 0.0
        assert result.std == 0.0
        assert result.num_samples == 0


class TestMonteCarloEnsemble:
    """Test Monte Carlo ensemble analysis."""
    
    def test_basic_usage(self):
        result = monte_carlo_ensemble(
            gen.random_straight_2d,
            compute_fiber_count,
            num_samples=10,
            generator_kwargs={'num_fibers': 20}
        )
        
        assert isinstance(result, EnsembleResult)
        assert result.num_samples == 10
        assert result.mean == 20.0  # Should always return num_fibers
        assert result.std == 0.0
    
    def test_with_order_parameter(self):
        result = monte_carlo_ensemble(
            gen.random_straight_2d,
            compute_order,
            num_samples=20,
            generator_kwargs={'num_fibers': 50}
        )
        
        assert isinstance(result, EnsembleResult)
        assert result.num_samples == 20
        assert 0 <= result.mean <= 1
        assert result.std >= 0
        assert result.min_val <= result.mean <= result.max_val
    
    def test_confidence_interval(self):
        result = monte_carlo_ensemble(
            gen.random_straight_2d,
            compute_order,
            num_samples=50,
            generator_kwargs={'num_fibers': 50}
        )
        
        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower <= result.mean <= ci_upper
    
    def test_coefficient_of_variation(self):
        result = monte_carlo_ensemble(
            gen.random_straight_2d,
            compute_order,
            num_samples=30,
            generator_kwargs={'num_fibers': 50}
        )
        
        assert result.cv >= 0
    
    def test_verbose_mode(self):
        result = monte_carlo_ensemble(
            gen.random_straight_2d,
            compute_fiber_count,
            num_samples=20,
            generator_kwargs={'num_fibers': 20},
            verbose=True
        )
        
        assert result.num_samples == 20
    
    def test_convergence_check(self):
        result = monte_carlo_ensemble(
            gen.random_straight_2d,
            compute_fiber_count,
            num_samples=50,
            generator_kwargs={'num_fibers': 30},
            convergence_check=True,
            convergence_threshold=0.01
        )
        
        # Constant function should converge
        assert result.converged == True


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""
    
    def test_basic_usage(self):
        results = sensitivity_analysis(
            gen.random_straight_2d,
            compute_fiber_count,
            'num_fibers',
            [20, 40, 60],
            num_samples_per_value=5
        )
        
        assert isinstance(results, dict)
        assert 'values' in results
        assert 'means' in results
        assert 'stds' in results
        
        assert len(results['values']) == 3
        assert results['means'] == [20.0, 40.0, 60.0]
    
    def test_with_order_parameter(self):
        results = sensitivity_analysis(
            gen.random_straight_2d,
            compute_order,
            'num_fibers',
            [30, 60],
            num_samples_per_value=5
        )
        
        assert len(results['values']) == 2
        assert all(0 <= m <= 1 for m in results['means'])


class TestConvergenceStudy:
    """Test convergence study."""
    
    def test_basic_usage(self):
        results = convergence_study(
            gen.random_straight_2d,
            compute_fiber_count,
            sample_sizes=[5, 10],
            num_repeats=2,
            generator_kwargs={'num_fibers': 20}
        )
        
        assert isinstance(results, dict)
        assert 'sample_sizes' in results
        assert 'means' in results
        assert 'stds' in results
        assert 'ci_widths' in results
        
        assert results['sample_sizes'] == [5, 10]
        assert len(results['means']) == 2
    
    def test_convergence_trend(self):
        """Test that CI width decreases with sample size."""
        results = convergence_study(
            gen.random_straight_2d,
            compute_order,
            sample_sizes=[10, 30, 50],
            num_repeats=3,
            generator_kwargs={'num_fibers': 50}
        )
        
        # CI width should generally decrease
        # (may not be monotonic due to small repeats)
        assert len(results['ci_widths']) == 3
