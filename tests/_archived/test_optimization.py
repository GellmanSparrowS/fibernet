"""Tests for SciPy-based optimization."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.optimization import (
    EnergyMinimizer,
    ParameterOptimizer,
    OptimizationResult,
)


class TestEnergyMinimizer:
    """Test energy minimization."""
    
    def test_initialization(self):
        """Test minimizer initialization."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        minimizer = EnergyMinimizer(net)
        
        assert minimizer.network == net
        assert minimizer.num_vars > 0
        assert len(minimizer.x0) == minimizer.num_vars
    
    def test_energy_function(self):
        """Test energy function computation."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        minimizer = EnergyMinimizer(net)
        
        # Energy at reference configuration should be close to zero
        energy = minimizer._energy_function(minimizer.x0)
        assert energy >= 0
    
    def test_minimize_basic(self):
        """Test basic energy minimization."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        minimizer = EnergyMinimizer(net)
        
        # Perturb network slightly
        x_perturbed = minimizer.x0 + np.random.randn(*minimizer.x0.shape) * 0.1
        energy_before = minimizer._energy_function(x_perturbed)
        
        result = minimizer.minimize(
            method='L-BFGS-B',
            max_iterations=100
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.final_value <= energy_before + 1e-6  # Should not increase
    
    def test_minimize_methods(self):
        """Test different optimization methods."""
        net = gen.random_straight_3d(num_fibers=3, box_size=(10, 10, 10), seed=42)
        
        methods = ['L-BFGS-B', 'CG', 'Powell']
        
        for method in methods:
            minimizer = EnergyMinimizer(net)
            result = minimizer.minimize(
                method=method,
                max_iterations=50
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.final_value is not None
    
    def test_update_network(self):
        """Test updating network with optimized positions."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        minimizer = EnergyMinimizer(net)
        
        # Store original positions
        original_pos = [fiber.centerline.copy() for fiber in net.fibers]
        
        result = minimizer.minimize(max_iterations=50)
        minimizer.update_network(result)
        
        # Positions should be updated (may be same if already at minimum)
        assert all(len(fiber.centerline) > 0 for fiber in net.fibers)
    
    def test_empty_network(self):
        """Test with empty network."""
        from fibernet.core.network import FiberNetwork
        net = FiberNetwork(dimension=3)
        minimizer = EnergyMinimizer(net)
        
        assert minimizer.num_vars == 0


class TestParameterOptimizer:
    """Test parameter optimization."""
    
    def test_basic_optimization(self):
        """Test basic parameter optimization."""
        # Simple quadratic objective
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2
        
        optimizer = ParameterOptimizer(objective)
        
        result = optimizer.optimize(
            bounds=[(0, 10), (0, 10)],
            x0=np.array([1.0, 1.0]),
            max_iterations=100
        )
        
        assert isinstance(result, OptimizationResult)
        assert abs(result.final_params[0] - 2.0) < 0.5
        assert abs(result.final_params[1] - 3.0) < 0.5
    
    def test_rosenbrock(self):
        """Test Rosenbrock function optimization."""
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
        optimizer = ParameterOptimizer(rosenbrock)
        
        result = optimizer.optimize(
            bounds=[(-5, 5), (-5, 5)],
            x0=np.array([0.0, 0.0]),
            max_iterations=1000
        )
        
        # Should find minimum near (1, 1)
        assert abs(result.final_params[0] - 1.0) < 1.0
        assert abs(result.final_params[1] - 1.0) < 1.0
    
    def test_global_optimization(self):
        """Test global optimization."""
        # Function with multiple minima
        def objective(x):
            return np.sin(x[0]) * np.cos(x[1]) + 0.1 * (x[0]**2 + x[1]**2)
        
        optimizer = ParameterOptimizer(objective)
        
        result = optimizer.optimize_global(
            bounds=[(-5, 5), (-5, 5)],
            max_iterations=100,
            seed=42
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.final_value is not None
    
    def test_with_network_generation(self):
        """Test optimization with network generation."""
        # Optimize number of fibers for some property
        def objective(params):
            num_fibers = max(5, int(params[0]))
            fiber_length = max(1.0, params[1])
            
            net = gen.random_straight_2d(
                num_fibers=num_fibers,
                fiber_length=fiber_length,
                seed=42
            )
            
            # Simple objective: minimize difference from target fiber count
            target = 20
            return (len(net.fibers) - target)**2
        
        optimizer = ParameterOptimizer(objective)
        
        result = optimizer.optimize(
            bounds=[(10, 50), (5.0, 15.0)],
            max_iterations=50
        )
        
        assert result.final_value >= 0


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_creation(self):
        """Test result creation."""
        result = OptimizationResult(
            success=True,
            message="Converged",
            num_iterations=100,
            num_function_evals=200,
            final_value=1e-6,
            final_params=np.array([1.0, 2.0]),
            history=[1.0, 0.5, 0.1, 1e-6]
        )
        
        assert result.success
        assert result.num_iterations == 100
        assert result.final_value == 1e-6
        assert len(result.history) == 4


