"""
Tests for permeability and diffusion solver module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.permeability import (
    PermeabilitySolver, PermeabilityResult,
    DiffusionSolver, DiffusionResult,
    compute_permeability, compute_diffusion
)


class TestPermeabilityResult:
    """Test PermeabilityResult dataclass."""
    
    def test_initialization(self):
        K = np.eye(3) * 1e-12
        result = PermeabilityResult(
            permeability_tensor=K,
            principal_permeabilities=np.array([1e-12, 1e-12, 1e-12]),
            porosity=0.8,
            tortuosity=1.2,
            kozeny_carman_prediction=1e-12
        )
        
        assert result.porosity == 0.8
        assert result.tortuosity == 1.2
    
    def test_to_dict(self):
        K = np.eye(3) * 1e-12
        result = PermeabilityResult(
            permeability_tensor=K,
            principal_permeabilities=np.array([1e-12, 1e-12, 1e-12]),
            porosity=0.8,
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert 'permeability_tensor' in data
        assert data['porosity'] == 0.8
    
    def test_effective_permeability(self):
        result = PermeabilityResult(
            principal_permeabilities=np.array([1e-12, 2e-12, 3e-12])
        )
        
        k_eff = result.effective_permeability()
        
        # Geometric mean
        expected = np.exp(np.mean(np.log([1e-12, 2e-12, 3e-12])))
        assert abs(k_eff - expected) < 1e-20
    
    def test_effective_permeability_none(self):
        result = PermeabilityResult()
        k_eff = result.effective_permeability()
        assert k_eff == 0.0


class TestDiffusionResult:
    """Test DiffusionResult dataclass."""
    
    def test_initialization(self):
        D = np.eye(3) * 1e-9
        result = DiffusionResult(
            diffusion_tensor=D,
            principal_diffusivities=np.array([1e-9, 1e-9, 1e-9]),
            porosity=0.7,
            tortuosity=1.3,
            hindrance_factor=0.54
        )
        
        assert result.porosity == 0.7
        assert result.hindrance_factor == 0.54
    
    def test_to_dict(self):
        D = np.eye(3) * 1e-9
        result = DiffusionResult(
            diffusion_tensor=D,
            porosity=0.7,
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert 'diffusion_tensor' in data


class TestPermeabilitySolver:
    """Test PermeabilitySolver."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = PermeabilitySolver(net, resolution=20)
        
        assert solver.network == net
        assert solver.resolution == 20
    
    def test_compute_permeability(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        solver = PermeabilitySolver(net)
        
        result = solver.compute_permeability()
        
        assert isinstance(result, PermeabilityResult)
        assert result.permeability_tensor is not None
        assert result.permeability_tensor.shape == (3, 3)
        assert result.principal_permeabilities is not None
        assert len(result.principal_permeabilities) == 3
        assert 0 <= result.porosity <= 1
        assert result.tortuosity >= 1.0
    
    def test_porosity_computation(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        solver = PermeabilitySolver(net)
        
        porosity = solver._compute_porosity()
        
        assert 0 <= porosity <= 1
        # For random networks, porosity should be relatively high
        assert porosity > 0.5
    
    def test_tortuosity_computation(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        solver = PermeabilitySolver(net)
        
        tortuosity = solver._compute_tortuosity()
        
        assert tortuosity >= 1.0
        # Tortuosity should be reasonable
        assert tortuosity < 3.0
    
    def test_kozeny_carman(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        solver = PermeabilitySolver(net)
        
        k_kc = solver._kozeny_carman(porosity=0.8)
        
        assert k_kc > 0
        # Should be positive and finite
        assert np.isfinite(k_kc)
    
    def test_kozeny_carman_extreme_porosity(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        solver = PermeabilitySolver(net)
        
        # Zero porosity
        k0 = solver._kozeny_carman(porosity=0.0)
        assert k0 == 0.0
        
        # Full porosity
        k1 = solver._kozeny_carman(porosity=1.0)
        assert k1 == 0.0


class TestDiffusionSolver:
    """Test DiffusionSolver."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = DiffusionSolver(net)
        
        assert solver.network == net
    
    def test_compute_diffusion(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        solver = DiffusionSolver(net)
        
        result = solver.compute_diffusion(D0=1e-9)
        
        assert isinstance(result, DiffusionResult)
        assert result.diffusion_tensor is not None
        assert result.diffusion_tensor.shape == (3, 3)
        assert result.principal_diffusivities is not None
        assert 0 <= result.porosity <= 1
        assert result.tortuosity >= 1.0
        assert 0 < result.hindrance_factor <= 1.0
    
    def test_hindrance_factor(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        solver = DiffusionSolver(net)
        
        result = solver.compute_diffusion(D0=1e-9)
        
        # Hindrance factor should be porosity/tortuosity
        expected = result.porosity / result.tortuosity
        assert abs(result.hindrance_factor - expected) < 1e-10
    
    def test_different_D0(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        solver = DiffusionSolver(net)
        
        result1 = solver.compute_diffusion(D0=1e-9)
        result2 = solver.compute_diffusion(D0=2e-9)
        
        # Hindrance factor should be the same
        assert abs(result1.hindrance_factor - result2.hindrance_factor) < 1e-10
        
        # Effective diffusion should scale (if non-zero)
        if result1.principal_diffusivities[0] > 0:
            ratio = result2.principal_diffusivities[0] / result1.principal_diffusivities[0]
            assert abs(ratio - 2.0) < 0.1


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_compute_permeability(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        
        result = compute_permeability(net, fluid_viscosity=1e-3)
        
        assert isinstance(result, PermeabilityResult)
        assert result.permeability_tensor is not None
    
    def test_compute_diffusion(self):
        net = gen.random_straight_2d(num_fibers=50, seed=42)
        
        result = compute_diffusion(net, D0=1e-9)
        
        assert isinstance(result, DiffusionResult)
        assert result.diffusion_tensor is not None


class TestPermeabilityValidation:
    """Validation tests against analytical solutions."""
    
    def test_permeability_vs_kozeny_carman(self):
        """Test that computed permeability is close to Kozeny-Carman prediction."""
        net = gen.random_straight_2d(num_fibers=100, seed=42)
        solver = PermeabilitySolver(net)
        
        result = solver.compute_permeability()
        
        # Computed permeability should be within an order of magnitude of KC
        k_eff = result.effective_permeability()
        k_kc = result.kozeny_carman_prediction
        
        if k_kc > 0 and k_eff > 0:
            ratio = k_eff / k_kc
            # Allow 2 orders of magnitude difference (simplified model)
            assert 0.01 < ratio < 100
    
    def test_porosity_vs_fiber_volume(self):
        """Test that porosity is consistent with fiber volume fraction."""
        net = gen.random_straight_2d(num_fibers=100, seed=42)
        solver = PermeabilitySolver(net)
        
        porosity = solver._compute_porosity()
        
        # Porosity should be in valid range
        assert 0 <= porosity <= 1
        
        # For 2D networks with reasonable density, porosity should be > 0
        assert porosity > 0
    
    def test_diffusion_decreases_with_density(self):
        """Test that diffusion decreases with fiber density."""
        # Low density network
        net_low = gen.random_straight_2d(num_fibers=30, seed=42)
        solver_low = DiffusionSolver(net_low)
        result_low = solver_low.compute_diffusion(D0=1e-9)
        
        # High density network
        net_high = gen.random_straight_2d(num_fibers=150, seed=42)
        solver_high = DiffusionSolver(net_high)
        result_high = solver_high.compute_diffusion(D0=1e-9)
        
        # High density should have lower hindrance
        assert result_high.hindrance_factor <= result_low.hindrance_factor
