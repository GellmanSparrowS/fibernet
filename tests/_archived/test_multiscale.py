"""
Tests for multiscale homogenization module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.multiscale import (
    HomogenizationSolver, RVEAnalyzer,
    HomogenizedProperties, RVEResult,
    compute_effective_properties, estimate_rve_size
)


class TestHomogenizedProperties:
    """Test HomogenizedProperties dataclass."""
    
    def test_creation(self):
        props = HomogenizedProperties(
            youngs_modulus_x=1e9,
            youngs_modulus_y=1e9,
            youngs_modulus_z=1e9,
            poissons_ratio_xy=0.3,
            poissons_ratio_xz=0.3,
            poissons_ratio_yz=0.3,
            shear_modulus_xy=0.4e9,
            shear_modulus_xz=0.4e9,
            shear_modulus_yz=0.4e9,
            thermal_conductivity_x=1.0,
            thermal_conductivity_y=1.0,
            thermal_conductivity_z=1.0,
            thermal_expansion_x=1e-5,
            thermal_expansion_y=1e-5,
            thermal_expansion_z=1e-5,
            density=1000.0,
            porosity=0.5,
        )
        assert props.youngs_modulus_x == 1e9
        assert props.is_isotropic
    
    def test_effective_modulus(self):
        props = HomogenizedProperties(
            youngs_modulus_x=2e9,
            youngs_modulus_y=1e9,
            youngs_modulus_z=1e9,
            poissons_ratio_xy=0.3,
            poissons_ratio_xz=0.3,
            poissons_ratio_yz=0.3,
            shear_modulus_xy=0.4e9,
            shear_modulus_xz=0.4e9,
            shear_modulus_yz=0.4e9,
            thermal_conductivity_x=1.0,
            thermal_conductivity_y=1.0,
            thermal_conductivity_z=1.0,
            thermal_expansion_x=1e-5,
            thermal_expansion_y=1e-5,
            thermal_expansion_z=1e-5,
            density=1000.0,
            porosity=0.5,
        )
        assert props.effective_youngs_modulus > 0


class TestHomogenizationSolver:
    """Test HomogenizationSolver."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = HomogenizationSolver(net)
        assert solver.network == net
    
    def test_homogenize(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = HomogenizationSolver(net)
        props = solver.homogenize()
        assert isinstance(props, HomogenizedProperties)
        assert props.youngs_modulus_x > 0
    
    def test_compute_elastic_properties(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        solver = HomogenizationSolver(net)
        result = solver.compute_elastic_properties()
        assert len(result) == 6  # Ex, Ey, Ez, nuxy, nuxz, nuyz


class TestRVEAnalyzer:
    """Test RVEAnalyzer."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = RVEAnalyzer(net)
        assert analyzer.network == net
    
    def test_compute_effective_stiffness(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = RVEAnalyzer(net)
        stiffness = analyzer.compute_effective_stiffness()
        assert isinstance(stiffness, np.ndarray)
        assert stiffness.shape == (6, 6)
    
    def test_apply_periodic_bc(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        analyzer = RVEAnalyzer(net)
        strain = np.array([0.001, 0, 0, 0, 0, 0])
        displacements = analyzer.apply_periodic_bc(strain)
        assert isinstance(displacements, np.ndarray)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_compute_effective_properties(self):
        net = gen.random_straight_2d(num_fibers=30, seed=42)
        props = compute_effective_properties(net)
        assert isinstance(props, HomogenizedProperties)
    
    def test_estimate_rve_size(self):
        rve_size = estimate_rve_size(
            generator_func=gen.random_straight_2d,
            generator_params={'num_fibers': 50, 'seed': 42},
        )
        assert isinstance(rve_size, (int, float))
        assert rve_size > 0
