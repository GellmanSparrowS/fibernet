"""Test multi-scale modeling module."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim import (
    HomogenizationSolver, RVEAnalyzer,
    HomogenizedProperties, compute_effective_properties,
)


class TestHomogenizationSolver:
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(30, 30), radius=0.1, seed=42)
        solver = HomogenizationSolver(net, fiber_youngs_modulus=1e9)
        assert solver.E_f == 1e9
        assert solver.network == net

    def test_compute_elastic_properties(self):
        net = gen.random_straight_3d(num_fibers=80, fiber_length=12.0, box_size=(40, 40, 40), radius=0.1, seed=42)
        solver = HomogenizationSolver(net, fiber_youngs_modulus=1e9)
        E_x, E_y, E_z, nu_xy, G_xy, G_xz = solver.compute_elastic_properties()
        assert E_x > 0
        assert E_y > 0
        assert E_z > 0
        assert G_xy > 0

    def test_compute_thermal_properties(self):
        net = gen.random_straight_3d(num_fibers=80, fiber_length=12.0, box_size=(40, 40, 40), radius=0.1, seed=42)
        solver = HomogenizationSolver(net, fiber_thermal_conductivity=0.5)
        k_x, k_y, k_z, alpha_x, alpha_y, alpha_z = solver.compute_thermal_properties()
        assert k_x >= 0
        assert k_y >= 0
        assert alpha_x >= 0

    def test_homogenize(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(30, 30), radius=0.1, seed=42)
        solver = HomogenizationSolver(net)
        props = solver.homogenize()
        assert isinstance(props, HomogenizedProperties)
        assert props.effective_youngs_modulus > 0
        assert 0 <= props.porosity <= 1.0
        assert props.density >= 0

    def test_empty_network(self):
        net = gen.random_straight_2d(num_fibers=3, fiber_length=5.0, box_size=(50, 50), radius=0.05, seed=42)
        solver = HomogenizationSolver(net)
        props = solver.homogenize()
        assert props.effective_youngs_modulus >= 0

    def test_isotropic_check(self):
        props = HomogenizedProperties(
            youngs_modulus_x=1e9, youngs_modulus_y=1e9, youngs_modulus_z=1e9,
            poissons_ratio_xy=0.3, poissons_ratio_xz=0.3, poissons_ratio_yz=0.3,
            shear_modulus_xy=3.85e8, shear_modulus_xz=3.85e8, shear_modulus_yz=3.85e8,
            thermal_conductivity_x=0.5, thermal_conductivity_y=0.5, thermal_conductivity_z=0.5,
            thermal_expansion_x=1e-5, thermal_expansion_y=1e-5, thermal_expansion_z=1e-5,
            density=500, porosity=0.5,
        )
        assert props.is_isotropic


class TestRVEAnalyzer:
    def test_apply_periodic_bc(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(30, 30), radius=0.1, seed=42)
        rve = RVEAnalyzer(net, youngs_modulus=1e9)
        macro_strain = np.array([0.01, 0, 0, 0, 0, 0])
        displacements = rve.apply_periodic_bc(macro_strain)
        assert displacements.shape == (net.num_fibers, 3)

    def test_compute_effective_stiffness(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(30, 30), radius=0.1, seed=42)
        rve = RVEAnalyzer(net, youngs_modulus=1e9)
        C_eff = rve.compute_effective_stiffness(num_tests=6)
        assert C_eff.shape == (6, 6)


class TestConvenienceFunctions:
    def test_compute_effective_properties(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(30, 30), radius=0.1, seed=42)
        props = compute_effective_properties(net)
        assert isinstance(props, HomogenizedProperties)
        assert props.effective_youngs_modulus > 0

    def test_compute_effective_properties_with_params(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(30, 30), radius=0.1, seed=42)
        props = compute_effective_properties(net, fiber_properties={
            'fiber_youngs_modulus': 2e9,
            'fiber_poissons_ratio': 0.35,
        })
        assert props.effective_youngs_modulus > 0
