"""Test fracture mechanics module."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim import (
    CrackPropagationSolver,
    compute_energy_release_rate,
    compute_fracture_toughness,
)


class TestCrackPropagationSolver:
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = CrackPropagationSolver(net, fracture_toughness=100.0, element_length=0.5)
        assert solver.network == net
        assert solver.G_c == 100.0
        assert solver.da == 0.5

    def test_initialize_crack(self):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = CrackPropagationSolver(net)
        tip = solver.initialize_crack(
            tip_position=np.array([10.0, 10.0, 0.0]),
            tip_direction=np.array([1.0, 0.0, 0.0]),
            initial_length=2.0,
        )
        assert tip is not None
        assert np.allclose(tip.position, [10.0, 10.0, 0.0])
        assert np.allclose(tip.direction, [1.0, 0.0, 0.0])
        assert tip.length == 2.0

    def test_compute_stress_intensity_factors(self):
        net = gen.random_straight_2d(num_fibers=40, fiber_length=9.0, box_size=(25, 25), seed=123)
        solver = CrackPropagationSolver(net)
        tip = solver.initialize_crack(
            tip_position=np.array([12.5, 12.5, 0.0]),
            tip_direction=np.array([1.0, 0.0, 0.0]),
            initial_length=3.0,
        )
        stress_field = np.zeros((net.num_fibers, 3))
        stress_field[:, 0] = 1e6
        K_I, K_II = solver.compute_stress_intensity_factors(tip, stress_field)
        assert isinstance(K_I, (int, float))
        assert isinstance(K_II, (int, float))

    def test_compute_growth_direction(self):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=456)
        solver = CrackPropagationSolver(net)
        tip = solver.initialize_crack(
            tip_position=np.array([10.0, 10.0, 0.0]),
            tip_direction=np.array([1.0, 0.0, 0.0]),
            initial_length=2.0,
        )
        growth_dir = solver.compute_crack_growth_direction(tip)
        assert growth_dir is not None
        assert len(growth_dir) == 3

    def test_simulate_propagation(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=10.0, box_size=(30, 30), seed=789)
        solver = CrackPropagationSolver(net, fracture_toughness=50.0, element_length=0.5)
        tip = solver.initialize_crack(
            tip_position=np.array([15.0, 15.0, 0.0]),
            tip_direction=np.array([1.0, 0.0, 0.0]),
            initial_length=2.0,
        )
        stress_field = np.zeros((net.num_fibers, 3))
        stress_field[:, 0] = 1e6
        result = solver.simulate_propagation(stress_field, max_steps=5)
        assert result is not None
        assert result.j_integral is not None
        assert result.fracture_toughness is not None
        assert result.energy_dissipated is not None
        assert result.crack_path is not None


class TestFractureFunctions:
    def test_compute_energy_release_rate(self):
        net = gen.random_straight_2d(num_fibers=40, fiber_length=9.0, box_size=(25, 25), seed=42)
        G = compute_energy_release_rate(net, crack_length=2.0, applied_stress=1e6)
        assert G > 0
        assert isinstance(G, (int, float))

    def test_compute_energy_release_rate_with_geometry(self):
        net = gen.random_straight_2d(num_fibers=35, fiber_length=8.5, box_size=(22, 22), seed=123)
        G = compute_energy_release_rate(net, crack_length=1.5, applied_stress=2e6, geometry_factor=1.2)
        assert G > 0

    def test_compute_fracture_toughness(self):
        net = gen.random_straight_2d(num_fibers=45, fiber_length=9.5, box_size=(28, 28), seed=456)
        K_c = compute_fracture_toughness(net, critical_load=1000.0, crack_length=2.0, specimen_width=30.0)
        assert K_c > 0
        assert isinstance(K_c, (int, float))

    def test_compute_fracture_toughness_with_geometry(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=10.0, box_size=(30, 30), seed=789)
        K_c = compute_fracture_toughness(net, critical_load=1500.0, crack_length=2.5, specimen_width=25.0, geometry_factor=1.15)
        assert K_c > 0
