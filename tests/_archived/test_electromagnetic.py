"""Test electromagnetic simulation module."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim import EMSolver, EMResult
from fibernet.core import Material


class TestEMSolver:
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = EMSolver(net)
        assert solver.network == net

    def test_solve_conductivity_basic(self):
        # Create network with conductive fibers
        mat = Material(name="copper", youngs_modulus=110e9, electrical_conductivity=5.96e7)
        net = gen.random_straight_2d(num_fibers=50, fiber_length=10.0, box_size=(30, 30), seed=42, material=mat)
        solver = EMSolver(net)
        result = solver.solve_conductivity(voltage=1.0, axis=0)
        assert isinstance(result, EMResult)
        assert result.potentials is not None
        assert result.effective_conductivity >= 0.0

    def test_solve_conductivity_non_conductive(self):
        mat = Material(name="rubber", youngs_modulus=0.01e9, electrical_conductivity=0.0)
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42, material=mat)
        solver = EMSolver(net)
        result = solver.solve_conductivity()
        assert result.effective_conductivity < 1e-12

    def test_empty_network(self):
        from fibernet.core import FiberNetwork
        net = FiberNetwork(dimension=2)
        solver = EMSolver(net)
        result = solver.solve_conductivity()
        assert result.effective_conductivity == 0.0
        assert result.is_percolating == False

    def test_percolation_analysis(self):
        mat = Material(name="carbon", youngs_modulus=230e9, electrical_conductivity=1e6)
        net = gen.random_straight_2d(num_fibers=80, fiber_length=12.0, box_size=(30, 30), seed=42, material=mat)
        solver = EMSolver(net)
        volumes, probs = solver.percolation_analysis(num_samples=10)
        assert len(volumes) == 10
        assert len(probs) == 10
        assert all(0 <= p <= 1 for p in probs)

    def test_different_axes(self):
        mat = Material(name="copper", youngs_modulus=110e9, electrical_conductivity=5.96e7)
        net = gen.random_straight_2d(num_fibers=40, fiber_length=10.0, box_size=(30, 30), seed=42, material=mat)
        solver = EMSolver(net)
        r0 = solver.solve_conductivity(axis=0)
        r1 = solver.solve_conductivity(axis=1)
        # Both should give valid results
        assert r0.effective_conductivity >= 0
        assert r1.effective_conductivity >= 0

    def test_is_percolating(self):
        mat = Material(name="carbon", youngs_modulus=230e9, electrical_conductivity=1e6)
        # Dense network should percolate
        net = gen.random_straight_2d(num_fibers=100, fiber_length=15.0, box_size=(30, 30), seed=42, material=mat)
        solver = EMSolver(net)
        result = solver.solve_conductivity()
        assert result.is_percolating == True

    def test_result_dataclass(self):
        result = EMResult(
            potentials=np.array([0, 0.5, 1.0]),
            effective_conductivity=100.0,
            is_percolating=True,
        )
        assert result.effective_conductivity == 100.0
        assert result.is_percolating == True
        assert len(result.potentials) == 3
