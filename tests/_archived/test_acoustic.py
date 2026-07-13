"""Tests for acoustic wave propagation simulation."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim.acoustic import AcousticSolver, AcousticResult


class TestAcousticResult:
    def test_fundamental_frequency(self):
        freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = AcousticResult(frequencies=freqs)
        assert result.fundamental_frequency() == 1.0

    def test_fundamental_frequency_empty(self):
        result = AcousticResult()
        assert result.fundamental_frequency() == 0.0

    def test_fundamental_frequency_all_zero(self):
        result = AcousticResult(frequencies=np.array([0.0, 0.0, 0.0]))
        assert result.fundamental_frequency() == 0.0

    def test_density_of_states(self):
        freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = AcousticResult(frequencies=freqs)
        centers, dos = result.density_of_states(num_bins=5)
        assert len(centers) == 5
        assert len(dos) == 5
        assert np.all(dos >= 0)

    def test_density_of_states_empty(self):
        result = AcousticResult()
        centers, dos = result.density_of_states()
        assert len(centers) == 0
        assert len(dos) == 0


class TestAcousticSolver:
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = AcousticSolver(net, segments_per_fiber=3)
        assert solver.network == net
        assert solver.segments == 3
        assert solver.num_nodes > 0

    def test_mesh_building(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = AcousticSolver(net, segments_per_fiber=3)
        assert solver.num_nodes > 0
        assert solver.num_elements > 0
        assert solver.num_dof == solver.num_nodes * 3

    def test_compute_modes_graceful_failure(self):
        """Test that compute_modes handles singular matrices gracefully."""
        net = gen.random_straight_2d(num_fibers=20, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = AcousticSolver(net, segments_per_fiber=3)
        result = solver.compute_modes(num_modes=10)
        assert isinstance(result, AcousticResult)
        # Result may be None if eigenvalue computation fails
        # The important thing is it doesn't crash

    def test_sound_velocity(self):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=10.0, box_size=(30, 30), seed=42)
        solver = AcousticSolver(net, segments_per_fiber=4)
        result = solver.compute_modes(num_modes=20)
        # Sound velocity should be non-negative
        assert result.sound_velocity >= 0

    def test_empty_network(self):
        from fibernet.core import FiberNetwork
        net = FiberNetwork(dimension=2)
        solver = AcousticSolver(net, segments_per_fiber=3)
        assert solver.num_nodes == 0
        assert solver.num_elements == 0
