"""Tests for dynamics simulation enhancements."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.dynamics import (
    FiberDynamics,
    DynamicsResult,
    TimeDependentLoading,
    compute_kinetic_energy,
    compute_temperature,
)


class TestFiberDynamics:
    """Test dynamics engine."""
    
    def test_initialization(self):
        """Test dynamics engine initialization."""
        net = gen.random_straight_3d(num_fibers=10, box_size=(20, 20, 20), seed=42)
        dynamics = FiberDynamics(net, dt=1e-6, damping=0.01)
        
        assert dynamics.network == net
        assert dynamics.dt == 1e-6
        assert dynamics.damping == 0.01
        assert dynamics.num_nodes > 0
    
    def test_build_nodes(self):
        """Test node building."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        dynamics = FiberDynamics(net)
        
        assert len(dynamics.positions) > 0
        assert len(dynamics.velocities) > 0
        assert len(dynamics.masses) > 0
        assert dynamics.positions.shape[1] == 3
    
    def test_run_verlet(self):
        """Test Verlet integration."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        dynamics = FiberDynamics(net, dt=1e-6, damping=0.1)
        
        result = dynamics.run_verlet(num_steps=100, save_interval=10)
        
        assert isinstance(result, DynamicsResult)
        assert len(result.positions) > 0
        assert len(result.velocities) > 0
        assert len(result.kinetic_energy) > 0
    
    def test_run_verlet_with_fixed_nodes(self):
        """Test Verlet with fixed nodes."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        dynamics = FiberDynamics(net, dt=1e-6, damping=0.1)
        
        fixed = [0, 1]
        initial_pos = dynamics.positions[fixed].copy()
        
        result = dynamics.run_verlet(
            num_steps=50,
            fixed_nodes=fixed,
            save_interval=10
        )
        
        # Fixed nodes should not move
        final_pos = result.positions[-1][fixed]
        assert np.allclose(initial_pos, final_pos)
    
    def test_minimize_energy(self):
        """Test energy minimization."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        dynamics = FiberDynamics(net, dt=1e-6)
        
        result = dynamics.minimize_energy(max_steps=100)
        
        assert isinstance(result, DynamicsResult)
        assert len(result.positions) > 0
    
    def test_empty_network(self):
        """Test with empty network."""
        from fibernet.core.network import FiberNetwork
        net = FiberNetwork(dimension=3)
        dynamics = FiberDynamics(net)
        
        assert dynamics.num_nodes == 0


class TestTimeDependentLoading:
    """Test time-dependent loading."""
    
    def test_constant_loading(self):
        """Test constant force loading."""
        force = np.array([1.0, 0.0, 0.0])
        loading = TimeDependentLoading.constant(force, node_indices=[0, 1])
        
        positions = np.zeros((5, 3))
        forces = loading(0, positions)
        
        assert forces.shape == positions.shape
        assert np.allclose(forces[0], force)
        assert np.allclose(forces[1], force)
        assert np.allclose(forces[2], np.zeros(3))
    
    def test_ramp_loading(self):
        """Test linear ramp loading."""
        loading = TimeDependentLoading.ramp(
            rate=1.0,
            direction=np.array([1, 0, 0]),
            dt=1.0
        )
        
        positions = np.zeros((3, 3))
        
        # At t=0, force should be 0
        forces_0 = loading(0, positions)
        assert np.allclose(forces_0[0], np.zeros(3))
        
        # At t=5, force should be 5
        forces_5 = loading(5, positions)
        assert np.allclose(forces_5[0][0], 5.0)
    
    def test_sinusoidal_loading(self):
        """Test sinusoidal loading."""
        loading = TimeDependentLoading.sinusoidal(
            amplitude=1.0,
            frequency=1.0,
            direction=np.array([1, 0, 0]),
            phase=0.0,
            dt=0.25
        )
        
        positions = np.zeros((3, 3))
        
        # At t=0, sin(0) = 0
        forces_0 = loading(0, positions)
        assert np.allclose(forces_0[0][0], 0.0, atol=1e-10)
        
        # At t=0.25, sin(pi/2) = 1
        forces_quarter = loading(1, positions)
        assert abs(forces_quarter[0][0] - 1.0) < 0.1
    
    def test_step_loading(self):
        """Test step loading."""
        force = np.array([1.0, 0.0, 0.0])
        loading = TimeDependentLoading.step_loading(
            force, step_time=10, node_indices=[0]
        )
        
        positions = np.zeros((3, 3))
        
        # Before step_time
        forces_before = loading(5, positions)
        assert np.allclose(forces_before[0], np.zeros(3))
        
        # After step_time
        forces_after = loading(15, positions)
        assert np.allclose(forces_after[0], force)


class TestComputeKineticEnergy:
    """Test kinetic energy computation."""
    
    def test_basic_kinetic_energy(self):
        """Test basic kinetic energy computation."""
        velocities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        masses = np.array([1.0, 1.0])
        
        ke = compute_kinetic_energy(velocities, masses)
        
        # KE = 0.5 * m * v^2 for each particle
        # = 0.5 * 1.0 * 1.0 + 0.5 * 1.0 * 1.0 = 1.0
        assert np.isclose(ke, 1.0)
    
    def test_zero_velocity(self):
        """Test with zero velocity."""
        velocities = np.zeros((3, 3))
        masses = np.array([1.0, 2.0, 3.0])
        
        ke = compute_kinetic_energy(velocities, masses)
        assert ke == 0.0


class TestComputeTemperature:
    """Test temperature computation."""
    
    def test_basic_temperature(self):
        """Test basic temperature computation."""
        velocities = np.array([[1000.0, 0.0, 0.0]])
        masses = np.array([1e-20])  # Small mass for reasonable temperature
        dof = 3
        
        T = compute_temperature(velocities, masses, dof)
        assert T >= 0
    
    def test_zero_temperature(self):
        """Test with zero velocity."""
        velocities = np.zeros((3, 3))
        masses = np.array([1.0, 2.0, 3.0])
        
        T = compute_temperature(velocities, masses, 9)
        assert T == 0.0
    
    def test_zero_dof(self):
        """Test with zero degrees of freedom."""
        velocities = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])
        
        T = compute_temperature(velocities, masses, 0)
        assert T == 0.0


class TestDynamicsIntegration:
    """Test dynamics integration scenarios."""
    
    def test_verlet_with_loading(self):
        """Test Verlet with external loading."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        dynamics = FiberDynamics(net, dt=1e-6, damping=0.1)
        
        # Apply constant force
        force = np.array([1e-10, 0.0, 0.0])
        loading = TimeDependentLoading.constant(force, node_indices=[0])
        
        result = dynamics.run_verlet(
            num_steps=50,
            external_force=loading,
            save_interval=10
        )
        
        assert len(result.positions) > 0
    
    def test_energy_conservation(self):
        """Test that damped dynamics reduces energy."""
        net = gen.random_straight_3d(num_fibers=5, box_size=(10, 10, 10), seed=42)
        dynamics = FiberDynamics(net, dt=1e-6, damping=0.5)
        
        # Give initial velocity
        dynamics.velocities = np.random.randn(*dynamics.velocities.shape) * 1e-3
        
        result = dynamics.run_verlet(num_steps=200, save_interval=10)
        
        # Kinetic energy should decrease with damping
        ke_initial = result.kinetic_energy[0]
        ke_final = result.kinetic_energy[-1]
        
        assert ke_final < ke_initial


