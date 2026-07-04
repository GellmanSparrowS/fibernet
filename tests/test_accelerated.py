"""Tests for Taichi-accelerated simulations."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestTaichiEngine:
    def test_import(self):
        from fibernet.sim.accelerated import TaichiEngine, HAS_TAICHI
        assert HAS_TAICHI
    
    def test_parallel_force_computation(self):
        from fibernet.sim.accelerated import TaichiEngine
        engine = TaichiEngine(arch="cpu")
        
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        edges = np.array([[0, 1], [1, 2]])
        rest_lengths = np.array([1.0, 1.0])
        stiffness = np.array([100.0, 100.0])
        
        forces = engine.parallel_force_computation(
            positions, rest_lengths, stiffness, edges
        )
        assert forces.shape == (3, 3)
    
    def test_parallel_force_stretched(self):
        from fibernet.sim.accelerated import TaichiEngine
        engine = TaichiEngine(arch="cpu")
        
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ])
        edges = np.array([[0, 1]])
        rest_lengths = np.array([1.0])
        stiffness = np.array([100.0])
        
        forces = engine.parallel_force_computation(
            positions, rest_lengths, stiffness, edges
        )
        assert forces[0, 0] > 0
        assert forces[1, 0] < 0
    
    def test_parallel_dynamics(self):
        from fibernet.sim.accelerated import TaichiEngine
        engine = TaichiEngine(arch="cpu")
        
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        velocities = np.zeros((3, 3))
        masses = np.array([1.0, 1.0, 1.0])
        edges = np.array([[0, 1], [1, 2]])
        rest_lengths = np.array([1.0, 1.0])
        stiffness = np.array([100.0, 100.0])
        
        result = engine.parallel_dynamics(
            positions, velocities, masses,
            rest_lengths, stiffness, edges,
            dt=1e-4, num_steps=100, save_interval=50,
            fixed_nodes=[0, 2],
        )
        assert len(result.positions) > 0
        assert result.time_seconds > 0
    
    def test_parallel_generate_random_3d(self):
        from fibernet.sim.accelerated import TaichiEngine
        engine = TaichiEngine(arch="cpu")
        
        endpoints = engine.parallel_generate_random_3d(
            num_fibers=100, fiber_length=10,
            box_size=(50, 50, 50), seed=42,
        )
        assert endpoints.shape == (100, 2, 3)
