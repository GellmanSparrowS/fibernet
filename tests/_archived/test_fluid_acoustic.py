"""Tests for fluid flow and acoustic modules."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim.fluid import DarcySolver, PoreNetworkModel
from fibernet.sim.acoustic import AcousticSolver


class TestDarcySolver:
    def test_porosity(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(30, 30), seed=42)
        solver = DarcySolver(net)
        
        porosity = solver.compute_porosity()
        assert 0.0 <= porosity <= 1.0
        assert porosity > 0.5  # Should be mostly void
    
    def test_tortuosity(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(30, 30), seed=42)
        solver = DarcySolver(net)
        
        tortuosity = solver.compute_tortuosity(axis=0)
        assert tortuosity >= 1.0
    
    def test_permeability(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(30, 30), seed=42)
        solver = DarcySolver(net)
        
        K = solver.kozeny_carman_permeability()
        assert K > 0.0
    
    def test_solve_flow(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(30, 30), seed=42)
        solver = DarcySolver(net)
        
        grad = np.array([1.0, 0.0, 0.0])
        result = solver.solve_flow(grad, viscosity=1e-3, axis=0)
        
        assert result.permeability > 0
        assert result.porosity > 0
        assert result.tortuosity >= 1.0


class TestPoreNetworkModel:
    def test_build_network(self):
        net = gen.random_straight_2d(num_fibers=15, fiber_length=8, box_size=(20, 20), seed=42)
        pore_net = PoreNetworkModel(net, num_pores=30)
        
        assert pore_net.num_pores > 0
        assert len(pore_net.pore_positions) > 0
    
    def test_permeability(self):
        net = gen.random_straight_2d(num_fibers=15, fiber_length=8, box_size=(20, 20), seed=42)
        pore_net = PoreNetworkModel(net, num_pores=30)
        
        K = pore_net.compute_permeability(axis=0, viscosity=1e-3)
        assert K >= 0.0


class TestAcousticSolver:
    def test_build_mesh(self):
        net = gen.square_lattice_2d(spacing=5, grid_size=(2, 2))
        solver = AcousticSolver(net, segments_per_fiber=3)
        
        assert solver.num_nodes > 0
        assert solver.num_elements > 0
    
    def test_compute_modes(self):
        net = gen.square_lattice_2d(spacing=5, grid_size=(2, 2))
        solver = AcousticSolver(net, segments_per_fiber=3)
        
        result = solver.compute_modes(num_modes=5)
        
        if result.frequencies is not None:
            assert len(result.frequencies) <= 5
            assert np.all(result.frequencies >= 0)
    
    def test_sound_velocity(self):
        net = gen.square_lattice_2d(spacing=5, grid_size=(2, 2))
        solver = AcousticSolver(net, segments_per_fiber=3)
        
        velocity = solver.compute_sound_velocity()
        assert velocity >= 0.0
    
    def test_frequency_response(self):
        net = gen.square_lattice_2d(spacing=5, grid_size=(2, 2))
        solver = AcousticSolver(net, segments_per_fiber=3)
        
        freq, response = solver.frequency_response(
            freq_range=(0, 100),
            num_points=20,
        )
        
        assert len(freq) == 20
        assert len(response) == 20
