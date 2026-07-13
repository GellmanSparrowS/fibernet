"""
Tests for molecular dynamics module.
"""

import pytest
import numpy as np
from fibernet import gen
from fibernet.sim.molecular_dynamics import (
    FiberMDSolver, MDParameters, MDTrajectory,
    run_fiber_md
)


class TestMDParameters:
    """Test MDParameters dataclass."""
    
    def test_defaults(self):
        params = MDParameters()
        assert params.temperature == 300.0
        assert params.timestep == 0.001
        assert params.nsteps == 10000
    
    def test_custom(self):
        params = MDParameters(temperature=400.0, nsteps=5000)
        assert params.temperature == 400.0
        assert params.nsteps == 5000


class TestMDTrajectory:
    """Test MDTrajectory dataclass."""
    
    def test_empty(self):
        traj = MDTrajectory()
        assert traj.num_frames() == 0
    
    def test_with_data(self):
        traj = MDTrajectory(
            positions=[np.zeros((10, 3))],
            times=[0.0],
        )
        assert traj.num_frames() == 1
    
    def test_to_dict(self):
        traj = MDTrajectory(
            positions=[np.zeros((10, 3))],
            times=[1.0],
            energies=[100.0],
        )
        data = traj.to_dict()
        assert data['num_frames'] == 1
        assert data['total_time'] == 1.0


class TestFiberMDSolver:
    """Test FiberMDSolver."""
    
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=20, seed=42)
        solver = FiberMDSolver(net)
        assert solver.network == net
        assert solver.n_beads > 0
        assert len(solver.bonds) > 0
    
    def test_build_model(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        solver = FiberMDSolver(net)
        assert solver.bead_positions.shape[1] == 3
        assert len(solver.bead_fiber_ids) == solver.n_beads
    
    def test_bond_forces(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        solver = FiberMDSolver(net)
        forces = solver._compute_bond_forces(solver.bead_positions)
        assert forces.shape == solver.bead_positions.shape
    
    def test_run_short(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        params = MDParameters(nsteps=10, dump_freq=5)
        solver = FiberMDSolver(net, params)
        traj = solver.run()
        assert traj.num_frames() > 0
        assert len(traj.times) == traj.num_frames()
    
    def test_compute_msd(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        params = MDParameters(nsteps=20, dump_freq=5)
        solver = FiberMDSolver(net, params)
        traj = solver.run()
        times, msd = solver.compute_msd(traj)
        assert len(times) == len(msd)
        # MSD should be non-negative
        assert np.all(msd >= 0)
    
    def test_compute_diffusion(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        params = MDParameters(nsteps=50, dump_freq=5)
        solver = FiberMDSolver(net, params)
        traj = solver.run()
        D = solver.compute_diffusion_coefficient(traj)
        assert isinstance(D, float)


class TestRunFiberMD:
    """Test convenience function."""
    
    def test_basic_usage(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        traj = run_fiber_md(net, temperature=300.0, nsteps=10, dump_freq=5)
        assert isinstance(traj, MDTrajectory)
        assert traj.num_frames() > 0
    
    def test_with_params(self):
        net = gen.random_straight_2d(num_fibers=10, seed=42)
        traj = run_fiber_md(
            net,
            temperature=400.0,
            nsteps=20,
            dump_freq=10,
            friction=2.0,
        )
        assert isinstance(traj, MDTrajectory)
