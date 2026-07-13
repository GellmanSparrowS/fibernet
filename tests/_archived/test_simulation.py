"""Tests for simulation engines."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fibernet.core.material import Material
from fibernet.gen.ordered import square_lattice_2d
from fibernet.gen.disordered import random_straight_2d
from fibernet.sim.mechanical import FiberFEM, stress_strain_curve
from fibernet.sim.dynamics import FiberDynamics
from fibernet.sim.thermal import ThermalSolver
from fibernet.sim.electromagnetic import EMSolver
from fibernet.analysis.morphology import MorphologyAnalyzer
from fibernet.analysis.topology import TopologyAnalyzer


class TestFEM:
    def test_mesh_building(self):
        net = square_lattice_2d(spacing=5, grid_size=(3, 3))
        fem = FiberFEM(net, segments_per_fiber=3)
        assert fem.num_nodes > 0
        assert fem.num_elements > 0
    
    def test_stiffness_assembly(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        fem = FiberFEM(net, segments_per_fiber=2)
        K = fem.assemble_stiffness()
        assert K.shape[0] == fem.num_dof
        assert K.shape[1] == fem.num_dof
    
    def test_static_solve(self):
        net = square_lattice_2d(spacing=5, grid_size=(3, 3))
        fem = FiberFEM(net, segments_per_fiber=3)
        result = fem.apply_uniaxial_strain(strain=0.001, axis=0)
        assert result.displacements is not None
        assert result.energy >= 0
    
    def test_effective_modulus(self):
        net = square_lattice_2d(spacing=5, grid_size=(3, 3))
        fem = FiberFEM(net, segments_per_fiber=3)
        E = fem.effective_modulus(strain=0.001, axis=0)
        assert E >= 0


class TestDynamics:
    def test_build_nodes(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        dyn = FiberDynamics(net, dt=1e-6)
        assert dyn.num_nodes > 0
    
    def test_verlet(self):
        net = square_lattice_2d(spacing=5, grid_size=(2, 2))
        dyn = FiberDynamics(net, dt=1e-8, damping=0.1)
        result = dyn.run_verlet(num_steps=10, save_interval=5)
        assert len(result.positions) > 0


class TestThermal:
    def test_steady_state(self):
        net = square_lattice_2d(spacing=5, grid_size=(3, 3))
        solver = ThermalSolver(net)
        result = solver.solve_steady_state(T_hot=100, T_cold=0, axis=0)
        assert result.temperatures is not None


class TestEM:
    def test_conductivity(self):
        net = square_lattice_2d(spacing=5, grid_size=(3, 3))
        solver = EMSolver(net)
        result = solver.solve_conductivity(voltage=1.0, axis=0)
        assert result.potentials is not None


class TestAnalysis:
    def test_morphology(self):
        net = random_straight_2d(num_fibers=50, seed=42)
        analyzer = MorphologyAnalyzer(net)
        report = analyzer.full_report()
        assert "num_fibers" in report
        assert report["num_fibers"] == 50
        assert "nematic_order" in report
    
    def test_topology(self):
        net = random_straight_2d(num_fibers=50, seed=42)
        analyzer = TopologyAnalyzer(net)
        report = analyzer.full_report()
        assert "num_fibers" in report
        assert "is_connected" in report
