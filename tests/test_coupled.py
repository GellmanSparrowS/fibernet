"""Tests for multi-physics coupled simulations."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim.coupled import (
    ThermoMechanicalSolver,
    ElectroMechanicalSolver,
    MultiPhysicsSolver,
    CoupledResult
)


class TestThermoMechanical:
    def test_basic_coupling(self):
        net = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
        thermo = ThermoMechanicalSolver(net, alpha=1e-5, segments_per_fiber=3)
        result = thermo.coupled_analysis(T_hot=100, T_cold=0, mechanical_strain=0.001, axis=0)
        
        assert result.thermal is not None
        assert result.mechanical is not None
        assert result.converged
    
    def test_thermal_only(self):
        net = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
        thermo = ThermoMechanicalSolver(net, alpha=1e-5, segments_per_fiber=3)
        result = thermo.coupled_analysis(T_hot=100, T_cold=0, mechanical_strain=0.0, axis=0)
        
        assert result.thermal is not None


class TestElectroMechanical:
    def test_basic_coupling(self):
        net = gen.square_lattice_2d(spacing=5, grid_size=(3, 3))
        electro = ElectroMechanicalSolver(net, piezo_coeff=1e-12, segments_per_fiber=3)
        result = electro.coupled_analysis(voltage=1.0, axis=0, mechanical_strain=0.001)
        
        assert result.electromagnetic is not None
        assert result.mechanical is not None


class TestCoupledResult:
    def test_default_values(self):
        result = CoupledResult()
        assert result.mechanical is None
        assert result.thermal is None
        assert result.electromagnetic is None
        assert result.converged is True
        assert result.iterations == 0
