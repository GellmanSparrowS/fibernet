"""Test coupled multi-physics simulations."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim import (
    ThermoMechanicalSolver,
    ElectroMechanicalSolver,
    run_thermo_mechanical_analysis,
)


class TestThermoMechanicalSolver:
    def test_steady_state_basic(self):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = ThermoMechanicalSolver(net, youngs_modulus=1e9, thermal_expansion=1e-5, thermal_conductivity=0.5)
        result = solver.solve_steady_state(
            mechanical_load={'strain': 0.01, 'axis': 0},
            thermal_boundary={'top': 350, 'bottom': 300},
        )
        assert result.temperature_field is not None
        assert len(result.temperature_field) == net.num_fibers
        assert result.max_temperature > 0
        assert result.max_stress >= 0
        assert result.total_energy is not None

    def test_steady_state_with_mechanical_load(self):
        net = gen.random_straight_2d(num_fibers=25, fiber_length=7.0, box_size=(18, 18), seed=123)
        solver = ThermoMechanicalSolver(net)
        result_tension = solver.solve_steady_state(
            mechanical_load={'strain': 0.02, 'axis': 0},
            thermal_boundary={'top': 320, 'bottom': 300},
        )
        result_compression = solver.solve_steady_state(
            mechanical_load={'strain': -0.01, 'axis': 0},
            thermal_boundary={'top': 320, 'bottom': 300},
        )
        assert result_tension.max_stress != result_compression.max_stress

    def test_steady_state_thermal_gradient(self):
        net = gen.random_straight_2d(num_fibers=40, fiber_length=9.0, box_size=(25, 25), seed=456)
        solver = ThermoMechanicalSolver(net, thermal_expansion=2e-5)
        result = solver.solve_steady_state(
            mechanical_load={'strain': 0.005, 'axis': 1},
            thermal_boundary={'top': 400, 'bottom': 300},
        )
        temp_range = result.max_temperature - result.min_temperature
        assert temp_range > 0

    def test_transient_basic(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=6.0, box_size=(15, 15), seed=789)
        solver = ThermoMechanicalSolver(net)
        result = solver.solve_transient(
            mechanical_load={'strain': 0.01, 'axis': 0},
            thermal_boundary={'top': 350, 'bottom': 300},
            initial_temperature=300,
            total_time=1.0,
            num_steps=5,
        )
        assert result.temperature_field is not None
        assert len(result.temperature_field) == net.num_fibers
        assert len(result.time_steps) == 6


class TestElectroMechanicalSolver:
    def test_basic_electromechanical(self):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = ElectroMechanicalSolver(net, youngs_modulus=1e9, piezoelectric_coefficient=1e-10)
        result = solver.solve(
            mechanical_load={'strain': 0.01, 'axis': 0},
            voltage_boundary={'top': 100, 'bottom': 0},
        )
        assert result.electric_potential is not None
        assert len(result.electric_potential) == net.num_fibers
        assert result.polarization is not None

    def test_voltage_gradient(self):
        net = gen.random_straight_2d(num_fibers=35, fiber_length=7.5, box_size=(22, 22), seed=123)
        solver = ElectroMechanicalSolver(net, piezoelectric_coefficient=2e-10)
        result = solver.solve(
            mechanical_load={'strain': 0.015, 'axis': 0},
            voltage_boundary={'top': 200, 'bottom': 0},
        )
        assert result.electric_potential is not None


class TestConvenienceFunctions:
    def test_run_thermo_mechanical_analysis(self):
        net = gen.random_straight_2d(num_fibers=30, fiber_length=8.0, box_size=(20, 20), seed=42)
        result = run_thermo_mechanical_analysis(net, strain=0.01, axis=0, temperature_diff=50)
        assert result.temperature_field is not None
        assert result.max_temperature > 0
        assert result.max_stress >= 0
