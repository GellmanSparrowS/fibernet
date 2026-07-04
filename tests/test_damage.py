"""Test damage mechanics and fatigue module."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.sim import (
    DamageMechanicsSolver, FatigueSolver,
    DamageState, compute_damage_tolerance,
)


class TestDamageMechanicsSolver:
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = DamageMechanicsSolver(net, youngs_modulus=1e9, tensile_strength=1e8)
        assert solver.E == 1e9
        assert solver.sigma_f == 1e8
        assert solver.state.global_damage == 0.0
        assert len(solver.state.broken_fibers) == 0

    def test_compute_fiber_stress(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = DamageMechanicsSolver(net)
        stresses = solver.compute_fiber_stress(strain=0.01, axis=0)
        assert len(stresses) == net.num_fibers
        assert np.all(stresses >= 0)

    def test_update_damage(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = DamageMechanicsSolver(net)
        stresses = np.full(net.num_fibers, 5e7)  # 50% of strength
        solver.update_damage(stresses)
        assert solver.state.global_damage >= 0.0

    def test_progressive_failure(self):
        net = gen.random_straight_2d(num_fibers=80, fiber_length=10.0, box_size=(30, 30), seed=42)
        solver = DamageMechanicsSolver(net)
        result = solver.progressive_failure(max_strain=0.1, num_steps=30, axis=0)
        assert result.load_displacement.shape == (31, 2)
        assert result.damage_evolution.shape == (31, 2)
        assert result.peak_load >= 0
        assert result.energy_absorbed >= 0

    def test_residual_stiffness(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = DamageMechanicsSolver(net, youngs_modulus=1e9)
        assert solver.residual_stiffness() == 1e9
        # Apply damage
        solver.state.fiber_damage[:] = 0.5
        solver.state.global_damage = 0.5
        assert solver.residual_stiffness() == 5e8

    def test_residual_strength(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        solver = DamageMechanicsSolver(net, tensile_strength=1e8)
        assert solver.residual_strength() == 1e8

    def test_damage_state_properties(self):
        state = DamageState(
            fiber_damage=np.array([0.0, 0.5, 1.0, 0.3]),
            crosslink_damage=np.array([0.0, 0.2]),
            broken_fibers=[2],
            broken_crosslinks=[],
        )
        assert state.fraction_broken_fibers == 0.25
        assert state.fraction_broken_crosslinks == 0.0


class TestFatigueSolver:
    def test_initialization(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        fatigue = FatigueSolver(net)
        assert fatigue.E == 1e9
        assert fatigue.sigma_e == 3e7

    def test_compute_cycles_to_failure(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        fatigue = FatigueSolver(net, fatigue_limit=3e7)
        # Below fatigue limit: infinite life
        N_f = fatigue.compute_cycles_to_failure(1e7)
        assert N_f == int(1e9)
        # Above fatigue limit: finite life
        N_f = fatigue.compute_cycles_to_failure(5e7)
        assert N_f < int(1e9)
        assert N_f > 0

    def test_generate_sn_curve(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        fatigue = FatigueSolver(net)
        sn_curve = fatigue.generate_sn_curve(stress_range=(0.3, 0.9), num_points=5)
        assert sn_curve.shape == (5, 2)
        assert np.all(sn_curve[:, 0] > 0)  # Positive stress
        assert np.all(sn_curve[:, 1] > 0)  # Positive cycles

    def test_miners_rule(self):
        net = gen.random_straight_2d(num_fibers=50, fiber_length=8.0, box_size=(20, 20), seed=42)
        fatigue = FatigueSolver(net)
        load_history = [(4e7, 100), (6e7, 50)]
        damage = fatigue.miners_rule(load_history)
        assert damage > 0


class TestDamageTolerance:
    def test_compute_damage_tolerance(self):
        net = gen.random_straight_2d(num_fibers=80, fiber_length=10.0, box_size=(30, 30), seed=42)
        tolerance = compute_damage_tolerance(net, initial_damage_fraction=0.1)
        assert 'stiffness_retention' in tolerance
        assert 'strength_retention' in tolerance
        assert tolerance['stiffness_retention'] <= 1.0
        assert tolerance['strength_retention'] <= 1.0
        assert tolerance['num_broken_fibers'] >= 0
