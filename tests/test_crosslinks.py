"""Tests for crosslink models."""

import numpy as np
import pytest
from fibernet.core.crosslinks import (
    RigidCrosslink,
    SpringCrosslink,
    BreakableCrosslink,
    FrictionCrosslink,
    BondedCrosslink,
    CrosslinkState,
)


class TestRigidCrosslink:
    def test_compute_force(self):
        xl = RigidCrosslink()
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        state = CrosslinkState()
        
        force, new_state = xl.compute_force(p1, p2, state)
        assert force.shape == (3,)
        assert new_state.displacement is not None
    
    def test_energy_zero(self):
        xl = RigidCrosslink()
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        
        energy = xl.energy(p1, p2)
        assert energy == 0.0


class TestSpringCrosslink:
    def test_compute_force_at_rest(self):
        xl = SpringCrosslink(stiffness=100.0, rest_length=1.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        state = CrosslinkState()
        
        force, _ = xl.compute_force(p1, p2, state)
        assert np.allclose(force, 0.0)
    
    def test_compute_force_stretched(self):
        xl = SpringCrosslink(stiffness=100.0, rest_length=1.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        state = CrosslinkState()
        
        force, _ = xl.compute_force(p1, p2, state)
        # Force should be 100 * (2 - 1) = 100 in x direction
        assert np.isclose(force[0], 100.0)
        assert np.isclose(force[1], 0.0)
        assert np.isclose(force[2], 0.0)
    
    def test_compute_force_compressed(self):
        xl = SpringCrosslink(stiffness=100.0, rest_length=1.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.5, 0.0, 0.0])
        state = CrosslinkState()
        
        force, _ = xl.compute_force(p1, p2, state)
        # Force should be 100 * (0.5 - 1) = -50 in x direction
        assert np.isclose(force[0], -50.0)
    
    def test_energy(self):
        xl = SpringCrosslink(stiffness=100.0, rest_length=1.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        
        energy = xl.energy(p1, p2)
        # Energy = 0.5 * 100 * (2 - 1)^2 = 50
        assert np.isclose(energy, 50.0)


class TestBreakableCrosslink:
    def test_intact_bond(self):
        xl = BreakableCrosslink(stiffness=100.0, critical_force=1000.0, rest_length=1.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.5, 0.0, 0.0])
        state = CrosslinkState()
        
        force, new_state = xl.compute_force(p1, p2, state)
        assert not new_state.broken
        assert np.linalg.norm(force) < 1000.0
    
    def test_broken_bond(self):
        xl = BreakableCrosslink(stiffness=100.0, critical_force=50.0, rest_length=1.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        state = CrosslinkState()
        
        force, new_state = xl.compute_force(p1, p2, state)
        assert new_state.broken
        
        # Once broken, should return zero force
        force2, _ = xl.compute_force(p1, p2, new_state)
        assert np.allclose(force2, 0.0)


class TestFrictionCrosslink:
    def test_below_friction(self):
        xl = FrictionCrosslink(stiffness=100.0, friction_force=1000.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.5, 0.0, 0.0])
        state = CrosslinkState(slip_displacement=0.0)
        
        force, _ = xl.compute_force(p1, p2, state)
        assert np.linalg.norm(force) < 1000.0
    
    def test_slip(self):
        xl = FrictionCrosslink(stiffness=100.0, friction_force=50.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        state = CrosslinkState(slip_displacement=0.0)
        
        force, new_state = xl.compute_force(p1, p2, state)
        # Force should be capped at friction force
        assert np.linalg.norm(force) <= 50.0 + 1e-6
        assert new_state.slip_displacement > 0.0


class TestBondedCrosslink:
    def test_stretch(self):
        xl = BondedCrosslink(stretch_stiffness=100.0, bend_stiffness=10.0, rest_length=1.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        state = CrosslinkState()
        
        force, _ = xl.compute_force(p1, p2, state)
        assert np.isclose(force[0], 100.0)
    
    def test_bending_energy(self):
        xl = BondedCrosslink(stretch_stiffness=100.0, bend_stiffness=10.0, rest_length=1.0)
        
        # Straight configuration (theta = pi)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        
        energy = xl.bending_energy(p1, p2, p3)
        assert np.isclose(energy, 0.0, atol=1e-10)
        
        # Bent configuration (theta = pi/2)
        p3_bent = np.array([1.0, 1.0, 0.0])
        energy_bent = xl.bending_energy(p1, p2, p3_bent)
        assert energy_bent > 0.0
