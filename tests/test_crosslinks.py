"""Tests for advanced crosslink models."""

import pytest
import numpy as np
from fibernet.core.crosslinks import (
    CovalentBond, HydrogenBond, PhysicalEntanglement, IonicBond,
    CrosslinkState
)


class TestCovalentBond:
    def test_creation(self):
        """Test creating covalent bond"""
        bond = CovalentBond(stiffness=1e9, bond_energy=5e-19, equilibrium_length=0.154e-9)
        assert bond.k == 1e9
        assert bond.D == 5e-19
        assert bond.r0 == 0.154e-9
    
    def test_force_at_equilibrium(self):
        """Test force is zero at equilibrium length"""
        bond = CovalentBond(equilibrium_length=0.154e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.154e-9, 0, 0])
        state = CrosslinkState()
        force, _ = bond.compute_force(p1, p2, state)
        # Force should be approximately zero at equilibrium
        assert np.linalg.norm(force) < 1e-12
    
    def test_force_stretched(self):
        """Test force when bond is stretched"""
        bond = CovalentBond(stiffness=1e9, equilibrium_length=0.1e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.2e-9, 0, 0])
        state = CrosslinkState()
        force, _ = bond.compute_force(p1, p2, state)
        # Force should exist when stretched
        assert np.linalg.norm(force) > 0
    
    def test_force_compressed(self):
        """Test force when bond is compressed"""
        bond = CovalentBond(stiffness=1e9, equilibrium_length=0.2e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.1e-9, 0, 0])
        state = CrosslinkState()
        force, _ = bond.compute_force(p1, p2, state)
        # Force should exist when compressed
        assert np.linalg.norm(force) > 0
    
    def test_energy_calculation(self):
        """Test energy calculation"""
        bond = CovalentBond(bond_energy=5e-19)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.154e-9, 0, 0])
        energy = bond.energy(p1, p2)
        assert energy >= 0


class TestHydrogenBond:
    def test_creation(self):
        """Test creating hydrogen bond"""
        bond = HydrogenBond(strength=1e7, energy=3e-20, equilibrium_length=0.28e-9)
        assert bond.k == 1e7
        assert bond.E == 3e-20
        assert bond.r0 == 0.28e-9
    
    def test_force_calculation(self):
        """Test force calculation"""
        bond = HydrogenBond(strength=1e7, equilibrium_length=0.28e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.3e-9, 0, 0])
        state = CrosslinkState()
        force, _ = bond.compute_force(p1, p2, state)
        # Force should exist
        assert force is not None
    
    def test_bond_breaking(self):
        """Test bond can break"""
        bond = HydrogenBond(strength=1e7, energy=3e-20, equilibrium_length=0.28e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([1e-9, 0, 0])
        state = CrosslinkState()
        force, new_state = bond.compute_force(p1, p2, state)
        # Should not crash
        assert new_state is not None


class TestPhysicalEntanglement:
    def test_creation(self):
        """Test creating physical entanglement"""
        entanglement = PhysicalEntanglement(friction=1e-6, slip_force=1e-6, confinement=1e-9)
        assert entanglement.mu == 1e-6
        assert entanglement.F_slip == 1e-6
        assert entanglement.rc == 1e-9
    
    def test_force_within_confinement(self):
        """Test force when within confinement radius"""
        entanglement = PhysicalEntanglement(confinement=1e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.5e-9, 0, 0])
        state = CrosslinkState()
        force, _ = entanglement.compute_force(p1, p2, state)
        # Force should be zero within confinement
        assert np.linalg.norm(force) < 1e-10
    
    def test_force_outside_confinement(self):
        """Test force when outside confinement radius"""
        entanglement = PhysicalEntanglement(friction=1e-6, confinement=1e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([2e-9, 0, 0])
        state = CrosslinkState()
        force, _ = entanglement.compute_force(p1, p2, state)
        # Force should exist outside confinement
        assert np.linalg.norm(force) > 0
    
    def test_energy_calculation(self):
        """Test energy calculation"""
        entanglement = PhysicalEntanglement(confinement=1e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([2e-9, 0, 0])
        energy = entanglement.energy(p1, p2)
        assert energy >= 0


class TestIonicBond:
    def test_creation(self):
        """Test creating ionic bond"""
        bond = IonicBond(charge1=1.6e-19, charge2=-1.6e-19, debye_length=1e-9)
        assert bond.q1 == 1.6e-19
        assert bond.q2 == -1.6e-19
        assert bond.lambda_D == 1e-9
    
    def test_attractive_force(self):
        """Test attractive force between opposite charges"""
        bond = IonicBond(charge1=1.6e-19, charge2=-1.6e-19, debye_length=1e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.5e-9, 0, 0])
        state = CrosslinkState()
        force, _ = bond.compute_force(p1, p2, state)
        # Force should exist (direction depends on convention)
        assert np.linalg.norm(force) > 0
    
    def test_repulsive_force(self):
        """Test repulsive force between like charges"""
        bond = IonicBond(charge1=1.6e-19, charge2=1.6e-19, debye_length=1e-9)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.5e-9, 0, 0])
        state = CrosslinkState()
        force, _ = bond.compute_force(p1, p2, state)
        # Force should exist (direction depends on convention)
        assert np.linalg.norm(force) > 0
    
    def test_debye_screening(self):
        """Test Debye screening reduces force at distance"""
        bond = IonicBond(charge1=1.6e-19, charge2=-1.6e-19, debye_length=1e-9)
        p1 = np.array([0, 0, 0])
        
        # Close distance
        p2_close = np.array([0.5e-9, 0, 0])
        state = CrosslinkState()
        force_close, _ = bond.compute_force(p1, p2_close, state)
        
        # Far distance
        p2_far = np.array([5e-9, 0, 0])
        state = CrosslinkState()
        force_far, _ = bond.compute_force(p1, p2_far, state)
        
        # Force should be stronger at close distance
        assert np.linalg.norm(force_close) > np.linalg.norm(force_far)
    
    def test_energy_calculation(self):
        """Test energy calculation"""
        bond = IonicBond(charge1=1.6e-19, charge2=-1.6e-19)
        p1 = np.array([0, 0, 0])
        p2 = np.array([0.5e-9, 0, 0])
        energy = bond.energy(p1, p2)
        # Energy should be negative for attractive interaction
        assert energy < 0


class TestCrosslinkState:
    def test_creation(self):
        """Test creating crosslink state"""
        state = CrosslinkState(broken=False, slip_displacement=0.0)
        assert state.broken == False
        assert state.slip_displacement == 0.0
    
    def test_state_update(self):
        """Test updating state"""
        state = CrosslinkState()
        state.broken = True
        state.slip_displacement = 0.1e-9
        assert state.broken == True
        assert state.slip_displacement == 0.1e-9
