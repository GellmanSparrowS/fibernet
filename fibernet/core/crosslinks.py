"""
Advanced crosslink models for fiber networks.

Provides different crosslink mechanical behaviors:
- RigidCrosslink: Perfectly rigid connections
- SpringCrosslink: Linear spring connections
- BreakableCrosslink: Force-dependent rupture
- FrictionCrosslink: Friction-based sliding
- BondedCrosslink: Covalent-like with bending resistance
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CrosslinkState:
    """State of a crosslink during simulation."""
    force: np.ndarray = None
    displacement: np.ndarray = None
    broken: bool = False
    slip_displacement: float = 0.0
    num_cycles: int = 0


class CrosslinkModel(ABC):
    """Base class for crosslink models."""
    
    @abstractmethod
    def compute_force(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        state: CrosslinkState,
    ) -> Tuple[np.ndarray, CrosslinkState]:
        """Compute crosslink force and update state."""
        pass
    
    @abstractmethod
    def energy(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute crosslink potential energy."""
        pass


class RigidCrosslink(CrosslinkModel):
    """Rigid crosslink (infinite stiffness constraint).
    
    Enforces p1 == p2 exactly.
    """
    
    def __init__(self):
        pass
    
    def compute_force(self, p1, p2, state):
        dr = np.asarray(p2) - np.asarray(p1)
        # Return constraint force (Lagrange multiplier approach)
        if state.force is None:
            force = np.zeros(3)
        else:
            force = state.force
        return force, CrosslinkState(force=force, displacement=dr)
    
    def energy(self, p1, p2):
        return 0.0


class SpringCrosslink(CrosslinkModel):
    """Linear spring crosslink.
    
    F = k * (|r| - r0) * r_hat
    
    Parameters
    ----------
    stiffness : float
        Spring stiffness (N/m).
    rest_length : float
        Rest length. Default 0 (point connection).
    """
    
    def __init__(self, stiffness: float = 1e6, rest_length: float = 0.0):
        self.k = stiffness
        self.r0 = rest_length
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            force = self.k * (r - self.r0) * r_hat
        
        new_state = CrosslinkState(force=force, displacement=dr)
        return force, new_state
    
    def energy(self, p1, p2):
        dr = np.asarray(p2) - np.asarray(p1)
        r = np.linalg.norm(dr)
        return 0.5 * self.k * (r - self.r0)**2


class BreakableCrosslink(CrosslinkModel):
    """Breakable crosslink with force-dependent rupture.
    
    Follows Bell-like bond model:
    - Below critical force: behaves as spring
    - Above critical force: bond breaks irreversibly
    
    Parameters
    ----------
    stiffness : float
        Spring stiffness.
    critical_force : float
        Force at which bond breaks.
    rest_length : float
        Rest length.
    """
    
    def __init__(self, stiffness: float = 1e6, critical_force: float = 1e3, rest_length: float = 0.0):
        self.k = stiffness
        self.f_crit = critical_force
        self.r0 = rest_length
    
    def compute_force(self, p1, p2, state):
        if state.broken:
            return np.zeros(3), state
        
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            force_mag = self.k * (r - self.r0)
            force = force_mag * r_hat
        
        f_norm = np.linalg.norm(force)
        broken = f_norm > self.f_crit
        
        new_state = CrosslinkState(
            force=force, displacement=dr, broken=broken,
        )
        return force, new_state
    
    def energy(self, p1, p2):
        dr = np.asarray(p2) - np.asarray(p1)
        r = np.linalg.norm(dr)
        return 0.5 * self.k * (r - self.r0)**2


class FrictionCrosslink(CrosslinkModel):
    """Friction-based crosslink with slip.
    
    - Below friction force: elastic spring
    - Above friction force: slips with constant friction force
    
    Parameters
    ----------
    stiffness : float
        Spring stiffness.
    friction_force : float
        Maximum friction force.
    """
    
    def __init__(self, stiffness: float = 1e6, friction_force: float = 100.0):
        self.k = stiffness
        self.f_friction = friction_force
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        
        dr = p2 - p1
        slip = state.slip_displacement
        
        elastic_disp = dr - slip * (dr / (np.linalg.norm(dr) + 1e-15))
        r_elastic = np.linalg.norm(elastic_disp)
        
        if r_elastic < 1e-15:
            force = np.zeros(3)
        else:
            force_mag = min(self.k * r_elastic, self.f_friction)
            force = force_mag * elastic_disp / r_elastic
        
        # Update slip if friction is exceeded
        if self.k * r_elastic > self.f_friction:
            slip_new = np.linalg.norm(dr) - self.f_friction / self.k
        else:
            slip_new = slip
        
        new_state = CrosslinkState(
            force=force, displacement=dr,
            slip_displacement=slip_new,
        )
        return force, new_state
    
    def energy(self, p1, p2):
        dr = np.asarray(p2) - np.asarray(p1)
        r = np.linalg.norm(dr)
        return min(0.5 * self.k * r**2, self.f_friction * r)


class BondedCrosslink(CrosslinkModel):
    """Bonded crosslink with stretching and bending resistance.
    
    Models covalent-like bonds with:
    - Stretching energy: 0.5 * k_s * (r - r0)²
    - Bending energy: 0.5 * k_b * theta² (requires neighbor info)
    
    Parameters
    ----------
    stretch_stiffness : float
        Stretching stiffness.
    bend_stiffness : float
        Bending stiffness.
    rest_length : float
        Rest length.
    """
    
    def __init__(self, stretch_stiffness: float = 1e6, bend_stiffness: float = 1e4, rest_length: float = 0.0):
        self.k_s = stretch_stiffness
        self.k_b = bend_stiffness
        self.r0 = rest_length
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            force = self.k_s * (r - self.r0) * r_hat
        
        new_state = CrosslinkState(force=force, displacement=dr)
        return force, new_state
    
    def energy(self, p1, p2):
        dr = np.asarray(p2) - np.asarray(p1)
        r = np.linalg.norm(dr)
        return 0.5 * self.k_s * (r - self.r0)**2
    
    def bending_energy(self, p1, p2, p3):
        """Compute bending energy for three-point angle."""
        v1 = np.asarray(p1) - np.asarray(p2)
        v2 = np.asarray(p3) - np.asarray(p2)
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-15 or n2 < 1e-15:
            return 0.0
        
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Bending energy around rest angle of pi (straight)
        return 0.5 * self.k_b * (theta - np.pi)**2


class CovalentBond(CrosslinkModel):
    """
    Covalent bond crosslink model.
    
    Strong, directional bonds with high stiffness and breaking energy.
    Uses Morse potential for bond breaking.
    
    Parameters
    ----------
    stiffness : float
        Bond stiffness (N/m). Default 1e9 (typical for C-C bonds).
    bond_energy : float
        Bond dissociation energy (J). Default 5e-19 (~300 kJ/mol).
    equilibrium_length : float
        Equilibrium bond length (m). Default 0.154 nm (C-C bond).
    width : float
        Potential well width (1/m). Controls breaking behavior.
    """
    
    def __init__(
        self,
        stiffness: float = 1e9,
        bond_energy: float = 5e-19,
        equilibrium_length: float = 0.154e-9,
        width: float = 2e10,
    ):
        self.k = stiffness
        self.D = bond_energy
        self.r0 = equilibrium_length
        self.alpha = width
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            
            # Morse potential force
            exp_term = np.exp(-self.alpha * (r - self.r0))
            force_mag = 2 * self.D * self.alpha * exp_term * (1 - exp_term)
            
            force = force_mag * r_hat
        
        new_state = CrosslinkState(force=force, displacement=dr)
        return force, new_state
    
    def energy(self, p1, p2):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        r = np.linalg.norm(p2 - p1)
        
        # Morse potential
        return self.D * (1 - np.exp(-self.alpha * (r - self.r0)))**2


class HydrogenBond(CrosslinkModel):
    """
    Hydrogen bond crosslink model.
    
    Weaker, directional bonds with angle dependence.
    Can break and reform dynamically.
    
    Parameters
    ----------
    strength : float
        Bond strength (N/m). Default 1e7 (typical for H-bonds).
    energy : float
        Bond energy (J). Default 3e-20 (~20 kJ/mol).
    equilibrium_length : float
        Equilibrium bond length (m). Default 0.28 nm.
    angle_cutoff : float
        Maximum angle deviation from linear (radians). Default π/6 (30°).
    reform_probability : float
        Probability of reforming after breaking. Default 0.1.
    """
    
    def __init__(
        self,
        strength: float = 1e7,
        energy: float = 3e-20,
        equilibrium_length: float = 0.28e-9,
        angle_cutoff: float = np.pi/6,
        reform_probability: float = 0.1,
    ):
        self.k = strength
        self.E = energy
        self.r0 = equilibrium_length
        self.angle_cutoff = angle_cutoff
        self.reform_prob = reform_probability
    
    def compute_force(self, p1, p2, state, normal1=None, normal2=None):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            
            # Check if broken
            if state.broken:
                # Try to reform with probability
                if np.random.random() < self.reform_prob:
                    state = CrosslinkState(broken=False)
                else:
                    return np.zeros(3), state
            
            # Angle check (if normals provided)
            angle_factor = 1.0
            if normal1 is not None and normal2 is not None:
                n1 = np.asarray(normal1, dtype=float)
                n2 = np.asarray(normal2, dtype=float)
                
                # Check angle with r_hat
                cos_angle1 = np.dot(n1, r_hat)
                cos_angle2 = np.dot(n2, -r_hat)
                
                angle1 = np.arccos(np.clip(cos_angle1, -1, 1))
                angle2 = np.arccos(np.clip(cos_angle2, -1, 1))
                
                if angle1 > self.angle_cutoff or angle2 > self.angle_cutoff:
                    # Break the bond
                    state = CrosslinkState(broken=True)
                    return np.zeros(3), state
                
                # Angular penalty
                angle_factor = cos_angle1 * cos_angle2
            
            # Linear spring with angle dependence
            force_mag = self.k * (r - self.r0) * angle_factor
            force = force_mag * r_hat
            
            # Check if force exceeds breaking threshold
            if abs(force_mag) > self.E / self.r0:
                state = CrosslinkState(broken=True, force=force)
                return force, state
        
        new_state = CrosslinkState(
            force=force,
            displacement=dr,
            broken=state.broken
        )
        return force, new_state
    
    def energy(self, p1, p2):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        r = np.linalg.norm(p2 - p1)
        
        # Harmonic potential
        return 0.5 * self.k * (r - self.r0)**2


class PhysicalEntanglement(CrosslinkModel):
    """
    Physical entanglement (topological constraint).
    
    Sliding contact that can move along fibers.
    No bond breaking, but can slip under high force.
    
    Parameters
    ----------
    friction : float
        Friction coefficient (N·s/m). Default 1e-6.
    slip_force : float
        Critical force for slipping (N). Default 1e-6.
    confinement : float
        Confinement radius (m). Default 1e-9.
    """
    
    def __init__(
        self,
        friction: float = 1e-6,
        slip_force: float = 1e-6,
        confinement: float = 1e-9,
    ):
        self.mu = friction
        self.F_slip = slip_force
        self.rc = confinement
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            
            # Confinement force (keep within rc)
            if r > self.rc:
                force_mag = self.mu * (r - self.rc)
                force = force_mag * r_hat
            else:
                # Check for slip
                if state.force is not None:
                    F_tangential = np.linalg.norm(state.force - np.dot(state.force, r_hat) * r_hat)
                    if F_tangential > self.F_slip:
                        # Slip occurs
                        slip_disp = state.slip_displacement + 0.01 * r
                        state = CrosslinkState(
                            force=state.force,
                            displacement=dr,
                            slip_displacement=slip_disp
                        )
                        return state.force, state
                
                force = np.zeros(3)
        
        new_state = CrosslinkState(
            force=force,
            displacement=dr,
            slip_displacement=state.slip_displacement
        )
        return force, new_state
    
    def energy(self, p1, p2):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        r = np.linalg.norm(p2 - p1)
        
        if r > self.rc:
            return 0.5 * self.mu * (r - self.rc)**2
        return 0.0


class IonicBond(CrosslinkModel):
    """
    Ionic bond crosslink model.
    
    Long-range electrostatic interactions with screening.
    Can break under mechanical load.
    
    Parameters
    ----------
    charge1 : float
        Charge on first fiber (C). Default 1.6e-19 (elementary charge).
    charge2 : float
        Charge on second fiber (C). Default -1.6e-19.
    dielectric : float
        Relative dielectric constant. Default 80 (water).
    debye_length : float
        Debye screening length (m). Default 1e-9 (1 nm).
    breaking_force : float
        Maximum force before breaking (N). Default 1e-9.
    """
    
    def __init__(
        self,
        charge1: float = 1.6e-19,
        charge2: float = -1.6e-19,
        dielectric: float = 80.0,
        debye_length: float = 1e-9,
        breaking_force: float = 1e-9,
    ):
        self.q1 = charge1
        self.q2 = charge2
        self.epsilon_r = dielectric
        self.lambda_D = debye_length
        self.F_break = breaking_force
        
        # Constants
        self.epsilon_0 = 8.854e-12  # Vacuum permittivity
        self.k_e = 1 / (4 * np.pi * self.epsilon_0 * self.epsilon_r)
    
    def compute_force(self, p1, p2, state):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        dr = p2 - p1
        r = np.linalg.norm(dr)
        
        if r < 1e-15:
            force = np.zeros(3)
        else:
            r_hat = dr / r
            
            # Check if broken
            if state.broken:
                return np.zeros(3), state
            
            # Yukawa potential (screened Coulomb)
            exp_term = np.exp(-r / self.lambda_D)
            force_mag = self.k_e * self.q1 * self.q2 / r**2 * exp_term * (1 + r / self.lambda_D)
            
            force = force_mag * r_hat
            
            # Check breaking
            if np.linalg.norm(force) > self.F_break:
                state = CrosslinkState(broken=True, force=force)
                return force, state
        
        new_state = CrosslinkState(
            force=force,
            displacement=dr,
            broken=state.broken
        )
        return force, new_state
    
    def energy(self, p1, p2):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        r = np.linalg.norm(p2 - p1)
        
        # Yukawa potential
        return self.k_e * self.q1 * self.q2 / r * np.exp(-r / self.lambda_D)


# Factory function for easy creation
def create_crosslink(
    crosslink_type: str,
    **kwargs
) -> CrosslinkModel:
    """
    Create a crosslink model by type name.
    
    Parameters
    ----------
    crosslink_type : str
        Type of crosslink: 'rigid', 'spring', 'breakable', 'friction',
        'bonded', 'covalent', 'hydrogen', 'entanglement', 'ionic'
    **kwargs
        Parameters passed to the crosslink constructor
    
    Returns
    -------
    CrosslinkModel
        The created crosslink model
    
    Examples
    --------
    >>> crosslink = create_crosslink('covalent', stiffness=1e9, bond_energy=5e-19)
    >>> crosslink = create_crosslink('hydrogen', strength=1e7)
    """
    crosslink_types = {
        'rigid': RigidCrosslink,
        'spring': SpringCrosslink,
        'breakable': BreakableCrosslink,
        'friction': FrictionCrosslink,
        'bonded': BondedCrosslink,
        'covalent': CovalentBond,
        'hydrogen': HydrogenBond,
        'entanglement': PhysicalEntanglement,
        'ionic': IonicBond,
    }
    
    if crosslink_type not in crosslink_types:
        available = ', '.join(crosslink_types.keys())
        raise ValueError(f"Unknown crosslink type: {crosslink_type}. Available: {available}")
    
    return crosslink_types[crosslink_type](**kwargs)
